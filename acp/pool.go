package acp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"
)

// sessionCounter assigns unique monotonic IDs across all pools.
var sessionCounter atomic.Uint64

// SpawnConfig holds the parameters needed to spawn new ACP sessions.
// Construct one and pass it to NewChatPool and NewGeneratePool.
type SpawnConfig struct {
	AgentArgs      []string
	Cwd            string
	SessionMode    string // ACP session mode ID to activate on spawn (empty = no mode set)
	Model          string
	Version        string
	Logger         *slog.Logger
	LogAgentOutput bool
}

// spawn creates an ACP session with a unique ID. The logger is pre-seeded
// with "session=<prefix-N>" so all log lines from this session carry the ID.
func (cfg SpawnConfig) spawn(ctx context.Context, prefix string) (*Session, string, error) {
	id := fmt.Sprintf("%s-%d", prefix, sessionCounter.Add(1))
	scoped := cfg
	scoped.Logger = cfg.Logger.With("session", id)
	sess, err := Spawn(ctx, scoped, cfg.Model)
	if err != nil {
		return nil, "", fmt.Errorf("spawning %s session: %w", prefix, err)
	}
	return sess, id, nil
}

// fingerprintMessages returns a SHA-256 fingerprint of an ordered message slice.
// The fingerprint is used to match incoming chat history to existing sessions.
// Returns "" for an empty slice.
func fingerprintMessages(messages []Message) string {
	if len(messages) == 0 {
		return ""
	}
	h := sha256.New()
	for _, m := range messages {
		h.Write([]byte(m.Role))
		h.Write([]byte{0}) // null separator prevents role+content collisions
		h.Write([]byte(m.Content))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ── ChatPool ──────────────────────────────────────────────────────────────────

// chatEntry is one slot in the chat pool: a session plus its conversation state.
type chatEntry struct {
	id       string
	log      *slog.Logger // pre-seeded with "session=<id>"
	sess     *Session
	history  []Message // full accumulated conversation sent to ACP so far
	fp       string    // SHA-256 fingerprint of history
	lastUsed time.Time
	inUse    bool
}

// ChatPool manages a bounded set of ACP sessions for /api/chat.
//
// Two distinct slices of entries are maintained:
//
//   - fresh: pre-warmed empty sessions ready for brand-new conversations.
//     Popped LIFO (most recently warmed first) so Acquire never waits for a
//     spawn on the happy path. freshTarget controls the steady-state size.
//
//   - sessions: sessions that have sent at least one prompt. Maintained in LRU
//     order (index 0 = oldest) and matched to incoming requests by history
//     fingerprint.
//
// When the pool needs capacity, sessions are evicted before fresh entries so
// the pre-warmed asset is preserved as long as possible.
// An optional TTL reaps idle sessions (not fresh) that have been idle too long.
type ChatPool struct {
	mu            sync.Mutex
	fresh         []*chatEntry // pre-warmed, empty history; LIFO pop for new conversations
	sessions      []*chatEntry // LRU-ordered (index 0 = oldest); may carry any history
	maxSize       int          // total cap: total() must never exceed this
	freshTarget   int          // desired steady-state len(fresh)
	pendingSpawn  int          // Acquire-path spawns in flight (counted against maxSize)
	pendingWarmup int          // spawnAndWarm-path spawns in flight
	ttl           time.Duration
	ctx           context.Context
	cfg           SpawnConfig
}

// NewChatPool creates a ChatPool. freshTarget pre-warmed empty sessions are
// spawned immediately in the background. If ttl > 0, a background goroutine
// reaps sessions idle longer than ttl.
func NewChatPool(ctx context.Context, cfg SpawnConfig, maxSize, freshTarget int, ttl time.Duration) *ChatPool {
	p := &ChatPool{
		maxSize:     maxSize,
		freshTarget: freshTarget,
		ttl:         ttl,
		ctx:         ctx,
		cfg:         cfg,
	}
	if ttl > 0 {
		go p.reapLoop()
	}
	if freshTarget > 0 {
		p.pendingWarmup = freshTarget
		for range freshTarget {
			go p.spawnAndWarm()
		}
	}
	return p
}

// total returns the count of all sessions and pending spawns. Must hold p.mu.
func (p *ChatPool) total() int {
	return len(p.fresh) + len(p.sessions) + p.pendingSpawn + p.pendingWarmup
}

// warmupNeed returns how many warm sessions to spawn to meet freshTarget.
// Must hold p.mu.
func (p *ChatPool) warmupNeed() int {
	need := p.freshTarget - len(p.fresh) - p.pendingWarmup
	if need <= 0 {
		return 0
	}
	if avail := p.maxSize - p.total(); need > avail {
		need = avail
	}
	if need < 0 {
		return 0
	}
	return need
}

// Acquire finds the best idle session for messages, or allocates a new one.
//
// Three paths, in priority order:
//  1. History match: the session in sessions whose accumulated history is the
//     longest prefix of messages — minimises new messages sent to ACP.
//  2. Warm: a pre-warmed empty session is available in fresh — O(1), no spawn.
//     The fresh queue is replenished immediately so the next caller doesn't wait.
//  3. Cold: spawn a new session. If the pool is at capacity, the LRU idle
//     session in sessions is evicted first; failing that, the oldest fresh
//     session. Returns an error only when every session is in use.
func (p *ChatPool) Acquire(messages []Message) (*Session, error) {
	p.mu.Lock()

	// 1. History match in sessions.
	var best *chatEntry
	for _, e := range p.sessions {
		if e.inUse || len(e.history) > len(messages) {
			continue
		}
		if fingerprintMessages(messages[:len(e.history)]) != e.fp {
			continue
		}
		if best == nil || len(e.history) > len(best.history) {
			best = e
		}
	}
	if best != nil {
		best.inUse = true
		best.lastUsed = time.Now()
		best.log.Debug("chat session matched",
			"history_len", len(best.history),
			"incoming_len", len(messages),
			"new_messages", len(messages)-len(best.history),
		)
		p.mu.Unlock()
		return best.sess, nil
	}

	// 2. Warm path: pop a pre-warmed session and immediately start replenishing.
	if len(p.fresh) > 0 {
		e := p.fresh[len(p.fresh)-1]
		p.fresh = p.fresh[:len(p.fresh)-1]
		e.inUse = true
		e.lastUsed = time.Now()
		p.sessions = append(p.sessions, e)
		e.log.Debug("chat session acquired (warm)",
			"incoming_len", len(messages),
			"fresh_remaining", len(p.fresh),
		)
		toSpawn := p.warmupNeed()
		p.pendingWarmup += toSpawn
		p.mu.Unlock()
		for range toSpawn {
			go p.spawnAndWarm()
		}
		return e.sess, nil
	}

	p.cfg.Logger.Debug("no warm chat session, spawning cold",
		"sessions", len(p.sessions),
		"max_size", p.maxSize,
		"pending_spawn", p.pendingSpawn,
	)

	// 3. Cold path: need capacity for a new spawn.
	if p.total() >= p.maxSize {
		if !p.evictLRU() {
			p.mu.Unlock()
			return nil, fmt.Errorf("chat pool exhausted: all %d sessions are busy", p.maxSize)
		}
	}

	p.pendingSpawn++
	p.mu.Unlock()

	// Spawn without holding the lock.
	sess, id, err := p.cfg.spawn(p.ctx, "chat")

	p.mu.Lock()
	defer p.mu.Unlock()
	p.pendingSpawn--

	if err != nil {
		return nil, err
	}

	// Re-check capacity: a concurrent spawnAndWarm may have filled the pool.
	if p.total() >= p.maxSize {
		if !p.evictLRU() {
			sess.Close()
			return nil, fmt.Errorf("chat pool exhausted (race during spawn)")
		}
	}

	entry := &chatEntry{
		id:       id,
		log:      p.cfg.Logger.With("session", id),
		sess:     sess,
		lastUsed: time.Now(),
		inUse:    true,
	}
	p.sessions = append(p.sessions, entry)
	entry.log.Debug("chat session added (cold)", "sessions", len(p.sessions))
	go p.monitorDeath(entry)
	return sess, nil
}

// Release marks a session idle after a request completes.
// Pass the full updated message history on success, or nil on error to leave
// the session's history unchanged. After release, spawnAndWarm may spawn new
// sessions to maintain freshTarget.
func (p *ChatPool) Release(sess *Session, history []Message) {
	p.mu.Lock()
	if e := p.findEntry(sess); e != nil {
		if history != nil {
			e.history = history
			e.fp = fingerprintMessages(history)
			e.log.Debug("chat session released", "history_len", len(history))
		} else {
			e.log.Debug("chat session released (history unchanged)")
		}
		e.inUse = false
		e.lastUsed = time.Now()
		p.sortLRU()
	}
	toSpawn := p.warmupNeed()
	p.pendingWarmup += toSpawn
	p.mu.Unlock()

	for range toSpawn {
		go p.spawnAndWarm()
	}
}

// evictLRU kills one idle session and frees its pool slot.
// Returns false only when every session is in use (nothing evictable).
//
// Prefers evicting from sessions (LRU order) before fresh, so the pre-warmed
// asset is consumed only as a last resort. Must hold p.mu.
func (p *ChatPool) evictLRU() bool {
	// sessions[0] is the LRU entry.
	for i, e := range p.sessions {
		if !e.inUse {
			e.log.Debug("evicting LRU chat session",
				"idle_for", time.Since(e.lastUsed).Round(time.Millisecond),
				"history_len", len(e.history),
			)
			e.sess.Close()
			p.sessions = append(p.sessions[:i], p.sessions[i+1:]...)
			return true
		}
	}
	// Fall back: evict the oldest fresh session (index 0 = appended first).
	if len(p.fresh) > 0 {
		e := p.fresh[0]
		p.fresh = p.fresh[1:]
		e.log.Debug("evicting oldest fresh session (no idle session available)")
		e.sess.Close()
		return true
	}
	p.cfg.Logger.Debug("LRU eviction skipped: all sessions are busy")
	return false
}

// sortLRU sorts sessions by lastUsed ascending so index 0 is the oldest.
// Insertion sort — O(n²) but fine for small slices (maxSize ≈ 10).
// Must hold p.mu.
func (p *ChatPool) sortLRU() {
	s := p.sessions
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j].lastUsed.Before(s[j-1].lastUsed); j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// monitorDeath watches for process exit and removes the entry from whichever
// slice holds it. If the exit is unexpected (not an intentional Close), it logs
// at error level. If the entry is a fresh session, a compensating warmup spawn
// is scheduled so the fresh target is maintained.
func (p *ChatPool) monitorDeath(entry *chatEntry) {
	<-entry.sess.Done()
	p.mu.Lock()

	for i, e := range p.sessions {
		if e == entry {
			p.sessions = append(p.sessions[:i], p.sessions[i+1:]...)
			entry.log.Error("chat agent process exited unexpectedly",
				"sessions_remaining", len(p.sessions))
			p.mu.Unlock()
			return
		}
	}
	for i, e := range p.fresh {
		if e == entry {
			p.fresh = append(p.fresh[:i], p.fresh[i+1:]...)
			entry.log.Error("pre-warmed chat agent process exited unexpectedly",
				"fresh_remaining", len(p.fresh))
			toSpawn := p.warmupNeed()
			p.pendingWarmup += toSpawn
			p.mu.Unlock()
			for range toSpawn {
				go p.spawnAndWarm()
			}
			return
		}
	}
	// Not found: already removed by evictLRU or reapExpired — intentional close.
	p.mu.Unlock()
}

// spawnAndWarm spawns a pre-warmed empty session and adds it to fresh.
// pendingWarmup must be incremented by the caller under the lock before calling.
// It is always decremented here once the fate of the session is decided.
func (p *ChatPool) spawnAndWarm() {
	sess, id, err := p.cfg.spawn(p.ctx, "chat")

	p.mu.Lock()
	defer p.mu.Unlock()
	p.pendingWarmup-- // always decrement, success or failure

	if err != nil {
		p.cfg.Logger.Error("failed to pre-warm chat session", "error", err)
		return
	}

	entry := &chatEntry{
		id:       id,
		log:      p.cfg.Logger.With("session", id),
		sess:     sess,
		lastUsed: time.Now(),
	}
	// After decrementing pendingWarmup, check total capacity again.
	if p.total() < p.maxSize {
		p.fresh = append(p.fresh, entry)
		entry.log.Debug("pre-warmed chat session added", "fresh_count", len(p.fresh))
		go p.monitorDeath(entry)
	} else {
		entry.log.Debug("discarding pre-warmed chat session (pool full)")
		entry.sess.Close()
	}
}

// reapLoop ticks at ttl/2 and calls reapExpired each time.
func (p *ChatPool) reapLoop() {
	ticker := time.NewTicker(p.ttl / 2)
	defer ticker.Stop()
	for {
		select {
		case <-p.ctx.Done():
			return
		case <-ticker.C:
			p.reapExpired()
		}
	}
}

// reapExpired closes sessions (not fresh) that have been idle longer than p.ttl.
// It collects sessions to close while holding the lock, then closes them after
// releasing it — so cmd.Wait() never blocks while the mutex is held.
// Fresh sessions are intentionally idle and are not subject to TTL reaping.
func (p *ChatPool) reapExpired() {
	p.mu.Lock()
	now := time.Now()
	var toClose []*Session
	surviving := p.sessions[:0] // reuse backing array
	for _, e := range p.sessions {
		if !e.inUse && now.Sub(e.lastUsed) > p.ttl {
			e.log.Info("TTL-evicting idle chat session",
				"idle_for", now.Sub(e.lastUsed).Round(time.Second),
				"history_len", len(e.history),
			)
			toClose = append(toClose, e.sess)
		} else {
			surviving = append(surviving, e)
		}
	}
	p.sessions = surviving
	p.mu.Unlock()

	for _, sess := range toClose {
		sess.Close()
	}
}

// findEntry returns the pool entry for sess within sessions. Must hold p.mu.
// At Release time the entry is always in sessions (it was moved there by Acquire).
func (p *ChatPool) findEntry(sess *Session) *chatEntry {
	for _, e := range p.sessions {
		if e.sess == sess {
			return e
		}
	}
	return nil
}

// ── GeneratePool ──────────────────────────────────────────────────────────────

// GeneratePool manages single-use ACP sessions for /api/generate.
//
// Each request gets a fresh session; after use the session is killed so its
// context cannot bleed into a future request. Up to maxSize requests may be
// in-flight simultaneously. When the pool is full, callers block (FIFO) until
// a slot frees up. idleTarget controls the desired number of idle sessions to
// keep ready — they are spawned at startup and replenished after each release.
type GeneratePool struct {
	mu         sync.Mutex
	idle       []*Session     // sessions ready for immediate use
	inFlight   int            // sessions currently serving a request
	maxSize    int
	idleTarget int
	waiters    []chan *Session // FIFO queue of blocked callers
	ctx        context.Context
	cfg        SpawnConfig
}

// NewGeneratePool creates a GeneratePool. If idleTarget > 0, that many
// sessions are spawned immediately into the idle pool in the background.
func NewGeneratePool(ctx context.Context, cfg SpawnConfig, maxSize, idleTarget int) *GeneratePool {
	p := &GeneratePool{
		maxSize:    maxSize,
		idleTarget: idleTarget,
		ctx:        ctx,
		cfg:        cfg,
	}
	for range idleTarget {
		go p.spawnAndWarm()
	}
	return p
}

// Acquire returns a session for a generate request. Three paths:
//
//   - Warm: an idle pre-warmed session is available → return it immediately.
//   - Cold: the pool has capacity → spawn a new session inline.
//   - Full: all slots are in-flight → block until one completes or ctx is cancelled.
func (p *GeneratePool) Acquire(ctx context.Context) (*Session, error) {
	p.mu.Lock()

	// Warm path.
	if len(p.idle) > 0 {
		sess := p.idle[len(p.idle)-1]
		p.idle = p.idle[:len(p.idle)-1]
		p.inFlight++
		inFlight := p.inFlight
		p.mu.Unlock()
		p.cfg.Logger.Debug("generate session acquired (warm)", "in_flight", inFlight)
		return sess, nil
	}

	// Cold path.
	if p.inFlight < p.maxSize {
		p.inFlight++
		inFlight := p.inFlight
		p.mu.Unlock()
		p.cfg.Logger.Debug("generate session spawning (cold start)", "in_flight", inFlight)
		sess, _, err := p.cfg.spawn(ctx, "gen")
		if err != nil {
			p.mu.Lock()
			p.inFlight--
			p.mu.Unlock()
			return nil, err
		}
		p.cfg.Logger.Debug("generate session acquired (cold start)", "in_flight", inFlight)
		return sess, nil
	}

	// Full path: enqueue a waiter channel.
	waiter := make(chan *Session, 1)
	p.waiters = append(p.waiters, waiter)
	depth := len(p.waiters)
	p.mu.Unlock()
	p.cfg.Logger.Debug("generate pool exhausted, request queued", "queue_depth", depth)

	select {
	case sess := <-waiter:
		if sess == nil {
			return nil, fmt.Errorf("generate pool: session spawn failed while queued")
		}
		p.cfg.Logger.Debug("generate session acquired (from queue)")
		return sess, nil
	case <-ctx.Done():
		// Remove our waiter from the queue so Release doesn't try to deliver to us.
		p.mu.Lock()
		for i, w := range p.waiters {
			if w == waiter {
				p.waiters = append(p.waiters[:i], p.waiters[i+1:]...)
				break
			}
		}
		p.mu.Unlock()
		p.cfg.Logger.Debug("generate request cancelled while queued")
		return nil, fmt.Errorf("generate pool: %w", ctx.Err())
	}
}

// Release kills the used session and refills the idle pool.
// If callers are queued, one replacement is delivered directly to the first waiter.
// Otherwise, up to p.idleTarget sessions are spawned to restore the idle pool.
func (p *GeneratePool) Release(sess *Session) {
	p.cfg.Logger.Debug("generate session released, killing process")
	sess.Close()

	p.mu.Lock()
	p.inFlight--

	if len(p.waiters) > 0 {
		waiter := p.waiters[0]
		p.waiters = p.waiters[1:]
		remaining := len(p.waiters)
		p.inFlight++
		p.mu.Unlock()
		p.cfg.Logger.Debug("spawning generate session for queued waiter",
			"queue_remaining", remaining)
		go p.spawnAndDeliver(waiter)
		return
	}
	p.mu.Unlock()

	for range p.idleTarget {
		go p.spawnAndWarm()
	}
}

// spawnAndDeliver spawns a session and sends it to a waiting caller.
// On spawn failure it sends nil so the caller can return an error.
func (p *GeneratePool) spawnAndDeliver(waiter chan *Session) {
	sess, _, err := p.cfg.spawn(p.ctx, "gen")
	if err != nil {
		p.cfg.Logger.Error("failed to spawn generate session for queued waiter", "error", err)
		p.mu.Lock()
		p.inFlight--
		p.mu.Unlock()
		waiter <- nil
		return
	}
	p.cfg.Logger.Debug("generate session delivered to queued waiter")
	waiter <- sess
}

// spawnAndWarm spawns a session for the idle pool.
// If a waiter arrives while spawning, the session is delivered to them instead.
func (p *GeneratePool) spawnAndWarm() {
	sess, _, err := p.cfg.spawn(p.ctx, "gen")
	if err != nil {
		p.cfg.Logger.Error("failed to spawn idle generate session", "error", err)
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	// A new waiter may have arrived while we were spawning. Deliver to them.
	if len(p.waiters) > 0 {
		waiter := p.waiters[0]
		p.waiters = p.waiters[1:]
		p.inFlight++
		p.cfg.Logger.Debug("idle generate session redirected to late waiter")
		go func() { waiter <- sess }()
		return
	}

	p.idle = append(p.idle, sess)
	p.cfg.Logger.Debug("generate session added to idle pool", "idle_count", len(p.idle))
}
