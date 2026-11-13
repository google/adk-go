// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package redis

import (
	"cmp"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	goredis "github.com/redis/go-redis/v9"

	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
)

// maxTxRetries bounds how often Create and AppendEvent retry a transaction
// that lost a race on a watched key. Each lost race means another writer
// committed, so the bound is reached only under sustained contention.
const maxTxRetries = 100

// Service is a Redis-backed implementation of [session.Service] and
// [session.UserStateReader].
//
// Service is safe for concurrent use by multiple goroutines. A zero Service
// is invalid; construct one with [NewService].
type Service struct {
	client    *goredis.Client
	keyPrefix string
	// ttl is the expiration passed to every SET; zero writes no expiration.
	ttl   time.Duration
	close func() error

	sessionLocksMu sync.Mutex
	// sessionLocks holds the lock of every session an AppendEvent call holds
	// or waits for. Guarded by sessionLocksMu; nil until the first append.
	sessionLocks map[sessionLockKey]*sessionLock
}

// sessionLockKey identifies a session in Service.sessionLocks.
type sessionLockKey struct {
	appName, userID, sessionID string
}

// sessionLock serializes appends to one session. refs counts the calls that
// hold or wait for mu, and is guarded by Service.sessionLocksMu.
type sessionLock struct {
	mu   sync.Mutex
	refs int
}

var (
	_ session.Service         = (*Service)(nil)
	_ session.UserStateReader = (*Service)(nil)
)

// NewService creates a new Redis-backed session [Service] from cfg.
//
// NewService does not dial Redis; connections are established lazily on the
// first operation. It returns an error if cfg.Client is nil and cfg.URI cannot
// be parsed, if cfg.Client is nil and cfg.URI is empty and cfg.Port or cfg.DB
// is negative, or if cfg.TTL is positive and not a whole number of seconds.
// Any error returned when parsing cfg.URI redacts credentials and never echoes
// the raw URI string.
func NewService(cfg Config) (*Service, error) {
	if cfg.TTL > 0 && cfg.TTL%time.Second != 0 {
		return nil, fmt.Errorf("redis: TTL must be a whole number of seconds, got %v", cfg.TTL)
	}
	client, owned, err := newClient(cfg)
	if err != nil {
		return nil, err
	}
	return &Service{
		client:    client,
		keyPrefix: cmp.Or(cfg.KeyPrefix, DefaultKeyPrefix),
		ttl:       resolveTTL(cfg.TTL),
		close: sync.OnceValue(func() error {
			if !owned {
				return nil
			}
			return client.Close()
		}),
	}, nil
}

// newClient returns the client cfg selects and whether the service owns it.
func newClient(cfg Config) (*goredis.Client, bool, error) {
	if cfg.Client != nil {
		return cfg.Client, false, nil
	}
	if cfg.URI != "" {
		opts, err := goredis.ParseURL(cfg.URI)
		if err != nil {
			// The parse error quotes the URI, and with it any password or
			// token the URI carries, so it is not wrapped.
			return nil, false, errors.New("redis: invalid connection URI; want redis://[[user]:password@]host[:port][/db] or rediss://")
		}
		return goredis.NewClient(opts), true, nil
	}
	if cfg.Port < 0 {
		return nil, false, fmt.Errorf("redis: Port must not be negative, got %d", cfg.Port)
	}
	if cfg.DB < 0 {
		return nil, false, fmt.Errorf("redis: DB must not be negative, got %d", cfg.DB)
	}
	host := cmp.Or(cfg.Host, DefaultHost)
	opts := &goredis.Options{
		Addr:     net.JoinHostPort(host, strconv.Itoa(cmp.Or(cfg.Port, DefaultPort))),
		Password: cfg.Password,
		DB:       cfg.DB,
	}
	if cfg.SSL {
		opts.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12, ServerName: host}
	}
	return goredis.NewClient(opts), true, nil
}

// resolveTTL maps Config.TTL to the expiration go-redis takes. A negative
// value must not reach go-redis, which reads -1 as KEEPTTL.
func resolveTTL(ttl time.Duration) time.Duration {
	switch {
	case ttl == 0:
		return DefaultTTL
	case ttl < 0:
		return 0
	default:
		return ttl
	}
}

func (s *Service) sessionKey(appName, userID, sessionID string) string {
	return s.keyPrefix + appName + ":" + userID + ":" + sessionID
}

func (s *Service) userStateKey(appName, userID string) string {
	return s.keyPrefix + "user_state:" + appName + ":" + userID
}

func (s *Service) appStateKey(appName string) string {
	return s.keyPrefix + "app_state:" + appName
}

// Create creates a new session in Redis and applies any initial "app:" and
// "user:" state deltas.
//
// req.AppName is required. An empty req.UserID is the user "", as in
// adk-python. An empty req.SessionID gets a generated ID; any other value is
// used verbatim. The session key, and the
// app and user state keys that req.State writes to, are written in one
// transaction, so when the session already exists Create returns an
// "already exists" error and writes nothing. The returned session holds the
// merged app, user and session state and, unlike the stored session, any
// "temp:" keys of req.State, as adk-python's RedisSessionService returns them.
func (s *Service) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if req == nil {
		return nil, errors.New("redis: create request is nil")
	}
	if req.AppName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}
	sid := req.SessionID
	if sid == "" {
		sid = platform.NewUUID(ctx)
	}
	now := platform.Now(ctx)
	appDelta, userDelta, sessionState := extractStateDeltas(req.State)
	sessionJSON, err := encodeSession(&storedSession{
		ID:             sid,
		AppName:        req.AppName,
		UserID:         req.UserID,
		State:          sessionState,
		Events:         []*session.Event{},
		LastUpdateTime: now,
	})
	if err != nil {
		return nil, err
	}

	sessionKey, appKey, userKey := s.sessionKey(req.AppName, req.UserID, sid), s.appStateKey(req.AppName), s.userStateKey(req.AppName, req.UserID)
	var appState, userState map[string]any
	err = s.watch(ctx, func(tx *goredis.Tx) error {
		err := tx.Get(ctx, sessionKey).Err()
		if err == nil {
			return fmt.Errorf("session %s already exists", sid)
		}
		if !errors.Is(err, goredis.Nil) {
			return fmt.Errorf("redis: read session %q: %w", sid, err)
		}
		if appState, err = readState(ctx, tx, appKey); err != nil {
			return err
		}
		if userState, err = readState(ctx, tx, userKey); err != nil {
			return err
		}
		maps.Copy(appState, appDelta)
		maps.Copy(userState, userDelta)
		writes := []write{{sessionKey, sessionJSON}}
		if writes, err = appendStateWrite(writes, appKey, appState, len(appDelta) > 0); err != nil {
			return err
		}
		if writes, err = appendStateWrite(writes, userKey, userState, len(userDelta) > 0); err != nil {
			return err
		}
		return s.commit(ctx, tx, writes)
	}, watchKeys(sessionKey, appKey, userKey, len(appDelta) > 0, len(userDelta) > 0)...)
	if err != nil {
		return nil, err
	}

	merged := mergeStates(appState, userState, sessionState)
	for k, v := range req.State {
		if strings.HasPrefix(k, session.KeyPrefixTemp) {
			merged[k] = v
		}
	}
	return &session.CreateResponse{Session: &localSession{
		appName:   req.AppName,
		userID:    req.UserID,
		sessionID: sid,
		state:     merged,
		updatedAt: now,
	}}, nil
}

// Get retrieves a session and its merged "app:", "user:", and session state
// from Redis. If the session does not exist or has expired, Get returns an
// error wrapping [session.ErrNotFound].
//
// req.AppName is required. An empty req.UserID is the user "", and an empty
// req.SessionID names no session, so Get returns [session.ErrNotFound] for
// it, as in adk-python. req.NumRecentEvents must not be negative, even when
// req.OmitEvents is true. When req.OmitEvents is true, the returned session
// has no events. Otherwise a positive NumRecentEvents keeps the last
// NumRecentEvents stored events, and a non-zero After then keeps the events
// whose timestamp is not before it. Get never refreshes a key's TTL.
//
// The returned session's app name, user ID and ID are those of the stored
// record, and its state merges the app and user state of req.AppName and
// req.UserID, as in adk-python. The record's identity differs from req's
// only when another app and user share the session's key, whose parts are
// joined with ":" unescaped.
func (s *Service) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	if req == nil {
		return nil, errors.New("redis: get request is nil")
	}
	if req.AppName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}
	if req.NumRecentEvents < 0 {
		return nil, fmt.Errorf("num_recent_events must not be negative, got %d", req.NumRecentEvents)
	}
	vals, err := s.client.MGet(ctx,
		s.sessionKey(req.AppName, req.UserID, req.SessionID),
		s.appStateKey(req.AppName),
		s.userStateKey(req.AppName, req.UserID),
	).Result()
	if err != nil {
		return nil, fmt.Errorf("redis: read session %q: %w", req.SessionID, err)
	}
	if vals[0] == nil {
		return nil, fmt.Errorf("%w: %q", session.ErrNotFound, req.SessionID)
	}
	stored, err := decodeSession(vals[0])
	if err != nil {
		return nil, err
	}
	appState, err := decodeState(vals[1])
	if err != nil {
		return nil, err
	}
	userState, err := decodeState(vals[2])
	if err != nil {
		return nil, err
	}

	// The events are stored in the session's value, so omitting them saves
	// no read.
	evs := stored.Events
	if req.OmitEvents {
		evs = nil
	} else if req.NumRecentEvents > 0 {
		evs = evs[max(len(evs)-req.NumRecentEvents, 0):]
	}
	filtered := make([]*session.Event, 0, len(evs))
	for _, ev := range evs {
		if !req.After.IsZero() && ev.Timestamp.Before(req.After) {
			continue
		}
		filtered = append(filtered, ev)
	}
	return &session.GetResponse{Session: &localSession{
		appName:   stored.AppName,
		userID:    stored.UserID,
		sessionID: stored.ID,
		events:    filtered,
		state:     mergeStates(appState, userState, stored.State),
		updatedAt: stored.LastUpdateTime,
	}}, nil
}

// List returns all sessions in Redis matching req.AppName and, when
// req.UserID is non-empty or req.FilterEmptyUserID is true, req.UserID,
// ordered by LastUpdateTime ascending with ties broken by UserID and then
// session ID. Listed sessions carry merged state and no events.
//
// req.AppName is required. Glob characters in req.AppName and req.UserID
// match literally. A key whose value does not decode as a session, or that
// belongs to another app or user (an ID can contain ":"), is skipped.
func (s *Service) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req == nil {
		return nil, errors.New("redis: list request is nil")
	}
	if req.AppName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}
	oneUser := req.UserID != "" || req.FilterEmptyUserID
	pattern := s.keyPrefix + escapeGlob(req.AppName) + ":"
	if oneUser {
		pattern += escapeGlob(req.UserID) + ":"
	}
	keys, err := s.scan(ctx, pattern+"*")
	if err != nil {
		return nil, err
	}
	sessions := make([]session.Session, 0, len(keys))
	if len(keys) == 0 {
		return &session.ListResponse{Sessions: sessions}, nil
	}
	vals, err := s.client.MGet(ctx, keys...).Result()
	if err != nil {
		return nil, fmt.Errorf("redis: read %d sessions: %w", len(keys), err)
	}
	appState, err := s.getState(ctx, s.appStateKey(req.AppName))
	if err != nil {
		return nil, err
	}
	userStates := map[string]map[string]any{}
	for _, v := range vals {
		stored, err := decodeSession(v)
		if err != nil || stored.ID == "" {
			// Expired since the scan, or not a session: skipped, as
			// adk-python skips a key whose JSON does not parse.
			continue
		}
		if stored.AppName != req.AppName || (oneUser && stored.UserID != req.UserID) {
			continue
		}
		userState, ok := userStates[stored.UserID]
		if !ok {
			if userState, err = s.getState(ctx, s.userStateKey(req.AppName, stored.UserID)); err != nil {
				return nil, err
			}
			userStates[stored.UserID] = userState
		}
		sessions = append(sessions, &localSession{
			appName:   stored.AppName,
			userID:    stored.UserID,
			sessionID: stored.ID,
			state:     mergeStates(appState, userState, stored.State),
			updatedAt: stored.LastUpdateTime,
		})
	}
	slices.SortFunc(sessions, func(a, b session.Session) int {
		return cmp.Or(
			a.LastUpdateTime().Compare(b.LastUpdateTime()),
			strings.Compare(a.UserID(), b.UserID()),
			strings.Compare(a.ID(), b.ID()),
		)
	})
	return &session.ListResponse{Sessions: sessions}, nil
}

// scan returns the keys matching pattern, each once: SCAN can return a key
// more than once.
func (s *Service) scan(ctx context.Context, pattern string) ([]string, error) {
	var keys []string
	it := s.client.Scan(ctx, 0, pattern, 0).Iterator()
	for it.Next(ctx) {
		keys = append(keys, it.Val())
	}
	if err := it.Err(); err != nil {
		return nil, fmt.Errorf("redis: scan sessions: %w", err)
	}
	slices.Sort(keys)
	return slices.Compact(keys), nil
}

// Delete removes the session key from Redis. Deleting a session that does not
// exist is a no-op and returns nil. App and user state are not removed.
//
// req.AppName is required. An empty req.UserID is the user "", and an empty
// req.SessionID names no session, so Delete deletes nothing for it.
func (s *Service) Delete(ctx context.Context, req *session.DeleteRequest) error {
	if req == nil {
		return errors.New("redis: delete request is nil")
	}
	if req.AppName == "" {
		return fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}
	if err := s.client.Del(ctx, s.sessionKey(req.AppName, req.UserID, req.SessionID)).Err(); err != nil {
		return fmt.Errorf("redis: delete session %q: %w", req.SessionID, err)
	}
	return nil
}

// AppendEvent appends event to curSession and persists the event and state
// deltas in Redis. curSession may be any [session.Session] implementation
// (identified by AppName, UserID, and ID). If the session does not exist or
// has expired in Redis, AppendEvent returns an error wrapping
// [session.ErrNotFound] without writing any state.
//
// A partial event is checked against storage and then ignored: nothing is
// stored and curSession is unchanged. Otherwise an event without an ID gets
// one, in place, and the stored session, its events and the app and user
// state the event's delta writes to are updated in one transaction, retried
// while another writer commits to the same keys first. The stored event and
// the session's update time take the event with its "temp:" keys removed and
// its Timestamp. After the commit, a session value this service, the
// in-memory or database service, or [session.CloneSession] made receives the
// event, the whole state delta with its "temp:" keys, and the update time; any
// other value only receives the state delta through its State, when State is
// non-nil, and the first error its Set returns is returned with the event
// already stored.
func (s *Service) AppendEvent(ctx context.Context, curSession session.Session, event *session.Event) error {
	if curSession == nil {
		return errors.New("session is nil")
	}
	if event == nil {
		return errors.New("event is nil")
	}
	// A request this service could not read back is refused, as the
	// in-memory and database services refuse it.
	if err := checkRequestedAuthConfigs(event.Actions.RequestedAuthConfigs); err != nil {
		return fmt.Errorf("cannot append event: %w", err)
	}
	appName, userID, sid := curSession.AppName(), curSession.UserID(), curSession.ID()
	sessionKey, appKey, userKey := s.sessionKey(appName, userID, sid), s.appStateKey(appName), s.userStateKey(appName, userID)
	notFound := fmt.Errorf("%w: %q, cannot apply event", session.ErrNotFound, sid)

	if event.Partial {
		n, err := s.client.Exists(ctx, sessionKey).Result()
		if err != nil {
			return fmt.Errorf("redis: read session %q: %w", sid, err)
		}
		if n == 0 {
			return notFound
		}
		return nil
	}

	if event.ID == "" {
		event.ID = platform.NewUUID(ctx)
	}
	trimmed := trimTempDeltaState(event)
	appDelta, userDelta, sessionDelta := extractStateDeltas(event.Actions.StateDelta)
	// The transaction orders appends in Redis; the lock makes curSession
	// receive them in that order too.
	unlock := s.lockSession(appName, userID, sid)
	defer unlock()
	err := s.watch(ctx, func(tx *goredis.Tx) error {
		// The stored session is read inside the transaction, so an append
		// through a filtered view keeps the full history, and a session
		// deleted or expired since the caller read it is not recreated.
		raw, err := tx.Get(ctx, sessionKey).Result()
		if errors.Is(err, goredis.Nil) {
			return notFound
		}
		if err != nil {
			return fmt.Errorf("redis: read session %q: %w", sid, err)
		}
		stored, err := decodeSession(raw)
		if err != nil {
			return err
		}
		if stored.State == nil {
			stored.State = map[string]any{}
		}
		maps.Copy(stored.State, sessionDelta)
		stored.Events = append(stored.Events, trimmed)
		stored.LastUpdateTime = event.Timestamp
		sessionJSON, err := encodeSession(stored)
		if err != nil {
			return err
		}
		writes := []write{{sessionKey, sessionJSON}}
		if writes, err = s.mergeStateWrite(ctx, tx, writes, appKey, appDelta); err != nil {
			return err
		}
		if writes, err = s.mergeStateWrite(ctx, tx, writes, userKey, userDelta); err != nil {
			return err
		}
		return s.commit(ctx, tx, writes)
	}, watchKeys(sessionKey, appKey, userKey, len(appDelta) > 0, len(userDelta) > 0)...)
	if err != nil {
		return err
	}
	if err := commitToHeldSession(curSession, trimmed, event.Actions.StateDelta, event.Timestamp); err != nil {
		return fmt.Errorf("event %q is stored in session %q, but the session value was not updated: %w", event.ID, sid, err)
	}
	return nil
}

// lockSession waits until no other AppendEvent call of s holds the session
// (appName, userID, sessionID), and returns the function that releases it.
// Appends through other Service values or other processes are ordered by
// the transaction alone.
func (s *Service) lockSession(appName, userID, sessionID string) func() {
	key := sessionLockKey{appName: appName, userID: userID, sessionID: sessionID}
	s.sessionLocksMu.Lock()
	if s.sessionLocks == nil {
		s.sessionLocks = make(map[sessionLockKey]*sessionLock)
	}
	l := s.sessionLocks[key]
	if l == nil {
		l = &sessionLock{}
		s.sessionLocks[key] = l
	}
	l.refs++
	s.sessionLocksMu.Unlock()

	l.mu.Lock()
	return func() {
		l.mu.Unlock()
		s.sessionLocksMu.Lock()
		defer s.sessionLocksMu.Unlock()
		// Drop the entry with its last user, so the map holds only the
		// sessions being appended to.
		l.refs--
		if l.refs == 0 {
			delete(s.sessionLocks, key)
		}
	}
}

// sessionCommitter has the method set of adk-go's internal MutableSession,
// which the sessions of the in-memory and database services, and those
// session.CloneSession returns, implement.
type sessionCommitter interface {
	CommitAppend(trimmedEvent any, rawStateDelta map[string]any, updatedAt time.Time)
}

// commitToHeldSession updates the session value a caller passed to
// AppendEvent after the append has been stored.
func commitToHeldSession(curSession session.Session, trimmedEvent *session.Event, stateDelta map[string]any, updatedAt time.Time) error {
	if c, ok := curSession.(sessionCommitter); ok {
		c.CommitAppend(trimmedEvent, stateDelta, updatedAt)
		return nil
	}
	st := curSession.State()
	if st == nil {
		return nil
	}
	for k, v := range stateDelta {
		if err := st.Set(k, v); err != nil {
			return err
		}
	}
	return nil
}

// GetUserState returns a detached snapshot of the user-scoped state stored for
// (req.AppName, req.UserID) with keys stripped of [session.KeyPrefixUser], or
// an empty non-nil map when no user state is stored. It implements
// [session.UserStateReader].
//
// Empty req.AppName and req.UserID are read as given. GetUserState never
// refreshes the key's TTL.
func (s *Service) GetUserState(ctx context.Context, req *session.GetUserStateRequest) (*session.GetUserStateResponse, error) {
	if req == nil {
		return nil, errors.New("redis: get user state request is nil")
	}
	st, err := s.getState(ctx, s.userStateKey(req.AppName, req.UserID))
	if err != nil {
		return nil, err
	}
	return &session.GetUserStateResponse{State: st}, nil
}

// Close closes the underlying Redis client if NewService created it from
// Config. If a pre-configured Config.Client was provided by the caller, Close
// is a no-op and returns nil. Calling Close multiple times is safe.
func (s *Service) Close() error {
	return s.close()
}

// watch runs fn in a transaction watching keys, retrying while another client
// changes a watched key before fn's writes commit.
func (s *Service) watch(ctx context.Context, fn func(*goredis.Tx) error, keys ...string) error {
	for range maxTxRetries {
		err := s.client.Watch(ctx, fn, keys...)
		if !errors.Is(err, goredis.TxFailedErr) {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return fmt.Errorf("redis: transaction lost to concurrent writers %d times: %w", maxTxRetries, goredis.TxFailedErr)
}

// watchKeys returns the keys a transaction must watch: the session key, and
// the app and user state keys when it writes them.
func watchKeys(sessionKey, appKey, userKey string, writesApp, writesUser bool) []string {
	keys := []string{sessionKey}
	if writesApp {
		keys = append(keys, appKey)
	}
	if writesUser {
		keys = append(keys, userKey)
	}
	return keys
}

type write struct {
	key   string
	value []byte
}

// commit writes every entry of writes with the service's TTL in one
// MULTI/EXEC.
func (s *Service) commit(ctx context.Context, tx *goredis.Tx, writes []write) error {
	_, err := tx.TxPipelined(ctx, func(p goredis.Pipeliner) error {
		for _, w := range writes {
			p.Set(ctx, w.key, w.value, s.ttl)
		}
		return nil
	})
	if err != nil && !errors.Is(err, goredis.TxFailedErr) {
		return fmt.Errorf("redis: write %d keys: %w", len(writes), err)
	}
	return err
}

// mergeStateWrite reads the state document at key, merges delta into it and
// adds the result to writes. An empty delta leaves the document, and its TTL,
// as they are.
func (s *Service) mergeStateWrite(ctx context.Context, tx *goredis.Tx, writes []write, key string, delta map[string]any) ([]write, error) {
	if len(delta) == 0 {
		return writes, nil
	}
	st, err := readState(ctx, tx, key)
	if err != nil {
		return nil, err
	}
	maps.Copy(st, delta)
	return appendStateWrite(writes, key, st, true)
}

// appendStateWrite adds st, encoded, to writes when changed is true.
func appendStateWrite(writes []write, key string, st map[string]any, changed bool) ([]write, error) {
	if !changed {
		return writes, nil
	}
	b, err := json.Marshal(st)
	if err != nil {
		return nil, fmt.Errorf("redis: encode state: %w", err)
	}
	return append(writes, write{key, b}), nil
}

func (s *Service) getState(ctx context.Context, key string) (map[string]any, error) {
	return readState(ctx, s.client, key)
}

// readState returns the state document at key, or an empty map when the key
// does not exist.
func readState(ctx context.Context, c goredis.Cmdable, key string) (map[string]any, error) {
	raw, err := c.Get(ctx, key).Result()
	if errors.Is(err, goredis.Nil) {
		return map[string]any{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("redis: read state: %w", err)
	}
	return decodeState(raw)
}

// decodeState decodes a state document as MGET or GET returned it; nil is a
// missing key and gives an empty map.
func decodeState(v any) (map[string]any, error) {
	st := map[string]any{}
	raw, ok := v.(string)
	if !ok {
		return st, nil
	}
	if err := json.Unmarshal([]byte(raw), &st); err != nil {
		return nil, fmt.Errorf("redis: decode state: %w", err)
	}
	if st == nil {
		// The document was JSON null.
		st = map[string]any{}
	}
	return st, nil
}

func decodeSession(v any) (*storedSession, error) {
	raw, ok := v.(string)
	if !ok {
		return nil, errors.New("redis: session key holds no value")
	}
	var stored storedSession
	if err := json.Unmarshal([]byte(raw), &stored); err != nil {
		return nil, fmt.Errorf("redis: decode session: %w", err)
	}
	return &stored, nil
}

func encodeSession(stored *storedSession) ([]byte, error) {
	b, err := json.Marshal(stored)
	if err != nil {
		return nil, fmt.Errorf("redis: encode session %q: %w", stored.ID, err)
	}
	return b, nil
}

// extractStateDeltas splits delta by scope: app: and user: keys lose their
// prefix, temp: keys are dropped, and the rest are session-scoped. Each
// result is a new map, even for an empty delta.
func extractStateDeltas(delta map[string]any) (appDelta, userDelta, sessionDelta map[string]any) {
	appDelta, userDelta, sessionDelta = map[string]any{}, map[string]any{}, map[string]any{}
	for k, v := range delta {
		if key, ok := strings.CutPrefix(k, session.KeyPrefixApp); ok {
			appDelta[key] = v
		} else if key, ok := strings.CutPrefix(k, session.KeyPrefixUser); ok {
			userDelta[key] = v
		} else if !strings.HasPrefix(k, session.KeyPrefixTemp) {
			sessionDelta[k] = v
		}
	}
	return appDelta, userDelta, sessionDelta
}

// mergeStates returns a new map holding sessionState and the app and user
// state with their prefixes restored.
func mergeStates(appState, userState, sessionState map[string]any) map[string]any {
	merged := make(map[string]any, len(appState)+len(userState)+len(sessionState))
	maps.Copy(merged, sessionState)
	for k, v := range appState {
		merged[session.KeyPrefixApp+k] = v
	}
	for k, v := range userState {
		merged[session.KeyPrefixUser+k] = v
	}
	return merged
}

// trimTempDeltaState returns event, or a shallow copy of it whose state delta
// has no temp: keys when it had any.
func trimTempDeltaState(event *session.Event) *session.Event {
	delta := event.Actions.StateDelta
	trimmed := make(map[string]any, len(delta))
	for k, v := range delta {
		if !strings.HasPrefix(k, session.KeyPrefixTemp) {
			trimmed[k] = v
		}
	}
	if len(trimmed) == len(delta) {
		return event
	}
	eventCopy := *event
	eventCopy.Actions.StateDelta = trimmed
	return &eventCopy
}

// checkRequestedAuthConfigs reports whether configs survive the JSON round
// trip a stored event takes, as adk-go's other session services check.
func checkRequestedAuthConfigs(configs map[string]toolauth.Config) error {
	if len(configs) == 0 {
		return nil
	}
	b, err := json.Marshal(configs)
	if err != nil {
		return fmt.Errorf("encode requested auth configs: %w", err)
	}
	var stored any
	if err := json.Unmarshal(b, &stored); err != nil {
		return fmt.Errorf("decode requested auth configs: %w", err)
	}
	if b, err = json.Marshal(stored); err != nil {
		return fmt.Errorf("encode requested auth configs: %w", err)
	}
	var decoded map[string]toolauth.Config
	if err := json.Unmarshal(b, &decoded); err != nil {
		return fmt.Errorf("requested auth configs do not decode: %w", err)
	}
	return nil
}

var globEscaper = strings.NewReplacer(`\`, `\\`, `*`, `\*`, `?`, `\?`, `[`, `\[`, `]`, `\]`)

// escapeGlob escapes the characters SCAN MATCH treats as a pattern.
func escapeGlob(s string) string {
	return globEscaper.Replace(s)
}
