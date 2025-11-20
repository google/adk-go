// Copyright 2025 The Go MCP SDK Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"math"
	"math/rand/v2"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/modelcontextprotocol/go-sdk/auth"
	"github.com/modelcontextprotocol/go-sdk/internal/jsonrpc2"
	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
)

const (
	protocolVersionHeader = "Mcp-Protocol-Version"
	sessionIDHeader       = "Mcp-Session-Id"
)

// A StreamableHTTPHandler is an http.Handler that serves streamable MCP
// sessions, as defined by the [MCP spec].
//
// [MCP spec]: https://modelcontextprotocol.io/2025/03/26/streamable-http-transport.html
type StreamableHTTPHandler struct {
	getServer func(*http.Request) *Server
	opts      StreamableHTTPOptions

	onTransportDeletion func(sessionID string) // for testing only

	mu sync.Mutex
	// TODO: we should store the ServerSession along with the transport, because
	// we need to cancel keepalive requests when closing the transport.
	transports map[string]*StreamableServerTransport // keyed by IDs (from Mcp-Session-Id header)
}

// StreamableHTTPOptions configures the StreamableHTTPHandler.
type StreamableHTTPOptions struct {
	// Stateless controls whether the session is 'stateless'.
	//
	// A stateless server does not validate the Mcp-Session-Id header, and uses a
	// temporary session with default initialization parameters. Any
	// server->client request is rejected immediately as there's no way for the
	// client to respond. Server->Client notifications may reach the client if
	// they are made in the context of an incoming request, as described in the
	// documentation for [StreamableServerTransport].
	Stateless bool

	// TODO(#148): support session retention (?)

	// JSONResponse causes streamable responses to return application/json rather
	// than text/event-stream ([§2.1.5] of the spec).
	//
	// [§2.1.5]: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#sending-messages-to-the-server
	JSONResponse bool
}

// NewStreamableHTTPHandler returns a new [StreamableHTTPHandler].
//
// The getServer function is used to create or look up servers for new
// sessions. It is OK for getServer to return the same server multiple times.
// If getServer returns nil, a 400 Bad Request will be served.
func NewStreamableHTTPHandler(getServer func(*http.Request) *Server, opts *StreamableHTTPOptions) *StreamableHTTPHandler {
	h := &StreamableHTTPHandler{
		getServer:  getServer,
		transports: make(map[string]*StreamableServerTransport),
	}
	if opts != nil {
		h.opts = *opts
	}
	return h
}

// closeAll closes all ongoing sessions.
//
// TODO(rfindley): investigate the best API for callers to configure their
// session lifecycle. (?)
//
// Should we allow passing in a session store? That would allow the handler to
// be stateless.
func (h *StreamableHTTPHandler) closeAll() {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, s := range h.transports {
		s.connection.Close()
	}
	h.transports = nil
}

func (h *StreamableHTTPHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// Allow multiple 'Accept' headers.
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Accept#syntax
	accept := strings.Split(strings.Join(req.Header.Values("Accept"), ","), ",")
	var jsonOK, streamOK bool
	for _, c := range accept {
		switch strings.TrimSpace(c) {
		case "application/json", "application/*":
			jsonOK = true
		case "text/event-stream", "text/*":
			streamOK = true
		case "*/*":
			jsonOK = true
			streamOK = true
		}
	}

	if req.Method == http.MethodGet {
		if !streamOK {
			http.Error(w, "Accept must contain 'text/event-stream' for GET requests", http.StatusBadRequest)
			return
		}
	} else if (!jsonOK || !streamOK) && req.Method != http.MethodDelete { // TODO: consolidate with handling of http method below.
		http.Error(w, "Accept must contain both 'application/json' and 'text/event-stream'", http.StatusBadRequest)
		return
	}

	sessionID := req.Header.Get(sessionIDHeader)
	var transport *StreamableServerTransport
	if sessionID != "" {
		h.mu.Lock()
		transport = h.transports[sessionID]
		h.mu.Unlock()
		if transport == nil && !h.opts.Stateless {
			// Unless we're in 'stateless' mode, which doesn't perform any Session-ID
			// validation, we require that the session ID matches a known session.
			//
			// In stateless mode, a temporary transport is be created below.
			http.Error(w, "session not found", http.StatusNotFound)
			return
		}
	}

	if req.Method == http.MethodDelete {
		if sessionID == "" {
			http.Error(w, "Bad Request: DELETE requires an Mcp-Session-Id header", http.StatusBadRequest)
			return
		}
		if transport != nil { // transport may be nil in stateless mode
			h.mu.Lock()
			delete(h.transports, transport.SessionID)
			h.mu.Unlock()
			transport.connection.Close()
		}
		w.WriteHeader(http.StatusNoContent)
		return
	}

	switch req.Method {
	case http.MethodPost, http.MethodGet:
		if req.Method == http.MethodGet && (h.opts.Stateless || sessionID == "") {
			http.Error(w, "GET requires an active session", http.StatusMethodNotAllowed)
			return
		}
	default:
		w.Header().Set("Allow", "GET, POST, DELETE")
		http.Error(w, "Method Not Allowed: streamable MCP servers support GET, POST, and DELETE requests", http.StatusMethodNotAllowed)
		return
	}

	// [§2.7] of the spec (2025-06-18) states:
	//
	// "If using HTTP, the client MUST include the MCP-Protocol-Version:
	// <protocol-version> HTTP header on all subsequent requests to the MCP
	// server, allowing the MCP server to respond based on the MCP protocol
	// version.
	//
	// For example: MCP-Protocol-Version: 2025-06-18
	// The protocol version sent by the client SHOULD be the one negotiated during
	// initialization.
	//
	// For backwards compatibility, if the server does not receive an
	// MCP-Protocol-Version header, and has no other way to identify the version -
	// for example, by relying on the protocol version negotiated during
	// initialization - the server SHOULD assume protocol version 2025-03-26.
	//
	// If the server receives a request with an invalid or unsupported
	// MCP-Protocol-Version, it MUST respond with 400 Bad Request."
	//
	// Since this wasn't present in the 2025-03-26 version of the spec, this
	// effectively means:
	//  1. IF the client provides a version header, it must be a supported
	//     version.
	//  2. In stateless mode, where we've lost the state of the initialize
	//     request, we assume that whatever the client tells us is the truth (or
	//     assume 2025-03-26 if the client doesn't say anything).
	//
	// This logic matches the typescript SDK.
	//
	// [§2.7]: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#protocol-version-header
	protocolVersion := req.Header.Get(protocolVersionHeader)
	if protocolVersion == "" {
		protocolVersion = protocolVersion20250326
	}
	if !slices.Contains(supportedProtocolVersions, protocolVersion) {
		http.Error(w, fmt.Sprintf("Bad Request: Unsupported protocol version (supported versions: %s)", strings.Join(supportedProtocolVersions, ",")), http.StatusBadRequest)
		return
	}

	if transport == nil {
		server := h.getServer(req)
		if server == nil {
			// The getServer argument to NewStreamableHTTPHandler returned nil.
			http.Error(w, "no server available", http.StatusBadRequest)
			return
		}
		if sessionID == "" {
			// In stateless mode, sessionID may be nonempty even if there's no
			// existing transport.
			sessionID = server.opts.GetSessionID()
		}
		transport = &StreamableServerTransport{
			SessionID:    sessionID,
			Stateless:    h.opts.Stateless,
			jsonResponse: h.opts.JSONResponse,
		}

		// To support stateless mode, we initialize the session with a default
		// state, so that it doesn't reject subsequent requests.
		var connectOpts *ServerSessionOptions
		if h.opts.Stateless {
			// Peek at the body to see if it is initialize or initialized.
			// We want those to be handled as usual.
			var hasInitialize, hasInitialized bool
			{
				// TODO: verify that this allows protocol version negotiation for
				// stateless servers.
				body, err := io.ReadAll(req.Body)
				if err != nil {
					http.Error(w, "failed to read body", http.StatusInternalServerError)
					return
				}
				req.Body.Close()

				// Reset the body so that it can be read later.
				req.Body = io.NopCloser(bytes.NewBuffer(body))

				msgs, _, err := readBatch(body)
				if err == nil {
					for _, msg := range msgs {
						if req, ok := msg.(*jsonrpc.Request); ok {
							switch req.Method {
							case methodInitialize:
								hasInitialize = true
							case notificationInitialized:
								hasInitialized = true
							}
						}
					}
				}
			}

			// If we don't have InitializeParams or InitializedParams in the request,
			// set the initial state to a default value.
			state := new(ServerSessionState)
			if !hasInitialize {
				state.InitializeParams = &InitializeParams{
					ProtocolVersion: protocolVersion,
				}
			}
			if !hasInitialized {
				state.InitializedParams = new(InitializedParams)
			}
			state.LogLevel = "info"
			connectOpts = &ServerSessionOptions{
				State: state,
			}
		} else {
			// Cleanup is only required in stateful mode, as transportation is
			// not stored in the map otherwise.
			connectOpts = &ServerSessionOptions{
				onClose: func() {
					h.mu.Lock()
					delete(h.transports, transport.SessionID)
					h.mu.Unlock()
					if h.onTransportDeletion != nil {
						h.onTransportDeletion(transport.SessionID)
					}
				},
			}
		}

		// Pass req.Context() here, to allow middleware to add context values.
		// The context is detached in the jsonrpc2 library when handling the
		// long-running stream.
		ss, err := server.Connect(req.Context(), transport, connectOpts)
		if err != nil {
			http.Error(w, "failed connection", http.StatusInternalServerError)
			return
		}
		if h.opts.Stateless {
			// Stateless mode: close the session when the request exits.
			defer ss.Close() // close the fake session after handling the request
		} else {
			// Otherwise, save the transport so that it can be reused
			h.mu.Lock()
			h.transports[transport.SessionID] = transport
			h.mu.Unlock()
		}
	}

	transport.ServeHTTP(w, req)
}

// A StreamableServerTransport implements the server side of the MCP streamable
// transport.
//
// Each StreamableServerTransport must be connected (via [Server.Connect]) at
// most once, since [StreamableServerTransport.ServeHTTP] serves messages to
// the connected session.
//
// Reads from the streamable server connection receive messages from http POST
// requests from the client. Writes to the streamable server connection are
// sent either to the hanging POST response, or to the hanging GET, according
// to the following rules:
//   - JSON-RPC responses to incoming requests are always routed to the
//     appropriate HTTP response.
//   - Requests or notifications made with a context.Context value derived from
//     an incoming request handler, are routed to the HTTP response
//     corresponding to that request, unless it has already terminated, in
//     which case they are routed to the hanging GET.
//   - Requests or notifications made with a detached context.Context value are
//     routed to the hanging GET.
type StreamableServerTransport struct {
	// SessionID is the ID of this session.
	//
	// If SessionID is the empty string, this is a 'stateless' session, which has
	// limited ability to communicate with the client. Otherwise, the session ID
	// must be globally unique, that is, different from any other session ID
	// anywhere, past and future. (We recommend using a crypto random number
	// generator to produce one, as with [crypto/rand.Text].)
	SessionID string

	// Stateless controls whether the eventstore is 'Stateless'. Server sessions
	// connected to a stateless transport are disallowed from making outgoing
	// requests.
	//
	// See also [StreamableHTTPOptions.Stateless].
	Stateless bool

	// Storage for events, to enable stream resumption.
	// If nil, a [MemoryEventStore] with the default maximum size will be used.
	EventStore EventStore

	// jsonResponse, if set, tells the server to prefer to respond to requests
	// using application/json responses rather than text/event-stream.
	//
	// Specifically, responses will be application/json whenever incoming POST
	// request contain only a single message. In this case, notifications or
	// requests made within the context of a server request will be sent to the
	// hanging GET request, if any.
	//
	// TODO(rfindley): jsonResponse should be exported, since
	// StreamableHTTPOptions.JSONResponse is exported.
	jsonResponse bool

	// connection is non-nil if and only if the transport has been connected.
	connection *streamableServerConn
}

// Connect implements the [Transport] interface.
func (t *StreamableServerTransport) Connect(ctx context.Context) (Connection, error) {
	if t.connection != nil {
		return nil, fmt.Errorf("transport already connected")
	}
	t.connection = &streamableServerConn{
		sessionID:      t.SessionID,
		stateless:      t.Stateless,
		eventStore:     t.EventStore,
		jsonResponse:   t.jsonResponse,
		incoming:       make(chan jsonrpc.Message, 10),
		done:           make(chan struct{}),
		streams:        make(map[string]*stream),
		requestStreams: make(map[jsonrpc.ID]string),
	}
	if t.connection.eventStore == nil {
		t.connection.eventStore = NewMemoryEventStore(nil)
	}
	// Stream 0 corresponds to the hanging 'GET'.
	//
	// It is always text/event-stream, since it must carry arbitrarily many
	// messages.
	var err error
	t.connection.streams[""], err = t.connection.newStream(ctx, "", false, false)
	if err != nil {
		return nil, err
	}
	return t.connection, nil
}

type streamableServerConn struct {
	sessionID    string
	stateless    bool
	jsonResponse bool
	eventStore   EventStore

	incoming chan jsonrpc.Message // messages from the client to the server

	mu sync.Mutex // guards all fields below

	// Sessions are closed exactly once.
	isDone bool
	done   chan struct{}

	// Sessions can have multiple logical connections (which we call streams),
	// corresponding to HTTP requests. Additionally, streams may be resumed by
	// subsequent HTTP requests, when the HTTP connection is terminated
	// unexpectedly.
	//
	// Therefore, we use a logical stream ID to key the stream state, and
	// perform the accounting described below when incoming HTTP requests are
	// handled.

	// streams holds the logical streams for this session, keyed by their ID.
	// TODO: streams are never deleted, so the memory for a connection grows without
	// bound. If we deleted a stream when the response is sent, we would lose the ability
	// to replay if there was a cut just before the response was transmitted.
	// Perhaps we could have a TTL for streams that starts just after the response.
	streams map[string]*stream

	// requestStreams maps incoming requests to their logical stream ID.
	//
	// Lifecycle: requestStreams persist for the duration of the session.
	//
	// TODO: clean up once requests are handled. See the TODO for streams above.
	requestStreams map[jsonrpc.ID]string
}

func (c *streamableServerConn) SessionID() string {
	return c.sessionID
}

// A stream is a single logical stream of SSE events within a server session.
// A stream begins with a client request, or with a client GET that has
// no Last-Event-ID header.
//
// A stream ends only when its session ends; we cannot determine its end otherwise,
// since a client may send a GET with a Last-Event-ID that references the stream
// at any time.
type stream struct {
	// id is the logical ID for the stream, unique within a session.
	// an empty string is used for messages that don't correlate with an incoming request.
	id string

	// If isInitialize is set, the stream is in response to an initialize request,
	// and therefore should include the session ID header.
	isInitialize bool

	// jsonResponse records whether this stream should respond with application/json
	// instead of text/event-stream.
	//
	// See [StreamableServerTransportOptions.JSONResponse].
	jsonResponse bool

	// signal is a 1-buffered channel, owned by an incoming HTTP request, that signals
	// that there are messages available to write into the HTTP response.
	// In addition, the presence of a channel guarantees that at most one HTTP response
	// can receive messages for a logical stream. After claiming the stream, incoming
	// requests should read from the event store, to ensure that no new messages are missed.
	//
	// To simplify locking, signal is an atomic. We need an atomic.Pointer, because
	// you can't set an atomic.Value to nil.
	//
	// Lifecycle: each channel value persists for the duration of an HTTP POST or
	// GET request for the given streamID.
	signal atomic.Pointer[chan struct{}]

	// The following mutable fields are protected by the mutex of the containing
	// StreamableServerTransport.

	// streamRequests is the set of unanswered incoming RPCs for the stream.
	//
	// Requests persist until their response data has been added to the event store.
	requests map[jsonrpc.ID]struct{}
}

func (c *streamableServerConn) newStream(ctx context.Context, id string, isInitialize, jsonResponse bool) (*stream, error) {
	if err := c.eventStore.Open(ctx, c.sessionID, id); err != nil {
		return nil, err
	}
	return &stream{
		id:           id,
		isInitialize: isInitialize,
		jsonResponse: jsonResponse,
		requests:     make(map[jsonrpc.ID]struct{}),
	}, nil
}

func signalChanPtr() *chan struct{} {
	c := make(chan struct{}, 1)
	return &c
}

// We track the incoming request ID inside the handler context using
// idContextValue, so that notifications and server->client calls that occur in
// the course of handling incoming requests are correlated with the incoming
// request that caused them, and can be dispatched as server-sent events to the
// correct HTTP request.
//
// Currently, this is implemented in [ServerSession.handle]. This is not ideal,
// because it means that a user of the MCP package couldn't implement the
// streamable transport, as they'd lack this privileged access.
//
// If we ever wanted to expose this mechanism, we have a few options:
//  1. Make ServerSession an interface, and provide an implementation of
//     ServerSession to handlers that closes over the incoming request ID.
//  2. Expose a 'HandlerTransport' interface that allows transports to provide
//     a handler middleware, so that we don't hard-code this behavior in
//     ServerSession.handle.
//  3. Add a `func ForRequest(context.Context) jsonrpc.ID` accessor that lets
//     any transport access the incoming request ID.
//
// For now, by giving only the StreamableServerTransport access to the request
// ID, we avoid having to make this API decision.
type idContextKey struct{}

// ServeHTTP handles a single HTTP request for the session.
func (t *StreamableServerTransport) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if t.connection == nil {
		http.Error(w, "transport not connected", http.StatusInternalServerError)
		return
	}
	switch req.Method {
	case http.MethodGet:
		t.connection.serveGET(w, req)
	case http.MethodPost:
		t.connection.servePOST(w, req)
	default:
		// Should not be reached, as this is checked in StreamableHTTPHandler.ServeHTTP.
		w.Header().Set("Allow", "GET, POST")
		http.Error(w, "unsupported method", http.StatusMethodNotAllowed)
		return
	}
}

// serveGET streams messages to a hanging http GET, with stream ID and last
// message parsed from the Last-Event-ID header.
//
// It returns an HTTP status code and error message.
func (c *streamableServerConn) serveGET(w http.ResponseWriter, req *http.Request) {
	// connID 0 corresponds to the default GET request.
	id := ""
	// By default, we haven't seen a last index. Since indices start at 0, we represent
	// that by -1. This is incremented just before each event is written, in streamResponse
	// around L407.
	lastIdx := -1
	if len(req.Header.Values("Last-Event-ID")) > 0 {
		eid := req.Header.Get("Last-Event-ID")
		var ok bool
		id, lastIdx, ok = parseEventID(eid)
		if !ok {
			http.Error(w, fmt.Sprintf("malformed Last-Event-ID %q", eid), http.StatusBadRequest)
			return
		}
	}

	c.mu.Lock()
	stream, ok := c.streams[id]
	c.mu.Unlock()
	if !ok {
		http.Error(w, "unknown stream", http.StatusBadRequest)
		return
	}
	if !stream.signal.CompareAndSwap(nil, signalChanPtr()) {
		// The CAS returned false, meaning that the comparison failed: stream.signal is not nil.
		http.Error(w, "stream ID conflicts with ongoing stream", http.StatusConflict)
		return
	}
	defer stream.signal.Store(nil)
	persistent := id == "" // Only the special stream "" is a hanging get.
	c.respondSSE(stream, w, req, lastIdx, persistent)
}

// servePOST handles an incoming message, and replies with either an outgoing
// message stream or single response object, depending on whether the
// jsonResponse option is set.
//
// It returns an HTTP status code and error message.
func (c *streamableServerConn) servePOST(w http.ResponseWriter, req *http.Request) {
	if len(req.Header.Values("Last-Event-ID")) > 0 {
		http.Error(w, "can't send Last-Event-ID for POST request", http.StatusBadRequest)
		return
	}

	// Read incoming messages.
	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	if len(body) == 0 {
		http.Error(w, "POST requires a non-empty body", http.StatusBadRequest)
		return
	}
	incoming, isBatch, err := readBatch(body)
	if err != nil {
		http.Error(w, fmt.Sprintf("malformed payload: %v", err), http.StatusBadRequest)
		return
	}

	protocolVersion := req.Header.Get(protocolVersionHeader)
	if protocolVersion == "" {
		protocolVersion = protocolVersion20250326
	}

	if isBatch && protocolVersion >= protocolVersion20250618 {
		http.Error(w, fmt.Sprintf("JSON-RPC batching is not supported in %s and later (request version: %s)", protocolVersion20250618, protocolVersion), http.StatusBadRequest)
		return
	}

	requests := make(map[jsonrpc.ID]struct{})
	tokenInfo := auth.TokenInfoFromContext(req.Context())
	isInitialize := false
	for _, msg := range incoming {
		if jreq, ok := msg.(*jsonrpc.Request); ok {
			// Preemptively check that this is a valid request, so that we can fail
			// the HTTP request. If we didn't do this, a request with a bad method or
			// missing ID could be silently swallowed.
			if _, err := checkRequest(jreq, serverMethodInfos); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if jreq.Method == methodInitialize {
				isInitialize = true
			}
			jreq.Extra = &RequestExtra{
				TokenInfo: tokenInfo,
				Header:    req.Header,
			}
			if jreq.IsCall() {
				requests[jreq.ID] = struct{}{}
			}
		}
	}

	var stream *stream // if non-nil, used to handle requests

	// If we have requests, we need to handle responses along with any
	// notifications or server->client requests made in the course of handling.
	// Update accounting for this incoming payload.
	if len(requests) > 0 {
		stream, err = c.newStream(req.Context(), randText(), isInitialize, c.jsonResponse)
		if err != nil {
			http.Error(w, fmt.Sprintf("storing stream: %v", err), http.StatusInternalServerError)
			return
		}
		c.mu.Lock()
		c.streams[stream.id] = stream
		stream.requests = requests
		for reqID := range requests {
			c.requestStreams[reqID] = stream.id
		}
		c.mu.Unlock()
		stream.signal.Store(signalChanPtr())
		defer stream.signal.Store(nil)
	}

	// Publish incoming messages.
	for _, msg := range incoming {
		c.incoming <- msg
	}

	if stream == nil {
		w.WriteHeader(http.StatusAccepted)
		return
	}

	if stream.jsonResponse {
		c.respondJSON(stream, w, req)
	} else {
		c.respondSSE(stream, w, req, -1, false)
	}
}

func (c *streamableServerConn) respondJSON(stream *stream, w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Content-Type", "application/json")
	if c.sessionID != "" && stream.isInitialize {
		w.Header().Set(sessionIDHeader, c.sessionID)
	}

	var msgs []json.RawMessage
	ctx := req.Context()
	for msg, err := range c.messages(ctx, stream, false, -1) {
		if err != nil {
			if ctx.Err() != nil {
				w.WriteHeader(http.StatusNoContent)
				return
			} else {
				http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
				return
			}
		}
		msgs = append(msgs, msg)
	}
	var data []byte
	if len(msgs) == 1 {
		data = []byte(msgs[0])
	} else {
		// TODO: add tests for batch responses, or disallow them entirely.
		var err error
		data, err = json.Marshal(msgs)
		if err != nil {
			http.Error(w, fmt.Sprintf("internal error marshalling response: %v", err), http.StatusInternalServerError)
			return
		}
	}
	_, _ = w.Write(data) // ignore error: client disconnected
}

// lastIndex is the index of the last seen event if resuming, else -1.
func (c *streamableServerConn) respondSSE(stream *stream, w http.ResponseWriter, req *http.Request, lastIndex int, persistent bool) {
	// Accept was checked in [StreamableHTTPHandler]
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Content-Type", "text/event-stream") // Accept checked in [StreamableHTTPHandler]
	w.Header().Set("Connection", "keep-alive")
	if c.sessionID != "" && stream.isInitialize {
		w.Header().Set(sessionIDHeader, c.sessionID)
	}
	if persistent {
		// Issue #410: the hanging GET is likely not to receive messages for a long
		// time. Ensure that headers are flushed.
		//
		// For non-persistent requests, delay the writing of the header in case we
		// may want to set an error status.
		// (see the TODO: this probably isn't worth it).
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}

	// write one event containing data.
	writes := 0
	write := func(data []byte) bool {
		lastIndex++
		e := Event{
			Name: "message",
			ID:   formatEventID(stream.id, lastIndex),
			Data: data,
		}
		if _, err := writeEvent(w, e); err != nil {
			// Connection closed or broken.
			// TODO(#170): log when we add server-side logging.
			return false
		}
		writes++
		return true
	}

	// Repeatedly collect pending outgoing events and send them.
	ctx := req.Context()
	for msg, err := range c.messages(ctx, stream, persistent, lastIndex) {
		if err != nil {
			if ctx.Err() == nil && writes == 0 && !persistent {
				// If we haven't yet written the header, we have an opportunity to
				// promote an error to an HTTP error.
				//
				// TODO: This may not matter in practice, in which case we should
				// simplify.
				http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
			} else {
				// TODO(#170): log when we add server-side logging
			}
			return
		}
		if !write(msg) {
			return
		}
	}
}

// messages iterates over messages sent to the current stream.
//
// persistent indicates if it is the main GET listener, which should never be
// terminated.
// lastIndex is the index of the last seen event, iteration begins at lastIndex+1.
//
// The first iterated value is the received JSON message. The second iterated
// value is an error value indicating whether the stream terminated normally.
// Iteration ends at the first non-nil error.
//
// If the stream did not terminate normally, it is either because ctx was
// cancelled, or the connection is closed: check the ctx.Err() to differentiate
// these cases.
func (c *streamableServerConn) messages(ctx context.Context, stream *stream, persistent bool, lastIndex int) iter.Seq2[json.RawMessage, error] {
	return func(yield func(json.RawMessage, error) bool) {
		for {
			c.mu.Lock()
			nOutstanding := len(stream.requests)
			c.mu.Unlock()
			for data, err := range c.eventStore.After(ctx, c.SessionID(), stream.id, lastIndex) {
				if err != nil {
					yield(nil, err)
					return
				}
				if !yield(data, nil) {
					return
				}
				lastIndex++
			}
			// If all requests have been handled and replied to, we should terminate this connection.
			// "After the JSON-RPC response has been sent, the server SHOULD close the SSE stream."
			// §6.4, https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#sending-messages-to-the-server
			// We only want to terminate POSTs, and GETs that are replaying. The general-purpose GET
			// (stream ID 0) will never have requests, and should remain open indefinitely.
			if nOutstanding == 0 && !persistent {
				return
			}

			select {
			case <-*stream.signal.Load(): // there are new outgoing messages
				// return to top of loop
			case <-c.done: // session is closed
				yield(nil, errors.New("session is closed"))
				return
			case <-ctx.Done():
				yield(nil, ctx.Err())
				return
			}
		}

	}
}

// Event IDs: encode both the logical connection ID and the index, as
// <streamID>_<idx>, to be consistent with the typescript implementation.

// formatEventID returns the event ID to use for the logical connection ID
// streamID and message index idx.
//
// See also [parseEventID].
func formatEventID(sid string, idx int) string {
	return fmt.Sprintf("%s_%d", sid, idx)
}

// parseEventID parses a Last-Event-ID value into a logical stream id and
// index.
//
// See also [formatEventID].
func parseEventID(eventID string) (streamID string, idx int, ok bool) {
	parts := strings.Split(eventID, "_")
	if len(parts) != 2 {
		return "", 0, false
	}
	streamID = parts[0]
	idx, err := strconv.Atoi(parts[1])
	if err != nil || idx < 0 {
		return "", 0, false
	}
	return streamID, idx, true
}

// Read implements the [Connection] interface.
func (c *streamableServerConn) Read(ctx context.Context) (jsonrpc.Message, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case msg, ok := <-c.incoming:
		if !ok {
			return nil, io.EOF
		}
		return msg, nil
	case <-c.done:
		return nil, io.EOF
	}
}

// Write implements the [Connection] interface.
func (c *streamableServerConn) Write(ctx context.Context, msg jsonrpc.Message) error {
	if req, ok := msg.(*jsonrpc.Request); ok && req.ID.IsValid() && (c.stateless || c.sessionID == "") {
		// Requests aren't possible with stateless servers, or when there's no session ID.
		return fmt.Errorf("%w: stateless servers cannot make requests", jsonrpc2.ErrRejected)
	}
	// Find the incoming request that this write relates to, if any.
	var forRequest jsonrpc.ID
	isResponse := false
	if resp, ok := msg.(*jsonrpc.Response); ok {
		// If the message is a response, it relates to its request (of course).
		forRequest = resp.ID
		isResponse = true
	} else {
		// Otherwise, we check to see if it request was made in the context of an
		// ongoing request. This may not be the case if the request was made with
		// an unrelated context.
		if v := ctx.Value(idContextKey{}); v != nil {
			forRequest = v.(jsonrpc.ID)
		}
	}

	// Find the logical connection corresponding to this request.
	//
	// For messages sent outside of a request context, this is the default
	// connection "".
	var forStream string
	if forRequest.IsValid() {
		c.mu.Lock()
		forStream = c.requestStreams[forRequest]
		c.mu.Unlock()
	}

	data, err := jsonrpc2.EncodeMessage(msg)
	if err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.isDone {
		return errors.New("session is closed")
	}

	stream := c.streams[forStream]
	if stream == nil {
		return fmt.Errorf("no stream with ID %s", forStream)
	}

	// Special case a few conditions where we fall back on stream 0 (the hanging GET):
	//
	//  - if forStream is known, but the associated stream is logically complete
	//  - if the stream is application/json, but the message is not a response
	//
	// TODO(rfindley): either of these, particularly the first, might be
	// considered a bug in the server. Report it through a side-channel?
	if len(stream.requests) == 0 && forStream != "" || stream.jsonResponse && !isResponse {
		stream = c.streams[""]
	}

	if err := c.eventStore.Append(ctx, c.SessionID(), stream.id, data); err != nil {
		return fmt.Errorf("error storing event: %w", err)
	}
	if isResponse {
		// Once we've put the reply on the queue, it's no longer outstanding.
		delete(stream.requests, forRequest)
	}

	// Signal streamResponse that new work is available.
	signalp := stream.signal.Load()
	if signalp != nil {
		select {
		case *signalp <- struct{}{}:
		default:
		}
	}
	return nil
}

// Close implements the [Connection] interface.
func (c *streamableServerConn) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.isDone {
		c.isDone = true
		close(c.done)
		// TODO: find a way to plumb a context here, or an event store with a long-running
		// close operation can take arbitrary time. Alternative: impose a fixed timeout here.
		return c.eventStore.SessionClosed(context.TODO(), c.sessionID)
	}
	return nil
}

// A StreamableClientTransport is a [Transport] that can communicate with an MCP
// endpoint serving the streamable HTTP transport defined by the 2025-03-26
// version of the spec.
type StreamableClientTransport struct {
	Endpoint   string
	HTTPClient *http.Client
	// MaxRetries is the maximum number of times to attempt a reconnect before giving up.
	// It defaults to 5. To disable retries, use a negative number.
	MaxRetries int

	// TODO(rfindley): propose exporting these.
	// If strict is set, the transport is in 'strict mode', where any violation
	// of the MCP spec causes a failure.
	strict bool
	// If logger is set, it is used to log aspects of the transport, such as spec
	// violations that were ignored.
	logger *slog.Logger
}

// These settings are not (yet) exposed to the user in
// StreamableClientTransport.
const (
	// reconnectGrowFactor is the multiplicative factor by which the delay increases after each attempt.
	// A value of 1.0 results in a constant delay, while a value of 2.0 would double it each time.
	// It must be 1.0 or greater if MaxRetries is greater than 0.
	reconnectGrowFactor = 1.5
	// reconnectInitialDelay is the base delay for the first reconnect attempt.
	reconnectInitialDelay = 1 * time.Second
	// reconnectMaxDelay caps the backoff delay, preventing it from growing indefinitely.
	reconnectMaxDelay = 30 * time.Second
)

// Connect implements the [Transport] interface.
//
// The resulting [Connection] writes messages via POST requests to the
// transport URL with the Mcp-Session-Id header set, and reads messages from
// hanging requests.
//
// When closed, the connection issues a DELETE request to terminate the logical
// session.
func (t *StreamableClientTransport) Connect(ctx context.Context) (Connection, error) {
	client := t.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	maxRetries := t.MaxRetries
	if maxRetries == 0 {
		maxRetries = 5
	} else if maxRetries < 0 {
		maxRetries = 0
	}
	// Create a new cancellable context that will manage the connection's lifecycle.
	// This is crucial for cleanly shutting down the background SSE listener by
	// cancelling its blocking network operations, which prevents hangs on exit.
	connCtx, cancel := context.WithCancel(ctx)
	conn := &streamableClientConn{
		url:        t.Endpoint,
		client:     client,
		incoming:   make(chan jsonrpc.Message, 10),
		done:       make(chan struct{}),
		maxRetries: maxRetries,
		strict:     t.strict,
		logger:     t.logger,
		ctx:        connCtx,
		cancel:     cancel,
		failed:     make(chan struct{}),
	}
	return conn, nil
}

type streamableClientConn struct {
	url        string
	client     *http.Client
	ctx        context.Context
	cancel     context.CancelFunc
	incoming   chan jsonrpc.Message
	maxRetries int
	strict     bool         // from [StreamableClientTransport.strict]
	logger     *slog.Logger // from [StreamableClientTransport.logger]

	// Guard calls to Close, as it may be called multiple times.
	closeOnce sync.Once
	closeErr  error
	done      chan struct{} // signal graceful termination

	// Logical reads are distributed across multiple http requests. Whenever any
	// of them fails to process their response, we must break the connection, by
	// failing the pending Read.
	//
	// Achieve this by storing the failure message, and signalling when reads are
	// broken. See also [streamableClientConn.fail] and
	// [streamableClientConn.failure].
	failOnce sync.Once
	_failure error
	failed   chan struct{} // signal failure

	// Guard the initialization state.
	mu                sync.Mutex
	initializedResult *InitializeResult
	sessionID         string
}

// errSessionMissing distinguishes if the session is known to not be present on
// the server (see [streamableClientConn.fail]).
//
// TODO(rfindley): should we expose this error value (and its corresponding
// API) to the user?
//
// The spec says that if the server returns 404, clients should reestablish
// a session. For now, we delegate that to the user, but do they need a way to
// differentiate a 'NotFound' error from other errors?
var errSessionMissing = errors.New("session not found")

var _ clientConnection = (*streamableClientConn)(nil)

func (c *streamableClientConn) sessionUpdated(state clientSessionState) {
	c.mu.Lock()
	c.initializedResult = state.InitializeResult
	c.mu.Unlock()

	// Start the persistent SSE listener as soon as we have the initialized
	// result.
	//
	// § 2.2: The client MAY issue an HTTP GET to the MCP endpoint. This can be
	// used to open an SSE stream, allowing the server to communicate to the
	// client, without the client first sending data via HTTP POST.
	//
	// We have to wait for initialized, because until we've received
	// initialized, we don't know whether the server requires a sessionID.
	//
	// § 2.5: A server using the Streamable HTTP transport MAY assign a session
	// ID at initialization time, by including it in an Mcp-Session-Id header
	// on the HTTP response containing the InitializeResult.
	go c.handleSSE("hanging GET", nil, true, nil)
}

// fail handles an asynchronous error while reading.
//
// If err is non-nil, it is terminal, and subsequent (or pending) Reads will
// fail.
//
// If err wraps errSessionMissing, the failure indicates that the session is no
// longer present on the server, and no final DELETE will be performed when
// closing the connection.
func (c *streamableClientConn) fail(err error) {
	if err != nil {
		c.failOnce.Do(func() {
			c._failure = err
			close(c.failed)
		})
	}
}

func (c *streamableClientConn) failure() error {
	select {
	case <-c.failed:
		return c._failure
	default:
		return nil
	}
}

func (c *streamableClientConn) SessionID() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sessionID
}

// Read implements the [Connection] interface.
func (c *streamableClientConn) Read(ctx context.Context) (jsonrpc.Message, error) {
	if err := c.failure(); err != nil {
		return nil, err
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-c.failed:
		return nil, c.failure()
	case <-c.done:
		return nil, io.EOF
	case msg := <-c.incoming:
		return msg, nil
	}
}

// Write implements the [Connection] interface.
func (c *streamableClientConn) Write(ctx context.Context, msg jsonrpc.Message) error {
	if err := c.failure(); err != nil {
		return err
	}

	var requestSummary string
	var isCall bool
	switch msg := msg.(type) {
	case *jsonrpc.Request:
		requestSummary = fmt.Sprintf("sending %q", msg.Method)
		isCall = msg.IsCall()
	case *jsonrpc.Response:
		requestSummary = fmt.Sprintf("sending jsonrpc response #%d", msg.ID)
	default:
		panic("unreachable")
	}

	data, err := jsonrpc.EncodeMessage(msg)
	if err != nil {
		return fmt.Errorf("%s: %v", requestSummary, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	c.setMCPHeaders(req)

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("%s: %v", requestSummary, err)
	}

	// §2.5.3: "The server MAY terminate the session at any time, after
	// which it MUST respond to requests containing that session ID with HTTP
	// 404 Not Found."
	if resp.StatusCode == http.StatusNotFound {
		// Fail the session immediately, rather than relying on jsonrpc2 to fail
		// (and close) it, because we want the call to Close to know that this
		// session is missing (and therefore not send the DELETE).
		err := fmt.Errorf("%s: failed to send: %w", requestSummary, errSessionMissing)
		c.fail(err)
		resp.Body.Close()
		return err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		resp.Body.Close()
		return fmt.Errorf("broken session: %v", resp.Status)
	}

	if sessionID := resp.Header.Get(sessionIDHeader); sessionID != "" {
		c.mu.Lock()
		hadSessionID := c.sessionID
		if hadSessionID == "" {
			c.sessionID = sessionID
		}
		c.mu.Unlock()
		if hadSessionID != "" && hadSessionID != sessionID {
			resp.Body.Close()
			return fmt.Errorf("mismatching session IDs %q and %q", hadSessionID, sessionID)
		}
	}
	if resp.StatusCode == http.StatusNoContent || resp.StatusCode == http.StatusAccepted {
		// [§2.1.4]: "If the input is a JSON-RPC response or notification:
		// If the server accepts the input, the server MUST return HTTP status code 202 Accepted with no body."
		//
		// [§2.1.4]: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#listening-for-messages-from-the-server
		resp.Body.Close()
		return nil
	} else if !isCall && !c.strict {
		// Some servers return 200, even with an empty json body.
		// Ignore this response in non-strict mode.
		if c.logger != nil {
			c.logger.Warn(fmt.Sprintf("unexpected status code %d from non-call", resp.StatusCode))
		}
		resp.Body.Close()
		return nil
	}

	contentType := strings.TrimSpace(strings.SplitN(resp.Header.Get("Content-Type"), ";", 2)[0])
	switch contentType {
	case "application/json":
		go c.handleJSON(requestSummary, resp)

	case "text/event-stream":
		jsonReq, _ := msg.(*jsonrpc.Request)
		go c.handleSSE(requestSummary, resp, false, jsonReq)

	default:
		resp.Body.Close()
		return fmt.Errorf("%s: unsupported content type %q", requestSummary, contentType)
	}
	return nil
}

// testAuth controls whether a fake Authorization header is added to outgoing requests.
// TODO: replace with a better mechanism when client-side auth is in place.
var testAuth atomic.Bool

func (c *streamableClientConn) setMCPHeaders(req *http.Request) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.initializedResult != nil {
		req.Header.Set(protocolVersionHeader, c.initializedResult.ProtocolVersion)
	}
	if c.sessionID != "" {
		req.Header.Set(sessionIDHeader, c.sessionID)
	}
	if testAuth.Load() {
		req.Header.Set("Authorization", "Bearer foo")
	}
}

func (c *streamableClientConn) handleJSON(requestSummary string, resp *http.Response) {
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		c.fail(fmt.Errorf("%s: failed to read body: %v", requestSummary, err))
		return
	}
	msg, err := jsonrpc.DecodeMessage(body)
	if err != nil {
		c.fail(fmt.Errorf("%s: failed to decode response: %v", requestSummary, err))
		return
	}
	select {
	case c.incoming <- msg:
	case <-c.done:
		// The connection was closed by the client; exit gracefully.
	}
}

// handleSSE manages the lifecycle of an SSE connection. It can be either
// persistent (for the main GET listener) or temporary (for a POST response).
//
// If forReq is set, it is the request that initiated the stream, and the
// stream is complete when we receive its response.
func (c *streamableClientConn) handleSSE(requestSummary string, initialResp *http.Response, persistent bool, forReq *jsonrpc2.Request) {
	resp := initialResp
	var lastEventID string
	for {
		if resp != nil {
			eventID, clientClosed := c.processStream(requestSummary, resp, forReq)
			lastEventID = eventID

			// If the connection was closed by the client, we're done.
			if clientClosed {
				return
			}
			// If the stream has ended, then do not reconnect if the stream is
			// temporary (POST initiated SSE).
			if lastEventID == "" && !persistent {
				return
			}
		}

		// The stream was interrupted or ended by the server. Attempt to reconnect.
		newResp, err := c.reconnect(lastEventID)
		if err != nil {
			// All reconnection attempts failed: fail the connection.
			c.fail(fmt.Errorf("%s: failed to reconnect (session ID: %v): %v", requestSummary, c.sessionID, err))
			return
		}
		resp = newResp
		if resp.StatusCode == http.StatusMethodNotAllowed && persistent {
			// [§2.2.3]: "The server MUST either return Content-Type:
			// text/event-stream in response to this HTTP GET, or else return HTTP
			// 405 Method Not Allowed, indicating that the server does not offer an
			// SSE stream at this endpoint."
			//
			// [§2.2.3]: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#listening-for-messages-from-the-server

			// The server doesn't support the hanging GET.
			resp.Body.Close()
			return
		}
		if resp.StatusCode == http.StatusNotFound && persistent && !c.strict {
			// modelcontextprotocol/gosdk#393: some servers return NotFound instead
			// of MethodNotAllowed for the persistent GET.
			//
			// Treat this like MethodNotAllowed in non-strict mode.
			if c.logger != nil {
				c.logger.Warn("got 404 instead of 405 for hanging GET")
			}
			resp.Body.Close()
			return
		}
		// (see equivalent handling in [streamableClientConn.Write]).
		if resp.StatusCode == http.StatusNotFound {
			c.fail(fmt.Errorf("%s: failed to reconnect (session ID: %v): %w", requestSummary, c.sessionID, errSessionMissing))
			return
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			resp.Body.Close()
			c.fail(fmt.Errorf("%s: failed to reconnect: %v", requestSummary, http.StatusText(resp.StatusCode)))
			return
		}
		// Reconnection was successful. Continue the loop with the new response.
	}
}

// processStream reads from a single response body, sending events to the
// incoming channel. It returns the ID of the last processed event and a flag
// indicating if the connection was closed by the client. If resp is nil, it
// returns "", false.
func (c *streamableClientConn) processStream(requestSummary string, resp *http.Response, forReq *jsonrpc.Request) (lastEventID string, clientClosed bool) {
	defer resp.Body.Close()
	for evt, err := range scanEvents(resp.Body) {
		if err != nil {
			return lastEventID, false
		}

		if evt.ID != "" {
			lastEventID = evt.ID
		}

		msg, err := jsonrpc.DecodeMessage(evt.Data)
		if err != nil {
			c.fail(fmt.Errorf("%s: failed to decode event: %v", requestSummary, err))
			return "", true
		}

		select {
		case c.incoming <- msg:
			if jsonResp, ok := msg.(*jsonrpc.Response); ok && forReq != nil {
				// TODO: we should never get a response when forReq is nil (the hanging GET).
				// We should detect this case, and eliminate the 'persistent' flag arguments.
				if jsonResp.ID == forReq.ID {
					return "", true
				}
			}
		case <-c.done:
			// The connection was closed by the client; exit gracefully.
			return "", true
		}
	}
	// The loop finished without an error, indicating the server closed the stream.
	return "", false
}

// reconnect handles the logic of retrying a connection with an exponential
// backoff strategy. It returns a new, valid HTTP response if successful, or
// an error if all retries are exhausted.
func (c *streamableClientConn) reconnect(lastEventID string) (*http.Response, error) {
	var finalErr error

	// We can reach the 'reconnect' path through the hanging GET, in which case
	// lastEventID will be "".
	//
	// In this case, we need an initial attempt.
	attempt := 0
	if lastEventID != "" {
		attempt = 1
	}

	for ; attempt <= c.maxRetries; attempt++ {
		select {
		case <-c.done:
			return nil, fmt.Errorf("connection closed by client during reconnect")
		case <-time.After(calculateReconnectDelay(attempt)):
			resp, err := c.establishSSE(lastEventID)
			if err != nil {
				finalErr = err // Store the error and try again.
				continue
			}
			return resp, nil
		}
	}
	// If the loop completes, all retries have failed.
	if finalErr != nil {
		return nil, fmt.Errorf("connection failed after %d attempts: %w", c.maxRetries, finalErr)
	}
	return nil, fmt.Errorf("connection failed after %d attempts", c.maxRetries)
}

// Close implements the [Connection] interface.
func (c *streamableClientConn) Close() error {
	c.closeOnce.Do(func() {
		if errors.Is(c.failure(), errSessionMissing) {
			// If the session is missing, no need to delete it.
		} else {
			req, err := http.NewRequestWithContext(c.ctx, http.MethodDelete, c.url, nil)
			if err != nil {
				c.closeErr = err
			} else {
				c.setMCPHeaders(req)
				if _, err := c.client.Do(req); err != nil {
					c.closeErr = err
				}
			}
		}

		// Cancel any hanging network requests after cleanup.
		c.cancel()
		close(c.done)
	})
	return c.closeErr
}

// establishSSE establishes the persistent SSE listening stream.
// It is used for reconnect attempts using the Last-Event-ID header to
// resume a broken stream where it left off.
func (c *streamableClientConn) establishSSE(lastEventID string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(c.ctx, http.MethodGet, c.url, nil)
	if err != nil {
		return nil, err
	}
	c.setMCPHeaders(req)
	if lastEventID != "" {
		req.Header.Set("Last-Event-ID", lastEventID)
	}
	req.Header.Set("Accept", "text/event-stream")

	return c.client.Do(req)
}

// calculateReconnectDelay calculates a delay using exponential backoff with full jitter.
func calculateReconnectDelay(attempt int) time.Duration {
	if attempt == 0 {
		return 0
	}
	// Calculate the exponential backoff using the grow factor.
	backoffDuration := time.Duration(float64(reconnectInitialDelay) * math.Pow(reconnectGrowFactor, float64(attempt-1)))
	// Cap the backoffDuration at maxDelay.
	backoffDuration = min(backoffDuration, reconnectMaxDelay)

	// Use a full jitter using backoffDuration
	jitter := rand.N(backoffDuration)

	return backoffDuration + jitter
}
