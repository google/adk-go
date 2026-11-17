// Copyright 2025 Google LLC
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

package llminternal

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log"
	"maps"
	"net"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/internal/plugininternal/plugincontext"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/internal/version"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// ErrModelNotConfigured is returned when the model is not configured.
var ErrModelNotConfigured = errors.New("model not configured; ensure Model is set in llmagent.Config")

type BeforeModelCallback func(ctx agent.Context, llmRequest *model.LLMRequest) (*model.LLMResponse, error)

type AfterModelCallback func(ctx agent.Context, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)

type OnModelErrorCallback func(ctx agent.Context, llmRequest *model.LLMRequest, llmResponseError error) (*model.LLMResponse, error)

type BeforeToolCallback func(ctx agent.Context, tool tool.Tool, args map[string]any) (map[string]any, error)

type AfterToolCallback func(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error)

type OnToolErrorCallback func(ctx agent.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error)

// Flow is a base implementation of the agent flow.
type Flow struct {
	Model model.LLM
	// stepModel is the model of the current step: Model when set, else the
	// agent's resolved model. runOneStep and RunLive set it before anything
	// reads it, so every reader within one step sees the same model.
	stepModel model.LLM

	Tools                 []tool.Tool
	RequestProcessors     []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]
	ResponseProcessors    []func(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) iter.Seq2[*session.Event, error]
	BeforeModelCallbacks  []BeforeModelCallback
	AfterModelCallbacks   []AfterModelCallback
	OnModelErrorCallbacks []OnModelErrorCallback
	BeforeToolCallbacks   []BeforeToolCallback
	AfterToolCallbacks    []AfterToolCallback
	OnToolErrorCallbacks  []OnToolErrorCallback

	// ownToolCount is how many leading entries of Tools are the agent's own
	// tools. toolsetTools holds the tools each of the agent's Toolsets listed
	// in the current step, index-aligned with them, so that a toolset's
	// request hook runs just before its tools'. toolProcessor sets Tools and
	// both fields together; toolsetTools is nil until it has run.
	ownToolCount int
	toolsetTools [][]tool.Tool

	// reconnect bounds RunLive's reconnect attempts. Nil means the defaults;
	// only tests set it, to shrink the delays.
	reconnect *liveReconnectPolicy
}

var (
	DefaultRequestProcessors = []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{
		basicRequestProcessor,
		toolProcessor,
		authPreprocessor,
		RequestConfirmationRequestProcessor,
		instructionsRequestProcessor,
		identityRequestProcessor,
		// Compaction must run before historyRequestProcessor so a summary it
		// appends is reflected in the history assembled for this very request.
		CompactionRequestProcessor,
		historyRequestProcessor,
		// The same position as in adk-python's single flow. The model fingerprints
		// the finished request when it is called, so processors after this one
		// still count towards the cached prefix.
		ContextCacheRequestProcessor,
		// Some implementations of NL Planning mark planning contents as thoughts in the post processor.
		// Since these need to be unmarked, NL Planning should be after historyRequestProcessor.
		nlPlanningRequestProcessor,
		// Code execution should be after historyRequestProcessor as it mutates the contents
		// to optimize data files.
		codeExecutionRequestProcessor,
		outputSchemaRequestProcessor,
		AgentTransferRequestProcessor,
		removeDisplayNameIfExists,
		// The continuation turn depends on the role of the last content, which
		// code execution changes when it turns a result into a user turn.
		continuationTurnRequestProcessor,
	}
	DefaultResponseProcessors = []func(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) iter.Seq2[*session.Event, error]{
		nlPlanningResponseProcessor,
		codeExecutionResponseProcessor,
	}
)

// maxConsecutiveThoughtOnlyTurns bounds how many thought-only ("thinking")
// turns in a row the flow accepts before giving up, so the model is re-called
// at most maxConsecutiveThoughtOnlyTurns-1 times waiting for an answer. A model
// that keeps emitting thoughts without ever surfacing an answer would otherwise
// spin forever, appending an event per turn and re-sending an ever-growing
// history. Any turn that is not a final response resets the count.
//
// This is a safety net against a degenerate model, not a tuning knob: it is
// deliberately not configurable, because a caller has no basis for choosing a
// value. It is set high on purpose. A bound that is too low never requests an
// answer the model was about to give, which is worse than a few wasted calls
// before a degenerate model is cut off.
const maxConsecutiveThoughtOnlyTurns = 10

func (f *Flow) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		thoughtOnlyTurns := 0
		for {
			var lastEvent *session.Event
			for ev, err := range f.runOneStep(ctx) {
				if err != nil {
					yield(nil, err)
					return
				}
				// forward the event first.
				if !yield(ev, nil) {
					return
				}
				lastEvent = ev
			}
			if lastEvent == nil {
				return
			}
			if lastEvent.IsFinalResponse() {
				// A thought-only ("thinking") turn reports as final but has no
				// answer; don't stop on it — call the model again. A
				// thought-only turn that finished with STOP is not one of these:
				// applyEmptyResponsePolicy gave it an error code, and it ends
				// the run. Give up once the model has produced only thoughts too
				// many times in a row, leaving the last thinking event as the
				// result.
				if !isThoughtOnlyTurn(lastEvent) {
					return
				}
				thoughtOnlyTurns++
				if thoughtOnlyTurns >= maxConsecutiveThoughtOnlyTurns {
					log.Printf("adk: model %q produced %d consecutive thought-only turns without an answer (limit %d) for agent %q (invocation %q); giving up and returning the last thinking event", //nolint:forbidigo // pre-slog call site
						f.stepModel.Name(), thoughtOnlyTurns, maxConsecutiveThoughtOnlyTurns, ctx.Agent().Name(), ctx.InvocationID())
					return
				}
			} else {
				thoughtOnlyTurns = 0
			}
			if lastEvent.LLMResponse.Partial {
				// The last event is a partial streaming response. The realistic cause
				// is a producer that ends a stream without a terminal aggregate (e.g.,
				// an a2a peer whose stream ends on an appended artifact chunk with no
				// terminal status). The turn was truncated, not completed, which is
				// not expected, so we log a warning and return instead of looping again.
				log.Printf("adk: agent %q (invocation %q): step ended on a partial event from %q; the producer did not close its stream with an aggregated final event, so the turn will not appear in session history", //nolint:forbidigo // pre-slog call site
					ctx.Agent().Name(), ctx.InvocationID(), lastEvent.Author)
				return
			}
		}
	}
}

// isThoughtOnlyTurn reports whether ev is a completed (non-partial) model turn
// whose parts are all model "thinking" (Thought) parts — no surfaced answer.
// Thought signatures ride on substantive parts (text, function calls), which
// are not Thought, so any such part makes the turn non-thought-only. A turn
// with an error code is never thought-only: it is a final response, as in
// adk-python (Event.is_final_response), whatever its parts are. This is what
// ends the run on a thought-only STOP turn that applyEmptyResponsePolicy marked.
func isThoughtOnlyTurn(ev *session.Event) bool {
	if ev == nil || ev.LLMResponse.Partial || ev.LLMResponse.ErrorCode != "" {
		return false
	}
	content := ev.LLMResponse.Content
	if content == nil || len(content.Parts) == 0 {
		return false
	}
	for _, p := range content.Parts {
		if p != nil && !p.Thought {
			return false
		}
	}
	return true
}

// isPythonWhitespace reports whether r is whitespace for Python's str.strip,
// which also strips the file, group, record and unit separators U+001C to
// U+001F that unicode.IsSpace does not report.
func isPythonWhitespace(r rune) bool {
	return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f)
}

// hasMeaningfulContent reports whether resp holds a part a caller can act on:
// a function call or response, executable code or its result, inline or file
// data, or text that is not a thought and not only whitespace. It follows
// adk-python's has_meaningful_content (flows/llm_flows/core/_finalizer.py).
func hasMeaningfulContent(resp *model.LLMResponse) bool {
	if !hasParts(resp) {
		return false
	}
	for _, p := range resp.Content.Parts {
		if p == nil {
			continue
		}
		if p.FunctionCall != nil || p.FunctionResponse != nil || p.ExecutableCode != nil ||
			p.CodeExecutionResult != nil || p.InlineData != nil || p.FileData != nil {
			return true
		}
		if !p.Thought && strings.TrimFunc(p.Text, isPythonWhitespace) != "" {
			return true
		}
	}
	return false
}

// hasParts reports whether resp has content with at least one part.
func hasParts(resp *model.LLMResponse) bool {
	return resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0
}

// streamingMode returns the streaming mode of the invocation.
func streamingMode(ctx agent.InvocationContext) agent.StreamingMode {
	if rc := ctx.RunConfig(); rc != nil {
		return rc.StreamingMode
	}
	return agent.StreamingModeNone
}

// agentName returns the name of the invocation's agent, or "" without one.
func agentName(ctx agent.InvocationContext) string {
	if ctx == nil || ctx.Agent() == nil {
		return ""
	}
	return ctx.Agent().Name()
}

// Error code and messages applyEmptyResponsePolicy sets, as in adk-python
// (flows/llm_flows/core/_model_call.py).
const (
	noContentErrorCode              = "MODEL_RETURNED_NO_CONTENT"
	noContentErrorMessage           = "The model returned no content (finish_reason=STOP with empty parts)."
	noMeaningfulContentErrorMessage = "The model returned no actionable content (finish_reason=STOP with thought-only or whitespace-only parts)."
)

// applyEmptyResponsePolicy marks a completed response that finished with STOP
// but holds no meaningful content (no parts, only thought parts, or only
// whitespace text) as an error, so that the turn ends with an error event the
// caller can act on instead of a silent empty answer. It sets resp.ErrorCode
// to MODEL_RETURNED_NO_CONTENT and, unless resp already has an error message,
// a message that says whether the response had no parts or only thought or
// whitespace parts. It follows adk-python's apply_empty_response_policy
// (flows/llm_flows/core/_model_call.py).
//
// It leaves resp unchanged when resp is partial, already has an error code, or
// finished for another reason. Under SSE streaming it also leaves a response
// without parts unchanged: the streaming aggregator ends a turn with a
// finish-only chunk, which adk-python does not mark either.
//
// It must run after the after-model callbacks and before the response
// processors: emptiness is a property of what the model returned, and a
// processor may clear the content on purpose, as the code execution processor
// does.
func applyEmptyResponsePolicy(ctx agent.InvocationContext, resp *model.LLMResponse) {
	if resp == nil || resp.Partial || resp.ErrorCode != "" || resp.FinishReason != genai.FinishReasonStop || hasMeaningfulContent(resp) {
		return
	}
	withParts := hasParts(resp)
	if !withParts && streamingMode(ctx) == agent.StreamingModeSSE {
		return
	}
	resp.ErrorCode = noContentErrorCode
	if resp.ErrorMessage != "" {
		return
	}
	resp.ErrorMessage = noContentErrorMessage
	if withParts {
		resp.ErrorMessage = noMeaningfulContentErrorMessage
	}
}

// stampAgentNameLabel records the name of the invocation's agent in
// req.Config.Labels under version.AgentNameLabel, so that billing reports can
// be split per agent, as adk-python's call_llm_async does
// (flows/llm_flows/core/_model_call.py). The name is used as it is, without
// sanitizing, as in adk-python. It keeps a value already set under the key.
// Neither the config nor the labels map found in the request is ever
// modified: the label is added to a copy of the map, set on a shallow copy of
// the config (a new config when req.Config is nil), and that copy replaces
// req.Config, because a before-model callback may have attached a config or
// map it shares with other invocations. Without an agent it does nothing.
func stampAgentNameLabel(ctx agent.InvocationContext, req *model.LLMRequest) {
	name := agentName(ctx)
	if name == "" {
		return
	}
	var cfg genai.GenerateContentConfig
	if req.Config != nil {
		if _, ok := req.Config.Labels[version.AgentNameLabel]; ok {
			return
		}
		cfg = *req.Config
	}
	labels := maps.Clone(cfg.Labels)
	if labels == nil {
		labels = make(map[string]string, 1)
	}
	labels[version.AgentNameLabel] = name
	cfg.Labels = labels
	req.Config = &cfg
}

type activeTask struct {
	callID string
	cancel context.CancelFunc
}

type liveSessionImpl struct {
	inputCh      chan agent.LiveRequest
	outputCh     chan eventOrError
	done         chan struct{}
	closeOnce    sync.Once
	audioMgr     *AudioCacheManager
	waitForYield bool
	mu           sync.Mutex
	activeTools  map[string][]activeTask
	// finalErr buffers one terminal error for recvIter to deliver after
	// done is closed: the iterator cannot rely on outputCh (unbuffered)
	// once Close has been called, because the consumer may be parked in Send.
	finalMu  sync.Mutex
	finalErr error
}

// errSiblingCallStopped is the cancellation cause of the calls of one model
// turn after one of them paused or failed the run: it tells a call canceled
// for that reason from one that failed on its own or because the run ended.
var errSiblingCallStopped = errors.New("a sibling function call stopped the step")

// siblingCallStopped reports whether ctx was canceled because another call of
// its model turn paused or failed the run.
func siblingCallStopped(ctx context.Context) bool {
	return errors.Is(context.Cause(ctx), errSiblingCallStopped)
}

func (s *liveSessionImpl) setFinalError(err error) {
	s.finalMu.Lock()
	defer s.finalMu.Unlock()
	if s.finalErr == nil {
		s.finalErr = err
	}
}

func (s *liveSessionImpl) takeFinalError() error {
	s.finalMu.Lock()
	defer s.finalMu.Unlock()
	err := s.finalErr
	s.finalErr = nil
	return err
}

func (s *liveSessionImpl) RegisterStreamingTool(toolName, callID string, cancel context.CancelFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.activeTools == nil {
		s.activeTools = make(map[string][]activeTask)
	}
	s.activeTools[toolName] = append(s.activeTools[toolName], activeTask{
		callID: callID,
		cancel: cancel,
	})
}

func (s *liveSessionImpl) UnregisterStreamingTool(toolName, callID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	tasks, exists := s.activeTools[toolName]
	if !exists {
		return
	}
	for i, task := range tasks {
		if task.callID == callID {
			s.activeTools[toolName] = append(tasks[:i], tasks[i+1:]...)
			break
		}
	}
	if len(s.activeTools[toolName]) == 0 {
		delete(s.activeTools, toolName)
	}
}

func (s *liveSessionImpl) CancelAllStreamingTools(toolName string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	tasks, exists := s.activeTools[toolName]
	if !exists || len(tasks) == 0 {
		return false
	}
	for _, task := range tasks {
		task.cancel()
	}
	delete(s.activeTools, toolName)
	return true
}

type eventOrError struct {
	event *session.Event
	err   error
	taken chan bool
}

func newLiveSessionImpl() *liveSessionImpl {
	return &liveSessionImpl{
		inputCh:  make(chan agent.LiveRequest),
		outputCh: make(chan eventOrError),
		done:     make(chan struct{}),
		audioMgr: NewAudioCacheManager(),
	}
}

func (s *liveSessionImpl) Send(req agent.LiveRequest) error {
	select {
	case s.inputCh <- req:
		return nil
	case <-s.done:
		return io.EOF
	}
}

func (s *liveSessionImpl) recvIter() iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for {
			if err := s.takeFinalError(); err != nil {
				// Deliver the buffered terminal error; no more events follow.
				_ = yield(nil, err)
				return
			}
			select {
			case res := <-s.outputCh:
				ok := yield(res.event, res.err)
				if res.taken != nil {
					res.taken <- ok
				}
				if !ok {
					return
				}
			case <-s.done:
				// done closing and finalErr being set are not ordered with
				// the iterator's first check, so drain the terminal error
				// here too instead of dropping it.
				if err := s.takeFinalError(); err != nil {
					_ = yield(nil, err)
				}
				return
			}
		}
	}
}

func (s *liveSessionImpl) Close() error {
	s.closeOnce.Do(func() {
		close(s.done)
	})
	return nil
}

func (s *liveSessionImpl) pushEvent(ev *session.Event) bool {
	return s.pushLiveEvent(ev, false)
}

func (s *liveSessionImpl) pushCallEvent(ev *session.Event) bool {
	return s.pushLiveEvent(ev, true)
}

func (s *liveSessionImpl) pushLiveEvent(ev *session.Event, waitBeforeCalls bool) bool {
	var taken chan bool
	if waitBeforeCalls && s.waitForYield && ev != nil && !ev.LLMResponse.Partial {
		taken = make(chan bool, 1)
	}
	select {
	case s.outputCh <- eventOrError{event: ev, taken: taken}:
	case <-s.done:
		return false
	}
	if taken != nil {
		select {
		case ok := <-taken:
			return ok
		case <-s.done:
			return false
		}
	}
	return true
}

func (s *liveSessionImpl) pushError(err error) bool {
	select {
	case s.outputCh <- eventOrError{err: err}:
		return true
	case <-s.done:
		return false
	}
}

// tornDown reports whether the session was closed or the invocation canceled.
func tornDown(ctx context.Context, sess *liveSessionImpl) bool {
	select {
	case <-sess.done:
		return true
	case <-ctx.Done():
		return true
	default:
		return false
	}
}

// waitBeforeReconnect paces a reconnect, reporting whether it should go ahead.
// Teardown during the wait cuts it short, so Close stays prompt.
//
// The re-check after the timer matters: select picks uniformly among ready
// cases, so a teardown landing as the timer fires would otherwise report
// "proceed" half the time and dial a socket nobody reads.
func waitBeforeReconnect(ctx context.Context, sess *liveSessionImpl, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return !tornDown(ctx, sess)
	case <-sess.done:
		return false
	case <-ctx.Done():
		return false
	}
}

// isResumable reports whether a live-connection error means the socket is
// gone and the flow should reconnect, rather than surface the error and stop.
//
// Both the reader and the sender goroutine report into the same errChan and
// the flow acts on whichever arrives first, so the two must classify the same
// connection loss the same way. They do not produce the same text. The reader
// usually sees the websocket close ("close 1006 ... unexpected EOF"), and on a
// reset it can get the raw socket error instead. The sender sees the raw
// socket write failure, whose wording is platform-specific ("write: broken
// pipe" on Linux, "wsasend: An established connection was aborted by the
// software in your host machine." on Windows), and over wss:// it arrives
// wrapped by crypto/tls. Matching the *net.OpError by type rather than by text
// gives a dropped or reset connection the same verdict on both goroutines and
// on every platform.
//
// A timed-out connection does not always get one verdict, because the kernel
// reports the timeout to only one call. On Linux every later call sees a
// closed connection instead: a read gets EOF, which gorilla reports as
// close 1006, and a write gets EPIPE, and both resume. The call that does get
// ETIMEDOUT is fatal if gorilla/websocket hides it, which it does for any
// net.Error whose Temporary() is true (hideTempErr). That is the case for a
// read over ws:// or wss:// and for a write over ws://. crypto/tls wraps a
// failed write in an error whose Temporary() is always false, so a timed-out
// write over wss:// keeps the *net.OpError and resumes. So on Linux a
// timed-out connection resumes or ends depending on which goroutine reaches
// errChan first, unless the timeout lands on a wss:// write. On Windows the
// timeout is WSAETIMEDOUT (10060) and the other call gets WSAECONNRESET
// (10054). Go treats neither as temporary, so both keep the *net.OpError and
// the connection resumes whichever goroutine reports first.
func isResumable(err error) bool {
	if err == nil {
		return false
	}
	if err == io.EOF {
		return true
	}
	// A failed read or write on the underlying socket of an already-established
	// connection. The dial lives on a different path, which handles its own
	// errors, so reaching here means a live connection dropped.
	if _, ok := errors.AsType[*net.OpError](err); ok {
		return true
	}
	// Match the close code rather than the text: gorilla puts the socket
	// addresses in its error text, so a port such as 41008 would match "1008".
	var closeErr *websocket.CloseError
	if errors.As(err, &closeErr) && closeErr.Code == websocket.ClosePolicyViolation {
		return true
	}
	errStr := err.Error()
	return strings.Contains(errStr, "broken pipe") ||
		strings.Contains(errStr, "connection reset") ||
		strings.Contains(errStr, "EOF") ||
		strings.Contains(errStr, "GoAway")
}

// hasRealtimeInput reports whether v should be passed to SendRealtime. A nil
// pointer of one of the types LiveRequest.RealtimeInput documents carries no
// input, as a None field in adk-python does.
func hasRealtimeInput(v any) bool {
	switch v := v.(type) {
	case nil:
		return false
	case *genai.Blob:
		return v != nil
	case *genai.ActivityStart:
		return v != nil
	case *genai.ActivityEnd:
		return v != nil
	}
	return true
}

// isEmptyLiveResponse reports whether r carries nothing worth an event, as
// the response built from the Live API's setupComplete does. adk-python's live
// flow skips the same responses.
func isEmptyLiveResponse(r *model.LLMResponse) bool {
	return r.Content == nil && r.ErrorCode == "" && !r.Interrupted && !r.TurnComplete &&
		r.InputTranscription == nil && r.OutputTranscription == nil && r.UsageMetadata == nil &&
		r.SessionResumptionHandle == "" && r.GroundingMetadata == nil
}

func (f *Flow) RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	opts := liveOptionsFrom(ctx, f)
	if err := f.setStepModel(ctx, ResolveLiveModel); err != nil {
		return nil, nil, err
	}
	clientProvider, ok := f.stepModel.(interface {
		Client() *genai.Client
	})
	if !ok {
		return nil, nil, fmt.Errorf("model does not support live connection")
	}
	client := clientProvider.Client()
	if client == nil {
		return nil, nil, fmt.Errorf("agent %q: model %q has no live client; check the model's credentials and configuration", ctx.Agent().Name(), f.stepModel.Name())
	}

	live := opts.live
	if live == nil {
		runCfg := runconfig.FromContext(ctx)
		if runCfg == nil || runCfg.Live == nil {
			return nil, nil, fmt.Errorf("live run config not found")
		}
		live = runCfg.Live
	}

	sess := newLiveSessionImpl()
	sess.waitForYield = opts.waitForYield

	go func() {
		defer func() {
			_ = sess.Close()
		}()

		nreq := &model.LLMRequest{
			Model: f.stepModel.Name(),
		}
		// The live callable map is built from f.Tools, so it needs the names
		// of the tools that registered themselves without a declaration.
		dropped := make(map[string]struct{})
		for ev, err := range f.preprocessRecording(ctx, nreq, dropped) {
			if err != nil {
				sess.pushError(err)
				return
			}
			if ev != nil {
				if !sess.pushEvent(ev) {
					return
				}
			}
		}

		// The reconnect loop writes the resumption handle into this config, so
		// it needs its own copy: the caller may reuse its LiveRunConfig.
		sessionResumption := live.SessionResumption
		if sessionResumption != nil {
			sessionResumption = new(*sessionResumption)
		}

		liveConnectConfig := &genai.LiveConnectConfig{
			ResponseModalities:       live.ResponseModalities,
			SpeechConfig:             live.SpeechConfig,
			SystemInstruction:        nreq.Config.SystemInstruction,
			Tools:                    nreq.Config.Tools,
			SessionResumption:        sessionResumption,
			InputAudioTranscription:  live.InputAudioTranscription,
			OutputAudioTranscription: live.OutputAudioTranscription,
			RealtimeInputConfig:      live.RealtimeInputConfig,
		}
		// Leave affective dialog and proactivity unset for the models
		// IsGemini3xLive matches. Affective dialog is not supported on them,
		// and proactive audio is either always on (Gemini 3.8 Live, where
		// sending proactive_audio: false is an error) or not supported
		// (Gemini 3.1 Flash Live). Live Translate is excluded by
		// IsGemini3xLive and still gets both.
		if !googlellm.IsGemini3xLive(f.stepModel.Name()) {
			if live.EnableAffectiveDialog {
				liveConnectConfig.EnableAffectiveDialog = new(true)
			}
			liveConnectConfig.Proactivity = live.Proactivity
		}

		iCtx, isIContext := ctx.(*icontext.InvocationContext)

		policy := f.reconnect
		if policy == nil {
			policy = defaultLiveReconnectPolicy()
		}
		// reconnectAttempts counts consecutive reconnects since a connection
		// last worked and drives the backoff; a connection works by delivering
		// content or by outliving healthyUptime. shortLivedReconnects counts
		// only the connections that died inside healthyUptime, and never
		// resets.
		reconnectAttempts := 0
		shortLivedReconnects := 0
		currentBackoff := policy.initialBackoff
		// lastConnWasBrief records whether the connection that just dropped
		// died inside healthyUptime. A connection the server cycles after
		// minutes of service is routine, so it spends neither budget and clears
		// the consecutive one; charging it would let a long call die of old
		// age.
		lastConnWasBrief := false
		var lastErr error
		// isReconnect is false only for the very first dial; every path back to
		// the loop top sets it. It is what makes a first-ever connect failure
		// fatal while a failed redial is charged to the budget.
		isReconnect := false

		for {
			if isReconnect {
				reconnectAttempts++
				if lastConnWasBrief {
					shortLivedReconnects++
				}
				if reconnectAttempts > policy.maxAttempts {
					// Resumable errors are swallowed while retrying, so
					// without this the caller's stream just goes quiet.
					sess.pushError(liveReconnectGaveUpError(
						fmt.Sprintf("%d consecutive attempts delivered no content", policy.maxAttempts), lastErr))
					return
				}
				if shortLivedReconnects > policy.maxTotal {
					sess.pushError(liveReconnectGaveUpError(
						fmt.Sprintf("%d short-lived connections in one invocation", policy.maxTotal), lastErr))
					return
				}
				sleepDuration := policy.jittered(currentBackoff)
				currentBackoff = policy.nextBackoff(currentBackoff)

				log.Printf("live session: reconnect attempt %d/%d (%d short-lived) in %v", //nolint:forbidigo // pre-slog call site
					reconnectAttempts, policy.maxAttempts, shortLivedReconnects, sleepDuration)
				if !waitBeforeReconnect(ctx, sess, sleepDuration) {
					// Cancellation reports itself, matching the consumer
					// loop's ctx.Done arm; Close is a clean teardown.
					if err := ctx.Err(); err != nil {
						sess.pushError(err)
					}
					return
				}
			}
			if isIContext {
				handle := iCtx.LiveSessionResumptionHandle()
				if handle != "" {
					if liveConnectConfig.SessionResumption == nil {
						liveConnectConfig.SessionResumption = &genai.SessionResumptionConfig{}
					}
					liveConnectConfig.SessionResumption.Handle = handle
					if googlellm.GetGoogleLLMVariant(f.stepModel) == genai.BackendVertexAI {
						liveConnectConfig.SessionResumption.Transparent = true
					}
				}
			}
			// TODO(kdroste): refactor underlying context
			connCtx, cancelConn := context.WithCancel(ctx)

			if liveConnectConfig.SessionResumption != nil {
				log.Printf("connecting with live session handle: %s\n", liveConnectConfig.SessionResumption.Handle) //nolint:forbidigo // pre-slog call site
			}
			liveSession, err := client.Live.Connect(connCtx, f.stepModel.Name(), liveConnectConfig)
			if err != nil {
				cancelConn()
				log.Printf("failed to connect live session: %v\n", err) //nolint:forbidigo // pre-slog call site
				if isReconnect {
					// genai returns without closing the socket it dialed when
					// the setup write fails, and it dials with a context-less
					// dialer, so cancelConn cannot release it either. Each
					// budgeted redial against such an endpoint strands one fd
					// until the process exits; the bounds above cap how many.
					lastErr = err
					lastConnWasBrief = true
					continue
				}
				sess.pushError(fmt.Errorf("failed to connect live session: %w", err))
				return
			}

			connectedAt := time.Now()
			liveConn := googlellm.NewLiveConnection(liveSession, f.stepModel.Name())

			cleanup := func() {
				cancelConn()
				_ = liveConn.Close()
			}

			eventsChan := make(chan *session.Event)
			// Producers must guard sends with connCtx: once a reconnect (or any
			// teardown) abandons this unbuffered channel, cleanup's cancelConn is
			// what unblocks them (issue #1152).
			errChan := make(chan liveConnError)

			// Send preprocessed content directly to model if any exists after early preprocessing
			if len(nreq.Contents) > 0 {
				if err := liveConn.SendHistory(ctx, nreq.Contents); err != nil {
					log.Printf("failed to send history: %v\n", err) //nolint:forbidigo // pre-slog call site
					sess.pushError(err)
					// cleanup, not a bare return: genai dials the live socket
					// with a context-less websocket.DefaultDialer.Dial, and
					// nothing in the SDK watches a context, so only an explicit
					// Close releases it. Returning without cleanup strands the
					// connection for the life of the process.
					cleanup()
					return
				}
			}

			// Reading from model loop
			go func() {
				for {
					resp, err := liveConn.Recv(connCtx)
					if err != nil {
						select {
						case errChan <- liveConnError{err: err, at: time.Now()}:
						case <-connCtx.Done():
						}
						return
					}
					if resp != nil {
						if resp.SessionResumptionHandle != "" {
							if isIContext {
								log.Printf("received session resumption handle: %s\n", resp.SessionResumptionHandle) //nolint:forbidigo // pre-slog call site
								iCtx.SetLiveSessionResumptionHandle(resp.SessionResumptionHandle)
							}
						}
						if live.SaveLiveBlob && resp.Content != nil {
							for _, part := range resp.Content.Parts {
								if part.InlineData != nil {
									sess.audioMgr.CacheOutput(ctx, part.InlineData.Data, part.InlineData.MIMEType)
								}
							}
						}
						if isEmptyLiveResponse(resp) {
							continue
						}
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = ctx.Agent().Name()
						ev.LLMResponse = *resp
						select {
						case eventsChan <- ev:
						case <-connCtx.Done():
							return
						}
					}
				}
			}()

			// Sending to model loop
			go func() {
				for {
					select {
					case <-connCtx.Done():
						return
					case req, ok := <-sess.inputCh:
						if !ok {
							return
						}
						if req.Content != nil {
							if err := liveConn.SendContent(connCtx, req.Content); err != nil {
								select {
								case errChan <- liveConnError{err: err, at: time.Now()}:
								case <-connCtx.Done():
								}
								return
							}
						}
						if hasRealtimeInput(req.RealtimeInput) {
							if blob, ok := req.RealtimeInput.(*genai.Blob); ok {
								sess.audioMgr.CacheInput(ctx, blob.Data, blob.MIMEType)
							}
							if err := liveConn.SendRealtime(connCtx, req.RealtimeInput); err != nil {
								select {
								case errChan <- liveConnError{err: err, at: time.Now()}:
								case <-connCtx.Done():
								}
								return
							}
						}
					}
				}
			}()

			// pendingSend holds a function response awaiting delivery to the
			// model, queued on sess.inputCh so the sender goroutine is the
			// connection's only writer; a direct write from this loop races
			// it and panics (#1819). It is declared inside the per-connection
			// loop, so a response pending when the connection drops is
			// discarded with the socket: carrying one across needs the value
			// to survive until the write is confirmed, which stays open on
			// #1819.
			var pendingSend *genai.Content

			reconnect := false
			for !reconnect {
				// Arm the inputCh send only while a response is pending, and
				// hold further events until it is delivered, so a second tool
				// response cannot overwrite the first (nil channel cases never
				// fire in a select).
				eventCase := eventsChan
				var inputCase chan agent.LiveRequest
				var inputReq agent.LiveRequest
				if pendingSend != nil {
					inputCase = sess.inputCh
					inputReq = agent.LiveRequest{Content: pendingSend}
					eventCase = nil
				}
				select {
				case ev := <-eventCase:
					// The response processors may rewrite ev's parts, and the
					// events they emit belong before ev, as in adk-python's
					// live flow. Their errors end the session the way a request
					// processor error does.
					for pev, err := range f.postprocess(ctx, nreq, &ev.LLMResponse) {
						if err != nil {
							sess.pushError(err)
							cleanup()
							return
						}
						if pev != nil && !sess.pushEvent(pev) {
							cleanup()
							return
						}
					}
					if opts.dropEmptyResponses && isEmptyLiveResponse(&ev.LLMResponse) {
						continue
					}
					// adk-python builds transcription and session resumption
					// events without the live session id
					// (live/_live_llm_flow.py:269-273, :422-449).
					if ev.LLMResponse.Content == nil && !ev.TurnComplete && !ev.Interrupted && (ev.InputTranscription != nil || ev.OutputTranscription != nil || ev.SessionResumptionHandle != "") {
						ev.LiveSessionID = ""
					}
					// After the response processors, as in adk-python: they see
					// the calls as the model sent them, and a call they add
					// still gets an id.
					utils.PopulateClientFunctionCallID(ctx, ev.LLMResponse.Content)
					fnCalls := utils.FunctionCalls(ev.LLMResponse.Content)
					var tools map[string]tool.Tool
					if len(fnCalls) > 0 {
						tools = make(map[string]tool.Tool, len(f.Tools)+len(nreq.Tools))
						for _, t := range f.Tools {
							if _, ok := dropped[t.Name()]; ok {
								continue
							}
							tools[t.Name()] = t
						}
						// Tools the request processors packed, such as
						// set_model_response, are callable too, as in the
						// normal loop; a name in f.Tools keeps its tool.
						for name, v := range nreq.Tools {
							if _, ok := tools[name]; ok {
								continue
							}
							t, ok := v.(tool.Tool)
							if !ok {
								if !sess.pushError(fmt.Errorf("unexpected tool type %T for tool %v", v, name)) {
									cleanup()
									return
								}
								continue
							}
							tools[name] = t
						}
						ev.LongRunningToolIDs = findLongRunningFunctionCallIDs(ev.LLMResponse.Content, tools)
					}
					var pushed bool
					if len(fnCalls) > 0 {
						pushed = sess.pushCallEvent(ev)
					} else {
						pushed = sess.pushEvent(ev)
					}
					if !pushed {
						cleanup()
						return
					}
					// Content proves this connection is serving, so the
					// consecutive budget starts over; shortLivedReconnects does
					// not, which is what bounds a backend that serves one frame
					// per connection and then hangs up.
					if ev != nil && ev.LLMResponse.Content != nil {
						reconnectAttempts = 0
						currentBackoff = policy.initialBackoff
					}
					// Flush caches if needed
					if live.SaveLiveBlob {
						var flushUser, flushModel bool
						if ev.LLMResponse.Interrupted {
							flushModel = true
						}
						if ev.LLMResponse.TurnComplete {
							flushUser = true
							flushModel = true
						}
						if flushUser || flushModel {
							flushedEvents, err := sess.audioMgr.FlushCaches(ctx, flushUser, flushModel)
							if err != nil {
								sess.pushError(err)
								cleanup()
								return
							}
							for _, fev := range flushedEvents {
								if !sess.pushEvent(fev) {
									cleanup()
									return
								}
							}
						}
					}
					if len(fnCalls) > 0 {
						respEv, err := f.handleFunctionCallsInOrder(ctx, tools, registryOrder(nreq, f.Tools, tools), &ev.LLMResponse, nil, sess)
						if err != nil {
							sess.pushError(err)
							cleanup()
							return
						}
						if respEv != nil {
							// The response event belongs to the connection that
							// sent the calls; a non-live run sets no id.
							respEv.LiveSessionID = ev.LiveSessionID
							if !sess.pushEvent(respEv) {
								cleanup()
								return
							}
							// A stored set_model_response answer ends the turn
							// with a final event. The model still gets its
							// function response below, as in adk-python's
							// live flow.
							final, err := structuredFinalEvent(ctx, respEv)
							if err != nil {
								sess.pushError(err)
								cleanup()
								return
							}
							if final != nil && !sess.pushEvent(final) {
								cleanup()
								return
							}
							// Check if task_completed was invoked.
							var isTaskCompleted bool
							if respEv.LLMResponse.Content != nil {
								for _, part := range respEv.LLMResponse.Content.Parts {
									if part.FunctionResponse != nil && part.FunctionResponse.Name == "task_completed" {
										isTaskCompleted = true
										break
									}
								}
							}
							if isTaskCompleted {
								time.Sleep(100 * time.Millisecond)
								cleanup()
								return
							}
							// Queue the function response for the sender
							// goroutine, the connection's only writer.
							pendingSend = respEv.LLMResponse.Content
						}
					}
				case inputCase <- inputReq:
					pendingSend = nil
				case ce := <-errChan:
					if isResumable(ce.err) {
						log.Printf("Connection error, attempting to resume: %v\n", ce.err) //nolint:forbidigo // pre-slog call site
						lastErr = ce.err
						// Score the connection's life from where the error was
						// produced, not from here: this loop also runs tools
						// and blocks on the caller, and crediting that time as
						// uptime would let a flapping backend pass as healthy.
						lastConnWasBrief = ce.at.Sub(connectedAt) < policy.healthyUptime
						if !lastConnWasBrief {
							// The connection served for as long as a healthy
							// one does, which is the only evidence a session
							// the model has nothing to say on ever produces.
							// Without this the consecutive budget ends such a
							// call after maxAttempts of the cycles the Live API
							// performs as ordinary lifecycle.
							reconnectAttempts = 0
							currentBackoff = policy.initialBackoff
						}
						reconnect = true
						break // Break the select
					}
					// pushError would park here until the consumer pulls from
					// outputCh, but the consumer may already be blocked in Send,
					// which only Close can release. Buffer the terminal error so
					// the iterator can deliver it; the deferred Close above ends
					// waits in Send and recvIter when this goroutine returns.
					sess.setFinalError(ce.err)
					cleanup()
					return
				case <-sess.done:
					cleanup()
					return
				case <-ctx.Done():
					sess.pushError(ctx.Err())
					cleanup()
					return
				}
				if reconnect {
					break // Break the for loop
				}
			}

			// Cleanup before reconnecting
			cleanup()

			if !reconnect {
				break
			}
			isReconnect = true
		}
	}()

	return sess, sess.recvIter(), nil
}

func (f *Flow) runOneStep(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if err := f.setStepModel(ctx, ResolveModel); err != nil {
			yield(nil, err)
			return
		}

		req := &model.LLMRequest{
			Model: f.stepModel.Name(),
		}

		// Preprocess before calling the LLM.
		preprocessYieldedFinalResponse := false
		for ev, err := range f.preprocess(ctx, req) {
			if err != nil {
				yield(nil, err)
				return
			}
			if ev != nil {
				if len(utils.FunctionResponses(ev.Content)) > 0 && ev.IsFinalResponse() {
					preprocessYieldedFinalResponse = true
				}
				if !yield(ev, nil) {
					return
				}
			}
		}
		// A preprocessor can run a tool itself, such as a call resumed after
		// confirmation. When that tool's response is final, because the tool
		// set SkipSummarization, the turn ends there, as it does when the tool
		// runs after a model call; calling the model now would summarize what
		// the tool asked to be the answer. adk-python stops on the same
		// condition (base_llm_flow.py).
		if ctx.Ended() || preprocessYieldedFinalResponse {
			return
		}
		tools, err := extractTools(req)
		if err != nil {
			yield(nil, err)
			return
		}
		// A resumed step first settles what its branch already holds: a call
		// still waiting for its answer pauses the step, and calls that never
		// ran run before the model is asked again, as in adk-python
		// (base_llm_flow.py).
		switch d := decideStepResume(ctx, tools); d.action {
		case resumeActionPause:
			return
		case resumeActionReplayCalls:
			f.postprocessFunctionCalls(ctx, tools, registryOrder(req, f.Tools, tools), d.event, yield)
			return
		}
		// Create event to pass to callback state delta
		stateDelta := make(map[string]any)
		artifactDelta := make(map[string]int64)
		// liveTransfer names the agent that a live event of a CFC model call
		// transfers to. The call ends after that event (callLLMOverLive).
		liveTransfer := ""
		// Calls the LLM.
		for resp, err := range f.callLLM(ctx, req, stateDelta, artifactDelta) {
			if err != nil {
				yield(nil, err)
				return
			}
			// RunLive has already post-processed a live event and run its
			// function calls, as adk-python's live flow does, so it is passed
			// on as it is.
			if resp.liveEvent != nil {
				if !yield(resp.liveEvent, nil) {
					return
				}
				if resp.liveEvent.Actions.TransferToAgent != "" {
					liveTransfer = resp.liveEvent.Actions.TransferToAgent
				}
				continue
			}
			applyEmptyResponsePolicy(ctx, resp.LLMResponse)
			for ev, err := range f.postprocess(ctx, req, resp.LLMResponse) {
				if err != nil {
					yield(nil, err)
					return
				}
				if ev != nil && !yield(ev, nil) {
					return
				}
			}
			// Skip the model response event if it carries no content, error code,
			// interruption or grounding metadata.
			// This is needed for the code executor to trigger another loop according to
			// adk-python src/google/adk/flows/llm_flows/base_llm_flow.py BaseLlmFlow._postprocess_async.
			if resp.Content == nil && resp.ErrorCode == "" && !resp.Interrupted && resp.GroundingMetadata == nil {
				continue
			}

			// A before-model callback may have registered tools after preprocess.
			if tools, err = extractTools(req); err != nil {
				yield(nil, err)
				return
			}

			// Build the event and yield.
			modelResponseEvent := f.finalizeModelResponseEvent(ctx, resp, tools, stateDelta, artifactDelta)
			if !yield(modelResponseEvent, nil) {
				return
			}

			if resp.Partial {
				continue
			}
			if !f.postprocessFunctionCalls(ctx, tools, registryOrder(req, f.Tools, tools), modelResponseEvent, yield) {
				return
			}
		}
		// The target runs once the CFC model call has ended, so that the
		// transferring agent's live connection is closed first, as
		// adk-python's live flow closes it before the transfer
		// (live/_live_llm_flow.py).
		if liveTransfer != "" {
			f.runTransfer(ctx, liveTransfer, yield)
		}
	}
}

// extractTools returns the tools of req by name, leaving out every entry
// without a declaration. It fails when an entry is not a tool.Tool.
func extractTools(req *model.LLMRequest) (map[string]tool.Tool, error) {
	tools := make(map[string]tool.Tool, len(req.Tools))
	for k, v := range req.Tools {
		t, ok := v.(tool.Tool)
		if !ok {
			return nil, fmt.Errorf("unexpected tool type %T for tool %v", v, k)
		}
		// An entry registered after preprocess, for example by a
		// before-model callback, is still subject to the rule.
		if undeclared(t) {
			continue
		}
		tools[k] = t
	}
	return tools, nil
}

// postprocessFunctionCalls runs the function calls of functionCallEvent with
// tools, whose names toolOrder lists in registry order, then yields their
// response event, any credential request, any tool confirmation request and
// any structured final answer, and runs the agent that a transfer_to_agent
// call names. A model turn and a replay of calls that never ran both go
// through it.
//
// It reports whether the step may go on to the model's next response: true
// when no call produced a response or when a transfer target ran to its end,
// false when the step ends, because the calls ran without a transfer, an
// error was yielded or the consumer stopped.
func (f *Flow) postprocessFunctionCalls(ctx agent.InvocationContext, tools map[string]tool.Tool, toolOrder []string, functionCallEvent *session.Event, yield func(*session.Event, error) bool) bool {
	ev, err := f.handleFunctionCallsInOrder(ctx, tools, toolOrder, &functionCallEvent.LLMResponse, nil, nil)
	if err != nil {
		yield(nil, err)
		return false
	}
	if ev == nil {
		// nothing to yield/process.
		return true
	}

	toolConfirmationEvent := generateRequestConfirmationEvent(ctx, functionCallEvent, ev)

	// Yield function responses before confirmation and credential requests
	// so consumers that pause for the user still persist completed tool
	// results.
	if !yieldWithAuthRequest(ctx, ev, yield) {
		return false
	}

	if toolConfirmationEvent != nil {
		if !yield(toolConfirmationEvent, nil) {
			return false
		}
	}

	// If the model response is structured, yield it as a final model response event.
	final, err := structuredFinalEvent(ctx, ev)
	if err != nil {
		yield(nil, err)
		return false
	}
	if final != nil {
		if !yield(final, nil) {
			return false
		}
	}
	// Actually handle "transfer_to_agent" tool. The function call sets the ev.Actions.TransferToAgent field.
	// We are following python's execution flow which is
	//   BaseLlmFlow._postprocess_async
	//    -> _postprocess_handle_function_calls_async
	// TODO(hakim): figure out why this isn't handled by the runner.
	if ev.Actions.TransferToAgent == "" {
		return false
	}
	return f.runTransfer(ctx, ev.Actions.TransferToAgent, yield)
}

// runTransfer runs the agent that a transfer from ctx's agent to the agent
// named agentName runs (see agentToRun) and yields its events. A unary model
// call and a CFC model call both transfer through it.
//
// It reports whether the target ran to its end: false when no agent may run,
// in which case the error is yielded, when the target yielded an error, or
// when the consumer stopped.
func (f *Flow) runTransfer(ctx agent.InvocationContext, agentName string, yield func(*session.Event, error) bool) bool {
	nextAgent, err := f.agentToRun(ctx, agentName)
	if err != nil {
		yield(nil, err)
		return false
	}
	// On a resumable run on the node path, the transferring agent ends
	// before its target runs, as adk-python's node wrapper ends it
	// (workflow/_llm_agent_wrapper.py). On the classic path its own run
	// writes its end after the target returns, as adk-python's legacy
	// path does.
	if rc := runconfig.FromContext(ctx); rc != nil && rc.Resumable && rc.NodePath {
		if !yield(NewEndOfAgentEvent(ctx, ctx.Agent().Name()), nil) {
			return false
		}
	}
	for ev, err := range RunTransferTarget(ctx, nextAgent) {
		if !yield(ev, err) || err != nil { // forward
			return false
		}
	}
	return true
}

// preprocess runs the request processors, then the tools and toolsets of the
// agent, over req. See preprocessRecording.
func (f *Flow) preprocess(ctx agent.InvocationContext, req *model.LLMRequest) iter.Seq2[*session.Event, error] {
	return f.preprocessRecording(ctx, req, nil)
}

// preprocessRecording is preprocess. When it has run the last tool and toolset
// ProcessRequest without error, it deletes from req.Tools every entry for
// which undeclared is true. When dropped is not nil it also records each
// deleted name in dropped.
func (f *Flow) preprocessRecording(ctx agent.InvocationContext, req *model.LLMRequest, dropped map[string]struct{}) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// apply request processor functions to the request in the configured order.
		for _, processor := range f.RequestProcessors {
			for ev, err := range processor(ctx, req, f) {
				if err != nil {
					yield(nil, err)
					return
				}
				if ev != nil && !yield(ev, nil) {
					return
				}
			}
		}

		// Tools declare themselves for the request's backend.
		toolCtx := ctx.WithContext(googlellm.ContextWithBackend(ctx, googlellm.GetGoogleLLMVariant(f.currentModel())))
		if err := processToolRequests(toolCtx, req, f); err != nil {
			yield(nil, err)
			return
		}
		// A tool without a declaration is not registered, as in adk-python's
		// LlmRequest.append_tools, so the model cannot call it.
		maps.DeleteFunc(req.Tools, func(name string, t any) bool {
			if !undeclared(t) {
				return false
			}
			if dropped != nil {
				dropped[name] = struct{}{}
			}
			return true
		})
	}
}

// undeclared reports whether t has a Declaration method that returns nil. A
// nil pointer, and a value without a Declaration method, are not undeclared.
func undeclared(t any) bool {
	d, ok := t.(interface {
		Declaration() *genai.FunctionDeclaration
	})
	if !ok {
		return false
	}
	if v := reflect.ValueOf(t); v.Kind() == reflect.Pointer && v.IsNil() {
		return false
	}
	return d.Declaration() == nil
}

// toolPreprocess runs tool preprocess on the given request
// If a tool set is encountered, it's expanded recursively in DFS fashion.
// TODO: check need/feasibility of running this concurrently.
func toolPreprocess(ctx agent.InvocationContext, req *model.LLMRequest, tools []tool.Tool) error {
	for _, t := range tools {
		requestProcessor, ok := t.(toolinternal.RequestProcessor)
		if !ok {
			return fmt.Errorf("tool %q does not implement RequestProcessor() method", t.Name())
		}
		// TODO: how to prevent mutation on this?
		toolCtx := agent.NewToolContext(ctx, "", &session.EventActions{}, nil)
		if err := requestProcessor.ProcessRequest(toolCtx, req); err != nil {
			return err
		}
	}
	return nil
}

// processToolRequests runs the request hooks of the agent's own tools, then,
// for each of the agent's toolsets in order, the toolset's hook followed by
// the hooks of the tools it listed in this step, as adk-python's flow does
// (flows/llm_flows/tools/_agent_tools.py). The groups are positional slices of
// f.Tools, so a tool that is both an own tool and listed by a toolset is
// processed in both groups. Before toolProcessor has run, all of f.Tools
// counts as own.
func processToolRequests(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) error {
	own := f.Tools
	if f.toolsetTools != nil {
		own = f.Tools[:f.ownToolCount]
	}
	if err := toolPreprocess(ctx, req, own); err != nil {
		return err
	}
	llmAgent, ok := ctx.Agent().(Agent)
	if !ok {
		return nil
	}
	for i, toolset := range Reveal(llmAgent).Toolsets {
		if processor, ok := toolset.(toolinternal.RequestProcessor); ok {
			toolCtx := agent.NewToolContext(ctx, "", nil, nil)
			if err := processor.ProcessRequest(toolCtx, req); err != nil {
				return fmt.Errorf("process request by toolset %q: %w", toolset.Name(), err)
			}
		}
		var group []tool.Tool
		if i < len(f.toolsetTools) {
			group = f.toolsetTools[i]
		}
		if err := toolPreprocess(ctx, req, group); err != nil {
			return err
		}
	}
	return nil
}

func newResponseWithEventID(ctx context.Context, resp *model.LLMResponse) *responseWithEventID {
	return &responseWithEventID{LLMResponse: resp, eventID: platform.NewUUID(ctx)}
}

func (f *Flow) callLLM(ctx agent.InvocationContext, req *model.LLMRequest, stateDelta map[string]any, artifactDelta map[string]int64) iter.Seq2[*responseWithEventID, error] {
	return func(yieldResponse func(*responseWithEventID, error) bool) {
		// Every model callback of this call records into one actions value, as
		// adk-python's share the model response event's actions. State and
		// artifact writes land in the caller's maps; the widgets rendered so far
		// travel with each response to its event.
		actions := &session.EventActions{StateDelta: stateDelta, ArtifactDelta: artifactDelta}
		yield := func(resp *responseWithEventID, err error) bool {
			if resp != nil {
				resp.uiWidgets = slices.Clone(actions.RenderUIWidgets)
			}
			return yieldResponse(resp, err)
		}

		pluginManager := pluginManagerFromContext(ctx)
		if pluginManager != nil {
			cctx := agent.NewCallbackContextWithArtifactTracking(ctx, actions)
			callbackResponse, callbackErr := pluginManager.RunBeforeModelCallback(cctx, req)
			if callbackResponse != nil || callbackErr != nil {
				yield(newResponseWithEventID(ctx, callbackResponse), callbackErr)
				return
			}
		}

		for _, callback := range f.BeforeModelCallbacks {
			cctx := agent.NewCallbackContextWithArtifactTracking(ctx, actions)
			callbackResponse, callbackErr := callback(cctx, req)

			if callbackResponse != nil || callbackErr != nil {
				yield(newResponseWithEventID(ctx, callbackResponse), callbackErr)
				return
			}
		}

		stampAgentNameLabel(ctx, req)

		// Streaming mode comes from the invocation context's RunConfig rather than
		// the runconfig context value: agent.Run() may be invoked directly (outside
		// the runner), in which case no runconfig is stored in the Go context and
		// runconfig.FromContext returns nil (issue #586).
		useStream := false
		if rc := ctx.RunConfig(); rc != nil {
			useStream = rc.StreamingMode == agent.StreamingModeSSE
		}

		// Count this call against the invocation's budget before making it. The
		// run config is absent when an agent is run directly rather than through
		// the runner (issue #586), and that imposes no limit.
		if err := runconfig.FromContext(ctx).IncrementAndEnforceLLMCallsLimit(); err != nil {
			yield(nil, err)
			return
		}

		if rc := ctx.RunConfig(); rc != nil && rc.SupportCFC {
			f.callLLMOverLive(ctx, req, actions, yield)
			return
		}

		for resp, err := range generateContent(ctx, f.currentModel(), req, useStream) {
			if err != nil {
				cbResp, cbErr := f.runOnModelErrorCallbacks(ctx, req, actions, err)
				if cbErr != nil {
					yield(nil, cbErr)
					return
				}
				if cbResp == nil {
					yield(nil, err)
					return
				}
				resp = &responseWithEventID{
					LLMResponse: cbResp,
					eventID:     resp.eventID,
				}
				err = cbErr
			}
			// Function call ID is optional in genai API and some models do not use the field.
			// Set it in case after model callbacks use it.
			utils.PopulateClientFunctionCallID(ctx, resp.Content)

			callbackResp, callbackErr := f.runAfterModelCallbacks(ctx, resp.LLMResponse, actions, err)
			// TODO: check if we should stop iterator on the first error from stream or continue yielding next results.
			if callbackErr != nil {
				yield(nil, callbackErr)
				return
			}

			if callbackResp != nil {
				resp := &responseWithEventID{
					LLMResponse: inheritModelFields(resp.LLMResponse, callbackResp),
					eventID:     resp.eventID,
				}
				if !yield(resp, nil) {
					return
				}
				continue
			}

			// TODO: check if we should stop iterator on the first error from stream or continue yielding next results.
			if err != nil {
				yield(nil, err)
				return
			}

			if !yield(resp, nil) {
				return
			}
		}
	}
}

// inheritModelFields carries fields that describe the model call, rather than
// the response content, from a response onto the after-model callback's
// replacement for it.
//
// A callback that rebuilds the response, for example to scrub its text, is
// still answering the same model call. If the replacement drops Partial, every
// streamed delta looks final: SSE clients render each delta as a complete
// turn and the runner persists each one as a separate session event. If it
// drops TurnComplete, clients never see the turn end. If it drops
// UsageMetadata, the call disappears from token accounting, and context
// caching, which sizes a cache from the previous prompt's token count, never
// creates one. If it drops CacheMetadata, the session loses track of the cache
// the call used: the next turn creates another instead of reusing it.
//
// A delta is never final while the stream continues, so a replacement that
// leaves Partial unset takes Partial and PartialSet from the original. Partial
// counts as set when it is true or PartialSet is true, so a replacement with
// Partial: false and PartialSet: true keeps its explicit false and becomes a
// non-partial event, while later chunks of the stream still follow.
// TurnComplete is a plain bool, so an unset value cannot be told apart from an
// explicit false; a response that ended the turn keeps its replacement ending
// it. UsageMetadata and CacheMetadata set on the replacement are respected.
// The replacement is copied rather than modified, and returned as is when
// nothing needs filling.
//
// adk-python carries the first three fields but not the cache state
// (flows/llm_flows/core/_finalizer.py, _inherit_unset_model_fields), so a
// callback that replaces every reply leaves it caching nothing.
func inheritModelFields(original, replacement *model.LLMResponse) *model.LLMResponse {
	if original == nil {
		return replacement
	}
	replacementHasPartial := replacement.Partial || replacement.PartialSet
	originalHasPartial := original.Partial || original.PartialSet
	inheritPartial := originalHasPartial && !replacementHasPartial
	inheritTurnComplete := original.TurnComplete && !replacement.TurnComplete
	inheritUsage := original.UsageMetadata != nil && replacement.UsageMetadata == nil
	inheritCache := original.CacheMetadata != nil && replacement.CacheMetadata == nil
	if !inheritPartial && !inheritTurnComplete && !inheritUsage && !inheritCache {
		return replacement
	}
	merged := *replacement
	if inheritPartial {
		merged.Partial = original.Partial
		merged.PartialSet = original.PartialSet
	}
	if inheritTurnComplete {
		merged.TurnComplete = true
	}
	if inheritUsage {
		merged.UsageMetadata = original.UsageMetadata
	}
	if inheritCache {
		merged.CacheMetadata = original.CacheMetadata
	}
	return &merged
}

type responseWithEventID struct {
	*model.LLMResponse
	eventID string
	// uiWidgets are the widgets the model callbacks of the call rendered
	// before this response was yielded, for its model response event.
	uiWidgets []session.UIWidget
	// liveEvent is set only by a CFC model call. It is the live event itself,
	// already post-processed by RunLive; runOneStep yields it as it is.
	liveEvent *session.Event
}

// generateContent wraps the LLM call with tracing and logging.
// The generate_content span should cover only calls to LLM. Plugins and callbacks should be outside of this span.
func generateContent(ctx agent.InvocationContext, m model.LLM, req *model.LLMRequest, useStream bool) iter.Seq2[*responseWithEventID, error] {
	params := telemetry.GenerateContentParams{
		ModelName:    m.Name(),
		InvocationID: ctx.InvocationID(),
		Request:      req,
		Backend:      googlellm.GetGoogleLLMVariant(m),
		Stream:       useStream,
	}
	response := func(r *responseWithEventID) (*model.LLMResponse, string) { return r.LLMResponse, r.eventID }
	return telemetry.InstrumentGenerateContent(ctx, params, response, func(spanCtx context.Context) iter.Seq2[*responseWithEventID, error] {
		ctx := ctx.WithContext(spanCtx)
		return func(yield func(*responseWithEventID, error) bool) {
			// The partial chunks of a streamed response share one event id with
			// the response that completes them. A complete response ends the id,
			// and so does an error, because an on-model-error callback may turn
			// it into a saved event. A new id is drawn when the next response
			// arrives, never ahead of one.
			var eventID string
			for resp, err := range m.GenerateContent(ctx, req, useStream) {
				if resp == nil && err == nil {
					// Third-party model implementations may yield an empty response.
					continue
				}
				if eventID == "" {
					eventID = platform.NewUUID(ctx)
				}
				if !yield(&responseWithEventID{LLMResponse: resp, eventID: eventID}, err) {
					return
				}
				// resp is non-nil whenever err is nil; the empty case is skipped above.
				if err != nil || !resp.Partial {
					eventID = ""
				}
			}
		}
	})
}

func (f *Flow) runAfterModelCallbacks(ctx agent.InvocationContext, llmResp *model.LLMResponse, actions *session.EventActions, llmErr error) (*model.LLMResponse, error) {
	pluginManager := pluginManagerFromContext(ctx)
	if pluginManager != nil {
		cctx := agent.NewCallbackContextWithArtifactTracking(ctx, actions)
		callbackResponse, callbackErr := pluginManager.RunAfterModelCallback(cctx, llmResp, llmErr)
		if callbackResponse != nil || callbackErr != nil {
			return callbackResponse, callbackErr
		}
	}

	for _, callback := range f.AfterModelCallbacks {
		cctx := agent.NewCallbackContextWithArtifactTracking(ctx, actions)
		callbackResponse, callbackErr := callback(cctx, llmResp, llmErr)

		if callbackResponse != nil || callbackErr != nil {
			return callbackResponse, callbackErr
		}
	}

	return nil, nil
}

func (f *Flow) runOnModelErrorCallbacks(ctx agent.InvocationContext, llmReq *model.LLMRequest, actions *session.EventActions, llmErr error) (*model.LLMResponse, error) {
	pluginManager := pluginManagerFromContext(ctx)
	if pluginManager != nil {
		cctx := agent.NewCallbackContextWithArtifactTracking(ctx, actions)
		callbackResponse, callbackErr := pluginManager.RunOnModelErrorCallback(cctx, llmReq, llmErr)
		if callbackResponse != nil || callbackErr != nil {
			return callbackResponse, callbackErr
		}
	}

	for _, callback := range f.OnModelErrorCallbacks {
		cctx := agent.NewCallbackContextWithArtifactTracking(ctx, actions)
		callbackResponse, callbackErr := callback(cctx, llmReq, llmErr)

		if callbackResponse != nil || callbackErr != nil {
			return callbackResponse, callbackErr
		}
	}

	return nil, nil
}

// postprocess applies the response processors to resp in the configured
// order and yields the events they emit, which belong before the model
// response event. It stops at the first error.
func (f *Flow) postprocess(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if resp == nil {
			return
		}
		for _, processor := range f.ResponseProcessors {
			for ev, err := range processor(ctx, req, resp) {
				if !yield(ev, err) {
					return
				}
				if err != nil {
					return
				}
			}
		}
	}
}

// agentToRun returns the agent that a transfer from ctx's agent to the agent
// named agentName runs, as adk-python's get_agent_to_run does
// (flows/llm_flows/core/_function_call_postprocessor.py). It looks first at
// the caller itself, then at the caller's transfer targets, so that a target
// wins over a same-named agent elsewhere in the tree.
//
// The caller's own name runs the caller again, except on the node path, where
// it is an error, as in adk-python's workflow scheduler
// (workflow/utils/_transfer_utils.py). Another agent of the tree that is not a
// target is an error matching [agent.ErrInvalidTransferTarget]. A name no
// agent of the tree has is an error too.
func (f *Flow) agentToRun(ctx agent.InvocationContext, agentName string) (agent.Agent, error) {
	cur := ctx.Agent()
	if agentName == cur.Name() {
		if rc := runconfig.FromContext(ctx); rc != nil && rc.NodePath {
			return nil, fmt.Errorf("Agent '%s' cannot transfer to itself.", agentName) //nolint:staticcheck // ST1005: the message text is adk-python's
		}
		return cur, nil
	}
	for _, target := range transferTargets(cur, parentOf(ctx, cur)) {
		if target.Name() == agentName {
			return target, nil
		}
	}
	if rootOf(ctx, cur).FindAgent(agentName) == nil {
		return nil, fmt.Errorf("failed to find agent: %s", agentName)
	}
	return nil, invalidTransferError{caller: cur.Name(), target: agentName}
}

// invalidTransferError is the error of a transfer to an agent of the tree
// that is not among the caller's transfer targets. Its text is adk-python's.
type invalidTransferError struct {
	caller, target string
}

func (e invalidTransferError) Error() string {
	return fmt.Sprintf("Agent %s is not allowed to transfer to agent %s.", e.caller, e.target)
}

// Is makes the error match agent.ErrInvalidTransferTarget.
func (e invalidTransferError) Is(target error) bool {
	return target == agent.ErrInvalidTransferTarget
}

func (f *Flow) finalizeModelResponseEvent(ctx agent.InvocationContext, resp *responseWithEventID, tools map[string]tool.Tool, stateDelta map[string]any, artifactDelta map[string]int64) *session.Event {
	// FunctionCall & FunctionResponse matching algorithm assumes non-empty function call IDs
	// but function call ID is optional in genai API and some models do not use the field.
	// Generate function call ids. (see functions.populate_client_function_call_id in python SDK)
	utils.PopulateClientFunctionCallID(ctx, resp.Content)

	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.ID = resp.eventID // TODO change NewEvent to accept event id
	ev.Author = ctx.Agent().Name()
	ev.Branch = ctx.Branch()
	ev.LLMResponse = *resp.LLMResponse
	ev.Actions.StateDelta = stateDelta
	ev.Actions.ArtifactDelta = artifactDelta
	ev.Actions.RenderUIWidgets = resp.uiWidgets

	// Populate ev.LongRunningToolIDs
	ev.LongRunningToolIDs = findLongRunningFunctionCallIDs(resp.Content, tools)

	return ev
}

// findLongRunningFunctionCallIDs iterates over the FunctionCalls and
// returns the callIDs of the long running functions
func findLongRunningFunctionCallIDs(c *genai.Content, tools map[string]tool.Tool) []string {
	set := make(map[string]struct{})
	// Iterate over function calls.
	for _, fc := range utils.FunctionCalls(c) {
		if tool, ok := tools[fc.Name]; ok && fc.ID != "" && tool.IsLongRunning() {
			// If the tool exists and is long-running, add its ID to the set.
			set[fc.ID] = struct{}{}
		}
	}
	// Transform the set (map keys) into a slice.
	return slices.Collect(maps.Keys(set))
}

type fakeTool struct {
	name string
}

func (f *fakeTool) Name() string      { return f.name }
func (*fakeTool) Description() string { return "Tool not found" }
func (*fakeTool) IsLongRunning() bool { return false }

var _ tool.Tool = (*fakeTool)(nil)

type cancelledToolContext struct {
	agent.Context
	// Marker: this is one of ADK's own context types, so the identity procedure
	// asks it rather than reading a session it does not have. Without it the
	// procedure would fall back to the embedded tool context's Session(), which
	// is nil by design, and a streaming tool that re-derives its context would
	// lose the acting user.
	adkcontext.Marker
	cancelCtx context.Context
}

func (c *cancelledToolContext) Done() <-chan struct{} {
	return c.cancelCtx.Done()
}

func (c *cancelledToolContext) Err() error {
	return c.cancelCtx.Err()
}

func (c *cancelledToolContext) Deadline() (deadline time.Time, ok bool) {
	return c.cancelCtx.Deadline()
}

// RenderUIWidget implements [agent.UIWidgetRenderer] for the tool context it
// embeds, which the embedding would otherwise hide.
func (c *cancelledToolContext) RenderUIWidget(widget session.UIWidget) error {
	return agent.RenderUIWidget(c.Context, widget)
}

func (c *cancelledToolContext) Value(key any) any {
	// Fails closed rather than dereferencing: this type carries the identity
	// marker, so the identity procedure asks it directly, and that ask can arrive
	// from inside http.RoundTripper where net/http does not recover.
	if c == nil || c.cancelCtx == nil {
		return nil
	}
	return c.cancelCtx.Value(key)
}

// displayableToolResultText renders a tool result as text for the case where
// SkipSummarization suppresses the LLM's own summary of it. It returns
// ok=false for results that carry nothing worth showing: error responses,
// and empty or null results such as those from control-flow-only tools
// (e.g. exitlooptool), which return no meaningful output of their own.
//
// Tools that wrap a single plain-text result under a "result" key (the
// convention used by agenttool) are shown as-is; anything else is rendered
// as JSON so structured results remain readable.
func displayableToolResultText(result map[string]any) (string, bool) {
	if len(result) == 0 {
		return "", false
	}
	if _, isErr := result["error"]; isErr {
		return "", false
	}
	if len(result) == 1 {
		if v, ok := result["result"]; ok {
			if v == nil {
				return "", false
			}
			if text, ok := v.(string); ok {
				return text, text != ""
			}
		}
	}
	b, err := marshalJSONNoHTMLEscape(result)
	if err != nil {
		return "", false
	}
	return string(b), true
}

// marshalJSONNoHTMLEscape is json.Marshal without HTML-escaping the output.
// json.Marshal escapes '<', '>' and '&' (e.g. a URL's "&" becomes "&"),
// which is meant for embedding JSON in HTML but only hurts readability of
// text meant for a chat transcript.
func marshalJSONNoHTMLEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// handleFunctionCalls is handleFunctionCallsInOrder for a toolsDict built
// from f.Tools, whose names are listed in f.Tools order.
func (f *Flow) handleFunctionCalls(ctx agent.InvocationContext, toolsDict map[string]tool.Tool, resp *model.LLMResponse, toolConfirmations map[string]*toolconfirmation.ToolConfirmation, liveSess agent.LiveSession) (*session.Event, error) {
	return f.handleFunctionCallsInOrder(ctx, toolsDict, registryOrder(nil, f.Tools, toolsDict), resp, toolConfirmations, liveSess)
}

// handleFunctionCallsInOrder calls the functions and returns the function
// response event. toolOrder is the names of toolsDict in registry order (see
// registryOrder); a response to an unknown name lists them in that order.
//
// The first call to pause or fail the run, by time, stops the step: the calls
// still running are canceled, handleFunctionCallsInOrder waits for them to
// return, and it returns that call's error with no response event.
//
// TODO: accept filters to include/exclude function calls.
// TODO: check feasibility of running tool.Run concurrently.
func (f *Flow) handleFunctionCallsInOrder(ctx agent.InvocationContext, toolsDict map[string]tool.Tool, toolOrder []string, resp *model.LLMResponse, toolConfirmations map[string]*toolconfirmation.ToolConfirmation, liveSess agent.LiveSession) (mergedEvent *session.Event, err error) {
	fnCalls := utils.FunctionCalls(resp.Content)

	// Merged span for parallel tool calls - create only if there is more than one tool call.
	if len(fnCalls) > 1 {
		mergedCtx, mergedToolCallSpan := telemetry.StartTrace(ctx, "execute_tool (merged)")
		ctx = ctx.WithContext(mergedCtx)
		defer func() {
			telemetry.TraceMergedToolCallsResult(mergedToolCallSpan, mergedEvent, err)
			mergedToolCallSpan.End()
		}()
	}

	fnResponseEvents := make([]*session.Event, len(fnCalls))
	// stop is the first error, by time, that keeps a call from being answered:
	// a pause, or an error that ends the run. stopMu guards it. Like
	// adk-python's _gather_or_cancel (_batch_executor.py:195-211), the first
	// stop cancels the calls still running and is the step's error; what the
	// canceled calls return is dropped. batchCtx is not canceled when no call
	// stops: work a tool starts with its context, such as a live streaming
	// tool, outlives the step.
	var (
		stopMu sync.Mutex
		stop   error
	)
	batchCtx, cancelBatch := context.WithCancelCause(ctx)
	stopBatch := func(err error) {
		stopMu.Lock()
		defer stopMu.Unlock()
		if stop == nil {
			stop = err
			cancelBatch(errSiblingCallStopped)
		}
	}

	// Tool calls run via the context's task runner: concurrent goroutines by
	// default, or a caller-installed runner (platform.WithTaskRunner).
	tasks := make([]func(context.Context), len(fnCalls))
	// Orders the calls by call index for the code that takes its turn: the
	// calls of a sequential tool (see tool.SequentialRunner and callTool) and
	// in-tree code that calls adkcontext.AwaitEarlierCalls. It does not order
	// State. A batch of one gets an order too, so that its call does not keep
	// the place of an enclosing batch.
	order := adkcontext.NewCallOrder(len(fnCalls))
	for i, fnCall := range fnCalls {
		tasks[i] = func(taskCtx context.Context) {
			// Deferred so that every return path, including an early return
			// without a response and a panic, releases the later calls.
			defer order.Finish(i)
			taskCtx = adkcontext.WithCallPlace(taskCtx, order, i)
			name := fnCall.Name
			if name == "" {
				name = unnamedTool
			}
			sctx, span := telemetry.StartExecuteToolSpan(taskCtx, telemetry.StartExecuteToolSpanParams{
				ToolName: name,
				Args:     fnCall.Args,
			})
			defer span.End()
			toolCallCtx := ctx.WithContext(sctx)
			var confirmation *toolconfirmation.ToolConfirmation
			if toolConfirmations != nil {
				confirmation = toolConfirmations[fnCall.ID]
			}
			toolCtx := agent.NewToolContext(toolCallCtx, fnCall.ID, &session.EventActions{StateDelta: make(map[string]any)}, confirmation)
			args := cloneArgs(fnCall.Args)

			var res toolResult
			var curTool tool.Tool
			if fnCall.Name == "stop_streaming" {
				funcToStop, _ := fnCall.Args["function_name"].(string)
				var status string
				if impl, ok := liveSess.(*liveSessionImpl); ok && impl.CancelAllStreamingTools(funcToStop) {
					status = fmt.Sprintf("Successfully stopped all running instances of %s", funcToStop)
				} else {
					status = fmt.Sprintf("No active streaming function named %s found", funcToStop)
				}
				res = toolResult{m: map[string]any{"status": status}}
			} else {
				curTool = toolsDict[fnCall.Name]
				if streamTool, ok := curTool.(toolinternal.StreamingFunctionTool); ok {
					if liveSess != nil {
						res = toolResult{m: map[string]any{"status": "The function is running asynchronously and the results are pending."}}
						cancelCtx, cancel := toolCtx.WithAgentCancel()
						cancelToolCtx := &cancelledToolContext{
							Context:   toolCtx,
							cancelCtx: cancelCtx,
						}
						if impl, ok := liveSess.(*liveSessionImpl); ok {
							impl.RegisterStreamingTool(streamTool.Name(), fnCall.ID, cancel)
						}
						go func() {
							defer func() {
								if impl, ok := liveSess.(*liveSessionImpl); ok {
									impl.UnregisterStreamingTool(streamTool.Name(), fnCall.ID)
								}
								cancel()
							}()
							for chunk, err := range streamTool.RunStream(cancelToolCtx, args) {
								select {
								case <-cancelCtx.Done():
									return
								default:
								}
								if err != nil {
									fmt.Printf("Error in streaming tool %s: %v\n", streamTool.Name(), err) //nolint:forbidigo // pre-slog call site
									return
								}
								updatedContent := &genai.Content{
									Role: "user",
									Parts: []*genai.Part{
										{
											Text: fmt.Sprintf("Function %s returned: %s", streamTool.Name(), chunk),
										},
									},
								}
								if err := liveSess.Send(agent.LiveRequest{Content: updatedContent}); err != nil {
									fmt.Printf("Failed to send content from streaming tool %s: %v\n", streamTool.Name(), err) //nolint:forbidigo // pre-slog call site
									return
								}
							}
						}()
					} else {
						var sb strings.Builder
						for chunk, err := range streamTool.RunStream(toolCtx, args) {
							if err != nil {
								errText, _ := toolinternal.ModelErrorText(streamTool.Name(), err)
								res = toolResult{m: map[string]any{"error": errText}}
								break
							}
							sb.WriteString(chunk)
						}
						if res.m == nil {
							res = toolResult{m: map[string]any{"result": sb.String()}}
						}
					}
				} else if _, ok := curTool.(toolinternal.FunctionTool); ok {
					res = f.callTool(toolCtx, curTool, args, nil)
				} else {
					// A name no runnable tool has is answered like a tool
					// call, with a placeholder tool, so that callbacks can
					// answer it (_caller.py:449-455).
					curTool = &fakeTool{name: name}
					res = f.callTool(toolCtx, curTool, args, newToolLookupFailure(name, toolOrder))
				}
			}

			if res.stop != nil {
				// Before the deferred order.Finish(i): a sequential call woken
				// by it must find batchCtx canceled.
				stopBatch(res.stop)
				return
			}

			if isFalsyResult(res) {
				if d, ok := curTool.(toolinternal.ResponseDeferrer); ok && d.DefersResponse() {
					return
				}
				if curTool != nil && curTool.IsLongRunning() {
					return
				}
			}
			// A tool that returns nothing is answered {"result": nil}, wrapped
			// after the callbacks saw nil, as adk-python wraps None.
			if res.m == nil {
				res = toolResult{m: map[string]any{"result": nil}}
			}

			part := buildFunctionResponse(curTool, fnCall.ID, name, res, "")
			parts := []*genai.Part{part}
			// SkipSummarization causes the parent agent loop to terminate on this
			// event (see session.Event.IsFinalResponse) without the LLM ever
			// seeing the function response. Since most UIs don't render function
			// response parts, attach the result as a visible text part so it
			// isn't silently dropped from the terminal event.
			//
			// This only applies to tools that opt in via
			// SkipSummarizationResultDisplayer (agenttool): SkipSummarization is
			// also set by tools whose result is an internal detail never meant
			// for display - e.g. a pending tool confirmation (see
			// RequestedToolConfirmations above) or a UI/widget tool's
			// acknowledgement - and those must not have their result leaked into
			// a visible text part.
			if actions := toolCtx.Actions(); actions.SkipSummarization {
				if displayer, ok := curTool.(toolinternal.SkipSummarizationResultDisplayer); ok && displayer.DisplayResultOnSkipSummarization() {
					if text, ok := displayableToolResultText(res.m); ok {
						parts = append(parts, &genai.Part{Text: text})
					}
				}
			}

			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.LLMResponse = model.LLMResponse{
				Content: &genai.Content{
					Role:  "user",
					Parts: parts,
				},
			}
			ev.Author = ctx.Agent().Name()
			ev.Branch = ctx.Branch()
			ev.Actions = *toolCtx.Actions()
			// A tool handler holds this EventActions for the whole call, and
			// everything on it lands on the persisted event. Compaction is the
			// framework's to write: see session.EventActions.Compaction.
			ev.Actions.Compaction = nil

			traceTool := curTool
			if traceTool == nil {
				traceTool = &fakeTool{name: name}
			}
			var toolErr error
			resultErr := part.FunctionResponse.Response["error"]
			if resultErr != nil {
				if err, ok := resultErr.(error); ok {
					toolErr = err
				} else if errStr, ok := resultErr.(string); ok {
					toolErr = errors.New(errStr)
				}
			}
			telemetry.TraceToolResult(span, telemetry.TraceToolResultParams{
				Description:   traceTool.Description(),
				ResponseEvent: ev,
				Error:         toolErr,
			})

			fnResponseEvents[i] = ev
		}
	}
	platform.RunTasks(batchCtx, tasks)
	// A stop ends the step with no response for any call, so a paused call
	// and its siblings are replayed together on resume.
	if stop != nil {
		return nil, stop
	}
	mergedEvent, err = mergeParallelResponses(ctx, fnResponseEvents)
	if err != nil {
		return mergedEvent, err
	}
	return mergedEvent, nil
}

func (f *Flow) runOnToolErrorCallbacks(toolCtx agent.Context, tool tool.Tool, fArgs map[string]any, err error) (map[string]any, error) {
	pluginManager := pluginManagerFromContext(toolCtx)
	if pluginManager != nil {
		result, err := pluginManager.RunOnToolErrorCallback(toolCtx, tool, fArgs, err)
		if result != nil || err != nil {
			return result, err
		}
	}
	return f.invokeOnToolErrorCallbacks(toolCtx, tool, fArgs, err)
}

// callTool runs one call of t: the before-tool callbacks, then the tool, the
// on-tool-error callbacks on an error, and the after-tool callbacks. A tool
// error of type *toolinternal.PausedError returns at once as the result's stop,
// with no further callback. A *toolinternal.FailRunError goes to the
// on-tool-error callbacks: a response answers the call as for any error, and
// otherwise the result's stop is the callback's error or the FailRunError, with
// no after-tool callback. lookup is
// non-nil when no runnable tool has the called name; t is then the
// placeholder, which only callbacks see. A before-tool callback can still
// answer such a call, and the after-tool callbacks then run as for a tool;
// otherwise the on-tool-error callbacks receive lookup.err and the after-tool
// callbacks are skipped (_caller.py:529-554).
//
// When t is a tool.SequentialRunner that answers true, callTool first waits
// until every earlier call of the batch has returned. If the call's context
// ends first, the call is answered with the wait's error, and neither the
// callbacks nor the tool run.
//
// Once another call of the batch has stopped the step (toolCtx canceled with
// errSiblingCallStopped), the result is a stop and nothing more of the call
// runs: no before-tool callback when the call has not begun, no tool after the
// before-tool callbacks, and no on-tool-error or after-tool callback after the
// tool failed.
func (f *Flow) callTool(toolCtx agent.Context, t tool.Tool, args map[string]any, lookup *toolLookupFailure) toolResult {
	if s, ok := t.(tool.SequentialRunner); ok && s.RunsSequentially() {
		if err := adkcontext.AwaitEarlierCalls(toolCtx); err != nil {
			errText, _ := toolinternal.ModelErrorText(t.Name(), fmt.Errorf("wait for earlier function calls: %w", err))
			return toolResult{m: map[string]any{"error": errText}}
		}
	}
	// A call that has not begun when a sibling stops the step runs nothing:
	// adk-python cancels its task before it starts (_batch_executor.py:188-192).
	if siblingCallStopped(toolCtx) {
		return toolResult{stop: context.Cause(toolCtx)}
	}
	var response map[string]any
	var err error
	pluginManager := pluginManagerFromContext(toolCtx)
	if pluginManager != nil {
		response, err = pluginManager.RunBeforeToolCallback(toolCtx, t, args)
	}
	if response == nil && err == nil {
		response, err = f.invokeBeforeToolCallbacks(toolCtx, t, args)
	}
	// Nor does a call go on past a before-tool callback that was running when
	// a sibling stopped the step: the cancellation reaches a Python call at the
	// callback's await, and nothing after it runs.
	if siblingCallStopped(toolCtx) {
		return toolResult{stop: context.Cause(toolCtx)}
	}
	if lookup != nil && response == nil && err == nil {
		return f.answerLookupFailure(toolCtx, t, args, lookup)
	}

	res := toolResult{m: response}
	var failRun *toolinternal.FailRunError
	if response == nil && err == nil {
		res, err = runTool(toolCtx, t, args)
		// A pause is not a tool error: no callback sees it, as adk-python's
		// NodeInterruptedError, a BaseException, passes _caller.py:572's
		// `except Exception`.
		if _, ok := errors.AsType[*toolinternal.PausedError](err); ok {
			return toolResult{stop: err}
		}
		// Nor is a failure after a sibling call stopped the step, which
		// canceled the tool: adk-python's CancelledError, a BaseException,
		// skips the error and after-tool callbacks. As a stop it gets no
		// response, and handleFunctionCallsInOrder ends the step without one.
		if err != nil && siblingCallStopped(toolCtx) {
			return toolResult{stop: err}
		}
		errors.As(err, &failRun)
	}

	if failRun != nil {
		// The callbacks may answer a run-failing error; otherwise it, or a
		// callback's own error, ends the run (_caller.py:580-585). cbErr is
		// err itself when every callback declines.
		errorResponse, cbErr := f.runOnToolErrorCallbacks(toolCtx, t, args, err)
		if cbErr != nil {
			return toolResult{stop: cbErr}
		}
		res, err = toolResult{m: errorResponse}, nil
	} else if err != nil {
		errorResponse, cbErr := f.runOnToolErrorCallbacks(toolCtx, t, args, err)
		if errorResponse != nil || cbErr != nil {
			res = toolResult{m: errorResponse}
			err = cbErr
		}
	}

	var altered map[string]any
	var alteredErr error
	if pluginManager != nil {
		altered, alteredErr = pluginManager.RunAfterToolCallback(toolCtx, t, args, res.m, err)
	}
	if altered == nil && alteredErr == nil {
		altered, alteredErr = f.invokeAfterToolCallbacks(toolCtx, t, args, res.m, err)
	}
	if altered != nil || alteredErr != nil {
		res.m = altered
		err = alteredErr
	}

	if err != nil {
		errText, _ := toolinternal.ModelErrorText(t.Name(), err)
		return toolResult{m: map[string]any{"error": errText}}
	}
	return res
}

func (f *Flow) invokeBeforeToolCallbacks(toolCtx agent.Context, tool tool.Tool, fArgs map[string]any) (map[string]any, error) {
	for _, callback := range f.BeforeToolCallbacks {
		result, err := callback(toolCtx, tool, fArgs)
		if err != nil {
			return nil, err
		}
		// When a list of callbacks is provided, the callbacks will be called in the
		// order they are listed while a callback returns nil.
		if result != nil {
			return result, nil
		}
	}
	return nil, nil
}

func (f *Flow) invokeAfterToolCallbacks(toolCtx agent.Context, tool tool.Tool, fArgs, fResult map[string]any, fErr error) (map[string]any, error) {
	for _, callback := range f.AfterToolCallbacks {
		result, err := callback(toolCtx, tool, fArgs, fResult, fErr)
		if err != nil {
			return nil, err
		}
		// When a list of callbacks is provided, the callbacks will be called in the
		// order they are listed while a callback returns nil.
		if result != nil {
			return result, nil
		}
	}
	// If no callback returned a result/error, return the original result/error.
	return fResult, fErr
}

func (f *Flow) invokeOnToolErrorCallbacks(toolCtx agent.Context, tool tool.Tool, fArgs map[string]any, fErr error) (map[string]any, error) {
	for _, callback := range f.OnToolErrorCallbacks {
		result, err := callback(toolCtx, tool, fArgs, fErr)
		if err != nil {
			return nil, err
		}
		// When a list of callbacks is provided, the callbacks will be called in the
		// order they are listed while a callback returns nil.
		if result != nil {
			return result, nil
		}
	}
	// If no callback returned a result/error, return the original result/error.
	return nil, fErr
}

func mergeParallelFunctionResponseEvents(events []*session.Event) (*session.Event, error) {
	switch len(events) {
	case 0:
		return nil, nil
	case 1:
		return events[0], nil
	}
	var parts []*genai.Part
	var actions *session.EventActions
	var result *session.Event // first non-nil event, reused as the merged result
	for _, ev := range events {
		if ev == nil || ev.LLMResponse.Content == nil {
			continue
		}
		if result == nil {
			result = ev
		}
		parts = append(parts, ev.LLMResponse.Content.Parts...)
		actions = mergeEventActions(actions, &ev.Actions)
	}
	// All entries were nil (e.g. every call was long-running/deferred).
	if result == nil {
		return nil, nil
	}
	result.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Role:  "user",
			Parts: parts,
		},
	}
	result.Actions = *actions
	return result, nil
}

func mergeEventActions(base, other *session.EventActions) *session.EventActions {
	// flows/llm_flows/functions.py merge_parallel_function_response_events
	if other == nil {
		return base
	}
	if base == nil {
		return other
	}
	if other.SkipSummarization {
		base.SkipSummarization = true
	}
	if other.TransferToAgent != "" {
		base.TransferToAgent = other.TransferToAgent
	}
	if other.TransferReason != "" {
		base.TransferReason = other.TransferReason
	}
	if other.Escalate {
		base.Escalate = true
	}
	if other.StateDelta != nil {
		base.StateDelta = deepMergeMap(base.StateDelta, other.StateDelta)
	}
	if other.ArtifactDelta != nil {
		base.ArtifactDelta = mergeArtifactDeltas(base.ArtifactDelta, other.ArtifactDelta)
	}
	if other.RequestedToolConfirmations != nil {
		if base.RequestedToolConfirmations == nil {
			base.RequestedToolConfirmations = make(map[string]toolconfirmation.ToolConfirmation)
		}
		maps.Copy(base.RequestedToolConfirmations, other.RequestedToolConfirmations)
	}
	if other.RequestedAuthConfigs != nil {
		if base.RequestedAuthConfigs == nil {
			base.RequestedAuthConfigs = make(map[string]toolauth.Config)
		}
		maps.Copy(base.RequestedAuthConfigs, other.RequestedAuthConfigs)
	}
	if other.SetModelResponse != nil {
		base.SetModelResponse = mergeSetModelResponse(base.SetModelResponse, other.SetModelResponse)
	}
	// Concatenated without a duplicate check, as in adk-python
	// (flows/llm_flows/tools/_batch_executor.py).
	base.RenderUIWidgets = append(base.RenderUIWidgets, other.RenderUIWidgets...)
	if other.RewindBeforeInvocationID != "" {
		base.RewindBeforeInvocationID = other.RewindBeforeInvocationID
	}
	if other.EndOfAgent != nil {
		base.EndOfAgent = other.EndOfAgent
	}
	if other.AgentState != nil {
		base.AgentState = deepMergeMap(base.AgentState, other.AgentState)
	}
	return base
}

// mergeArtifactDeltas merges artifact deltas, preferring higher versions.
// This is a deliberate divergance from adk-python which uses last-write-wins.
func mergeArtifactDeltas(dst, src map[string]int64) map[string]int64 {
	if dst == nil {
		return maps.Clone(src)
	}
	for key, value := range src {
		if dstVal, ok := dst[key]; ok {
			dst[key] = max(dstVal, value)
		} else {
			dst[key] = value
		}
	}
	return dst
}

func deepMergeMap(dst, src map[string]any) map[string]any {
	if dst == nil {
		dst = make(map[string]any)
	}
	for key, value := range src {
		if srcMap, ok := value.(map[string]any); ok {
			if dstMap, ok := dst[key].(map[string]any); ok {
				dst[key] = deepMergeMap(dstMap, srcMap)
				continue
			}
		}
		dst[key] = value
	}
	return dst
}

// mergeSetModelResponse combines the answers two parallel tool calls stored.
// Two maps are deep merged into dst, as Python's deep_merge_dicts does; any
// other pairing (a list, a scalar, or no earlier answer) takes src, because
// Python overwrites the key unless both values are dicts
// (flows/llm_flows/tools/_batch_executor.py).
func mergeSetModelResponse(dst, src any) any {
	dstMap, dstIsMap := dst.(map[string]any)
	srcMap, srcIsMap := src.(map[string]any)
	if dstIsMap && srcIsMap {
		return deepMergeMap(dstMap, srcMap)
	}
	return src
}

func pluginManagerFromContext(ctx context.Context) pluginManager {
	m, ok := ctx.Value(plugincontext.PluginManagerCtxKey).(pluginManager)
	if !ok {
		return nil
	}
	return m
}

type pluginManager interface {
	RunBeforeModelCallback(cctx agent.Context, llmRequest *model.LLMRequest) (*model.LLMResponse, error)
	RunAfterModelCallback(cctx agent.Context, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)
	RunOnModelErrorCallback(ctx agent.Context, llmRequest *model.LLMRequest, llmResponseError error) (*model.LLMResponse, error)
	RunBeforeToolCallback(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error)
	RunAfterToolCallback(ctx agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error)
	RunOnToolErrorCallback(ctx agent.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error)
}

// Asserted rather than left to the [adkcontext.Marker] embed: dropping the embed
// would also drop the import, breaking the build for an unrelated reason. This
// fails on the type.
var _ adkcontext.Source = (*cancelledToolContext)(nil)
