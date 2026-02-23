// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package llminternal

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/runconfig"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

type liveResult struct {
	event     *session.Event
	err       error
	reconnect bool
}

// RunLive runs the flow in live mode, connecting to the model and handling live interactions.
func (f *Flow) RunLive(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if f.Model == nil {
			yield(nil, fmt.Errorf("agent %q: %w", ctx.Agent().Name(), ErrModelNotConfigured))
			return
		}

		defer func() {
			log.Debug().Msg("RunLive closed")
		}()

		req := &model.LLMRequest{
			Model:             f.Model.Name(),
			LiveConnectConfig: runconfig.FromContext(ctx).LiveConnectConfig,
		}

		// Run preprocessors to setup request (e.g. tools, auth)
		// We only run them once at start for now.
		// TODO: Re-evaluate if some processors need to run per-turn or if they make sense in live.
		// Preprocess before calling the LLM.
		for _, err := range f.preprocess(ctx, req) {
			if err != nil {
				yield(nil, err)
				return
			}
			if ctx.Ended() {
				return
			}
		}

		attempt := 0
		for {
			// On subsequent attempts, use the saved token to reconnect
			if ctx.LiveSessionResumptionHandle() != "" {
				log.Info().Int("attempt", attempt).
					Str("handle", ctx.LiveSessionResumptionHandle()).
					Msg("Attempting to reconnect live session with handle")
				attempt++
				if req.LiveConnectConfig == nil {
					req.LiveConnectConfig = &genai.LiveConnectConfig{}
				}
				if req.LiveConnectConfig.SessionResumption == nil {
					req.LiveConnectConfig.SessionResumption = &genai.SessionResumptionConfig{
						Handle: ctx.LiveSessionResumptionHandle(),
					}
				}
				req.LiveConnectConfig.SessionResumption.Handle = ctx.LiveSessionResumptionHandle()
				// req.LiveConnectConfig.SessionResumption.Transparent = true
			}

			// Connect to the model
			conn, err := f.Model.Connect(ctx, req)
			if err != nil {
				yield(nil, fmt.Errorf("failed to connect to live model: %w", err))
				return
			}
			defer conn.Close()

			// TODO: this is python implementation and we need to send the history to the model to continue the conversation
			// if llm_request.contents:
			//   # Sends the conversation history to the model.
			//   with tracer.start_as_current_span('send_data'):
			//     # Combine regular contents with audio/transcription from session
			//     logger.debug('Sending history to model: %s', llm_request.contents)
			//       await llm_connection.send_history(llm_request.contents)
			//       trace_send_data(
			//           invocation_context, event_id, llm_request.contents
			//       )

			queue := ctx.LiveRequestQueue()
			if queue == nil {
				yield(nil, fmt.Errorf("LiveRequestQueue not found in context"))
				return
			}

			// Decouple sender/receiver from the yield loop using a channel
			results := make(chan liveResult, 64)

			connCtx, cancel := context.WithCancel(ctx)
			connInvCtx := ctx.WithContext(connCtx)
			var wg sync.WaitGroup

			wg.Add(2)
			go f.runSender(connInvCtx, &wg, conn, results)
			go f.runReceiver(connInvCtx, &wg, conn, req, results)

			// Close the channel when both goroutines are finished
			go func() {
				wg.Wait()
				close(results)
			}()

			// Process results from background goroutines safely in this thread
			shouldReconnect := false
			for res := range results {
				if res.err != nil {
					if !yield(nil, res.err) {
						cancel()
						return
					}
					shouldReconnect = true
					break
				}

				if res.reconnect {
					log.Info().Msg("Received reconnection signal from background")
					shouldReconnect = true
					break
				}

				if res.event != nil {
					// Handle tool response if needed (special logic from previous loop)
					if len(res.event.FunctionResponses()) > 0 {
						toolResponses := &genai.LiveToolResponseInput{
							FunctionResponses: res.event.FunctionResponses(),
						}
						queue.SendToolResponse(toolResponses)
					}

					// # We handle agent transfer here in `run_live` rather than
					// # in `_postprocess_live` to prevent duplication of function
					// # response processing. If agent transfer were handled in
					// # `_postprocess_live`, events yielded from child agent's
					// # `run_live` would bubble up to parent agent's `run_live`,
					// # causing `event.get_function_responses()` to be true in both
					// # child and parent, and `send_content()` to be called twice for
					// # the same function response. By handling agent transfer here,
					// # we ensure that only child agent processes its own function
					// # responses after the transfer.
					// if (
					//     event.content
					//     and event.content.parts
					//     and event.content.parts[0].function_response
					//     and event.content.parts[0].function_response.name
					//     == 'transfer_to_agent'
					// ):
					//   await asyncio.sleep(DEFAULT_TRANSFER_AGENT_DELAY)
					//   # cancel the tasks that belongs to the closed connection.
					//   send_task.cancel()
					//   logger.debug('Closing live connection')
					//   await llm_connection.close()
					//   logger.debug('Live connection closed.')
					//   # transfer to the sub agent.
					//   transfer_to_agent = event.actions.transfer_to_agent
					//   if transfer_to_agent:
					//     logger.debug('Transferring to agent: %s', transfer_to_agent)
					//     agent_to_run = self._get_agent_to_run(
					//         invocation_context, transfer_to_agent
					//     )
					//     async with Aclosing(
					//         agent_to_run.run_live(invocation_context)
					//     ) as agen:
					//       async for item in agen:
					//         yield item
					// if (
					//     event.content
					//     and event.content.parts
					//     and event.content.parts[0].function_response
					//     and event.content.parts[0].function_response.name
					//     == 'task_completed'
					// ):
					//   # this is used for sequential agent to signal the end of the agent.
					//   await asyncio.sleep(DEFAULT_TASK_COMPLETION_DELAY)
					//   # cancel the tasks that belongs to the closed connection.
					//   send_task.cancel()
					//   return

					if !yield(res.event, nil) {
						cancel()
						return
					}
				}
			}

			cancel()
			conn.Close()

			if !shouldReconnect || ctx.Err() != nil {
				return
			}
		}
	}
}

func (f *Flow) runSender(ctx agent.InvocationContext, wg *sync.WaitGroup, conn model.LiveConnection, results chan<- liveResult) {
	defer wg.Done()
	queue := ctx.LiveRequestQueue()
	ch := queue.Channel()

	defer func() {
		log.Debug().Msg("runSender closed")
	}()

	for {
		select {
		case <-ctx.Done():
			return
		case liveReq, ok := <-ch:
			if !ok {
				return
			}
			if liveReq.Close {
				conn.Close()
				return
			}

			if liveReq.ActivityStart != nil || liveReq.ActivityEnd != nil {
				if err := conn.Send(liveReq); err != nil {
					sendResult(ctx, results, liveResult{err: fmt.Errorf("failed to send to live connection: %w", err)})
					return
				}
			} else if liveReq.RealtimeInput != nil && liveReq.RealtimeInput.Audio != nil {
				err := f.AudioCacheManager.CacheAudio(ctx, liveReq.RealtimeInput.Audio, "input")
				if err != nil {
					sendResult(ctx, results, liveResult{err: fmt.Errorf("failed to cache audio: %w", err)})
					return
				}
				if err := conn.Send(liveReq); err != nil {
					sendResult(ctx, results, liveResult{err: fmt.Errorf("failed to send to live connection: %w", err)})
					return
				}
			}

			if liveReq.Content != nil {
				content := liveReq.Content
				// Persist user text content to session (similar to non-live mode)
				// Skip function responses - they are already handled separately
				isFunctionResponse := false
				for _, part := range content.Parts {
					if part.FunctionResponse != nil {
						isFunctionResponse = true
						break
					}
				}
				if !isFunctionResponse {
					if content.Role == "" {
						content.Role = "user"
					}
					userContentEvent := session.NewEvent(ctx.InvocationID())
					userContentEvent.Content = content
					userContentEvent.Author = "user"

					sendResult(ctx, results, liveResult{event: userContentEvent})

					if err := conn.Send(liveReq); err != nil {
						sendResult(ctx, results, liveResult{err: fmt.Errorf("failed to send to live connection: %w", err)})
						return
					}
				}
			}

			if liveReq.ToolResponse != nil {
				if err := conn.Send(liveReq); err != nil {
					sendResult(ctx, results, liveResult{err: fmt.Errorf("failed to send to live connection: %w", err)})
					return
				}
			}
		}
	}
}

func (f *Flow) runReceiver(ctx agent.InvocationContext, wg *sync.WaitGroup, conn model.LiveConnection, req *model.LLMRequest, results chan<- liveResult) {
	defer wg.Done()

	defer func() {
		log.Debug().Msg("runReceiver closed")
	}()

	getAuthorForEvent := func(llmResponse *model.LLMResponse) string {
		if llmResponse != nil && llmResponse.Content != nil && llmResponse.Content.Role == "user" {
			return "user"
		}
		return ctx.Agent().Name()
	}

	resumable := false
	if cfg := ctx.ResumabilityConfig(); cfg != nil {
		resumable = cfg.IsResumable
	}

	var goAway *genai.LiveServerGoAway
	var goAwayTimer *time.Timer
	defer func() {
		if goAwayTimer != nil {
			goAwayTimer.Stop()
		}
	}()

	resps, errs := conn.Receive(ctx)
	for {
		select {
		case <-time.After(30 * time.Second):
			log.Debug().Msg("runReceiver dummy timeout. closing")
			sendResult(ctx, results, liveResult{reconnect: true})
			return
		case resp, ok := <-resps:
			if !ok {
				log.Debug().Str("func", "runReceiver").Msg("response channel is closed. returning")
				return
			}
			if resp.LiveSessionResumptionUpdate != nil {
				log.Info().Str("handle", resp.LiveSessionResumptionUpdate.NewHandle).Msg("Live session resumption update")
				ctx.SetLiveSessionResumptionHandle(resp.LiveSessionResumptionUpdate.NewHandle)
			}

			if resp.LiveGoAway != nil {
				if resumable {
					log.Info().Interface("go_away", resp.LiveGoAway).Msg("Received GoAway from model (Resumable). Setting up deferred reconnect.")
					goAway = resp.LiveGoAway

					// Reconnect slightly before it expires
					reconnectDelay := goAway.TimeLeft - 1*time.Second
					if reconnectDelay < 0 {
						reconnectDelay = 0
					}

					if goAwayTimer != nil {
						goAwayTimer.Stop()
					}
					goAwayTimer = time.AfterFunc(reconnectDelay, func() {
						log.Info().Msg("GoAway timer expired. Triggering reconnect.")
						sendResult(ctx, results, liveResult{reconnect: true})
					})
					continue
				} else {
					log.Info().Interface("go_away", resp.LiveGoAway).Msg("Received GoAway from model (Not Resumable). Notifying client.")
				}
			}

			modelResponseEvent := session.NewEvent(ctx.InvocationID())
			modelResponseEvent.Content = resp.Content
			modelResponseEvent.Author = getAuthorForEvent(resp)
			modelResponseEvent.OutputTranscription = resp.OutputTranscription
			modelResponseEvent.InputTranscription = resp.InputTranscription

			log.Info().Interface("resp", resp).Msg("resp from live model before postprocess")

			for ev, err := range f.postprocessLive(ctx, req, resp, modelResponseEvent) {
				if err != nil {
					sendResult(ctx, results, liveResult{err: err})
					return
				}

				if ctx.RunConfig().SaveLiveBlob &&
					ev.Content != nil &&
					ev.Content.Parts != nil &&
					ev.Content.Parts[0].InlineData != nil &&
					strings.HasPrefix(ev.Content.Parts[0].InlineData.MIMEType, "audio/") {

					audioBlob := &genai.Blob{
						Data:     ev.Content.Parts[0].InlineData.Data,
						MIMEType: ev.Content.Parts[0].InlineData.MIMEType,
					}

					if err := f.AudioCacheManager.CacheAudio(ctx, audioBlob, "output"); err != nil {
						log.Error().Err(err).Msg("Failed to cache audio")
					}
					log.Info().Int("audioblob_length", len(audioBlob.Data)).Msg("Cached audio")
				}

				sendResult(ctx, results, liveResult{event: ev})
			}

			if resumable && goAway != nil && resp.TurnComplete {
				log.Info().Msg("Model turn complete after GoAway. Reconnecting now.")
				if goAwayTimer != nil {
					goAwayTimer.Stop()
				}
				sendResult(ctx, results, liveResult{reconnect: true})
				return
			}

			if ctx.Err() != nil {
				return
			}

		case err, ok := <-errs:
			if !ok {
				log.Debug().Str("func", "runReceiver").Msg("error channel is closed. returning")
				return
			}
			if err != nil {
				log.Debug().Str("func", "runReceiver").Err(err).Msg("error received from live connection")
				sendResult(ctx, results, liveResult{err: err})
				return
			}

		case <-ctx.Done():
			return
		}
	}
}

func sendResult(ctx context.Context, results chan<- liveResult, res liveResult) {
	select {
	case results <- res:
	case <-ctx.Done():
	}
}

func (f *Flow) postprocessLive(ctx agent.InvocationContext, llmRequest *model.LLMRequest, llmResponse *model.LLMResponse, modelResponseEvent *session.Event) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if err := f.postprocess(ctx, llmRequest, llmResponse); err != nil {
			yield(nil, err)
			return
		}

		if llmResponse.Content == nil &&
			llmResponse.ErrorCode == "" &&
			!llmResponse.Interrupted &&
			!llmResponse.TurnComplete &&
			llmResponse.InputTranscription == nil &&
			llmResponse.OutputTranscription == nil &&
			llmResponse.UsageMetadata == nil &&
			llmResponse.LiveGoAway == nil {
			return
		}

		if llmResponse.InputTranscription != nil {
			modelResponseEvent.InputTranscription = llmResponse.InputTranscription
			modelResponseEvent.Partial = llmResponse.Partial
			yield(modelResponseEvent, nil)
			return
		}

		if llmResponse.OutputTranscription != nil {
			modelResponseEvent.OutputTranscription = llmResponse.OutputTranscription
			modelResponseEvent.Partial = llmResponse.Partial
			log.Info().Interface("modelResponseEvent", modelResponseEvent).Msg("Yielding modelResponseEvent from outputTranscription postprocess")
			yield(modelResponseEvent, nil)
			return
		}

		// Flush audio caches based on control events using configurable settings
		if ctx.RunConfig().SaveLiveBlob {
			flushedEvents := f.handleControlEventFlush(ctx, llmResponse)
			for _, ev := range flushedEvents {
				if !yield(ev, nil) {
					return
				}
			}
			// if len(flushedEvents) > 0 {
			// 	// NOTE below return is O.K. for now, because currently we only flush
			// 	// events on interrupted or turn_complete. turn_complete is a pure
			// 	// control event and interrupted is not with content but those content
			// 	// is ignorable because model is already interrupted. If we have other
			// 	// case to flush events in the future that are not pure control events,
			// 	// we should not return here.
			// 	return
			// }
		}

		// Builds the event.
		modelResponseEvent = f.finalizeLiveModelResponseEvent(ctx, llmRequest, llmResponse, modelResponseEvent)
		log.Info().Interface("modelResponseEvent", modelResponseEvent).Msg("Yielding modelResponseEvent from finalizeLiveModelResponseEvent postprocess")
		if !yield(modelResponseEvent, nil) {
			return
		}

		// Resolve tools
		// TODO: Reuse tool resolution logic or cache it?
		tools := make(map[string]tool.Tool)
		for k, v := range llmRequest.Tools {
			tool, ok := v.(tool.Tool)
			if !ok {
				yield(nil, fmt.Errorf("unexpected tool type %T for tool %v", v, k))
				return
			}
			tools[k] = tool
		}
		if len(modelResponseEvent.FunctionCalls()) > 0 {
			log.Debug().Interface("functionCalls", modelResponseEvent.FunctionCalls()).Msg("handling function calls during postprocessing")
			functionResponseEvent, err := f.handleFunctionCalls(ctx, tools, llmResponse, nil)
			if err != nil {
				yield(nil, err)
				return
			}
			if functionResponseEvent != nil {
				if !yield(functionResponseEvent, nil) {
					return
				}
			}
		}
	}
}

func (f *Flow) finalizeLiveModelResponseEvent(ctx agent.InvocationContext, llmRequest *model.LLMRequest, llmResponse *model.LLMResponse, modelResponseEvent *session.Event) *session.Event {
	utils.PopulateClientFunctionCallID(llmResponse.Content)
	modelResponseEvent.Author = ctx.Agent().Name()
	modelResponseEvent.Branch = ctx.Branch()
	modelResponseEvent.LLMResponse = *llmResponse
	// ev.Actions.StateDelta = stateDelta

	// Populate ev.LongRunningToolIDs
	// ev.LongRunningToolIDs = findLongRunningFunctionCallIDs(resp.Content, tools)

	return modelResponseEvent
}

// handleControlEventFlush flushes audio caches based on control events using configurable settings
func (f *Flow) handleControlEventFlush(ctx agent.InvocationContext, llmResponse *model.LLMResponse) []*session.Event {
	stats := f.AudioCacheManager.GetCacheStats(ctx)
	log.Debug().Interface("stats", stats).Msg("audio cache stats")

	if llmResponse.Interrupted {
		events, err := f.AudioCacheManager.FlushCaches(ctx, false, true)
		if err != nil {
			log.Error().Err(err).Msg("failed to flush audio caches")
		}
		return events
	} else if llmResponse.TurnComplete {
		events, err := f.AudioCacheManager.FlushCaches(ctx, true, true)
		if err != nil {
			log.Error().Err(err).Msg("failed to flush audio caches")
		}
		return events
	}
	// TODO: Once generation_complete is surfaced on LlmResponse, we can flush
	// model audio here (flush_user_audio=False, flush_model_audio=True).
	return nil
}
