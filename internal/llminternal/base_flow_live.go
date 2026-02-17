package llminternal

import (
	"fmt"
	"iter"
	"log/slog"

	"github.com/rs/zerolog/log"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/runconfig"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// RunLive runs the flow in live mode, connecting to the model and handling live interactions.
func (f *Flow) RunLive(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if f.Model == nil {
			yield(nil, fmt.Errorf("agent %q: %w", ctx.Agent().Name(), ErrModelNotConfigured))
			return
		}

		req := &model.LLMRequest{
			Model:             f.Model.Name(),
			LiveConnectConfig: runconfig.FromContext(ctx).LiveConnectConfig,
		}

		// Run preprocessors to setup request (e.g. tools, auth)
		// We only run them once at start for now.
		// TODO: Re-evaluate if some processors need to run per-turn or if they make sense in live.
		// Preprocess before calling the LLM.
		if err := f.preprocess(ctx, req); err != nil {
			yield(nil, err)
			return
		}
		if ctx.Ended() {
			return
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
				req.LiveConnectConfig.SessionResumption.Transparent = true
			}

			// TODO pindahin connect to model ke dalam for loop ini
			// Connect to the model
			conn, err := f.Model.Connect(ctx, req)
			if err != nil {
				yield(nil, fmt.Errorf("failed to connect to live model: %w", err))
				return
			}
			defer conn.Close()

			queue := ctx.LiveRequestQueue()
			if queue == nil {
				yield(nil, fmt.Errorf("LiveRequestQueue not found in context"))
				return
			}

			// Start sender goroutine
			go func() {
				ch := queue.Channel()
				for {
					select {
					case liveReq, ok := <-ch:
						if !ok {
							return // Queue closed
						}
						if err := conn.Send(liveReq); err != nil {
							// TODO: Handle send error. Maybe log or signal main loop?
							// For now we just log/ignore as the main loop might catch connection issues too.
							fmt.Printf("Error sending to live connection: %v\n", err)
							return
						}
						if liveReq.Close {
							return
						}
					case <-ctx.Done():
						return
					}
				}
			}()

			// Main receive loop
			for {
				resp, err := conn.Receive()
				if err != nil {
					yield(nil, err)
					return
				}

				if resp == nil {
					continue
				}

				// Prepare event
				stateDelta := make(map[string]any)

				// Resolve tools
				// TODO: Reuse tool resolution logic or cache it?
				tools := make(map[string]tool.Tool)
				for k, v := range req.Tools {
					tool, ok := v.(tool.Tool)
					if !ok {
						yield(nil, fmt.Errorf("unexpected tool type %T for tool %v", v, k))
						return
					}
					tools[k] = tool
				}

				modelResponseEvent := f.finalizeModelResponseEvent(ctx, resp, tools, stateDelta)

				// Trace?
				// telemetry.TraceLLMCall(callLLMSpan, ctx, req, modelResponseEvent)

				if !yield(modelResponseEvent, nil) {
					return
				}

				// Handle function calls
				if resp.Content != nil {
					ev, err := f.handleFunctionCalls(ctx, tools, resp)
					if err != nil {
						yield(nil, err)
						return
					}

					if ev != nil {
						slog.Info("Function calls handled", "event", ev)
						// Yield the function response event (execution result)
						if !yield(ev, nil) {
							return
						}

						// Send tool response back to model
						// We need to extract the parts and construct a LiveRequest
						if ev.LLMResponse.Content != nil {
							toolResponses := &genai.LiveToolResponseInput{
								FunctionResponses: make([]*genai.FunctionResponse, 0),
							}
							for _, part := range ev.LLMResponse.Content.Parts {
								if part.FunctionResponse != nil {
									toolResponses.FunctionResponses = append(toolResponses.FunctionResponses, part.FunctionResponse)
								}
							}

							if len(toolResponses.FunctionResponses) > 0 {
								// Send directly to connection (bypass queue to avoid latency/ordering issues mixed with user input?)
								// Typically tool outputs corresponding to model calls should go ASAP.
								// But using queue ensures serialization if necessary.
								// ADK Python sends it directly or via queue?
								// Let's use the queue's helper if available or just construct request.
								// The queue is thread safe.

								queue.SendToolResponse(toolResponses)
							}
						}
					}
				}

				// Check for interruptions or turn completion if needed
				// The loop continues until error or context cancel.
				if ctx.Err() != nil {
					return
				}
			}

		}

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
			llmResponse.UsageMetadata == nil {
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
			yield(modelResponseEvent, nil)
			return
		}

		if ctx.RunConfig().SaveLiveBlob {
			// TODO: 
		}

	}
}
