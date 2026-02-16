package llminternal

import (
	"fmt"
	"iter"
	"log/slog"

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
							slog.Info("Sending tool responses", "toolResponses", toolResponses)
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
