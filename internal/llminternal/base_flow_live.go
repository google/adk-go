package llminternal

import (
	"fmt"
	"iter"
	"strings"

	"github.com/rs/zerolog/log"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/runconfig"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
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
						if liveReq.Close {
							conn.Close()
							return
						}

						if liveReq.ActivityStart != nil || liveReq.ActivityEnd != nil {
							if err := conn.Send(liveReq); err != nil {
								// TODO: Handle send error.
								fmt.Printf("Error sending to live connection: %v\n", err)
								return
							}
						} else if liveReq.RealtimeInput != nil && liveReq.RealtimeInput.Audio != nil {
							err := f.AudioCacheManager.CacheAudio(
								ctx,
								liveReq.RealtimeInput.Audio,
								"input",
							)
							if err != nil {
								// TODO: Handle send error.
								fmt.Printf("Error caching audio: %v\n", err)
							}
							if err := conn.Send(liveReq); err != nil {
								// TODO: Handle send error.
								fmt.Printf("Error sending to live connection: %v\n", err)
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

								yield(userContentEvent, nil)

								if err := conn.Send(liveReq); err != nil {
									// TODO: Handle send error.
									fmt.Printf("Error sending to live connection: %v\n", err)
									return
								}
							}

						}
					case <-ctx.Done():
						return
					}
				}
			}()

			getAuthorForEvent := func(llmResponse *model.LLMResponse) string {
				if llmResponse != nil && llmResponse.Content != nil && llmResponse.Content.Role == "user" {
					return "user"
				}
				return ctx.Agent().Name()
			}

			resps, errs := conn.Receive(ctx)
			for {
				select {
				case resp, ok := <-resps:
					if !ok {
						break
					}

					log.Info().Interface("resp", resp).Msg("Received response from conn.Receive")

					if resp.LiveSessionResumptionUpdate != nil {
						log.Info().Str("handle", resp.LiveSessionResumptionUpdate.NewHandle).Msg("Live session resumption update")
						ctx.SetLiveSessionResumptionHandle(resp.LiveSessionResumptionUpdate.NewHandle)
					}

					modelResponseEvent := session.NewEvent(ctx.InvocationID())
					modelResponseEvent.Content = resp.Content
					modelResponseEvent.Author = getAuthorForEvent(resp)
					modelResponseEvent.OutputTranscription = resp.OutputTranscription
					modelResponseEvent.InputTranscription = resp.InputTranscription

					for ev, err := range f.postprocessLive(ctx, req, resp, modelResponseEvent) {
						if err != nil {
							yield(nil, err)
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

						log.Info().Interface("ev", ev).Msg("Yielding event")

						if !yield(ev, nil) {
							return
						}

					}
				case err, ok := <-errs:
					if !ok {
						break
					}
					if err != nil {
						yield(nil, err)
						return
					}

				case <-ctx.Done():
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

		// Flush audio caches based on control events using configurable settings
		if ctx.RunConfig().SaveLiveBlob {
			flushedEvents := f.handleControlEventFlush(ctx, llmResponse)
			for _, ev := range flushedEvents {
				yield(ev, nil)
			}
			if len(flushedEvents) > 0 {
				// NOTE below return is O.K. for now, because currently we only flush
				// events on interrupted or turn_complete. turn_complete is a pure
				// control event and interrupted is not with content but those content
				// is ignorable because model is already interrupted. If we have other
				// case to flush events in the future that are not pure control events,
				// we should not return here.
				return
			}
		}

		// Builds the event.
		modelResponseEvent = f.finalizeLiveModelResponseEvent(ctx, llmRequest, llmResponse, modelResponseEvent)
		yield(modelResponseEvent, nil)

		// TODO: handle function calls
	}
}

func (f *Flow) finalizeLiveModelResponseEvent(ctx agent.InvocationContext, llmRequest *model.LLMRequest, llmResponse *model.LLMResponse, modelResponseEvent *session.Event) *session.Event {
	// TODO: not sure what to modify here
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
