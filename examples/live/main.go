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

package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"

	"net/http"
	"os"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/examples/live/models"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/session/database"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
	"gorm.io/driver/postgres"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for this example
	},
}

type ClientMessage struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Data     string `json:"data,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
}

func main() {

	var stdOut io.Writer = os.Stdout
	writers := []io.Writer{stdOut}
	var runLogFile *os.File

	runLogFile, err := os.OpenFile(
		"test.log",
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0666,
	)
	if err != nil {
		log.Fatal().Err(err).Msg("unable to open log file")
	}

	writers = append(writers, runLogFile)

	zerolog.TimeFieldFormat = time.RFC3339Nano

	multi := zerolog.MultiLevelWriter(writers...)
	log.Logger = zerolog.New(multi).With().Timestamp().Logger()
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		log.Fatal().Msg("GOOGLE_API_KEY environment variable is not set")
	}

	dsn := fmt.Sprintf("host=127.0.0.1 user=adk password=adk dbname=adk port=5432 sslmode=disable")
	sessionService, err := database.NewSessionService(postgres.Open(dsn))
	if err != nil {
		log.Printf("Failed to create session service: %v", err)
		return
	}

	if err := database.AutoMigrate(sessionService); err != nil {
		log.Printf("Failed to auto migrate session service: %v", err)
		return
	}

	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash-native-audio-preview-09-2025", &genai.ClientConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Printf("Failed to create model: %v", err)
		return
	}

	poemAgentModel, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatal().Msgf("Failed to create model: %v", err)
	}

	type Input struct {
		LineCount int `json:"lineCount"`
	}
	type Output struct {
		Poem string `json:"poem"`
	}
	// handler := func(ctx tool.Context, input Input) (Output, error) {
	// 	return Output{
	// 		Poem: strings.Repeat("A line of a poem,", input.LineCount) + "\n",
	// 	}, nil
	// }
	// poemTool, err := functiontool.New(functiontool.Config{
	// 	Name:        "poem",
	// 	Description: "Returns poem",
	// }, handler)
	// if err != nil {
	// 	log.Fatalf("Failed to create tool: %v", err)
	// }
	poemAgent, err := llmagent.New(llmagent.Config{
		Name:        "poem_agent",
		Model:       poemAgentModel,
		Description: "Generate a love poem about cats.",
		Instruction: "Generate a love poem about cats.",
		// Tools: []tool.Tool{
		// 	poemTool,
		// },
	})
	if err != nil {
		log.Fatal().Msgf("Failed to create agent: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "live_agent",
		Model:       model,
		Description: "A live agent that echoes what you say and generates poems.",
		Instruction: "You are a live assistant. Respond briefly to the user. If asked for a poem, use the poem_agent to generate a poem and read it.",
		Tools: []tool.Tool{
			agenttool.New(poemAgent, nil),
		},
	})
	if err != nil {
		log.Printf("Failed to create agent: %v", err)
		return
	}

	runn, err := runner.New(runner.Config{
		AppName:        "live_sample",
		Agent:          a,
		SessionService: sessionService,
		ResumabilityConfig: &agent.ResumabilityConfig{
			IsResumable: false,
		},
		ArtifactService: artifact.InMemoryService(),
	})
	if err != nil {
		log.Printf("Failed to create runner: %v", err)
		return
	}

	server := &Server{
		sessionService: sessionService,
		runner:         runn,
	}

	r := mux.NewRouter()

	// WebSocket handler with userId and sessionId parameters
	r.HandleFunc("/ws/{userId}/{sessionId}", server.websocketHandler())

	// Static file server with no-cache headers
	staticDir := "static"
	fileServer := http.FileServer(http.Dir(staticDir))
	noCacheHandler := func(h http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
			w.Header().Set("Pragma", "no-cache")
			w.Header().Set("Expires", "0")
			h.ServeHTTP(w, r)
		})
	}
	r.PathPrefix("/static/").Handler(noCacheHandler(http.StripPrefix("/static/", fileServer)))
	r.PathPrefix("/").Handler(noCacheHandler(fileServer))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	fmt.Printf("Server starting on http://localhost:%s\n", port)
	if err := http.ListenAndServe(":"+port, r); err != nil {
		log.Fatal().Err(err).Msg("Server failed")
	}
}

type Server struct {
	sessionService session.Service
	runner         *runner.Runner
}

func (s *Server) websocketHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		userID := vars["userId"]
		sessionID := vars["sessionId"]

		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("Failed to upgrade connection: %v", err)
			return
		}
		defer conn.Close()

		ctx := context.Background()

		// Create session for the user
		_, err = s.sessionService.Create(ctx, &session.CreateRequest{
			AppName:   "live_sample",
			UserID:    userID,
			SessionID: sessionID,
		})
		if err != nil {
			log.Printf("Session creation (might already exist): %v", err)
		}

		// Phase 2 - 3
		runConfig := agent.RunConfig{
			StreamingMode:             agent.StreamingModeBidi,
			ResponseModalities:        []genai.Modality{genai.ModalityAudio},
			InputAudioTranscription:   &genai.AudioTranscriptionConfig{},
			OutputAudioTranscription:  &genai.AudioTranscriptionConfig{},
			SaveLiveBlob:              true,
			SaveInputBlobsAsArtifacts: true,
		}

		// Phase 2 - 4
		liveRequestQueue := agent.NewLiveRequestQueue()

		// Channel to signal the reading loop to stop
		done := make(chan struct{})

		// Start receiving ADK events and sending them to the WebSocket
		go func() {
			defer close(done)

			// Phase 2 - 5
			for ev, err := range s.runner.RunLive(ctx, userID, sessionID, liveRequestQueue, runConfig) {
				if err != nil {
					log.Printf("Runner error: %v", err)
					return
				}

				// Convert ADK event to JSON and send to client
				evJSON, err := json.Marshal(models.FromSessionEvent(*ev))
				if err != nil {
					log.Printf("Failed to marshal event: %v", err)
					continue
				}

				if err := conn.WriteMessage(websocket.TextMessage, evJSON); err != nil {
					log.Printf("WebSocket write error: %v", err)
					return
				}
			}
		}()

		// Read from WebSocket and send to the ADK LiveRequestQueue
		for {
			messageType, p, err := conn.ReadMessage()
			if err != nil {
				log.Printf("WebSocket read error: %v", err)
				break
			}

			if messageType == websocket.BinaryMessage {
				log.Printf("Received binary message: %d bytes", len(p))
				// Binary data is assumed to be PCM audio from the client
				err := liveRequestQueue.SendContent(&genai.Content{
					Role: "user",
					Parts: []*genai.Part{
						{
							InlineData: &genai.Blob{
								MIMEType: "audio/pcm;rate=16000",
								Data:     p,
							},
						},
					},
				})
				if err != nil {
					log.Printf("Error sending audio to queue: %v", err)
				}
			} else if messageType == websocket.TextMessage {
				var msg ClientMessage
				if err := json.Unmarshal(p, &msg); err != nil {
					log.Printf("Failed to unmarshal text message: %v", err)
					continue
				}

				switch msg.Type {
				case "text":
					// Phase 3 Send content / Send real time
					err := liveRequestQueue.SendContent(genai.NewContentFromText(msg.Text, genai.RoleUser))
					if err != nil {
						log.Printf("Error sending text to queue: %v", err)
					}
				case "image":
					data, err := base64.StdEncoding.DecodeString(msg.Data)
					if err != nil {
						log.Printf("Failed to decode base64 image: %v", err)
						continue
					}
					// Phase 3 Send content / Send real time
					err = liveRequestQueue.SendRealtimeInput(&genai.LiveRealtimeInput{
						Media: &genai.Blob{
							Data:     data,
							MIMEType: msg.MimeType,
						},
					})
					if err != nil {
						log.Printf("Error sending image to queue: %v", err)
					}
				}
			}

			select {
			case <-done:
				return
			default:
			}
		}

		log.Printf("Closing queue")
		liveRequestQueue.Close()
	}
}
