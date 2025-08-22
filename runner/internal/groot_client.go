package internal

import (
	"iter"

	"github.com/gorilla/websocket"
)

type Client struct {
	conn *websocket.Conn
}

type Port struct {
	Name     string `json:"name,omitempty"`
	StreamID string `json:"streamId,omitempty"`
}

type ActionGraph struct {
	Actions []*Action `json:"actions,omitempty"`
	Outputs []*Port   `json:"outputs,omitempty"`
}

type Action struct {
	Name    string  `json:"name,omitempty"`
	Inputs  []*Port `json:"inputs,omitempty"`
	Outputs []*Port `json:"outputs,omitempty"`
	// TODO: Add configs.
}

type Chunk struct {
	MIMEType string `json:"mimeType,omitempty"`
	Data     []byte `json:"data,omitempty"`
	// TODO: Add metadata.
}

type StreamFrame struct {
	StreamID  string `json:"streamId,omitempty"`
	Data      *Chunk `json:"data,omitempty"`
	Continued bool   `json:"continued,omitempty"`
}

type executeActionsMsg struct {
	SessionID    string         `json:"sessionId,omitempty"`
	ActionGraph  *ActionGraph   `json:"actionGraph,omitempty"`
	StreamFrames []*StreamFrame `json:"streamFrames,omitempty"`
}

type ShadowStatus string

const (
	ShadowStatusPending   = "pending"
	ShadowStatusCompleted = "completed"
)

type Shadow struct{}

func NewClient(endpoint string, apiKey string) (*Client, error) {
	c, _, err := websocket.DefaultDialer.Dial(endpoint+"?key="+apiKey, nil)
	if err != nil {
		return nil, err
	}
	return &Client{conn: c}, nil
}

type Session struct {
	c         *Client
	sessionID string
}

func (c *Client) OpenSession(sessionID string) (*Session, error) {
	// TODO(jbd) Start session for real.
	return &Session{c: c, sessionID: sessionID}, nil
}

func (s *Session) ShadowADK(name string, input string, output string) (*Shadow, ShadowStatus, error) {
	panic("not implemented")
}

func (s *Session) ExecuteActions(actions []*Action, outputs []string) iter.Seq2[*StreamFrame, error] {
	outputs = []string{"test"}
	return func(yield func(*StreamFrame, error) bool) {
		if err := s.c.conn.WriteJSON(&executeActionsMsg{
			SessionID: s.sessionID,
			ActionGraph: &ActionGraph{
				Actions: []*Action{
					{
						Name:    "save_stream",
						Inputs:  []*Port{{Name: "input", StreamID: "test"}},
						Outputs: []*Port{{Name: "output", StreamID: "save_output"}},
					},
				},
				Outputs: []*Port{{Name: "output", StreamID: "test"}},
			},
			StreamFrames: []*StreamFrame{
				{StreamID: "test", Data: &Chunk{MIMEType: "text/plain", Data: []byte("hello world")}},
			},
		}); err != nil {
			yield(nil, err)
			return
		}

		waiting := make(map[string]struct{})
		for _, output := range outputs {
			waiting[output] = struct{}{}
		}
		for {
			var resp executeActionsMsg
			if err := s.c.conn.ReadJSON(&resp); err != nil {
				yield(nil, err)
				return
			}
			for _, frame := range resp.StreamFrames {
				if !frame.Continued {
					delete(waiting, frame.StreamID)
				}
				if !yield(frame, nil) {
					return
				}
			}
			if len(waiting) == 0 {
				break
			}
		}
	}
}

// func (s *Session) ExecuteADK(name string, inputs)
