package agent

import (
	"errors"
	"sync"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// LiveRequestQueue is a queue used to send LiveRequest in a live(bidirectional streaming) way.
type LiveRequestQueue struct {
	queue  chan *model.LiveRequest
	closed bool
	mu     sync.Mutex
}

// NewLiveRequestQueue creates a new LiveRequestQueue.
func NewLiveRequestQueue() *LiveRequestQueue {
	return &LiveRequestQueue{
		queue: make(chan *model.LiveRequest, 100), // Buffered channel to prevent blocking on small bursts
	}
}

// Close closes the queue and sends a close signal.
func (q *LiveRequestQueue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	if !q.closed {
		q.closed = true
		// We define Close=true as a signal in the stream, but we also close the channel
		// to signal end of iteration if needed.
		// However, to follow the Python pattern where "close" is a request type:
		q.sendLocked(&model.LiveRequest{Close: true})
		close(q.queue)
	}
}

// SendContent sends content to the queue.
func (q *LiveRequestQueue) SendContent(content *genai.Content) error {
	return q.Send(&model.LiveRequest{Content: content})
}

// SendRealtimeInput sends realtime input (audio/video) to the queue.
func (q *LiveRequestQueue) SendRealtimeInput(input *genai.LiveRealtimeInput) error {
	return q.Send(&model.LiveRequest{RealtimeInput: input})
}

// SendToolResponse sends tool response to the queue.
func (q *LiveRequestQueue) SendToolResponse(resp *genai.LiveToolResponseInput) error {
	return q.Send(&model.LiveRequest{ToolResponse: resp})
}

func (q *LiveRequestQueue) SendActivityStart() error {
	return q.Send(&model.LiveRequest{ActivityStart: &genai.ActivityStart{}})
}

func (q *LiveRequestQueue) SendActivityEnd() error {
	return q.Send(&model.LiveRequest{ActivityEnd: &genai.ActivityEnd{}})
}

// Send sends a generic LiveRequest to the queue.
func (q *LiveRequestQueue) Send(req *model.LiveRequest) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.sendLocked(req)
}

func (q *LiveRequestQueue) sendLocked(req *model.LiveRequest) error {
	if q.closed {
		return errors.New("queue is closed")
	}
	// TODO: Handle blocking or dropping if full? For now, we assume buffer is enough or we block.
	// Non-blocking send could be:
	// select {
	// case q.queue <- req:
	// default:
	//     return errors.New("queue full")
	// }
	// But blocking is usually safer for reliable delivery unless we have strict latency requirements to drop.
	q.queue <- req
	return nil
}

// Next returns the next request from the queue.
// It returns nil, false if the queue is closed and empty.
func (q *LiveRequestQueue) Next(ctxTimeout time.Duration) (*model.LiveRequest, bool) {
	select {
	case req, ok := <-q.queue:
		return req, ok
	case <-time.After(ctxTimeout):
		// This is just a helper, caller might use select on channel directly.
		return nil, true // Keep alive?
	}
}

// Channel returns the underlying channel for reading.
func (q *LiveRequestQueue) Channel() <-chan *model.LiveRequest {
	return q.queue
}
