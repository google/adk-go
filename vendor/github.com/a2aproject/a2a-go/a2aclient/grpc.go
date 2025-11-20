// Copyright 2025 The A2A Authors
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

package a2aclient

import (
	"context"
	"io"
	"iter"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2apb"
	"github.com/a2aproject/a2a-go/a2apb/pbconv"
)

// WithGRPCTransport create a gRPC transport implementation which will use the provided [grpc.DialOption]s during connection establishment.
func WithGRPCTransport(opts ...grpc.DialOption) FactoryOption {
	return WithTransport(
		a2a.TransportProtocolGRPC,
		TransportFactoryFn(func(ctx context.Context, url string, card *a2a.AgentCard) (Transport, error) {
			interceptors := []grpc.DialOption{
				grpc.WithStreamInterceptor(newStreamGRPCInterceptor()),
				grpc.WithUnaryInterceptor(newUnaryGRPCInterceptor()),
			}
			conn, err := grpc.NewClient(
				url,
				append(interceptors, opts...)...,
			)
			if err != nil {
				return nil, err
			}
			return NewGRPCTransport(conn), nil
		}),
	)
}

// NewGRPCTransport exposes a method for direct A2A gRPC protocol handler.
func NewGRPCTransport(conn *grpc.ClientConn) Transport {
	return &grpcTransport{
		client:      a2apb.NewA2AServiceClient(conn),
		closeConnFn: func() error { return conn.Close() },
	}
}

// grpcTransport implements Transport by delegating to a2apb.A2AServiceClient.
type grpcTransport struct {
	client      a2apb.A2AServiceClient
	closeConnFn func() error
}

// A2A protocol methods

func (c *grpcTransport) GetTask(ctx context.Context, query *a2a.TaskQueryParams) (*a2a.Task, error) {
	req, err := pbconv.ToProtoGetTaskRequest(query)
	if err != nil {
		return nil, err
	}

	pResp, err := c.client.GetTask(ctx, req)
	if err != nil {
		return nil, err
	}

	return pbconv.FromProtoTask(pResp)
}

func (c *grpcTransport) CancelTask(ctx context.Context, id *a2a.TaskIDParams) (*a2a.Task, error) {
	req, err := pbconv.ToProtoCancelTaskRequest(id)
	if err != nil {
		return nil, err
	}

	pResp, err := c.client.CancelTask(ctx, req)
	if err != nil {
		return nil, err
	}

	return pbconv.FromProtoTask(pResp)
}

func (c *grpcTransport) SendMessage(ctx context.Context, message *a2a.MessageSendParams) (a2a.SendMessageResult, error) {
	req, err := pbconv.ToProtoSendMessageRequest(message)
	if err != nil {
		return nil, err
	}

	pResp, err := c.client.SendMessage(ctx, req)
	if err != nil {
		return nil, err
	}

	return pbconv.FromProtoSendMessageResponse(pResp)
}

func (c *grpcTransport) ResubscribeToTask(ctx context.Context, id *a2a.TaskIDParams) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		req, err := pbconv.ToProtoTaskSubscriptionRequest(id)
		if err != nil {
			yield(nil, err)
			return
		}

		stream, err := c.client.TaskSubscription(ctx, req)
		if err != nil {
			yield(nil, err)
			return
		}

		drainEventStream(stream, yield)
	}
}

func (c *grpcTransport) SendStreamingMessage(ctx context.Context, message *a2a.MessageSendParams) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		req, err := pbconv.ToProtoSendMessageRequest(message)
		if err != nil {
			yield(nil, err)
			return
		}

		stream, err := c.client.SendStreamingMessage(ctx, req)
		if err != nil {
			yield(nil, err)
			return
		}

		drainEventStream(stream, yield)
	}
}

func drainEventStream(stream grpc.ServerStreamingClient[a2apb.StreamResponse], yield func(a2a.Event, error) bool) {
	for {
		pResp, err := stream.Recv()
		if err == io.EOF {
			return
		}
		if err != nil {
			yield(nil, err)
			return
		}

		resp, err := pbconv.FromProtoStreamResponse(pResp)
		if err != nil {
			yield(nil, err)
			return
		}

		if !yield(resp, nil) {
			return
		}
	}
}

func (c *grpcTransport) GetTaskPushConfig(ctx context.Context, params *a2a.GetTaskPushConfigParams) (*a2a.TaskPushConfig, error) {
	req, err := pbconv.ToProtoGetTaskPushConfigRequest(params)
	if err != nil {
		return nil, err
	}

	pResp, err := c.client.GetTaskPushNotificationConfig(ctx, req)
	if err != nil {
		return nil, err
	}

	return pbconv.FromProtoTaskPushConfig(pResp)
}

func (c *grpcTransport) ListTaskPushConfig(ctx context.Context, params *a2a.ListTaskPushConfigParams) ([]*a2a.TaskPushConfig, error) {
	req, err := pbconv.ToProtoListTaskPushConfigRequest(params)
	if err != nil {
		return nil, err
	}

	pResp, err := c.client.ListTaskPushNotificationConfig(ctx, req)
	if err != nil {
		return nil, err
	}

	return pbconv.FromProtoListTaskPushConfig(pResp)
}

func (c *grpcTransport) SetTaskPushConfig(ctx context.Context, params *a2a.TaskPushConfig) (*a2a.TaskPushConfig, error) {
	req, err := pbconv.ToProtoCreateTaskPushConfigRequest(params)
	if err != nil {
		return nil, err
	}

	pResp, err := c.client.CreateTaskPushNotificationConfig(ctx, req)
	if err != nil {
		return nil, err
	}

	return pbconv.FromProtoTaskPushConfig(pResp)
}

func (c *grpcTransport) DeleteTaskPushConfig(ctx context.Context, params *a2a.DeleteTaskPushConfigParams) error {
	req, err := pbconv.ToProtoDeleteTaskPushConfigRequest(params)
	if err != nil {
		return err
	}

	_, err = c.client.DeleteTaskPushNotificationConfig(ctx, req)

	return err
}

func (c *grpcTransport) GetAgentCard(ctx context.Context) (*a2a.AgentCard, error) {
	pCard, err := c.client.GetAgentCard(ctx, &a2apb.GetAgentCardRequest{})

	if err != nil {
		return nil, err
	}

	return pbconv.FromProtoAgentCard(pCard)
}

func (c *grpcTransport) Destroy() error {
	return c.closeConnFn()
}

func newUnaryGRPCInterceptor() grpc.UnaryClientInterceptor {
	return func(
		ctx context.Context,
		method string,
		req, reply interface{},
		cc *grpc.ClientConn,
		invoker grpc.UnaryInvoker,
		opts ...grpc.CallOption,
	) error {
		return invoker(withGRPCMetadata(ctx), method, req, reply, cc, opts...)
	}
}

func newStreamGRPCInterceptor() grpc.StreamClientInterceptor {
	return func(
		ctx context.Context,
		desc *grpc.StreamDesc,
		cc *grpc.ClientConn,
		method string,
		streamer grpc.Streamer,
		opts ...grpc.CallOption,
	) (grpc.ClientStream, error) {
		return streamer(withGRPCMetadata(ctx), desc, cc, method, opts...)
	}
}

func withGRPCMetadata(ctx context.Context) context.Context {
	callMeta, ok := CallMetaFrom(ctx)
	if !ok || len(callMeta) == 0 {
		return ctx
	}
	meta := metadata.MD{}
	for k, vals := range callMeta {
		meta[strings.ToLower(k)] = vals
	}
	return metadata.NewOutgoingContext(ctx, meta)
}
