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

package a2agrpc

import (
	"context"
	"strings"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2apb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"

	"github.com/a2aproject/a2a-go/a2apb/pbconv"
	"github.com/a2aproject/a2a-go/a2asrv"
)

// Handler implements protobuf translation layer and delegates the actual method handling to [a2asrv.RequestHandler].
type Handler struct {
	a2apb.UnimplementedA2AServiceServer
	handler a2asrv.RequestHandler
}

// RegisterWith registers as an A2AService implementation with the provided [grpc.Server].
func (h *Handler) RegisterWith(s *grpc.Server) {
	a2apb.RegisterA2AServiceServer(s, h)
}

// NewHandler is a [Handler] constructor function.
func NewHandler(handler a2asrv.RequestHandler) *Handler {
	return &Handler{handler: handler}
}

func (h *Handler) SendMessage(ctx context.Context, req *a2apb.SendMessageRequest) (*a2apb.SendMessageResponse, error) {
	if req.GetRequest() == nil {
		return nil, status.Error(codes.InvalidArgument, "request message is missing")
	}
	params, err := pbconv.FromProtoSendMessageRequest(req)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to convert request: %v", err)
	}

	ctx, callCtx := withCallContext(ctx)
	resp, err := h.handler.OnSendMessage(ctx, params)
	if err != nil {
		return nil, toGRPCError(err)
	}
	if err := grpc.SetTrailer(ctx, toTrailer(callCtx)); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to send active extensions: %v", err)
	}

	result, err := pbconv.ToProtoSendMessageResponse(resp)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to convert response: %v", err)
	}

	return result, nil
}

func (h *Handler) SendStreamingMessage(req *a2apb.SendMessageRequest, stream grpc.ServerStreamingServer[a2apb.StreamResponse]) error {
	if req.GetRequest() == nil {
		return status.Error(codes.InvalidArgument, "request message is missing")
	}
	params, err := pbconv.FromProtoSendMessageRequest(req)
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "failed to convert request: %v", err)
	}

	ctx, callCtx := withCallContext(stream.Context())
	for event, err := range h.handler.OnSendMessageStream(ctx, params) {
		if err != nil {
			return toGRPCError(err)
		}
		resp, err := pbconv.ToProtoStreamResponse(event)
		if err != nil {
			return status.Errorf(codes.Internal, "failed to convert response: %v", err)
		}
		err = stream.Send(resp)
		if err != nil {
			return status.Errorf(codes.Aborted, "failed to send response: %v", err)
		}
	}
	stream.SetTrailer(toTrailer(callCtx))

	return nil
}

func (h *Handler) GetTask(ctx context.Context, req *a2apb.GetTaskRequest) (*a2apb.Task, error) {
	params, err := pbconv.FromProtoGetTaskRequest(req)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to convert request: %v", err)
	}

	ctx, callCtx := withCallContext(ctx)
	task, err := h.handler.OnGetTask(ctx, params)
	if err != nil {
		return nil, toGRPCError(err)
	}
	if err := grpc.SetTrailer(ctx, toTrailer(callCtx)); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to send active extensions: %v", err)
	}

	result, err := pbconv.ToProtoTask(task)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to convert task: %v", err)
	}
	return result, nil
}

func (h *Handler) CancelTask(ctx context.Context, req *a2apb.CancelTaskRequest) (*a2apb.Task, error) {
	taskID, err := pbconv.ExtractTaskID(req.GetName())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to extract task id: %v", err)
	}

	ctx, callCtx := withCallContext(ctx)
	task, err := h.handler.OnCancelTask(ctx, &a2a.TaskIDParams{ID: taskID})
	if err != nil {
		return nil, toGRPCError(err)
	}
	if err := grpc.SetTrailer(ctx, toTrailer(callCtx)); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to send active extensions: %v", err)
	}

	result, err := pbconv.ToProtoTask(task)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to convert task: %v", err)
	}
	return result, nil
}

func (h *Handler) TaskSubscription(req *a2apb.TaskSubscriptionRequest, stream grpc.ServerStreamingServer[a2apb.StreamResponse]) error {
	taskID, err := pbconv.ExtractTaskID(req.GetName())
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "failed to extract task id: %v", err)
	}

	ctx, callCtx := withCallContext(stream.Context())
	for event, err := range h.handler.OnResubscribeToTask(ctx, &a2a.TaskIDParams{ID: taskID}) {
		if err != nil {
			return toGRPCError(err)
		}
		resp, err := pbconv.ToProtoStreamResponse(event)
		if err != nil {
			return status.Errorf(codes.Internal, "failed to convert response: %v", err)
		}
		err = stream.Send(resp)
		if err != nil {
			return status.Errorf(codes.Aborted, "failed to send response: %v", err)
		}
	}
	stream.SetTrailer(toTrailer(callCtx))

	return nil
}

func (h *Handler) CreateTaskPushNotificationConfig(ctx context.Context, req *a2apb.CreateTaskPushNotificationConfigRequest) (*a2apb.TaskPushNotificationConfig, error) {
	params, err := pbconv.FromProtoCreateTaskPushConfigRequest(req)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to convert request: %v", err)
	}

	ctx, callCtx := withCallContext(ctx)
	config, err := h.handler.OnSetTaskPushConfig(ctx, params)
	if err != nil {
		return nil, toGRPCError(err)
	}
	if err := grpc.SetTrailer(ctx, toTrailer(callCtx)); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to send active extensions: %v", err)
	}

	result, err := pbconv.ToProtoTaskPushConfig(config)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to convert push config: %v", err)
	}
	return result, nil
}

func (h *Handler) GetTaskPushNotificationConfig(ctx context.Context, req *a2apb.GetTaskPushNotificationConfigRequest) (*a2apb.TaskPushNotificationConfig, error) {
	params, err := pbconv.FromProtoGetTaskPushConfigRequest(req)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to convert request: %v", err)
	}

	ctx, callCtx := withCallContext(ctx)
	config, err := h.handler.OnGetTaskPushConfig(ctx, params)
	if err != nil {
		return nil, toGRPCError(err)
	}
	if err := grpc.SetTrailer(ctx, toTrailer(callCtx)); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to send active extensions: %v", err)
	}

	result, err := pbconv.ToProtoTaskPushConfig(config)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to convert push config: %v", err)
	}
	return result, nil
}

func (h *Handler) ListTaskPushNotificationConfig(ctx context.Context, req *a2apb.ListTaskPushNotificationConfigRequest) (*a2apb.ListTaskPushNotificationConfigResponse, error) {
	taskID, err := pbconv.ExtractTaskID(req.GetParent())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to extract task id: %v", err)
	}

	ctx, callCtx := withCallContext(ctx)
	// todo: handling pagination
	configs, err := h.handler.OnListTaskPushConfig(ctx, &a2a.ListTaskPushConfigParams{TaskID: taskID})
	if err != nil {
		return nil, toGRPCError(err)
	}
	if err := grpc.SetTrailer(ctx, toTrailer(callCtx)); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to send active extensions: %v", err)
	}

	result, err := pbconv.ToProtoListTaskPushConfig(configs)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to convert list of push configs: %v", err)
	}
	return result, nil
}

func (h *Handler) GetAgentCard(ctx context.Context, req *a2apb.GetAgentCardRequest) (*a2apb.AgentCard, error) {
	card, err := h.handler.OnGetExtendedAgentCard(ctx)
	if err != nil {
		return nil, toGRPCError(err)
	}
	result, err := pbconv.ToProtoAgentCard(card)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to convert agent card: %v", err)
	}
	return result, err
}

func (h *Handler) DeleteTaskPushNotificationConfig(ctx context.Context, req *a2apb.DeleteTaskPushNotificationConfigRequest) (*emptypb.Empty, error) {
	params, err := pbconv.FromProtoDeleteTaskPushConfigRequest(req)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to convert request: %v", err)
	}

	ctx, callCtx := withCallContext(ctx)
	if err := h.handler.OnDeleteTaskPushConfig(ctx, params); err != nil {
		return nil, toGRPCError(err)
	}
	if err := grpc.SetTrailer(ctx, toTrailer(callCtx)); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to send active extensions: %v", err)
	}

	return &emptypb.Empty{}, nil
}

func withCallContext(ctx context.Context) (context.Context, *a2asrv.CallContext) {
	var reqMeta *a2asrv.RequestMeta
	if meta, ok := metadata.FromIncomingContext(ctx); ok {
		reqMeta = a2asrv.NewRequestMeta(meta)
	}
	return a2asrv.WithCallContext(ctx, reqMeta)
}

func toTrailer(callCtx *a2asrv.CallContext) metadata.MD {
	activated := callCtx.Extensions().ActivatedURIs()
	if len(activated) == 0 {
		return metadata.MD{}
	}
	return metadata.MD{strings.ToLower(a2asrv.ExtensionsMetaKey): activated}
}
