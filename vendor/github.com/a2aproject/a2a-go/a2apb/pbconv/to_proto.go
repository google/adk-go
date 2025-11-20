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

package pbconv

import (
	"fmt"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2apb"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func toProtoMap(meta map[string]any) (*structpb.Struct, error) {
	if meta == nil {
		return nil, nil
	}
	s, err := structpb.NewStruct(meta)
	if err != nil {
		return nil, fmt.Errorf("failed to convert metadata to proto struct: %w", err)
	}
	return s, nil
}

func ToProtoSendMessageRequest(params *a2a.MessageSendParams) (*a2apb.SendMessageRequest, error) {
	if params == nil {
		return nil, nil
	}

	pMsg, err := toProtoMessage(params.Message)
	if err != nil {
		return nil, err
	}

	pConf, err := toProtoSendMessageConfig(params.Config)
	if err != nil {
		return nil, err
	}

	pMeta, err := structpb.NewStruct(params.Metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to convert metadata to proto struct: %w", err)
	}

	req := &a2apb.SendMessageRequest{
		Request:       pMsg,
		Configuration: pConf,
		Metadata:      pMeta,
	}
	return req, nil
}

func toProtoPushConfig(config *a2a.PushConfig) (*a2apb.PushNotificationConfig, error) {
	if config == nil {
		return nil, nil
	}

	pConf := &a2apb.PushNotificationConfig{
		Id:    config.ID,
		Url:   config.URL,
		Token: config.Token,
	}
	if config.Auth != nil {
		pConf.Authentication = &a2apb.AuthenticationInfo{
			Schemes:     config.Auth.Schemes,
			Credentials: config.Auth.Credentials,
		}
	}
	return pConf, nil
}

func toProtoSendMessageConfig(config *a2a.MessageSendConfig) (*a2apb.SendMessageConfiguration, error) {
	if config == nil {
		return nil, nil
	}

	pushConf, err := toProtoPushConfig(config.PushConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to convert push config: %w", err)
	}

	pConf := &a2apb.SendMessageConfiguration{
		AcceptedOutputModes: config.AcceptedOutputModes,
		PushNotification:    pushConf,
	}
	if config.Blocking != nil {
		pConf.Blocking = *config.Blocking
	}
	if config.HistoryLength != nil {
		pConf.HistoryLength = int32(*config.HistoryLength)
	}
	return pConf, nil
}

func ToProtoGetTaskRequest(params *a2a.TaskQueryParams) (*a2apb.GetTaskRequest, error) {
	if params == nil {
		return nil, nil
	}

	req := &a2apb.GetTaskRequest{Name: MakeTaskName(params.ID)}
	if params.HistoryLength != nil {
		req.HistoryLength = int32(*params.HistoryLength)
	}
	return req, nil
}

func ToProtoCancelTaskRequest(params *a2a.TaskIDParams) (*a2apb.CancelTaskRequest, error) {
	if params == nil {
		return nil, nil
	}
	return &a2apb.CancelTaskRequest{Name: MakeTaskName(params.ID)}, nil
}

func ToProtoTaskSubscriptionRequest(params *a2a.TaskIDParams) (*a2apb.TaskSubscriptionRequest, error) {
	if params == nil {
		return nil, nil
	}
	return &a2apb.TaskSubscriptionRequest{Name: MakeTaskName(params.ID)}, nil
}

func ToProtoCreateTaskPushConfigRequest(config *a2a.TaskPushConfig) (*a2apb.CreateTaskPushNotificationConfigRequest, error) {
	if config == nil {
		return nil, nil
	}

	pnc, err := toProtoPushConfig(&config.Config)
	if err != nil {
		return nil, fmt.Errorf("failed to convert push config: %w", err)
	}

	return &a2apb.CreateTaskPushNotificationConfigRequest{
		Parent: MakeTaskName(config.TaskID),
		Config: &a2apb.TaskPushNotificationConfig{PushNotificationConfig: pnc},
	}, nil
}

func ToProtoGetTaskPushConfigRequest(params *a2a.GetTaskPushConfigParams) (*a2apb.GetTaskPushNotificationConfigRequest, error) {
	if params == nil {
		return nil, nil
	}
	return &a2apb.GetTaskPushNotificationConfigRequest{
		Name: MakeConfigName(params.TaskID, params.ConfigID),
	}, nil
}

func ToProtoDeleteTaskPushConfigRequest(params *a2a.DeleteTaskPushConfigParams) (*a2apb.DeleteTaskPushNotificationConfigRequest, error) {
	if params == nil {
		return nil, nil
	}
	return &a2apb.DeleteTaskPushNotificationConfigRequest{
		Name: MakeConfigName(params.TaskID, params.ConfigID),
	}, nil
}

func ToProtoSendMessageResponse(result a2a.SendMessageResult) (*a2apb.SendMessageResponse, error) {
	resp := &a2apb.SendMessageResponse{}
	switch r := result.(type) {
	case *a2a.Message:
		pMsg, err := toProtoMessage(r)
		if err != nil {
			return nil, err
		}
		resp.Payload = &a2apb.SendMessageResponse_Msg{Msg: pMsg}
	case *a2a.Task:
		pTask, err := ToProtoTask(r)
		if err != nil {
			return nil, err
		}
		resp.Payload = &a2apb.SendMessageResponse_Task{Task: pTask}
	default:
		return nil, fmt.Errorf("unsupported SendMessageResult type: %T", result)
	}
	return resp, nil
}

func ToProtoStreamResponse(event a2a.Event) (*a2apb.StreamResponse, error) {
	resp := &a2apb.StreamResponse{}
	switch e := event.(type) {
	case *a2a.Message:
		pMsg, err := toProtoMessage(e)
		if err != nil {
			return nil, err
		}
		resp.Payload = &a2apb.StreamResponse_Msg{Msg: pMsg}
	case *a2a.Task:
		pTask, err := ToProtoTask(e)
		if err != nil {
			return nil, err
		}
		resp.Payload = &a2apb.StreamResponse_Task{Task: pTask}
	case *a2a.TaskStatusUpdateEvent:
		pStatus, err := toProtoTaskStatus(e.Status)
		if err != nil {
			return nil, err
		}
		metadata, err := toProtoMap(e.Metadata)
		if err != nil {
			return nil, err
		}
		resp.Payload = &a2apb.StreamResponse_StatusUpdate{StatusUpdate: &a2apb.TaskStatusUpdateEvent{
			ContextId: e.ContextID,
			Final:     e.Final,
			Status:    pStatus,
			TaskId:    string(e.TaskID),
			Metadata:  metadata,
		}}
	case *a2a.TaskArtifactUpdateEvent:
		pArtifact, err := toProtoArtifact(e.Artifact)
		if err != nil {
			return nil, err
		}
		metadata, err := toProtoMap(e.Metadata)
		if err != nil {
			return nil, err
		}
		resp.Payload = &a2apb.StreamResponse_ArtifactUpdate{
			ArtifactUpdate: &a2apb.TaskArtifactUpdateEvent{
				Append:    e.Append,
				Artifact:  pArtifact,
				ContextId: e.ContextID,
				LastChunk: e.LastChunk,
				TaskId:    string(e.TaskID),
				Metadata:  metadata,
			}}
	default:
		return nil, fmt.Errorf("unsupported Event type: %T", event)
	}
	return resp, nil
}

func toProtoMessage(msg *a2a.Message) (*a2apb.Message, error) {
	if msg == nil {
		return nil, nil
	}

	parts, err := toProtoParts(msg.Parts)
	if err != nil {
		return nil, fmt.Errorf("failed to convert parts: %w", err)
	}

	pMetadata, err := toProtoMap(msg.Metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to convert metadata to proto struct: %w", err)
	}
	var taskIDs []string
	if msg.ReferenceTasks != nil {
		taskIDs = make([]string, len(msg.ReferenceTasks))
		for i, tid := range msg.ReferenceTasks {
			taskIDs[i] = string(tid)
		}
	}

	return &a2apb.Message{
		MessageId:        msg.ID,
		ContextId:        msg.ContextID,
		Extensions:       msg.Extensions,
		Parts:            parts,
		Role:             toProtoRole(msg.Role),
		TaskId:           string(msg.TaskID),
		Metadata:         pMetadata,
		ReferenceTaskIds: taskIDs,
	}, nil
}

func toProtoMessages(msgs []*a2a.Message) ([]*a2apb.Message, error) {
	pMsgs := make([]*a2apb.Message, len(msgs))
	for i, msg := range msgs {
		pMsg, err := toProtoMessage(msg)
		if err != nil {
			return nil, fmt.Errorf("failed to convert message: %w", err)
		}
		pMsgs[i] = pMsg
	}
	return pMsgs, nil
}

func toProtoFilePart(part a2a.FilePart) (*a2apb.Part, error) {
	meta, err := toProtoMap(part.Metadata)
	if err != nil {
		return nil, err
	}
	switch fc := part.File.(type) {
	case a2a.FileBytes:
		return &a2apb.Part{
			Part: &a2apb.Part_File{File: &a2apb.FilePart{
				MimeType: fc.MimeType,
				Name:     fc.Name,
				File:     &a2apb.FilePart_FileWithBytes{FileWithBytes: []byte(fc.Bytes)},
			}},
			Metadata: meta,
		}, nil
	case a2a.FileURI:
		return &a2apb.Part{
			Part: &a2apb.Part_File{File: &a2apb.FilePart{
				MimeType: fc.MimeType,
				Name:     fc.Name,
				File:     &a2apb.FilePart_FileWithUri{FileWithUri: fc.URI},
			}},
			Metadata: meta,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported FilePartContent type: %T", fc)
	}
}

func toProtoDataPart(part a2a.DataPart) (*a2apb.Part, error) {
	s, err := structpb.NewStruct(part.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to convert data to proto struct: %w", err)
	}
	meta, err := toProtoMap(part.Metadata)
	if err != nil {
		return nil, err
	}
	return &a2apb.Part{
		Part:     &a2apb.Part_Data{Data: &a2apb.DataPart{Data: s}},
		Metadata: meta,
	}, nil
}

func toProtoPart(part a2a.Part) (*a2apb.Part, error) {
	switch p := part.(type) {
	case a2a.TextPart:
		meta, err := toProtoMap(p.Metadata)
		if err != nil {
			return nil, err
		}
		return &a2apb.Part{Part: &a2apb.Part_Text{Text: p.Text}, Metadata: meta}, nil
	case a2a.DataPart:
		return toProtoDataPart(p)
	case a2a.FilePart:
		return toProtoFilePart(p)
	default:
		return nil, fmt.Errorf("unsupported part type: %T", p)
	}
}

func toProtoParts(parts []a2a.Part) ([]*a2apb.Part, error) {
	pParts := make([]*a2apb.Part, len(parts))
	for i, part := range parts {
		pPart, err := toProtoPart(part)
		if err != nil {
			return nil, fmt.Errorf("failed to convert part: %w", err)
		}
		pParts[i] = pPart
	}
	return pParts, nil
}

func toProtoRole(role a2a.MessageRole) a2apb.Role {
	switch role {
	case a2a.MessageRoleUser:
		return a2apb.Role_ROLE_USER
	case a2a.MessageRoleAgent:
		return a2apb.Role_ROLE_AGENT
	default:
		return a2apb.Role_ROLE_UNSPECIFIED
	}
}

func toProtoTaskState(state a2a.TaskState) a2apb.TaskState {
	switch state {
	case a2a.TaskStateAuthRequired:
		return a2apb.TaskState_TASK_STATE_AUTH_REQUIRED
	case a2a.TaskStateCanceled:
		return a2apb.TaskState_TASK_STATE_CANCELLED
	case a2a.TaskStateCompleted:
		return a2apb.TaskState_TASK_STATE_COMPLETED
	case a2a.TaskStateFailed:
		return a2apb.TaskState_TASK_STATE_FAILED
	case a2a.TaskStateInputRequired:
		return a2apb.TaskState_TASK_STATE_INPUT_REQUIRED
	case a2a.TaskStateRejected:
		return a2apb.TaskState_TASK_STATE_REJECTED
	case a2a.TaskStateSubmitted:
		return a2apb.TaskState_TASK_STATE_SUBMITTED
	case a2a.TaskStateWorking:
		return a2apb.TaskState_TASK_STATE_WORKING
	default:
		return a2apb.TaskState_TASK_STATE_UNSPECIFIED
	}
}

func toProtoTaskStatus(status a2a.TaskStatus) (*a2apb.TaskStatus, error) {
	message, err := toProtoMessage(status.Message)
	if err != nil {
		return nil, fmt.Errorf("failed to convert message for task status: %w", err)
	}

	pStatus := &a2apb.TaskStatus{
		State:  toProtoTaskState(status.State),
		Update: message,
	}
	if status.Timestamp != nil {
		pStatus.Timestamp = timestamppb.New(*status.Timestamp)
	}

	return pStatus, nil
}

func toProtoArtifact(artifact *a2a.Artifact) (*a2apb.Artifact, error) {
	if artifact == nil {
		return nil, nil
	}

	metadata, err := toProtoMap(artifact.Metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to convert metadata to proto struct: %w", err)
	}

	parts, err := toProtoParts(artifact.Parts)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to proto parts: %w", err)
	}

	return &a2apb.Artifact{
		ArtifactId:  string(artifact.ID),
		Name:        artifact.Name,
		Description: artifact.Description,
		Parts:       parts,
		Metadata:    metadata,
		Extensions:  artifact.Extensions,
	}, nil
}

func toProtoArtifacts(artifacts []*a2a.Artifact) ([]*a2apb.Artifact, error) {
	result := make([]*a2apb.Artifact, len(artifacts))
	for i, artifact := range artifacts {
		pArtifact, err := toProtoArtifact(artifact)
		if err != nil {
			return nil, fmt.Errorf("failed to convert artifact: %w", err)
		}
		if pArtifact != nil {
			result[i] = pArtifact
		}
	}
	return result, nil
}

func ToProtoTask(task *a2a.Task) (*a2apb.Task, error) {
	if task == nil {
		return nil, nil
	}

	status, err := toProtoTaskStatus(task.Status)
	if err != nil {
		return nil, fmt.Errorf("failed to convert status: %w", err)
	}

	artifacts, err := toProtoArtifacts(task.Artifacts)
	if err != nil {
		return nil, fmt.Errorf("failed to convert artifacts: %w", err)
	}

	history, err := toProtoMessages(task.History)
	if err != nil {
		return nil, fmt.Errorf("failed to convert history: %w", err)
	}

	metadata, err := toProtoMap(task.Metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to convert metadata to proto struct: %w", err)
	}

	result := &a2apb.Task{
		Id:        string(task.ID),
		ContextId: task.ContextID,
		Status:    status,
		Artifacts: artifacts,
		History:   history,
		Metadata:  metadata,
	}
	return result, nil
}

func ToProtoTaskPushConfig(config *a2a.TaskPushConfig) (*a2apb.TaskPushNotificationConfig, error) {
	if config == nil {
		return nil, nil
	}

	if config.TaskID == "" {
		return nil, fmt.Errorf("taskID is required on TaskPushConfig")
	}

	pConfig, err := toProtoPushConfig(&config.Config)
	if err != nil {
		return nil, err
	}

	return &a2apb.TaskPushNotificationConfig{
		Name:                   MakeConfigName(config.TaskID, pConfig.GetId()),
		PushNotificationConfig: pConfig,
	}, nil
}

func ToProtoListTaskPushConfig(configs []*a2a.TaskPushConfig) (*a2apb.ListTaskPushNotificationConfigResponse, error) {
	pConfigs := make([]*a2apb.TaskPushNotificationConfig, len(configs))
	for i, config := range configs {
		pConfig, err := ToProtoTaskPushConfig(config)
		if err != nil {
			return nil, fmt.Errorf("failed to convert config: %w", err)
		}
		pConfigs[i] = pConfig
	}
	return &a2apb.ListTaskPushNotificationConfigResponse{
		Configs:       pConfigs,
		NextPageToken: "", // todo: add pagination
	}, nil
}

func ToProtoListTaskPushConfigRequest(req *a2a.ListTaskPushConfigParams) (*a2apb.ListTaskPushNotificationConfigRequest, error) {
	if req == nil {
		return nil, nil
	}
	return &a2apb.ListTaskPushNotificationConfigRequest{Parent: MakeTaskName(req.TaskID)}, nil
}

func toProtoAdditionalInterfaces(interfaces []a2a.AgentInterface) []*a2apb.AgentInterface {
	pInterfaces := make([]*a2apb.AgentInterface, len(interfaces))
	for i, iface := range interfaces {
		pInterfaces[i] = &a2apb.AgentInterface{
			Transport: string(iface.Transport),
			Url:       iface.URL,
		}
	}
	return pInterfaces
}

func toProtoAgentProvider(provider *a2a.AgentProvider) *a2apb.AgentProvider {
	if provider == nil {
		return nil
	}
	return &a2apb.AgentProvider{Organization: provider.Org, Url: provider.URL}
}

func toProtoAgentExtensions(extensions []a2a.AgentExtension) ([]*a2apb.AgentExtension, error) {
	pExtensions := make([]*a2apb.AgentExtension, len(extensions))
	for i, ext := range extensions {
		params, err := structpb.NewStruct(ext.Params)
		if err != nil {
			return nil, fmt.Errorf("failed to convert extension params: %w", err)
		}
		pExtensions[i] = &a2apb.AgentExtension{
			Uri:         ext.URI,
			Description: ext.Description,
			Required:    ext.Required,
			Params:      params,
		}
	}
	return pExtensions, nil
}

func toProtoCapabilities(capabilities a2a.AgentCapabilities) (*a2apb.AgentCapabilities, error) {
	extensions, err := toProtoAgentExtensions(capabilities.Extensions)
	if err != nil {
		return nil, fmt.Errorf("failed to convert extensions: %w", err)
	}

	result := &a2apb.AgentCapabilities{
		PushNotifications:      capabilities.PushNotifications,
		Streaming:              capabilities.Streaming,
		StateTransitionHistory: capabilities.StateTransitionHistory,
		Extensions:             extensions,
	}
	return result, nil
}

func toProtoAuthzOAuthCodeFlow(f *a2a.AuthorizationCodeOAuthFlow) *a2apb.OAuthFlows {
	return &a2apb.OAuthFlows{
		Flow: &a2apb.OAuthFlows_AuthorizationCode{
			AuthorizationCode: &a2apb.AuthorizationCodeOAuthFlow{
				AuthorizationUrl: f.AuthorizationURL,
				TokenUrl:         f.TokenURL,
				RefreshUrl:       f.RefreshURL,
				Scopes:           f.Scopes,
			},
		},
	}
}

func toProtoCredentialsOAuthFlow(f *a2a.ClientCredentialsOAuthFlow) *a2apb.OAuthFlows {
	return &a2apb.OAuthFlows{
		Flow: &a2apb.OAuthFlows_ClientCredentials{
			ClientCredentials: &a2apb.ClientCredentialsOAuthFlow{
				TokenUrl:   f.TokenURL,
				RefreshUrl: f.RefreshURL,
				Scopes:     f.Scopes,
			},
		},
	}
}

func toProtoImplicitOAuthFlow(f *a2a.ImplicitOAuthFlow) *a2apb.OAuthFlows {
	return &a2apb.OAuthFlows{
		Flow: &a2apb.OAuthFlows_Implicit{
			Implicit: &a2apb.ImplicitOAuthFlow{
				AuthorizationUrl: f.AuthorizationURL,
				RefreshUrl:       f.RefreshURL,
				Scopes:           f.Scopes,
			},
		},
	}
}

func toProtoPasswordOAuthFlows(f *a2a.PasswordOAuthFlow) *a2apb.OAuthFlows {
	return &a2apb.OAuthFlows{
		Flow: &a2apb.OAuthFlows_Password{
			Password: &a2apb.PasswordOAuthFlow{
				TokenUrl:   f.TokenURL,
				RefreshUrl: f.RefreshURL,
				Scopes:     f.Scopes,
			},
		},
	}
}

func toProtoOAuthFlows(flows a2a.OAuthFlows) (*a2apb.OAuthFlows, error) {
	var result []*a2apb.OAuthFlows

	if flows.AuthorizationCode != nil {
		result = append(result, toProtoAuthzOAuthCodeFlow(flows.AuthorizationCode))
	}
	if flows.ClientCredentials != nil {
		result = append(result, toProtoCredentialsOAuthFlow(flows.ClientCredentials))
	}
	if flows.Implicit != nil {
		result = append(result, toProtoImplicitOAuthFlow(flows.Implicit))
	}
	if flows.Password != nil {
		result = append(result, toProtoPasswordOAuthFlows(flows.Password))
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("no OAuthFlows found")
	}

	if len(result) > 1 {
		return nil, fmt.Errorf("only one OAuthFlow is allowed")
	}

	return result[0], nil
}

func toProtoSecurityScheme(scheme a2a.SecurityScheme) (*a2apb.SecurityScheme, error) {
	switch s := scheme.(type) {
	case a2a.APIKeySecurityScheme:
		return &a2apb.SecurityScheme{
			Scheme: &a2apb.SecurityScheme_ApiKeySecurityScheme{
				ApiKeySecurityScheme: &a2apb.APIKeySecurityScheme{
					Name:        s.Name,
					Location:    string(s.In),
					Description: s.Description,
				},
			},
		}, nil
	case a2a.HTTPAuthSecurityScheme:
		return &a2apb.SecurityScheme{
			Scheme: &a2apb.SecurityScheme_HttpAuthSecurityScheme{
				HttpAuthSecurityScheme: &a2apb.HTTPAuthSecurityScheme{
					Scheme:       string(s.Scheme),
					Description:  s.Description,
					BearerFormat: s.BearerFormat,
				},
			},
		}, nil
	case a2a.OpenIDConnectSecurityScheme:
		return &a2apb.SecurityScheme{
			Scheme: &a2apb.SecurityScheme_OpenIdConnectSecurityScheme{
				OpenIdConnectSecurityScheme: &a2apb.OpenIdConnectSecurityScheme{
					OpenIdConnectUrl: s.OpenIDConnectURL,
					Description:      s.Description,
				},
			},
		}, nil
	case a2a.MutualTLSSecurityScheme:
		return &a2apb.SecurityScheme{
			Scheme: &a2apb.SecurityScheme_MtlsSecurityScheme{
				MtlsSecurityScheme: &a2apb.MutualTlsSecurityScheme{
					Description: s.Description,
				},
			},
		}, nil
	case a2a.OAuth2SecurityScheme:
		flows, err := toProtoOAuthFlows(s.Flows)
		if err != nil {
			return nil, fmt.Errorf("failed to convert OAuthFlows: %w", err)
		}
		return &a2apb.SecurityScheme{
			Scheme: &a2apb.SecurityScheme_Oauth2SecurityScheme{
				Oauth2SecurityScheme: &a2apb.OAuth2SecurityScheme{
					Flows:             flows,
					Description:       s.Description,
					Oauth2MetadataUrl: s.Oauth2MetadataURL,
				},
			},
		}, nil
	default:
		return nil, fmt.Errorf("unsupported security scheme type: %T", s)
	}
}

func toProtoSecuritySchemes(schemes a2a.NamedSecuritySchemes) (map[string]*a2apb.SecurityScheme, error) {
	pSchemes := make(map[string]*a2apb.SecurityScheme, len(schemes))
	for name, scheme := range schemes {
		pScheme, err := toProtoSecurityScheme(scheme)
		if err != nil {
			return nil, fmt.Errorf("failed to convert security scheme: %w", err)
		}
		if pScheme != nil {
			pSchemes[string(name)] = pScheme
		}
	}
	return pSchemes, nil
}

func toProtoSecurity(security []a2a.SecurityRequirements) []*a2apb.Security {
	pSecurity := make([]*a2apb.Security, len(security))
	for i, sec := range security {
		pSchemes := make(map[string]*a2apb.StringList)
		for name, scopes := range sec {
			pSchemes[string(name)] = &a2apb.StringList{List: scopes}
		}
		pSecurity[i] = &a2apb.Security{Schemes: pSchemes}
	}
	return pSecurity
}

func toProtoSkills(skills []a2a.AgentSkill) []*a2apb.AgentSkill {
	pSkills := make([]*a2apb.AgentSkill, len(skills))
	for i, skill := range skills {
		pSkills[i] = &a2apb.AgentSkill{
			Id:          skill.ID,
			Name:        skill.Name,
			Description: skill.Description,
			Tags:        skill.Tags,
			Examples:    skill.Examples,
			InputModes:  skill.InputModes,
			OutputModes: skill.OutputModes,
			Security:    toProtoSecurity(skill.Security),
		}
	}
	return pSkills
}

func toProtoAgentCardSignatures(in []a2a.AgentCardSignature) ([]*a2apb.AgentCardSignature, error) {
	if in == nil {
		return nil, nil
	}
	out := make([]*a2apb.AgentCardSignature, len(in))
	for i, v := range in {
		header, err := toProtoMap(v.Header)
		if err != nil {
			return nil, err
		}
		out[i] = &a2apb.AgentCardSignature{
			Protected: v.Protected,
			Signature: v.Signature,
			Header:    header,
		}
	}
	return out, nil
}

func ToProtoAgentCard(card *a2a.AgentCard) (*a2apb.AgentCard, error) {
	if card == nil {
		return nil, nil
	}

	capabilities, err := toProtoCapabilities(card.Capabilities)
	if err != nil {
		return nil, fmt.Errorf("failed to convert agent capabilities: %w", err)
	}

	schemes, err := toProtoSecuritySchemes(card.SecuritySchemes)
	if err != nil {
		return nil, fmt.Errorf("failed to convert security schemes: %w", err)
	}

	signatures, err := toProtoAgentCardSignatures(card.Signatures)
	if err != nil {
		return nil, fmt.Errorf("failed to convert signatures: %w", err)
	}

	result := &a2apb.AgentCard{
		ProtocolVersion:                   card.ProtocolVersion,
		Name:                              card.Name,
		Description:                       card.Description,
		Url:                               card.URL,
		PreferredTransport:                string(card.PreferredTransport),
		Version:                           card.Version,
		DocumentationUrl:                  card.DocumentationURL,
		Capabilities:                      capabilities,
		DefaultInputModes:                 card.DefaultInputModes,
		DefaultOutputModes:                card.DefaultOutputModes,
		SupportsAuthenticatedExtendedCard: card.SupportsAuthenticatedExtendedCard,
		SecuritySchemes:                   schemes,
		Provider:                          toProtoAgentProvider(card.Provider),
		AdditionalInterfaces:              toProtoAdditionalInterfaces(card.AdditionalInterfaces),
		Security:                          toProtoSecurity(card.Security),
		Skills:                            toProtoSkills(card.Skills),
		IconUrl:                           card.IconURL,
		Signatures:                        signatures,
	}

	return result, nil
}
