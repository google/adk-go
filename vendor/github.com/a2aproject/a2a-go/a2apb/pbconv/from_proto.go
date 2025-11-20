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
	"google.golang.org/protobuf/proto"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

func fromProtoMap(meta *structpb.Struct) map[string]any {
	if meta == nil {
		return nil
	}
	return meta.AsMap()
}

func FromProtoSendMessageRequest(req *a2apb.SendMessageRequest) (*a2a.MessageSendParams, error) {
	if req == nil {
		return nil, nil
	}

	msg, err := FromProtoMessage(req.GetRequest())
	if err != nil {
		return nil, err
	}

	config, err := fromProtoSendMessageConfig(req.GetConfiguration())
	if err != nil {
		return nil, err
	}

	params := &a2a.MessageSendParams{
		Message:  msg,
		Config:   config,
		Metadata: fromProtoMap(req.GetMetadata()),
	}
	return params, nil
}

func FromProtoMessage(pMsg *a2apb.Message) (*a2a.Message, error) {
	if pMsg == nil {
		return nil, nil
	}

	parts, err := fromProtoParts(pMsg.GetParts())
	if err != nil {
		return nil, err
	}

	msg := &a2a.Message{
		ID:         pMsg.GetMessageId(),
		ContextID:  pMsg.GetContextId(),
		Extensions: pMsg.GetExtensions(),
		Parts:      parts,
		TaskID:     a2a.TaskID(pMsg.GetTaskId()),
		Role:       fromProtoRole(pMsg.GetRole()),
		Metadata:   fromProtoMap(pMsg.GetMetadata()),
	}

	taskIDs := pMsg.GetReferenceTaskIds()
	if taskIDs != nil {
		msg.ReferenceTasks = make([]a2a.TaskID, len(taskIDs))
		for i, tid := range taskIDs {
			msg.ReferenceTasks[i] = a2a.TaskID(tid)
		}
	}

	return msg, nil
}

func fromProtoFilePart(pPart *a2apb.FilePart, meta map[string]any) (a2a.FilePart, error) {
	switch f := pPart.GetFile().(type) {
	case *a2apb.FilePart_FileWithBytes:
		return a2a.FilePart{
			File: a2a.FileBytes{
				FileMeta: a2a.FileMeta{MimeType: pPart.GetMimeType(), Name: pPart.GetName()},
				Bytes:    string(f.FileWithBytes),
			},
			Metadata: meta,
		}, nil
	case *a2apb.FilePart_FileWithUri:
		return a2a.FilePart{
			File: a2a.FileURI{
				FileMeta: a2a.FileMeta{MimeType: pPart.GetMimeType(), Name: pPart.GetName()},
				URI:      f.FileWithUri,
			},
			Metadata: meta,
		}, nil
	default:
		return a2a.FilePart{}, fmt.Errorf("unsupported FilePart type: %T", f)
	}
}

func fromProtoPart(p *a2apb.Part) (a2a.Part, error) {
	meta := fromProtoMap(p.Metadata)
	switch part := p.GetPart().(type) {
	case *a2apb.Part_Text:
		return a2a.TextPart{Text: part.Text, Metadata: meta}, nil
	case *a2apb.Part_Data:
		return a2a.DataPart{Data: part.Data.GetData().AsMap(), Metadata: meta}, nil
	case *a2apb.Part_File:
		return fromProtoFilePart(part.File, meta)
	default:
		return nil, fmt.Errorf("unsupported part type: %T", part)
	}
}

func fromProtoRole(role a2apb.Role) a2a.MessageRole {
	switch role {
	case a2apb.Role_ROLE_USER:
		return a2a.MessageRoleUser
	case a2apb.Role_ROLE_AGENT:
		return a2a.MessageRoleAgent
	default:
		return a2a.MessageRoleUnspecified
	}
}

func fromProtoPushConfig(pConf *a2apb.PushNotificationConfig) (*a2a.PushConfig, error) {
	if pConf == nil {
		return nil, nil
	}

	result := &a2a.PushConfig{
		ID:    pConf.GetId(),
		URL:   pConf.GetUrl(),
		Token: pConf.GetToken(),
	}
	if pConf.GetAuthentication() != nil {
		result.Auth = &a2a.PushAuthInfo{
			Schemes:     pConf.GetAuthentication().GetSchemes(),
			Credentials: pConf.GetAuthentication().GetCredentials(),
		}
	}
	return result, nil
}

func fromProtoSendMessageConfig(conf *a2apb.SendMessageConfiguration) (*a2a.MessageSendConfig, error) {
	if conf == nil {
		return nil, nil
	}

	pConf, err := fromProtoPushConfig(conf.GetPushNotification())
	if err != nil {
		return nil, fmt.Errorf("failed to convert push config: %w", err)
	}

	result := &a2a.MessageSendConfig{
		AcceptedOutputModes: conf.GetAcceptedOutputModes(),
		Blocking:            proto.Bool(conf.GetBlocking()),
		PushConfig:          pConf,
	}

	// TODO: consider the approach after resolving https://github.com/a2aproject/A2A/issues/1072
	if conf.HistoryLength > 0 {
		hl := int(conf.HistoryLength)
		result.HistoryLength = &hl
	}
	return result, nil
}

func FromProtoGetTaskRequest(req *a2apb.GetTaskRequest) (*a2a.TaskQueryParams, error) {
	if req == nil {
		return nil, nil
	}

	// TODO: consider throwing an error when the path - req.GetName() is unexpected, e.g. tasks/taskID/someExtraText
	taskID, err := ExtractTaskID(req.GetName())
	if err != nil {
		return nil, fmt.Errorf("failed to extract task id: %w", err)
	}

	params := &a2a.TaskQueryParams{ID: taskID}
	if req.GetHistoryLength() > 0 {
		historyLength := int(req.GetHistoryLength())
		params.HistoryLength = &historyLength
	}
	return params, nil
}

func FromProtoCreateTaskPushConfigRequest(req *a2apb.CreateTaskPushNotificationConfigRequest) (*a2a.TaskPushConfig, error) {
	if req == nil {
		return nil, nil
	}

	config := req.GetConfig()
	if config.GetPushNotificationConfig() == nil {
		return nil, fmt.Errorf("invalid config")
	}

	pConf, err := fromProtoPushConfig(config.GetPushNotificationConfig())
	if err != nil {
		return nil, fmt.Errorf("failed to convert push config: %w", err)
	}

	taskID, err := ExtractTaskID(req.GetParent())
	if err != nil {
		return nil, fmt.Errorf("failed to extract task id: %w", err)
	}

	return &a2a.TaskPushConfig{TaskID: taskID, Config: *pConf}, nil
}

func FromProtoGetTaskPushConfigRequest(req *a2apb.GetTaskPushNotificationConfigRequest) (*a2a.GetTaskPushConfigParams, error) {
	if req == nil {
		return nil, nil
	}

	taskID, err := ExtractTaskID(req.GetName())
	if err != nil {
		return nil, fmt.Errorf("failed to extract task id: %w", err)
	}

	configID, err := ExtractConfigID(req.GetName())
	if err != nil {
		return nil, fmt.Errorf("failed to extract config id: %w", err)
	}

	return &a2a.GetTaskPushConfigParams{TaskID: taskID, ConfigID: configID}, nil
}

func FromProtoDeleteTaskPushConfigRequest(req *a2apb.DeleteTaskPushNotificationConfigRequest) (*a2a.DeleteTaskPushConfigParams, error) {
	if req == nil {
		return nil, nil
	}

	taskID, err := ExtractTaskID(req.GetName())
	if err != nil {
		return nil, fmt.Errorf("failed to extract task id: %w", err)
	}

	configID, err := ExtractConfigID(req.GetName())
	if err != nil {
		return nil, fmt.Errorf("failed to extract config id: %w", err)
	}

	return &a2a.DeleteTaskPushConfigParams{TaskID: taskID, ConfigID: configID}, nil
}

func FromProtoSendMessageResponse(resp *a2apb.SendMessageResponse) (a2a.SendMessageResult, error) {
	if resp == nil {
		return nil, nil
	}

	switch p := resp.Payload.(type) {
	case *a2apb.SendMessageResponse_Msg:
		return FromProtoMessage(p.Msg)
	case *a2apb.SendMessageResponse_Task:
		return FromProtoTask(p.Task)
	default:
		return nil, fmt.Errorf("unsupported SendMessageResponse payload type: %T", resp.Payload)
	}
}

func FromProtoStreamResponse(resp *a2apb.StreamResponse) (a2a.Event, error) {
	if resp == nil {
		return nil, nil
	}

	switch p := resp.Payload.(type) {
	case *a2apb.StreamResponse_Msg:
		return FromProtoMessage(p.Msg)
	case *a2apb.StreamResponse_Task:
		return FromProtoTask(p.Task)
	case *a2apb.StreamResponse_StatusUpdate:
		status, err := fromProtoTaskStatus(p.StatusUpdate.GetStatus())
		if err != nil {
			return nil, err
		}
		return &a2a.TaskStatusUpdateEvent{
			ContextID: p.StatusUpdate.GetContextId(),
			Final:     p.StatusUpdate.GetFinal(),
			Status:    status,
			TaskID:    a2a.TaskID(p.StatusUpdate.GetTaskId()),
			Metadata:  fromProtoMap(p.StatusUpdate.GetMetadata()),
		}, nil
	case *a2apb.StreamResponse_ArtifactUpdate:
		artifact, err := fromProtoArtifact(p.ArtifactUpdate.GetArtifact())
		if err != nil {
			return nil, err
		}
		return &a2a.TaskArtifactUpdateEvent{
			Append:    p.ArtifactUpdate.GetAppend(),
			Artifact:  artifact,
			ContextID: p.ArtifactUpdate.GetContextId(),
			LastChunk: p.ArtifactUpdate.GetLastChunk(),
			TaskID:    a2a.TaskID(p.ArtifactUpdate.GetTaskId()),
			Metadata:  fromProtoMap(p.ArtifactUpdate.GetMetadata()),
		}, nil
	default:
		return nil, fmt.Errorf("unsupported StreamResponse payload type: %T", resp.Payload)
	}
}

func fromProtoMessages(pMsgs []*a2apb.Message) ([]*a2a.Message, error) {
	msgs := make([]*a2a.Message, len(pMsgs))
	for i, pMsg := range pMsgs {
		msg, err := FromProtoMessage(pMsg)
		if err != nil {
			return nil, fmt.Errorf("failed to convert message: %w", err)
		}
		msgs[i] = msg
	}
	return msgs, nil
}

func fromProtoParts(pParts []*a2apb.Part) ([]a2a.Part, error) {
	parts := make([]a2a.Part, len(pParts))
	for i, pPart := range pParts {
		part, err := fromProtoPart(pPart)
		if err != nil {
			return nil, fmt.Errorf("failed to convert part: %w", err)
		}
		parts[i] = part
	}
	return parts, nil
}

func fromProtoTaskState(state a2apb.TaskState) a2a.TaskState {
	switch state {
	case a2apb.TaskState_TASK_STATE_AUTH_REQUIRED:
		return a2a.TaskStateAuthRequired
	case a2apb.TaskState_TASK_STATE_CANCELLED:
		return a2a.TaskStateCanceled
	case a2apb.TaskState_TASK_STATE_COMPLETED:
		return a2a.TaskStateCompleted
	case a2apb.TaskState_TASK_STATE_FAILED:
		return a2a.TaskStateFailed
	case a2apb.TaskState_TASK_STATE_INPUT_REQUIRED:
		return a2a.TaskStateInputRequired
	case a2apb.TaskState_TASK_STATE_REJECTED:
		return a2a.TaskStateRejected
	case a2apb.TaskState_TASK_STATE_SUBMITTED:
		return a2a.TaskStateSubmitted
	case a2apb.TaskState_TASK_STATE_WORKING:
		return a2a.TaskStateWorking
	default:
		return a2a.TaskStateUnspecified
	}
}

func fromProtoTaskStatus(pStatus *a2apb.TaskStatus) (a2a.TaskStatus, error) {
	if pStatus == nil {
		return a2a.TaskStatus{}, fmt.Errorf("invalid status")
	}

	message, err := FromProtoMessage(pStatus.GetUpdate())
	if err != nil {
		return a2a.TaskStatus{}, fmt.Errorf("failed to convert message for task status: %w", err)
	}

	status := a2a.TaskStatus{
		State:   fromProtoTaskState(pStatus.GetState()),
		Message: message,
	}

	if pStatus.Timestamp != nil {
		t := pStatus.Timestamp.AsTime()
		status.Timestamp = &t
	}

	return status, nil
}

func fromProtoArtifact(pArtifact *a2apb.Artifact) (*a2a.Artifact, error) {
	if pArtifact == nil {
		return nil, nil
	}

	parts, err := fromProtoParts(pArtifact.GetParts())
	if err != nil {
		return nil, fmt.Errorf("failed to convert from proto parts: %w", err)
	}

	return &a2a.Artifact{
		ID:          a2a.ArtifactID(pArtifact.GetArtifactId()),
		Name:        pArtifact.GetName(),
		Description: pArtifact.GetDescription(),
		Parts:       parts,
		Extensions:  pArtifact.GetExtensions(),
		Metadata:    fromProtoMap(pArtifact.GetMetadata()),
	}, nil
}

func fromProtoArtifacts(pArtifacts []*a2apb.Artifact) ([]*a2a.Artifact, error) {
	result := make([]*a2a.Artifact, len(pArtifacts))
	for i, pArtifact := range pArtifacts {
		artifact, err := fromProtoArtifact(pArtifact)
		if err != nil {
			return nil, fmt.Errorf("failed to convert artifact: %w", err)
		}
		result[i] = artifact
	}
	return result, nil
}

func FromProtoTask(pTask *a2apb.Task) (*a2a.Task, error) {
	if pTask == nil {
		return nil, nil
	}

	status, err := fromProtoTaskStatus(pTask.Status)
	if err != nil {
		return nil, fmt.Errorf("failed to convert status: %w", err)
	}

	artifacts, err := fromProtoArtifacts(pTask.Artifacts)
	if err != nil {
		return nil, fmt.Errorf("failed to convert artifacts: %w", err)
	}

	history, err := fromProtoMessages(pTask.History)
	if err != nil {
		return nil, fmt.Errorf("failed to convert history: %w", err)
	}

	result := &a2a.Task{
		ID:        a2a.TaskID(pTask.GetId()),
		ContextID: pTask.GetContextId(),
		Status:    status,
		Artifacts: artifacts,
		History:   history,
		Metadata:  fromProtoMap(pTask.GetMetadata()),
	}

	return result, nil
}

func FromProtoTaskPushConfig(pTaskConfig *a2apb.TaskPushNotificationConfig) (*a2a.TaskPushConfig, error) {
	if pTaskConfig == nil {
		return nil, nil
	}

	taskID, err := ExtractTaskID(pTaskConfig.GetName())
	if err != nil {
		return nil, fmt.Errorf("failed to extract task id: %w", err)
	}

	configID, err := ExtractConfigID(pTaskConfig.GetName())
	if err != nil {
		return nil, fmt.Errorf("failed to extract config id: %w", err)
	}

	pConf := pTaskConfig.GetPushNotificationConfig()
	if pConf == nil {
		return nil, fmt.Errorf("push notification config is nil")
	}

	if pConf.GetId() != configID {
		return nil, fmt.Errorf("config id mismatch: %q != %q", pConf.GetId(), configID)
	}

	config, err := fromProtoPushConfig(pConf)
	if err != nil {
		return nil, fmt.Errorf("failed to convert push config: %w", err)
	}

	return &a2a.TaskPushConfig{TaskID: taskID, Config: *config}, nil
}

func FromProtoListTaskPushConfig(resp *a2apb.ListTaskPushNotificationConfigResponse) ([]*a2a.TaskPushConfig, error) {
	if resp == nil {
		return nil, fmt.Errorf("response is nil")
	}

	configs := make([]*a2a.TaskPushConfig, len(resp.GetConfigs()))
	for i, pConfig := range resp.GetConfigs() {
		config, err := FromProtoTaskPushConfig(pConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to convert config: %w", err)
		}
		configs[i] = config
	}
	return configs, nil
}

func fromProtoAdditionalInterfaces(pInterfaces []*a2apb.AgentInterface) []a2a.AgentInterface {
	interfaces := make([]a2a.AgentInterface, len(pInterfaces))
	for i, pIface := range pInterfaces {
		interfaces[i] = a2a.AgentInterface{
			Transport: a2a.TransportProtocol(pIface.GetTransport()),
			URL:       pIface.GetUrl(),
		}
	}
	return interfaces
}

func fromProtoAgentProvider(pProvider *a2apb.AgentProvider) *a2a.AgentProvider {
	if pProvider == nil {
		return nil
	}
	return &a2a.AgentProvider{Org: pProvider.GetOrganization(), URL: pProvider.GetUrl()}
}

func fromProtoAgentExtensions(pExtensions []*a2apb.AgentExtension) ([]a2a.AgentExtension, error) {
	extensions := make([]a2a.AgentExtension, len(pExtensions))
	for i, pExt := range pExtensions {
		extensions[i] = a2a.AgentExtension{
			URI:         pExt.GetUri(),
			Description: pExt.GetDescription(),
			Required:    pExt.GetRequired(),
			Params:      pExt.GetParams().AsMap(),
		}
	}
	return extensions, nil
}

func fromProtoCapabilities(pCapabilities *a2apb.AgentCapabilities) (a2a.AgentCapabilities, error) {
	extensions, err := fromProtoAgentExtensions(pCapabilities.GetExtensions())
	if err != nil {
		return a2a.AgentCapabilities{}, fmt.Errorf("failed to convert extensions: %w", err)
	}

	result := a2a.AgentCapabilities{
		PushNotifications:      pCapabilities.GetPushNotifications(),
		Streaming:              pCapabilities.GetStreaming(),
		StateTransitionHistory: pCapabilities.GetStateTransitionHistory(),
		Extensions:             extensions,
	}
	return result, nil
}

func fromProtoOAuthFlows(pFlows *a2apb.OAuthFlows) (a2a.OAuthFlows, error) {
	flows := a2a.OAuthFlows{}
	if pFlows == nil {
		return flows, fmt.Errorf("oauth flows is nil")
	}
	switch f := pFlows.Flow.(type) {
	case *a2apb.OAuthFlows_AuthorizationCode:
		flows.AuthorizationCode = &a2a.AuthorizationCodeOAuthFlow{
			AuthorizationURL: f.AuthorizationCode.GetAuthorizationUrl(),
			TokenURL:         f.AuthorizationCode.GetTokenUrl(),
			RefreshURL:       f.AuthorizationCode.GetRefreshUrl(),
			Scopes:           f.AuthorizationCode.GetScopes(),
		}
	case *a2apb.OAuthFlows_ClientCredentials:
		flows.ClientCredentials = &a2a.ClientCredentialsOAuthFlow{
			TokenURL:   f.ClientCredentials.GetTokenUrl(),
			RefreshURL: f.ClientCredentials.GetRefreshUrl(),
			Scopes:     f.ClientCredentials.GetScopes(),
		}
	case *a2apb.OAuthFlows_Implicit:
		flows.Implicit = &a2a.ImplicitOAuthFlow{
			AuthorizationURL: f.Implicit.GetAuthorizationUrl(),
			RefreshURL:       f.Implicit.GetRefreshUrl(),
			Scopes:           f.Implicit.GetScopes(),
		}
	case *a2apb.OAuthFlows_Password:
		flows.Password = &a2a.PasswordOAuthFlow{
			TokenURL:   f.Password.GetTokenUrl(),
			RefreshURL: f.Password.GetRefreshUrl(),
			Scopes:     f.Password.GetScopes(),
		}
	default:
		return flows, fmt.Errorf("unsupported oauth flow type: %T", f)
	}
	return flows, nil
}

func fromProtoSecurityScheme(pScheme *a2apb.SecurityScheme) (a2a.SecurityScheme, error) {
	if pScheme == nil {
		return nil, fmt.Errorf("security scheme is nil")
	}

	switch s := pScheme.Scheme.(type) {
	case *a2apb.SecurityScheme_ApiKeySecurityScheme:
		return a2a.APIKeySecurityScheme{
			Name:        s.ApiKeySecurityScheme.GetName(),
			In:          a2a.APIKeySecuritySchemeIn(s.ApiKeySecurityScheme.GetLocation()),
			Description: s.ApiKeySecurityScheme.GetDescription(),
		}, nil
	case *a2apb.SecurityScheme_HttpAuthSecurityScheme:
		return a2a.HTTPAuthSecurityScheme{
			Scheme:       s.HttpAuthSecurityScheme.GetScheme(),
			Description:  s.HttpAuthSecurityScheme.GetDescription(),
			BearerFormat: s.HttpAuthSecurityScheme.GetBearerFormat(),
		}, nil
	case *a2apb.SecurityScheme_OpenIdConnectSecurityScheme:
		return a2a.OpenIDConnectSecurityScheme{
			OpenIDConnectURL: s.OpenIdConnectSecurityScheme.GetOpenIdConnectUrl(),
			Description:      s.OpenIdConnectSecurityScheme.GetDescription(),
		}, nil
	case *a2apb.SecurityScheme_Oauth2SecurityScheme:
		flows, err := fromProtoOAuthFlows(s.Oauth2SecurityScheme.GetFlows())
		if err != nil {
			return nil, fmt.Errorf("failed to convert OAuthFlows: %w", err)
		}
		return a2a.OAuth2SecurityScheme{
			Flows:             flows,
			Oauth2MetadataURL: s.Oauth2SecurityScheme.Oauth2MetadataUrl,
			Description:       s.Oauth2SecurityScheme.GetDescription(),
		}, nil
	case *a2apb.SecurityScheme_MtlsSecurityScheme:
		return a2a.MutualTLSSecurityScheme{
			Description: s.MtlsSecurityScheme.GetDescription(),
		}, nil
	default:
		return nil, fmt.Errorf("unsupported security scheme type: %T", s)
	}
}

func fromProtoSecuritySchemes(pSchemes map[string]*a2apb.SecurityScheme) (a2a.NamedSecuritySchemes, error) {
	schemes := make(a2a.NamedSecuritySchemes, len(pSchemes))
	for name, pScheme := range pSchemes {
		scheme, err := fromProtoSecurityScheme(pScheme)
		if err != nil {
			return nil, fmt.Errorf("failed to convert security scheme: %w", err)
		}
		schemes[a2a.SecuritySchemeName(name)] = scheme
	}
	return schemes, nil
}

func fromProtoSecurity(pSecurity []*a2apb.Security) []a2a.SecurityRequirements {
	security := make([]a2a.SecurityRequirements, len(pSecurity))
	for i, pSec := range pSecurity {
		schemes := make(a2a.SecurityRequirements)
		for name, scopes := range pSec.Schemes {
			schemes[a2a.SecuritySchemeName(name)] = scopes.GetList()
		}
		security[i] = schemes
	}
	return security
}

func fromProtoSkills(pSkills []*a2apb.AgentSkill) []a2a.AgentSkill {
	skills := make([]a2a.AgentSkill, len(pSkills))
	for i, pSkill := range pSkills {
		skills[i] = a2a.AgentSkill{
			ID:          pSkill.GetId(),
			Name:        pSkill.GetName(),
			Description: pSkill.GetDescription(),
			Tags:        pSkill.GetTags(),
			Examples:    pSkill.GetExamples(),
			InputModes:  pSkill.GetInputModes(),
			OutputModes: pSkill.GetOutputModes(),
			Security:    fromProtoSecurity(pSkill.GetSecurity()),
		}
	}
	return skills
}

func fromProtoAgentCardSignatures(in []*a2apb.AgentCardSignature) []a2a.AgentCardSignature {
	if in == nil {
		return nil
	}
	out := make([]a2a.AgentCardSignature, len(in))
	for i, v := range in {
		out[i] = a2a.AgentCardSignature{
			Protected: v.GetProtected(),
			Signature: v.GetSignature(),
			Header:    fromProtoMap(v.GetHeader()),
		}
	}
	return out
}

func FromProtoAgentCard(pCard *a2apb.AgentCard) (*a2a.AgentCard, error) {
	if pCard == nil {
		return nil, nil
	}

	capabilities, err := fromProtoCapabilities(pCard.GetCapabilities())
	if err != nil {
		return nil, fmt.Errorf("failed to convert agent capabilities: %w", err)
	}

	schemes, err := fromProtoSecuritySchemes(pCard.GetSecuritySchemes())
	if err != nil {
		return nil, fmt.Errorf("failed to convert security schemes: %w", err)
	}

	result := &a2a.AgentCard{
		ProtocolVersion:                   pCard.GetProtocolVersion(),
		Name:                              pCard.GetName(),
		Description:                       pCard.GetDescription(),
		URL:                               pCard.GetUrl(),
		PreferredTransport:                a2a.TransportProtocol(pCard.GetPreferredTransport()),
		Version:                           pCard.GetVersion(),
		DocumentationURL:                  pCard.GetDocumentationUrl(),
		Capabilities:                      capabilities,
		DefaultInputModes:                 pCard.GetDefaultInputModes(),
		DefaultOutputModes:                pCard.GetDefaultOutputModes(),
		SupportsAuthenticatedExtendedCard: pCard.GetSupportsAuthenticatedExtendedCard(),
		SecuritySchemes:                   schemes,
		Provider:                          fromProtoAgentProvider(pCard.GetProvider()),
		AdditionalInterfaces:              fromProtoAdditionalInterfaces(pCard.GetAdditionalInterfaces()),
		Security:                          fromProtoSecurity(pCard.GetSecurity()),
		Skills:                            fromProtoSkills(pCard.GetSkills()),
		IconURL:                           pCard.GetIconUrl(),
		Signatures:                        fromProtoAgentCardSignatures(pCard.GetSignatures()),
	}

	return result, nil
}
