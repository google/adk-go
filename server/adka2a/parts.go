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

package adka2a

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/a2aproject/a2a-go/a2a"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/converters"
)

var (
	a2aDataPartMetaTypeKey        = ToA2AMetaKey("type")
	a2aDataPartMetaLongRunningKey = ToA2AMetaKey("is_long_running")
)

const (
	a2aDataPartTypeFunctionCall       = "function_call"
	a2aDataPartTypeFunctionResponse   = "function_response"
	a2aDataPartTypeCodeExecResult     = "code_execution_result"
	a2aDataPartTypeCodeExecutableCode = "executable_code"
)

// ToA2AParts converts the provided genai parts to A2A equivalents. Long running tool IDs are used for attaching metadata to
// the relevant data parts.
func ToA2AParts(parts []*genai.Part, longRunningToolIDs []string) ([]a2a.Part, error) {
	result := make([]a2a.Part, len(parts))
	for i, part := range parts {
		if part.Text != "" {
			r := a2a.TextPart{Text: part.Text}
			if part.Thought {
				r.Metadata = map[string]any{ToA2AMetaKey("thought"): true}
			}
			result[i] = r
		} else if part.InlineData != nil || part.FileData != nil {
			r, err := toA2AFilePart(part)
			if err != nil {
				return nil, err
			}
			result[i] = r
		} else {
			r, err := toA2ADataPart(part, longRunningToolIDs)
			if err != nil {
				return nil, err
			}
			result[i] = r
		}
	}
	return result, nil
}

func toA2AFilePart(v *genai.Part) (a2a.FilePart, error) {
	if v == nil || (v.FileData == nil && v.InlineData == nil) {
		return a2a.FilePart{}, fmt.Errorf("not a file part: %v", v)
	}

	if v.FileData != nil {
		return a2a.FilePart{
			File: a2a.FileURI{
				FileMeta: a2a.FileMeta{
					Name:     v.FileData.DisplayName,
					MimeType: v.FileData.MIMEType,
				},
				URI: v.FileData.FileURI,
			},
		}, nil
	}

	part := a2a.FilePart{
		File: a2a.FileBytes{
			FileMeta: a2a.FileMeta{
				Name:     v.InlineData.DisplayName,
				MimeType: v.InlineData.MIMEType,
			},
			Bytes: base64.StdEncoding.EncodeToString(v.InlineData.Data),
		},
	}

	if v.VideoMetadata != nil {
		data, err := converters.ToMapStructure(v.VideoMetadata)
		if err != nil {
			return a2a.FilePart{}, err
		}
		part.Metadata = map[string]any{"video_metadata": data}
	}

	return part, nil
}

func toA2ADataPart(part *genai.Part, longRunningToolIDs []string) (a2a.DataPart, error) {
	if part.CodeExecutionResult != nil {
		data, err := converters.ToMapStructure(part.CodeExecutionResult)
		if err != nil {
			return a2a.DataPart{}, err
		}
		return a2a.DataPart{
			Data:     data,
			Metadata: map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeCodeExecResult},
		}, nil
	}

	if part.FunctionResponse != nil {
		data, err := converters.ToMapStructure(part.FunctionResponse)
		if err != nil {
			return a2a.DataPart{}, err
		}
		return a2a.DataPart{
			Data:     data,
			Metadata: map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeFunctionResponse},
		}, nil
	}

	if part.ExecutableCode != nil {
		data, err := converters.ToMapStructure(part.ExecutableCode)
		if err != nil {
			return a2a.DataPart{}, err
		}
		return a2a.DataPart{
			Data:     data,
			Metadata: map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeCodeExecutableCode},
		}, nil
	}

	if part.FunctionCall != nil {
		data, err := converters.ToMapStructure(part.FunctionCall)
		if err != nil {
			return a2a.DataPart{}, err
		}
		return a2a.DataPart{
			Data: data,
			Metadata: map[string]any{
				a2aDataPartMetaTypeKey:        a2aDataPartTypeFunctionCall,
				a2aDataPartMetaLongRunningKey: slices.Contains(longRunningToolIDs, part.FunctionCall.ID),
			},
		}, nil
	}

	return a2a.DataPart{Data: map[string]any{}}, nil
}

func toGenAIContent(msg *a2a.Message) (*genai.Content, error) {
	parts, err := ToGenAIParts(msg.Parts)
	if err != nil {
		return nil, err
	}
	return &genai.Content{Role: genai.RoleUser, Parts: parts}, nil
}

// ToGenAIParts converts the provided A2A parts to genai equivalents.
func ToGenAIParts(parts []a2a.Part) ([]*genai.Part, error) {
	result := make([]*genai.Part, len(parts))
	for i, part := range parts {
		switch v := part.(type) {
		case a2a.TextPart:
			r := genai.NewPartFromText(v.Text)
			if v.Metadata != nil {
				if thought, ok := v.Metadata[ToA2AMetaKey("thought")].(bool); ok {
					r.Thought = thought
				}
			}
			result[i] = r

		case a2a.DataPart:
			r, err := toGenAIDataPart(v)
			if err != nil {
				return nil, err
			}
			result[i] = r

		case a2a.FilePart:
			r, err := toGenAIFilePart(v)
			if err != nil {
				return nil, err
			}
			result[i] = r

		default:
			return nil, fmt.Errorf("unknown part type: %T", v)
		}
	}
	return result, nil
}

func toGenAIFilePart(part a2a.FilePart) (*genai.Part, error) {
	switch v := part.File.(type) {
	case a2a.FileBytes:
		bytes, err := base64.StdEncoding.DecodeString(v.Bytes)
		if err != nil {
			return nil, err
		}
		data := &genai.Blob{Data: bytes, MIMEType: v.MimeType, DisplayName: v.Name}
		return &genai.Part{InlineData: data}, nil

	case a2a.FileURI:
		data := &genai.FileData{FileURI: v.URI, MIMEType: v.MimeType, DisplayName: v.Name}
		return &genai.Part{FileData: data}, nil

	default:
		return nil, fmt.Errorf("unknown file content type: %T", v)
	}
}

func toGenAIDataPart(part a2a.DataPart) (*genai.Part, error) {
	if part.Metadata == nil {
		return toGenAITextPart(part)
	}
	adkMetaType, ok := part.Metadata[a2aDataPartMetaTypeKey]
	if !ok {
		return toGenAITextPart(part)
	}

	bytes, err := json.Marshal(part.Data)
	if err != nil {
		return nil, err
	}

	switch adkMetaType {
	case a2aDataPartTypeCodeExecResult:
		var val genai.CodeExecutionResult
		if err := json.Unmarshal(bytes, &val); err != nil {
			return nil, err
		}
		return &genai.Part{CodeExecutionResult: &val}, nil

	case a2aDataPartTypeFunctionCall:
		var val genai.FunctionCall
		if err := json.Unmarshal(bytes, &val); err != nil {
			return nil, err
		}
		return &genai.Part{FunctionCall: &val}, nil

	case a2aDataPartTypeCodeExecutableCode:
		var val genai.ExecutableCode
		if err := json.Unmarshal(bytes, &val); err != nil {
			return nil, err
		}
		return &genai.Part{ExecutableCode: &val}, nil

	case a2aDataPartTypeFunctionResponse:
		var val genai.FunctionResponse
		if err := json.Unmarshal(bytes, &val); err != nil {
			return nil, err
		}
		return &genai.Part{FunctionResponse: &val}, nil

	default:
		return &genai.Part{Text: string(bytes)}, nil
	}
}

func toGenAITextPart(part a2a.DataPart) (*genai.Part, error) {
	bytes, err := json.Marshal(part.Data)
	if err != nil {
		return nil, err
	}
	return &genai.Part{Text: string(bytes)}, nil
}
