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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/a2aproject/a2a-go/v2/a2a"
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

// IsPartial takes metadata of an A2A object (eg. a2a.Part, a2a.Artifact) and returns true if
// it was marked as partial based on the ADK partial flag set on the original ADK object.
func IsPartial(meta map[string]any) bool {
	if meta == nil {
		return false
	}
	isPartial, _ := meta[metadataPartialKey].(bool)
	return isPartial
}

// IsPartialFlagSet takes metadata of an A2A object (eg. a2a.Part, a2a.Artifact) and returns true if
// the ADK partial flag was set on it.
func IsPartialFlagSet(meta map[string]any) bool {
	if meta == nil {
		return false
	}
	_, isSet := meta[metadataPartialKey].(bool)
	return isSet
}

func validateDataPartJSON(d *genai.Part) ([]byte, bool) {
	if d.InlineData == nil || d.InlineData.MIMEType != "text/plain" {
		return nil, false
	}
	if noPrefix, ok := bytes.CutPrefix(d.InlineData.Data, []byte("<a2a_datapart_json>")); ok {
		if result, ok := bytes.CutSuffix(noPrefix, []byte("</a2a_datapart_json>")); ok {
			return result, true
		}
	}
	return nil, false
}

// ToA2APart converts the provided genai part to A2A equivalent. Long running tool IDs are used for attaching metadata to
// the relevant data parts.
func ToA2APart(part *genai.Part, longRunningToolIDs []string) (*a2a.Part, error) {
	parts, err := ToA2AParts([]*genai.Part{part}, longRunningToolIDs)
	if err != nil {
		return nil, err
	}
	return parts[0], nil
}

// ToA2AParts converts the provided genai parts to A2A equivalents. Long running tool IDs are used for attaching metadata to
// the relevant data parts.
func ToA2AParts(parts []*genai.Part, longRunningToolIDs []string) ([]*a2a.Part, error) {
	result := make([]*a2a.Part, len(parts))
	for i, part := range parts {
		if part.Text != "" {
			r := a2a.NewTextPart(part.Text)
			if part.Thought {
				r.SetMeta(ToA2AMetaKey("thought"), true)
			}
			result[i] = r
		} else if jsonBytes, ok := validateDataPartJSON(part); ok {
			var data map[string]any
			if err := json.Unmarshal(jsonBytes, &data); err != nil {
				return nil, err
			}
			result[i] = a2a.NewDataPart(data)
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

func updatePartsMetadata(parts []*a2a.Part, update map[string]any) {
	for _, part := range parts {
		if part.Metadata == nil {
			part.Metadata = make(map[string]any)
		}
		maps.Copy(part.Metadata, update)
	}
}

func toA2AFilePart(v *genai.Part) (*a2a.Part, error) {
	if v == nil || (v.FileData == nil && v.InlineData == nil) {
		return nil, fmt.Errorf("not a file part: %v", v)
	}

	if v.FileData != nil {
		r := a2a.NewFileURLPart(a2a.URL(v.FileData.FileURI), v.FileData.MIMEType)
		r.Filename = v.FileData.DisplayName
		return r, nil
	}

	r := a2a.NewRawPart(v.InlineData.Data)
	r.MediaType = v.InlineData.MIMEType
	r.Filename = v.InlineData.DisplayName

	if v.VideoMetadata != nil {
		data, err := converters.ToMapStructure(v.VideoMetadata)
		if err != nil {
			return nil, err
		}
		r.SetMeta("video_metadata", data)
	}

	return r, nil
}

func toA2ADataPart(part *genai.Part, longRunningToolIDs []string) (*a2a.Part, error) {
	if part.CodeExecutionResult != nil {
		data, err := converters.ToMapStructure(part.CodeExecutionResult)
		if err != nil {
			return nil, err
		}
		r := a2a.NewDataPart(data)
		r.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeCodeExecResult)
		return r, nil
	}

	if part.FunctionResponse != nil {
		data, err := converters.ToMapStructure(part.FunctionResponse)
		if err != nil {
			return nil, err
		}
		r := a2a.NewDataPart(data)
		r.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeFunctionResponse)
		return r, nil
	}

	if part.ExecutableCode != nil {
		data, err := converters.ToMapStructure(part.ExecutableCode)
		if err != nil {
			return nil, err
		}
		r := a2a.NewDataPart(data)
		r.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeCodeExecutableCode)
		return r, nil
	}

	if part.FunctionCall != nil {
		data, err := converters.ToMapStructure(part.FunctionCall)
		if err != nil {
			return nil, err
		}
		r := a2a.NewDataPart(data)
		r.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeFunctionCall)
		r.SetMeta(a2aDataPartMetaLongRunningKey, slices.Contains(longRunningToolIDs, part.FunctionCall.ID))
		return r, nil
	}

	mapStruct, err := converters.ToMapStructure(part)
	if err != nil {
		return nil, err
	}
	return a2a.NewDataPart(mapStruct), nil
}

func toGenAIContent(ctx context.Context, msg *a2a.Message, converter A2APartConverter) (*genai.Content, error) {
	if converter == nil {
		parts, err := ToGenAIParts(msg.Parts)
		if err != nil {
			return nil, err
		}
		return genai.NewContentFromParts(parts, toGenAIRole(msg.Role)), nil
	}

	parts := make([]*genai.Part, 0, len(msg.Parts))
	for _, part := range msg.Parts {
		cp, err := converter(ctx, a2a.Event(msg), part)
		if err != nil {
			return nil, err
		}
		if cp == nil {
			continue
		}
		parts = append(parts, cp)
	}
	return genai.NewContentFromParts(parts, toGenAIRole(msg.Role)), nil
}

// ToGenAIPart converts the provided A2A part to a genai equivalent.
func ToGenAIPart(part *a2a.Part) (*genai.Part, error) {
	parts, err := ToGenAIParts([]*a2a.Part{part})
	if err != nil {
		return nil, err
	}
	return parts[0], nil
}

// ToGenAIParts converts the provided A2A parts to genai equivalents.
func ToGenAIParts(parts []*a2a.Part) ([]*genai.Part, error) {
	result := make([]*genai.Part, len(parts))
	for i, part := range parts {
		if text := part.Text(); text != "" {
			r := genai.NewPartFromText(text)
			if thought, ok := part.Meta()[ToA2AMetaKey("thought")].(bool); ok {
				r.Thought = thought
			}
			result[i] = r
		} else if data := part.Data(); data != nil {
			r, err := toGenAIDataPart(part)
			if err != nil {
				return nil, err
			}
			result[i] = r
		} else if raw := part.Raw(); raw != nil {
			r, err := toGenAIFilePart(part)
			if err != nil {
				return nil, err
			}
			result[i] = r
		} else if url := part.URL(); url != "" {
			r, err := toGenAIFilePart(part)
			if err != nil {
				return nil, err
			}
			result[i] = r
		} else {
			return nil, fmt.Errorf("unknown part type: %v", part)
		}
	}
	return result, nil
}

func toGenAIFilePart(part *a2a.Part) (*genai.Part, error) {
	if raw := part.Raw(); raw != nil {
		data := &genai.Blob{Data: raw, MIMEType: part.MediaType, DisplayName: part.Filename}
		return &genai.Part{InlineData: data}, nil
	}

	if url := part.URL(); url != "" {
		data := &genai.FileData{FileURI: string(url), MIMEType: part.MediaType, DisplayName: part.Filename}
		return &genai.Part{FileData: data}, nil
	}

	return nil, fmt.Errorf("no file content in part")
}

func toGenAIDataPart(part *a2a.Part) (*genai.Part, error) {
	data := part.Data()
	bytes, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	adkMetaType := part.Metadata[a2aDataPartMetaTypeKey]
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
		var jsonData []byte
		prefix, suffix := []byte("<a2a_datapart_json>"), []byte("</a2a_datapart_json>")
		jsonData = make([]byte, 0, len(prefix)+len(bytes)+len(suffix))
		jsonData = append(jsonData, prefix...)
		jsonData = append(jsonData, bytes...)
		jsonData = append(jsonData, suffix...)

		return &genai.Part{InlineData: &genai.Blob{Data: jsonData, MIMEType: "text/plain"}}, nil
	}
}
