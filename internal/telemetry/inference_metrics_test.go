// Copyright 2026 Google LLC
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

package telemetry

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func TestSplitByModality(t *testing.T) {
	details := func(counts ...any) []*genai.ModalityTokenCount {
		var out []*genai.ModalityTokenCount
		for i := 0; i < len(counts); i += 2 {
			out = append(out, &genai.ModalityTokenCount{Modality: genai.MediaModality(counts[i].(string)), TokenCount: int32(counts[i+1].(int))})
		}
		return out
	}
	for _, tc := range []struct {
		name    string
		total   int64
		details []*genai.ModalityTokenCount
		want    modalitySplit
	}{
		{"no details", 25, nil, modalitySplit{unknown: 25}},
		{"details sum to total", 20, details("TEXT", 12, "IMAGE", 8), modalitySplit{text: 12, image: 8}},
		{"remainder", 25, details("TEXT", 12, "AUDIO", 8), modalitySplit{text: 12, audio: 8, unknown: 5}},
		{"video and document", 30, details("VIDEO", 10, "DOCUMENT", 20), modalitySplit{video: 10, document: 20}},
		{"unspecified is unknown", 10, details("TEXT", 3, "TEXT", 2, "MODALITY_UNSPECIFIED", 5), modalitySplit{text: 5, unknown: 5}},
		{"unrecognized is unknown", 10, details("TEXT", 4, "HOLOGRAM", 6), modalitySplit{text: 4, unknown: 6}},
		{"lowercase modality", 10, details("text", 10), modalitySplit{text: 10}},
		{"zero modality", 8, details("TEXT", 8, "IMAGE", 0), modalitySplit{text: 8}},
		{"no modality skipped", 8, details("", 3, "TEXT", 5), modalitySplit{text: 5, unknown: 3}},
		{"details exceed total", 5, details("TEXT", 8), modalitySplit{text: 8}},
		{"zero total", 0, nil, modalitySplit{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, splitByModality(tc.total, tc.details), cmp.AllowUnexported(modalitySplit{})); diff != "" {
				t.Errorf("splitByModality() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// partFieldIsOutput says, for every genai.Part field, whether a part carrying
// only that field is chunk output.
var partFieldIsOutput = map[string]bool{
	"Text":                true,  // text, refusals and reasoning
	"InlineData":          true,  // audio and images
	"FunctionCall":        true,  // a tool call's name or arguments
	"ToolCall":            true,  // a server-side tool call
	"Thought":             false, // marks Text as reasoning
	"ThoughtSignature":    false, // a signature
	"FunctionResponse":    false, // the caller's tool result
	"ToolResponse":        false, // the caller's tool result
	"FileData":            false, // a reference, not new output
	"ExecutableCode":      false, // not among the output kinds a chunk counts
	"CodeExecutionResult": false, // not among the output kinds a chunk counts
	"AudioTranscription":  false, // describes audio carried in InlineData
	"VideoMetadata":       false, // qualifies media in another field
	"MediaResolution":     false, // qualifies media in another field
	"MediaProcessing":     false, // qualifies media in another field
	"SpeechMetadata":      false, // qualifies Text
	"PartMetadata":        false, // caller bookkeeping
}

func TestCarriesOutput_EveryPartField(t *testing.T) {
	partType := reflect.TypeFor[genai.Part]()
	for i := range partType.NumField() {
		field := partType.Field(i)
		t.Run(field.Name, func(t *testing.T) {
			want, ok := partFieldIsOutput[field.Name]
			if !ok {
				t.Fatalf("genai.Part gained field %s; decide whether it is chunk output, then add it to partFieldIsOutput and, if it is, to carriesOutput", field.Name)
			}
			part := &genai.Part{}
			reflect.ValueOf(part).Elem().Field(i).Set(filledValue(t, field.Type))
			resp := &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{part}}}
			if got := carriesOutput(resp); got != want {
				t.Errorf("carriesOutput() = %v for a part carrying only %s, want %v", got, field.Name, want)
			}
		})
	}
	for name := range partFieldIsOutput {
		if _, ok := partType.FieldByName(name); !ok {
			t.Errorf("partFieldIsOutput lists %s, which genai.Part no longer has", name)
		}
	}
}

// filledValue is a value of typ whose fields, down to the first level, are
// set, so that a pointer to a struct carries data rather than only existing.
func filledValue(t *testing.T, typ reflect.Type) reflect.Value {
	t.Helper()
	switch typ.Kind() {
	case reflect.Pointer:
		v := reflect.New(typ.Elem())
		if typ.Elem().Kind() == reflect.Struct {
			for i := range typ.Elem().NumField() {
				if f := v.Elem().Field(i); f.CanSet() {
					f.Set(scalarValue(t, f.Type()))
				}
			}
		}
		return v
	default:
		return scalarValue(t, typ)
	}
}

func scalarValue(t *testing.T, typ reflect.Type) reflect.Value {
	t.Helper()
	switch typ.Kind() {
	case reflect.String:
		return reflect.ValueOf("x").Convert(typ)
	case reflect.Bool:
		return reflect.ValueOf(true).Convert(typ)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return reflect.ValueOf(int64(1)).Convert(typ)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return reflect.ValueOf(uint64(1)).Convert(typ)
	case reflect.Float32, reflect.Float64:
		return reflect.ValueOf(1.0).Convert(typ)
	case reflect.Slice:
		v := reflect.MakeSlice(typ, 1, 1)
		if typ.Elem().Kind() == reflect.Pointer {
			v.Index(0).Set(reflect.New(typ.Elem().Elem()))
		} else if typ.Elem().Kind() != reflect.Struct && typ.Elem().Kind() != reflect.Interface {
			v.Index(0).Set(scalarValue(t, typ.Elem()))
		}
		return v
	case reflect.Map:
		v := reflect.MakeMap(typ)
		v.SetMapIndex(scalarValue(t, typ.Key()), reflect.Zero(typ.Elem()))
		return v
	case reflect.Pointer:
		return reflect.New(typ.Elem())
	case reflect.Interface, reflect.Struct:
		return reflect.Zero(typ)
	default:
		t.Fatalf("no test value for kind %s; extend scalarValue", typ.Kind())
		return reflect.Value{}
	}
}
