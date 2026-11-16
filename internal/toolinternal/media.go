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

package toolinternal

import "google.golang.org/genai"

// MediaPart converts v to a function response part when v is a media part: a
// non-nil *genai.Part or a genai.Part whose InlineData has non-nil Data and a
// MIME type, or whose FileData has a URI and a MIME type. Inline data is
// checked first. DisplayName is not copied, as adk-python's
// _as_function_response_part does not copy it (_caller.py:174-193). It
// returns nil for any other value, including a part without a MIME type.
func MediaPart(v any) *genai.FunctionResponsePart {
	var p *genai.Part
	switch t := v.(type) {
	case *genai.Part:
		p = t
	case genai.Part:
		p = &t
	}
	if p == nil {
		return nil
	}
	if b := p.InlineData; b != nil && b.Data != nil && b.MIMEType != "" {
		return genai.NewFunctionResponsePartFromBytes(b.Data, b.MIMEType)
	}
	if f := p.FileData; f != nil && f.FileURI != "" && f.MIMEType != "" {
		return genai.NewFunctionResponsePartFromURI(f.FileURI, f.MIMEType)
	}
	return nil
}
