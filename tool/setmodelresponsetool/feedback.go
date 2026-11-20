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

package setmodelresponsetool

import (
	"fmt"
	"strings"
)

// issue is one validation failure, in pydantic's terms.
type issue struct {
	loc  string // dotted location: "age", "0.id", "address.zip"; empty at the root of a record list
	msg  string // pydantic's message for the kind
	kind string // pydantic's error type: "missing", "int_parsing", ...
}

// validationError is the feedback the model gets for an invalid answer. Its
// text is adk-python's wrapper around pydantic's ValidationError text, without
// the input values and the documentation URL pydantic adds. It never contains
// the model's input.
type validationError struct {
	title  string // the record's title, "list[<item title>]", or "set_model_response"
	issues []issue
}

func (e *validationError) Error() string {
	var b strings.Builder
	b.WriteString("Validation Error found:\n")
	plural := ""
	if len(e.issues) != 1 {
		plural = "s"
	}
	fmt.Fprintf(&b, "%d validation error%s for %s\n", len(e.issues), plural, e.title)
	for _, is := range e.issues {
		if is.loc != "" {
			b.WriteString(is.loc + "\n")
		}
		fmt.Fprintf(&b, "  %s [type=%s]\n", is.msg, is.kind)
	}
	b.WriteString("Recall the set_model_response function correctly, fix the errors, and call it again with all required fields using the correct types.")
	return b.String()
}

// missingParameter reports an absent wrapper parameter, which Python does not
// detect.
func missingParameter(name string) *validationError {
	return &validationError{
		title:  "set_model_response",
		issues: []issue{{loc: name, msg: "Field required", kind: "missing"}},
	}
}
