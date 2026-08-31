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

package conformance

import (
	"fmt"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/configurable"
	"google.golang.org/adk/v2/internal/typeutil"
)

func uppercaseFormatter(ctx agent.Context, input any) (any, error) {
	switch v := input.(type) {
	case string:
		return strings.ToUpper(v), nil
	case map[string]any:
		res := make(map[string]any, len(v))
		for k, val := range v {
			if s, ok := val.(string); ok {
				res[k] = strings.ToUpper(s)
			} else {
				res[k] = val
			}
		}
		return res, nil
	default:
		m, err := typeutil.ConvertToWithJSONSchema[any, map[string]any](input, nil)
		if err == nil {
			res := make(map[string]any, len(m))
			for k, val := range m {
				if s, ok := val.(string); ok {
					res[k] = strings.ToUpper(s)
				} else {
					res[k] = val
				}
			}
			return res, nil
		}
		return nil, fmt.Errorf("node_input must be a string or a map/dict")
	}
}

func RegisterNodeFunctions() error {
	configurable.RegisterNodeFunction("conformance.uppercase_formatter", uppercaseFormatter)
	return nil
}
