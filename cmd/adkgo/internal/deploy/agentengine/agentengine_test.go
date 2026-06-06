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

package agentengine

import (
	"testing"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"
)

func TestParseEnvVars(t *testing.T) {
	tests := []struct {
		name    string
		raw     []string
		want    []*aiplatformpb.EnvVar
		wantErr bool
	}{
		{
			name: "nil input returns empty",
			raw:  nil,
			want: []*aiplatformpb.EnvVar{},
		},
		{
			name: "single key=value",
			raw:  []string{"FOO=bar"},
			want: []*aiplatformpb.EnvVar{{Name: "FOO", Value: "bar"}},
		},
		{
			name: "value contains equals sign",
			raw:  []string{"URL=http://example.com?a=1&b=2"},
			want: []*aiplatformpb.EnvVar{{Name: "URL", Value: "http://example.com?a=1&b=2"}},
		},
		{
			name: "empty value",
			raw:  []string{"EMPTY="},
			want: []*aiplatformpb.EnvVar{{Name: "EMPTY", Value: ""}},
		},
		{
			name: "multiple vars",
			raw:  []string{"A=1", "B=2", "C=three"},
			want: []*aiplatformpb.EnvVar{
				{Name: "A", Value: "1"},
				{Name: "B", Value: "2"},
				{Name: "C", Value: "three"},
			},
		},
		{
			name:    "missing equals returns error",
			raw:     []string{"NOEQUALS"},
			wantErr: true,
		},
		{
			name:    "empty string returns error",
			raw:     []string{""},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseEnvVars(tc.raw)
			if (err != nil) != tc.wantErr {
				t.Fatalf("parseEnvVars(%v) error = %v, wantErr %v", tc.raw, err, tc.wantErr)
			}
			if tc.wantErr {
				return
			}
			if diff := cmp.Diff(tc.want, got, protocmp.Transform()); diff != "" {
				t.Errorf("parseEnvVars(%v) mismatch (-want +got):\n%s", tc.raw, diff)
			}
		})
	}
}
