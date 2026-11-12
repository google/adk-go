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

package feature_test

import (
	"context"
	"testing"

	"google.golang.org/adk/v2/exp/feature"
)

// TestLegacyFunctionDeclarationSchema fails if the unset default is true, if
// the stored value ignores v, or if the lookup does not read the nearest value
// in ctx.
func TestLegacyFunctionDeclarationSchema(t *testing.T) {
	bg := context.Background()
	tests := []struct {
		name string
		ctx  context.Context
		want bool
	}{
		{name: "unset", ctx: bg, want: false},
		{name: "set true", ctx: feature.WithLegacyFunctionDeclarationSchema(bg, true), want: true},
		{
			name: "true then false",
			ctx:  feature.WithLegacyFunctionDeclarationSchema(feature.WithLegacyFunctionDeclarationSchema(bg, true), false),
			want: false,
		},
		{
			name: "false then true",
			ctx:  feature.WithLegacyFunctionDeclarationSchema(feature.WithLegacyFunctionDeclarationSchema(bg, false), true),
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := feature.LegacyFunctionDeclarationSchema(tt.ctx); got != tt.want {
				t.Errorf("LegacyFunctionDeclarationSchema() = %v, want %v", got, tt.want)
			}
		})
	}
}
