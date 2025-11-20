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

package taskstore

import (
	"fmt"

	"github.com/a2aproject/a2a-go/a2a"
)

func validateTask(task *a2a.Task) error {
	if task == nil {
		return nil
	}
	if err := validateMessage(task.Status.Message); err != nil {
		return err
	}
	for _, msg := range task.History {
		if err := validateMessage(msg); err != nil {
			return err
		}
	}
	for _, a := range task.Artifacts {
		if err := validateArtifact(a); err != nil {
			return err
		}
	}
	if err := validateMeta(task.Metadata); err != nil {
		return err
	}
	return nil
}

func validateArtifact(artifact *a2a.Artifact) error {
	if artifact == nil {
		return nil
	}
	if err := validateParts(artifact.Parts); err != nil {
		return err
	}
	if err := validateMeta(artifact.Metadata); err != nil {
		return err
	}
	return nil
}

func validateMessage(msg *a2a.Message) error {
	if msg == nil {
		return nil
	}
	if err := validateParts(msg.Parts); err != nil {
		return err
	}
	if err := validateMeta(msg.Metadata); err != nil {
		return err
	}
	return nil
}

func validateParts(parts a2a.ContentParts) error {
	if parts == nil {
		return nil
	}
	for _, p := range parts {
		if err := validateMeta(p.Meta()); err != nil {
			return err
		}
	}
	return nil
}

func validateMeta(meta map[string]any) error {
	return validateMetaRecursive(meta, map[string]struct{}{})
}

func validateMetaRecursive(value any, processing map[string]struct{}) error {
	if value == nil {
		return nil
	}

	switch value.(type) {
	// Exclude uint because unsigned types won't play well with the spec
	case bool, int, int8, int16, int32, int64, float32, float64, string:
		return nil
	}

	key := fmt.Sprintf("%p", value)
	if _, ok := processing[key]; ok {
		return fmt.Errorf("circular reference in Metadata")
	}
	processing[key] = struct{}{}
	defer delete(processing, key)

	if arr, ok := value.([]any); ok {
		for _, elem := range arr {
			if err := validateMetaRecursive(elem, processing); err != nil {
				return err
			}
		}
		return nil
	}

	if m, ok := value.(map[string]any); ok {
		for _, elem := range m {
			if err := validateMetaRecursive(elem, processing); err != nil {
				return err
			}
		}
		return nil
	}

	return fmt.Errorf("%T is not permitted in Metadata, must be one of nil, bool, int, float, string, []any, map[string]any", value)
}
