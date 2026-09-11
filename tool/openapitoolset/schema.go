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

package openapitoolset

import (
	"fmt"
	"sort"

	"github.com/getkin/kin-openapi/openapi3"
	"google.golang.org/genai"
)

func convertSchema(ref *openapi3.SchemaRef) (*genai.Schema, error) {
	return (&schemaConverter{visiting: make(map[*openapi3.Schema]bool)}).convert(ref)
}

type schemaConverter struct {
	visiting map[*openapi3.Schema]bool
}

func (c *schemaConverter) convert(ref *openapi3.SchemaRef) (*genai.Schema, error) {
	if ref == nil {
		return nil, fmt.Errorf("schema is missing")
	}
	if ref.Value == nil {
		if ref.Ref != "" {
			return nil, fmt.Errorf("schema reference %q was not resolved", ref.Ref)
		}
		return nil, fmt.Errorf("schema is missing")
	}
	source := ref.Value
	if c.visiting[source] {
		return nil, fmt.Errorf("recursive schemas are not supported")
	}
	c.visiting[source] = true
	defer delete(c.visiting, source)

	result := &genai.Schema{
		Title:       source.Title,
		Description: source.Description,
		Format:      source.Format,
		Default:     source.Default,
		Example:     source.Example,
		Pattern:     source.Pattern,
		Minimum:     source.Min,
		Maximum:     source.Max,
	}
	if source.Nullable || (source.Type != nil && source.Type.IncludesNull()) {
		nullable := true
		result.Nullable = &nullable
	}
	for _, value := range source.Enum {
		result.Enum = append(result.Enum, fmt.Sprint(value))
	}
	if source.MinLength > 0 {
		value := int64(source.MinLength)
		result.MinLength = &value
	}
	if source.MaxLength != nil {
		value := int64(*source.MaxLength)
		result.MaxLength = &value
	}
	if source.MinItems > 0 {
		value := int64(source.MinItems)
		result.MinItems = &value
	}
	if source.MaxItems != nil {
		value := int64(*source.MaxItems)
		result.MaxItems = &value
	}
	if source.MinProps > 0 {
		value := int64(source.MinProps)
		result.MinProperties = &value
	}
	if source.MaxProps != nil {
		value := int64(*source.MaxProps)
		result.MaxProperties = &value
	}

	schemaType, err := convertSchemaType(source)
	if err != nil {
		return nil, err
	}
	result.Type = schemaType
	if source.Items != nil {
		result.Items, err = c.convert(source.Items)
		if err != nil {
			return nil, fmt.Errorf("array items: %w", err)
		}
	}
	if len(source.Properties) > 0 {
		result.Properties = make(map[string]*genai.Schema, len(source.Properties))
		result.PropertyOrdering = make([]string, 0, len(source.Properties))
		for name := range source.Properties {
			result.PropertyOrdering = append(result.PropertyOrdering, name)
		}
		sort.Strings(result.PropertyOrdering)
		for _, name := range result.PropertyOrdering {
			property, err := c.convert(source.Properties[name])
			if err != nil {
				return nil, fmt.Errorf("property %q: %w", name, err)
			}
			result.Properties[name] = property
		}
		result.Required = append([]string(nil), source.Required...)
	}

	alternatives := append(openapi3.SchemaRefs(nil), source.OneOf...)
	alternatives = append(alternatives, source.AnyOf...)
	for i, alternative := range alternatives {
		converted, err := c.convert(alternative)
		if err != nil {
			return nil, fmt.Errorf("alternative %d: %w", i, err)
		}
		result.AnyOf = append(result.AnyOf, converted)
	}
	for i, component := range source.AllOf {
		converted, err := c.convert(component)
		if err != nil {
			return nil, fmt.Errorf("allOf component %d: %w", i, err)
		}
		if err := mergeObjectSchema(result, converted); err != nil {
			return nil, fmt.Errorf("allOf component %d: %w", i, err)
		}
	}
	return result, nil
}

func convertSchemaType(schema *openapi3.Schema) (genai.Type, error) {
	if schema.Type == nil || schema.Type.IsEmpty() {
		switch {
		case len(schema.Properties) > 0 || len(schema.AllOf) > 0:
			return genai.TypeObject, nil
		case schema.Items != nil:
			return genai.TypeArray, nil
		default:
			return "", nil
		}
	}
	var concrete []string
	for _, value := range schema.Type.Slice() {
		if value != "null" {
			concrete = append(concrete, value)
		}
	}
	if len(concrete) != 1 {
		return "", fmt.Errorf("schema type %v is not supported", schema.Type.Slice())
	}
	switch concrete[0] {
	case "string":
		return genai.TypeString, nil
	case "number":
		return genai.TypeNumber, nil
	case "integer":
		return genai.TypeInteger, nil
	case "boolean":
		return genai.TypeBoolean, nil
	case "array":
		return genai.TypeArray, nil
	case "object":
		return genai.TypeObject, nil
	default:
		return "", fmt.Errorf("schema type %q is not supported", concrete[0])
	}
}

func mergeObjectSchema(target, source *genai.Schema) error {
	if source.Type != "" && source.Type != genai.TypeObject {
		return fmt.Errorf("non-object schema type %q is not supported", source.Type)
	}
	target.Type = genai.TypeObject
	if target.Properties == nil {
		target.Properties = make(map[string]*genai.Schema)
	}
	for name, property := range source.Properties {
		if _, exists := target.Properties[name]; exists {
			return fmt.Errorf("duplicate property %q", name)
		}
		target.Properties[name] = property
		target.PropertyOrdering = append(target.PropertyOrdering, name)
	}
	sort.Strings(target.PropertyOrdering)
	target.Required = appendUnique(target.Required, source.Required...)
	return nil
}

func appendUnique(values []string, additions ...string) []string {
	seen := make(map[string]bool, len(values)+len(additions))
	for _, value := range values {
		seen[value] = true
	}
	for _, value := range additions {
		if !seen[value] {
			values = append(values, value)
			seen[value] = true
		}
	}
	return values
}
