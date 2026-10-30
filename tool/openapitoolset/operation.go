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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/getkin/kin-openapi/openapi3"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/toolutils"
)

type argumentBinding struct {
	argumentName  string
	parameterName string
	location      string
	required      bool
	defaultValue  any
	style         string
	explode       bool
}

type bodyBinding struct {
	mediaType string
	required  bool
	wholeArg  string
	fields    []argumentBinding
}

type operationTool struct {
	name             string
	description      string
	method           string
	path             string
	baseURL          string
	client           *http.Client
	maxResponseBytes int64
	declaration      *genai.FunctionDeclaration
	parameters       []argumentBinding
	body             *bodyBinding
}

func newOperationTool(method, path, baseURL string, pathParameters openapi3.Parameters, operation *openapi3.Operation, client *http.Client, maxResponseBytes int64) (*operationTool, error) {
	nameSource := operation.OperationID
	if nameSource == "" {
		nameSource = strings.ToLower(method) + "_" + path
	}
	name := sanitizeIdentifier(nameSource, 128)
	if name == "" {
		return nil, fmt.Errorf("operation has no usable tool name")
	}
	description := operation.Description
	if description == "" {
		description = operation.Summary
	}
	if description == "" {
		description = method + " " + path
	}

	namer := &argumentNamer{used: make(map[string]bool)}
	parameters, properties, required, err := buildParameterBindings(pathParameters, operation.Parameters, namer)
	if err != nil {
		return nil, err
	}
	body, err := buildBodyBinding(operation.RequestBody, namer, properties, &required)
	if err != nil {
		return nil, err
	}
	ordering := make([]string, 0, len(properties))
	for property := range properties {
		ordering = append(ordering, property)
	}
	sort.Strings(ordering)
	sort.Strings(required)

	declaration := &genai.FunctionDeclaration{
		Name:        name,
		Description: description,
		Parameters: &genai.Schema{
			Type:             genai.TypeObject,
			Properties:       properties,
			PropertyOrdering: ordering,
			Required:         required,
		},
	}
	return &operationTool{
		name:             name,
		description:      description,
		method:           method,
		path:             path,
		baseURL:          baseURL,
		client:           client,
		maxResponseBytes: maxResponseBytes,
		declaration:      declaration,
		parameters:       parameters,
		body:             body,
	}, nil
}

func buildParameterBindings(pathParameters, operationParameters openapi3.Parameters, namer *argumentNamer) ([]argumentBinding, map[string]*genai.Schema, []string, error) {
	merged, err := mergeParameters(pathParameters, operationParameters)
	if err != nil {
		return nil, nil, nil, err
	}
	properties := make(map[string]*genai.Schema, len(merged))
	var (
		bindings []argumentBinding
		required []string
	)
	for _, ref := range merged {
		parameter := ref.Value
		if parameter.Schema == nil {
			return nil, nil, nil, fmt.Errorf("parameter %q must use a schema", parameter.Name)
		}
		schema, err := convertSchema(parameter.Schema)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("parameter %q: %w", parameter.Name, err)
		}
		if parameter.Description != "" {
			schema.Description = parameter.Description
		}
		serialization, err := parameter.SerializationMethod()
		if err != nil {
			return nil, nil, nil, fmt.Errorf("parameter %q: %w", parameter.Name, err)
		}
		argumentName := namer.claim(parameter.Name)
		properties[argumentName] = schema
		bindings = append(bindings, argumentBinding{
			argumentName:  argumentName,
			parameterName: parameter.Name,
			location:      parameter.In,
			required:      parameter.Required,
			defaultValue:  parameter.Schema.Value.Default,
			style:         serialization.Style,
			explode:       serialization.Explode,
		})
		if parameter.Required {
			required = append(required, argumentName)
		}
	}
	return bindings, properties, required, nil
}

func mergeParameters(pathParameters, operationParameters openapi3.Parameters) (openapi3.Parameters, error) {
	merged := append(openapi3.Parameters(nil), pathParameters...)
	index := make(map[string]int, len(merged))
	for i, ref := range merged {
		if ref == nil || ref.Value == nil {
			return nil, fmt.Errorf("path parameter reference was not resolved")
		}
		index[parameterKey(ref.Value)] = i
	}
	for _, ref := range operationParameters {
		if ref == nil || ref.Value == nil {
			return nil, fmt.Errorf("operation parameter reference was not resolved")
		}
		key := parameterKey(ref.Value)
		if i, ok := index[key]; ok {
			merged[i] = ref
			continue
		}
		index[key] = len(merged)
		merged = append(merged, ref)
	}
	return merged, nil
}

func parameterKey(parameter *openapi3.Parameter) string {
	return parameter.In + "\x00" + parameter.Name
}

func buildBodyBinding(ref *openapi3.RequestBodyRef, namer *argumentNamer, properties map[string]*genai.Schema, required *[]string) (*bodyBinding, error) {
	if ref == nil {
		return nil, nil
	}
	if ref.Value == nil {
		return nil, fmt.Errorf("request body reference was not resolved")
	}
	var mediaType string
	contentTypes := make([]string, 0, len(ref.Value.Content))
	for contentType := range ref.Value.Content {
		contentTypes = append(contentTypes, contentType)
	}
	sort.Strings(contentTypes)
	for _, contentType := range contentTypes {
		normalized := strings.ToLower(strings.TrimSpace(strings.Split(contentType, ";")[0]))
		if normalized == "application/json" || strings.HasSuffix(normalized, "+json") {
			mediaType = contentType
			break
		}
	}
	if mediaType == "" {
		return nil, fmt.Errorf("request body has no JSON media type")
	}
	media := ref.Value.Content[mediaType]
	if media == nil || media.Schema == nil {
		return nil, fmt.Errorf("JSON request body schema is missing")
	}
	schema, err := convertSchema(media.Schema)
	if err != nil {
		return nil, fmt.Errorf("request body: %w", err)
	}
	body := &bodyBinding{mediaType: mediaType, required: ref.Value.Required}
	if schema.Type != genai.TypeObject || len(schema.Properties) == 0 {
		argumentName := namer.claim("body")
		properties[argumentName] = schema
		body.wholeArg = argumentName
		if ref.Value.Required {
			*required = append(*required, argumentName)
		}
		return body, nil
	}
	names := append([]string(nil), schema.PropertyOrdering...)
	for _, propertyName := range names {
		argumentName := namer.claim(propertyName)
		propertySchema := schema.Properties[propertyName]
		properties[argumentName] = propertySchema
		isRequired := containsString(schema.Required, propertyName)
		body.fields = append(body.fields, argumentBinding{
			argumentName:  argumentName,
			parameterName: propertyName,
			location:      "body",
			required:      isRequired,
			defaultValue:  propertySchema.Default,
		})
		if ref.Value.Required && isRequired {
			*required = append(*required, argumentName)
		}
	}
	return body, nil
}

type argumentNamer struct {
	used map[string]bool
}

func (n *argumentNamer) claim(source string) string {
	base := sanitizeIdentifier(source, 64)
	if base == "" {
		base = "value"
	}
	candidate := base
	for suffix := 2; n.used[candidate]; suffix++ {
		text := "_" + strconv.Itoa(suffix)
		candidate = truncateRunes(base, 64-utf8.RuneCountInString(text)) + text
	}
	n.used[candidate] = true
	return candidate
}

func sanitizeIdentifier(source string, maxRunes int) string {
	var (
		builder        strings.Builder
		previousLetter bool
		previousLower  bool
		underscore     bool
	)
	for _, r := range strings.TrimSpace(source) {
		isLower := r >= 'a' && r <= 'z'
		isUpper := r >= 'A' && r <= 'Z'
		isDigit := r >= '0' && r <= '9'
		if isLower || isUpper || isDigit {
			if isUpper && previousLetter && previousLower && !underscore {
				builder.WriteByte('_')
			}
			if builder.Len() == 0 && isDigit {
				builder.WriteByte('_')
			}
			if isUpper {
				r += 'a' - 'A'
			}
			builder.WriteRune(r)
			previousLower = isLower || isDigit
			previousLetter = true
			underscore = false
			continue
		}
		if builder.Len() > 0 && !underscore {
			builder.WriteByte('_')
			underscore = true
		}
		previousLetter = false
		previousLower = false
	}
	result := strings.Trim(builder.String(), "_")
	if result == "" {
		return ""
	}
	if result[0] >= '0' && result[0] <= '9' {
		result = "_" + result
	}
	return truncateRunes(result, maxRunes)
}

func truncateRunes(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[:limit])
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func (t *operationTool) Name() string {
	return t.name
}

func (t *operationTool) Description() string {
	return t.description
}

func (t *operationTool) IsLongRunning() bool {
	return false
}

func (t *operationTool) Declaration() *genai.FunctionDeclaration {
	return t.declaration
}

func (t *operationTool) ProcessRequest(_ agent.Context, request *model.LLMRequest) error {
	return toolutils.PackTool(request, t)
}

func (t *operationTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	if ctx == nil {
		return nil, fmt.Errorf("openapi tool %q: context must not be nil", t.name)
	}
	if args == nil {
		args = map[string]any{}
	}
	arguments, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("openapi tool %q: arguments have type %T, want map[string]any", t.name, args)
	}
	known := make(map[string]bool, len(t.parameters)+len(t.declaration.Parameters.Properties))
	for _, parameter := range t.parameters {
		known[parameter.argumentName] = true
	}
	if t.body != nil {
		if t.body.wholeArg != "" {
			known[t.body.wholeArg] = true
		}
		for _, field := range t.body.fields {
			known[field.argumentName] = true
		}
	}
	for name := range arguments {
		if !known[name] {
			return nil, fmt.Errorf("openapi tool %q: unknown argument %q", t.name, name)
		}
	}

	path := t.path
	query := make(url.Values)
	headers := make(http.Header)
	var cookies []*http.Cookie
	for _, parameter := range t.parameters {
		value, present := arguments[parameter.argumentName]
		if !present && parameter.defaultValue != nil {
			value, present = parameter.defaultValue, true
		}
		if !present {
			if parameter.required {
				return nil, fmt.Errorf("openapi tool %q: required argument %q is missing", t.name, parameter.argumentName)
			}
			continue
		}
		switch parameter.location {
		case openapi3.ParameterInPath:
			serialized, err := serializePathParameter(parameter, value)
			if err != nil {
				return nil, fmt.Errorf("openapi tool %q: path argument %q: %w", t.name, parameter.argumentName, err)
			}
			escaped := url.PathEscape(serialized)
			placeholder := "{" + parameter.parameterName + "}"
			if !strings.Contains(path, placeholder) {
				return nil, fmt.Errorf("openapi tool %q: path has no placeholder for %q", t.name, parameter.parameterName)
			}
			path = strings.ReplaceAll(path, placeholder, escaped)
		case openapi3.ParameterInQuery:
			if err := addQueryParameter(query, parameter, value); err != nil {
				return nil, fmt.Errorf("openapi tool %q: query argument %q: %w", t.name, parameter.argumentName, err)
			}
		case openapi3.ParameterInHeader:
			serialized, err := serializeSimple(value, parameter.explode)
			if err != nil {
				return nil, fmt.Errorf("openapi tool %q: header argument %q: %w", t.name, parameter.argumentName, err)
			}
			headers.Set(parameter.parameterName, serialized)
		case openapi3.ParameterInCookie:
			serialized, err := serializeCookies(parameter, value)
			if err != nil {
				return nil, fmt.Errorf("openapi tool %q: cookie argument %q: %w", t.name, parameter.argumentName, err)
			}
			cookies = append(cookies, serialized...)
		default:
			return nil, fmt.Errorf("openapi tool %q: parameter location %q is not supported", t.name, parameter.location)
		}
	}
	if strings.Contains(path, "{") {
		return nil, fmt.Errorf("openapi tool %q: path contains unresolved parameters", t.name)
	}

	body, err := t.buildRequestBody(arguments)
	if err != nil {
		return nil, err
	}
	endpoint, err := url.Parse(strings.TrimRight(t.baseURL, "/") + "/" + strings.TrimLeft(path, "/"))
	if err != nil {
		return nil, fmt.Errorf("openapi tool %q: build request URL: %w", t.name, err)
	}
	endpoint.RawQuery = query.Encode()
	request, err := http.NewRequestWithContext(ctx, t.method, endpoint.String(), body)
	if err != nil {
		return nil, fmt.Errorf("openapi tool %q: create request: %w", t.name, err)
	}
	request.Header = headers
	for _, cookie := range cookies {
		request.AddCookie(cookie)
	}
	if body != nil {
		request.Header.Set("Content-Type", t.body.mediaType)
	}
	response, err := t.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("openapi tool %q: execute request: %w", t.name, err)
	}
	defer func() {
		_ = response.Body.Close()
	}()
	responseBody, tooLarge, err := readBounded(response.Body, t.maxResponseBytes)
	if err != nil {
		return nil, fmt.Errorf("openapi tool %q: read response: %w", t.name, err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("openapi tool %q: request failed: %s: %s", t.name, response.Status, strings.TrimSpace(string(responseBody)))
	}
	if tooLarge {
		return nil, fmt.Errorf("openapi tool %q: response exceeds %d bytes", t.name, t.maxResponseBytes)
	}
	return decodeResponse(response, responseBody)
}

func (t *operationTool) buildRequestBody(arguments map[string]any) (io.Reader, error) {
	if t.body == nil {
		return nil, nil
	}
	var value any
	present := false
	if t.body.wholeArg != "" {
		value, present = arguments[t.body.wholeArg]
		if !present && t.body.required {
			return nil, fmt.Errorf("openapi tool %q: required argument %q is missing", t.name, t.body.wholeArg)
		}
	} else {
		object := make(map[string]any)
		presentFields := make(map[string]bool, len(t.body.fields))
		for _, field := range t.body.fields {
			fieldValue, fieldPresent := arguments[field.argumentName]
			if !fieldPresent && field.defaultValue != nil {
				fieldValue, fieldPresent = field.defaultValue, true
			}
			if !fieldPresent {
				continue
			}
			object[field.parameterName] = fieldValue
			presentFields[field.argumentName] = true
			present = true
		}
		if present || t.body.required {
			for _, field := range t.body.fields {
				if field.required && !presentFields[field.argumentName] {
					return nil, fmt.Errorf("openapi tool %q: required argument %q is missing", t.name, field.argumentName)
				}
			}
		}
		if present || t.body.required {
			value = object
			present = true
		}
	}
	if !present {
		return nil, nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("openapi tool %q: encode JSON request body: %w", t.name, err)
	}
	return bytes.NewReader(encoded), nil
}

func addQueryParameter(query url.Values, parameter argumentBinding, value any) error {
	switch parameter.style {
	case openapi3.SerializationForm:
		if parameter.explode && isMap(value) {
			entries, err := mapEntries(value)
			if err != nil {
				return err
			}
			for _, entry := range entries {
				query.Add(entry[0], entry[1])
			}
			return nil
		}
		values, err := flattenValue(value, false)
		if err != nil {
			return err
		}
		if parameter.explode && isSlice(value) {
			for _, item := range values {
				query.Add(parameter.parameterName, item)
			}
			return nil
		}
		query.Set(parameter.parameterName, strings.Join(values, ","))
		return nil
	case openapi3.SerializationSpaceDelimited, openapi3.SerializationPipeDelimited:
		if !isSlice(value) {
			return fmt.Errorf("style %q requires an array value", parameter.style)
		}
		values, err := flattenValue(value, false)
		if err != nil {
			return err
		}
		separator := " "
		if parameter.style == openapi3.SerializationPipeDelimited {
			separator = "|"
		}
		query.Set(parameter.parameterName, strings.Join(values, separator))
		return nil
	case openapi3.SerializationDeepObject:
		entries, err := mapEntries(value)
		if err != nil {
			return fmt.Errorf("deepObject requires an object value: %w", err)
		}
		for _, entry := range entries {
			query.Add(parameter.parameterName+"["+entry[0]+"]", entry[1])
		}
		return nil
	default:
		return fmt.Errorf("style %q is not supported", parameter.style)
	}
}

func serializePathParameter(parameter argumentBinding, value any) (string, error) {
	values, err := flattenValue(value, parameter.explode)
	if err != nil {
		return "", err
	}
	switch parameter.style {
	case openapi3.SerializationSimple:
		return strings.Join(values, ","), nil
	case openapi3.SerializationLabel:
		separator := ","
		if parameter.explode {
			separator = "."
		}
		return "." + strings.Join(values, separator), nil
	case openapi3.SerializationMatrix:
		if parameter.explode && isSlice(value) {
			var builder strings.Builder
			for _, item := range values {
				builder.WriteString(";")
				builder.WriteString(parameter.parameterName)
				builder.WriteString("=")
				builder.WriteString(item)
			}
			return builder.String(), nil
		}
		if parameter.explode && isMap(value) {
			return ";" + strings.Join(values, ";"), nil
		}
		return ";" + parameter.parameterName + "=" + strings.Join(values, ","), nil
	default:
		return "", fmt.Errorf("style %q is not supported", parameter.style)
	}
}

func serializeCookies(parameter argumentBinding, value any) ([]*http.Cookie, error) {
	if parameter.style != openapi3.SerializationForm {
		return nil, fmt.Errorf("style %q is not supported", parameter.style)
	}
	if parameter.explode && isMap(value) {
		entries, err := mapEntries(value)
		if err != nil {
			return nil, err
		}
		cookies := make([]*http.Cookie, 0, len(entries))
		for _, entry := range entries {
			cookies = append(cookies, &http.Cookie{Name: entry[0], Value: entry[1]})
		}
		return cookies, nil
	}
	values, err := flattenValue(value, false)
	if err != nil {
		return nil, err
	}
	if parameter.explode && isSlice(value) {
		cookies := make([]*http.Cookie, 0, len(values))
		for _, item := range values {
			cookies = append(cookies, &http.Cookie{Name: parameter.parameterName, Value: item})
		}
		return cookies, nil
	}
	return []*http.Cookie{{Name: parameter.parameterName, Value: strings.Join(values, ",")}}, nil
}

func serializeSimple(value any, explode bool) (string, error) {
	values, err := flattenValue(value, explode)
	if err != nil {
		return "", err
	}
	separator := ","
	if explode && isMap(value) {
		separator = ","
	}
	return strings.Join(values, separator), nil
}

func flattenValue(value any, explode bool) ([]string, error) {
	if value == nil {
		return []string{""}, nil
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Slice, reflect.Array:
		values := make([]string, 0, reflected.Len())
		for i := 0; i < reflected.Len(); i++ {
			values = append(values, fmt.Sprint(reflected.Index(i).Interface()))
		}
		return values, nil
	case reflect.Map:
		if reflected.Type().Key().Kind() != reflect.String {
			return nil, fmt.Errorf("map keys must be strings")
		}
		keys := reflected.MapKeys()
		sort.Slice(keys, func(i, j int) bool {
			return keys[i].String() < keys[j].String()
		})
		values := make([]string, 0, len(keys)*2)
		for _, key := range keys {
			mapValue := reflected.MapIndex(key).Interface()
			if explode {
				values = append(values, key.String()+"="+fmt.Sprint(mapValue))
			} else {
				values = append(values, key.String(), fmt.Sprint(mapValue))
			}
		}
		return values, nil
	default:
		return []string{fmt.Sprint(value)}, nil
	}
}

func mapEntries(value any) ([][2]string, error) {
	if !isMap(value) {
		return nil, fmt.Errorf("value is not an object")
	}
	reflected := reflect.ValueOf(value)
	if reflected.Type().Key().Kind() != reflect.String {
		return nil, fmt.Errorf("map keys must be strings")
	}
	keys := reflected.MapKeys()
	sort.Slice(keys, func(i, j int) bool {
		return keys[i].String() < keys[j].String()
	})
	entries := make([][2]string, 0, len(keys))
	for _, key := range keys {
		entries = append(entries, [2]string{key.String(), fmt.Sprint(reflected.MapIndex(key).Interface())})
	}
	return entries, nil
}

func isSlice(value any) bool {
	if value == nil {
		return false
	}
	kind := reflect.ValueOf(value).Kind()
	return kind == reflect.Slice || kind == reflect.Array
}

func isMap(value any) bool {
	return value != nil && reflect.ValueOf(value).Kind() == reflect.Map
}

func decodeResponse(response *http.Response, body []byte) (map[string]any, error) {
	if len(body) == 0 {
		return map[string]any{"status_code": response.StatusCode}, nil
	}
	contentType := response.Header.Get("Content-Type")
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil && contentType != "" {
		return nil, fmt.Errorf("parse response content type: %w", err)
	}
	isJSON := mediaType == "application/json" || strings.HasSuffix(mediaType, "+json")
	if contentType == "" {
		trimmed := bytes.TrimSpace(body)
		isJSON = len(trimmed) > 0 && (trimmed[0] == '{' || trimmed[0] == '[')
	}
	if !isJSON {
		return map[string]any{"status_code": response.StatusCode, "body": string(body)}, nil
	}
	var decoded any
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, fmt.Errorf("decode JSON response: %w", err)
	}
	if object, ok := decoded.(map[string]any); ok {
		return object, nil
	}
	return map[string]any{"result": decoded}, nil
}

var _ interface {
	Name() string
	Description() string
	IsLongRunning() bool
	Declaration() *genai.FunctionDeclaration
	Run(agent.Context, any) (map[string]any, error)
	ProcessRequest(agent.Context, *model.LLMRequest) error
} = (*operationTool)(nil)
