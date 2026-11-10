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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// inputValidationError is a request that cannot be built from the tool-call
// arguments, such as a missing required path parameter or a traversal attempt.
// It is reported to the model as a tool error, not raised, so the model can
// correct its arguments. It is adk-python's InputValidationError.
type inputValidationError struct {
	msg string
}

func (e *inputValidationError) Error() string { return e.msg }

// preparedRequest is everything needed to issue one HTTP request, built from a
// tool's endpoint and the call arguments.
type preparedRequest struct {
	method  string // lowercase, as adk-python holds it
	url     string // base URL and path, without query or fragment
	query   url.Values
	headers map[string]string
	cookies map[string]string
	// body is the encoded request body, or nil for none. contentType is its
	// Content-Type, left empty for multipart so net/http sets the boundary.
	body        []byte
	contentType string
	multipart   bool
}

var pathPlaceholder = regexp.MustCompile(`\{([^}]+)\}`)

// prepareRequest builds the HTTP request for a call. It is adk-python's
// RestApiTool._prepare_request_params. args is not modified.
func (t *RESTAPITool) prepareRequest(args map[string]any) (*preparedRequest, error) {
	method := strings.ToLower(t.endpoint.Method)
	if method == "" {
		return nil, fmt.Errorf("operation method not found")
	}

	// Fill in required parameters that were not supplied but have a default.
	filled := make(map[string]any, len(args))
	for k, v := range args {
		filled[k] = v
	}
	for _, p := range t.params {
		if _, ok := filled[p.pyName]; ok {
			continue
		}
		if p.required {
			if def, ok := p.schema["default"]; ok && def != nil {
				filled[p.pyName] = def
			}
		}
	}

	byPyName := make(map[string]*apiParameter, len(t.params))
	for _, p := range t.params {
		byPyName[p.pyName] = p
	}

	pathParams := map[string]string{}
	query := url.Values{}
	spread := map[string]bool{}
	headers := map[string]string{"User-Agent": userAgent(t.name)}
	cookies := map[string]string{}

	for k, v := range filled {
		p := byPyName[k]
		if p == nil {
			continue // An argument that is not a known parameter is ignored.
		}
		switch p.location {
		case "path":
			pathParams[p.originalName] = paramString(v)
		case "query":
			if v == nil {
				continue
			}
			otherNames := otherQueryNames(t.params, p)
			for _, entry := range serializeQueryParam(p, v) {
				switch {
				case entry.key == p.originalName:
					query[entry.key] = entry.values
					delete(spread, entry.key)
				case !otherNames[entry.key]:
					query[entry.key] = entry.values
					spread[entry.key] = true
				}
			}
		case "header":
			if v != nil {
				headers[p.originalName] = paramString(v)
			}
		case "cookie":
			if v != nil {
				cookies[p.originalName] = paramString(v)
			}
		}
	}

	rawURL, err := t.buildURL(pathParams)
	if err != nil {
		return nil, err
	}
	rawURL, err = extractEmbeddedQuery(rawURL, query, spread)
	if err != nil {
		return nil, err
	}

	req := &preparedRequest{
		method:  method,
		url:     rawURL,
		query:   query,
		headers: headers,
		cookies: cookies,
	}
	if err := t.buildBody(req, filled); err != nil {
		return nil, err
	}
	return req, nil
}

// buildURL fills the path template with the encoded path parameters and
// prepends the base URL.
func (t *RESTAPITool) buildURL(pathParams map[string]string) (string, error) {
	base := strings.TrimSuffix(t.endpoint.BaseURL, "/")

	var buildErr error
	formatted := pathPlaceholder.ReplaceAllStringFunc(t.endpoint.Path, func(match string) string {
		name := match[1 : len(match)-1]
		raw, ok := pathParams[name]
		if !ok {
			if buildErr == nil {
				buildErr = t.missingPathParam(name)
			}
			return match
		}
		encoded, err := encodePathParam(name, raw)
		if err != nil && buildErr == nil {
			buildErr = err
		}
		return encoded
	})
	if buildErr != nil {
		return "", buildErr
	}
	return base + formatted, nil
}

// missingPathParam returns the error for a path placeholder with no value: the
// required-parameter error naming the argument when the placeholder is a
// declared path parameter, otherwise a plain error for an unmatched template.
func (t *RESTAPITool) missingPathParam(name string) error {
	for _, p := range t.params {
		if p.location == "path" && p.originalName == name {
			return &inputValidationError{fmt.Sprintf("Missing required path parameter %q.", p.pyName)}
		}
	}
	return fmt.Errorf("unmatched path placeholder %q", name)
}

// extractEmbeddedQuery moves a query string or fragment written into the path
// template itself into query, and returns the URL without them. A key already
// set explicitly wins, but one only placed there by an exploded object (in
// spread) is overridden. It is the "move embedded query params" step of
// adk-python's _prepare_request_params.
func extractEmbeddedQuery(rawURL string, query url.Values, spread map[string]bool) (string, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return "", fmt.Errorf("invalid request URL: %w", err)
	}
	for _, part := range []string{parsed.RawQuery, parsed.Fragment} {
		if part == "" {
			continue
		}
		values, err := url.ParseQuery(part)
		if err != nil {
			continue
		}
		for key, vals := range values {
			if _, set := query[key]; !set || spread[key] {
				query[key] = vals
				delete(spread, key)
			}
		}
	}
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}

// buildBody encodes the request body onto req from the body arguments.
func (t *RESTAPITool) buildBody(req *preparedRequest, args map[string]any) error {
	if t.body == nil {
		return nil
	}
	mediaType := t.body.mediaType

	var bodyData any
	if t.body.object {
		// An object body is sent as a map of all its body arguments.
		obj := map[string]any{}
		for _, p := range t.params {
			if p.location != "body" {
				continue
			}
			if v, ok := args[p.pyName]; ok {
				obj[p.originalName] = v
			}
		}
		bodyData = obj
	} else {
		// A single body argument carries the whole body; absent means none.
		for _, p := range t.params {
			if p.location != "body" {
				continue
			}
			if v, ok := args[p.pyName]; ok {
				bodyData = v
			}
			break
		}
		if bodyData == nil {
			return nil
		}
	}

	switch {
	case mediaType == "application/json" || strings.HasSuffix(mediaType, "+json"):
		encoded, err := json.Marshal(bodyData)
		if err != nil {
			return fmt.Errorf("encode JSON body: %w", err)
		}
		req.body = encoded
		req.contentType = mediaType
	case mediaType == "application/x-www-form-urlencoded":
		form, err := formValues(bodyData)
		if err != nil {
			return err
		}
		req.body = []byte(form.Encode())
		req.contentType = mediaType
	case mediaType == "multipart/form-data":
		body, contentType, err := multipartBody(bodyData)
		if err != nil {
			return err
		}
		req.body = body
		req.contentType = contentType
		req.multipart = true
	case mediaType == "application/octet-stream" || mediaType == "text/plain":
		req.body = rawBody(bodyData)
		req.contentType = mediaType
	default:
		// An unhandled media type still sets the declared Content-Type, as
		// adk-python does, but sends no body.
		req.contentType = mediaType
	}
	return nil
}

// httpRequest builds the net/http request. The method is sent uppercased, as
// servers expect on the wire, even though adk-python keeps it lowercase.
func (p *preparedRequest) httpRequest(ctx context.Context) (*http.Request, error) {
	var body io.Reader
	if p.body != nil {
		body = bytes.NewReader(p.body)
	}
	req, err := http.NewRequestWithContext(ctx, strings.ToUpper(p.method), p.url, body)
	if err != nil {
		return nil, err
	}
	if len(p.query) > 0 {
		req.URL.RawQuery = p.query.Encode()
	}
	for k, v := range p.headers {
		req.Header.Set(k, v)
	}
	if p.contentType != "" {
		req.Header.Set("Content-Type", p.contentType)
	}
	for k, v := range p.cookies {
		req.AddCookie(&http.Cookie{Name: k, Value: v})
	}
	return req, nil
}

// otherQueryNames is the set of spec names of the query parameters other than
// p, so an exploded object cannot inject another parameter's key.
func otherQueryNames(params []*apiParameter, p *apiParameter) map[string]bool {
	names := map[string]bool{}
	for _, other := range params {
		if other.location == "query" && other != p {
			names[other.originalName] = true
		}
	}
	return names
}

// queryEntry is one query key with its string values; several values repeat
// the key.
type queryEntry struct {
	key    string
	values []string
}

// serializeQueryParam applies a query parameter's OpenAPI style and explode to
// its value. It is adk-python's _serialize_query_param. An exploded object with
// declared properties and no additionalProperties drops undeclared keys.
func serializeQueryParam(p *apiParameter, value any) []queryEntry {
	name := p.originalName
	style := p.style
	if style == "" {
		style = "form"
	}
	explode := style == "form"
	if p.explode != nil {
		explode = *p.explode
	}

	switch v := value.(type) {
	case map[string]any:
		clean := nonNilMap(v)
		if len(clean) == 0 {
			return nil
		}
		keys := slices.Sorted(mapKeys(clean))
		if style == "deepObject" {
			entries := make([]queryEntry, 0, len(keys))
			for _, k := range keys {
				entries = append(entries, queryEntry{fmt.Sprintf("%s[%s]", name, k), []string{paramString(clean[k])}})
			}
			return entries
		}
		if explode {
			if declared := declaredProps(p.schema); declared != nil {
				var entries []queryEntry
				for _, k := range keys {
					if declared[k] {
						entries = append(entries, queryEntry{k, []string{paramString(clean[k])}})
					}
				}
				return entries
			}
			entries := make([]queryEntry, 0, len(keys))
			for _, k := range keys {
				entries = append(entries, queryEntry{k, []string{paramString(clean[k])}})
			}
			return entries
		}
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			parts = append(parts, fmt.Sprintf("%s,%s", k, paramString(clean[k])))
		}
		return []queryEntry{{name, []string{strings.Join(parts, ",")}}}
	case []any:
		clean := nonNilSlice(v)
		if len(clean) == 0 {
			return nil
		}
		if !explode {
			sep := ","
			switch style {
			case "spaceDelimited":
				sep = " "
			case "pipeDelimited":
				sep = "|"
			}
			parts := make([]string, len(clean))
			for i, item := range clean {
				parts[i] = paramString(item)
			}
			return []queryEntry{{name, []string{strings.Join(parts, sep)}}}
		}
		vals := make([]string, len(clean))
		for i, item := range clean {
			vals[i] = paramString(item)
		}
		return []queryEntry{{name, vals}}
	default:
		return []queryEntry{{name, []string{paramString(value)}}}
	}
}

// declaredProps returns the set of a schema's declared property names when it
// has properties and no additionalProperties, so undeclared keys are dropped;
// otherwise nil, meaning every key is kept.
func declaredProps(schema map[string]any) map[string]bool {
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		return nil
	}
	if ap, ok := schema["additionalProperties"]; ok && ap != false {
		return nil
	}
	declared := make(map[string]bool, len(props))
	for k := range props {
		declared[k] = true
	}
	return declared
}

// paramString serializes a path, header, cookie or query value for the wire. It
// is adk-python's _to_param_string: a boolean is its JSON spelling, a list or
// object is comma-joined (dropping nil entries), and anything else is its
// string form.
func paramString(v any) string {
	switch v := v.(type) {
	case nil:
		return "None" // Python str(None); reached only for a nil path value.
	case bool:
		if v {
			return "true"
		}
		return "false"
	case string:
		return v
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case int:
		return strconv.Itoa(v)
	case int64:
		return strconv.FormatInt(v, 10)
	case json.Number:
		return v.String()
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			if item != nil {
				parts = append(parts, paramString(item))
			}
		}
		return strings.Join(parts, ",")
	case map[string]any:
		clean := nonNilMap(v)
		keys := slices.Sorted(mapKeys(clean))
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			parts = append(parts, fmt.Sprintf("%s,%s", k, paramString(clean[k])))
		}
		return strings.Join(parts, ",")
	}
	return fmt.Sprintf("%v", v)
}

// encodePathParam percent-encodes a path parameter and rejects RFC 3986
// dot-segments, so a value cannot introduce a query, fragment or new path
// segment. It is adk-python's _encode_path_param: every byte outside the
// unreserved set is encoded (so "/" becomes %2F), while "." stays literal, so
// "." and ".." segments are rejected first.
func encodePathParam(name, value string) (string, error) {
	for _, segment := range strings.Split(strings.ReplaceAll(value, `\`, "/"), "/") {
		if segment == "." || segment == ".." {
			return "", &inputValidationError{fmt.Sprintf(
				"Path parameter %q must not contain current-directory or parent-directory segments.", name)}
		}
	}
	return percentEncode(value), nil
}

// percentEncode encodes every byte except the RFC 3986 unreserved set
// (A-Z a-z 0-9 - . _ ~), matching Python's urllib.parse.quote(value, safe="").
func percentEncode(s string) string {
	const upperhex = "0123456789ABCDEF"
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' ||
			c == '-' || c == '.' || c == '_' || c == '~' {
			b.WriteByte(c)
			continue
		}
		b.WriteByte('%')
		b.WriteByte(upperhex[c>>4])
		b.WriteByte(upperhex[c&0xf])
	}
	return b.String()
}

// formValues builds form fields from an object body.
func formValues(bodyData any) (url.Values, error) {
	obj, ok := bodyData.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("form body must be an object, got %T", bodyData)
	}
	form := url.Values{}
	for _, k := range slices.Sorted(mapKeys(obj)) {
		form.Set(k, paramString(obj[k]))
	}
	return form, nil
}

// multipartBody writes an object body as multipart/form-data and returns the
// body and its Content-Type (which carries the boundary).
func multipartBody(bodyData any) ([]byte, string, error) {
	obj, ok := bodyData.(map[string]any)
	if !ok {
		return nil, "", fmt.Errorf("multipart body must be an object, got %T", bodyData)
	}
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, k := range slices.Sorted(mapKeys(obj)) {
		if err := w.WriteField(k, string(rawBody(obj[k]))); err != nil {
			return nil, "", fmt.Errorf("write multipart field: %w", err)
		}
	}
	if err := w.Close(); err != nil {
		return nil, "", fmt.Errorf("close multipart writer: %w", err)
	}
	return buf.Bytes(), w.FormDataContentType(), nil
}

// rawBody returns the bytes of a scalar body value.
func rawBody(v any) []byte {
	switch v := v.(type) {
	case nil:
		return nil
	case []byte:
		return v
	case string:
		return []byte(v)
	}
	return []byte(paramString(v))
}

func nonNilMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if v != nil {
			out[k] = v
		}
	}
	return out
}

func nonNilSlice(s []any) []any {
	out := make([]any, 0, len(s))
	for _, v := range s {
		if v != nil {
			out = append(out, v)
		}
	}
	return out
}

func mapKeys(m map[string]any) func(func(string) bool) {
	return func(yield func(string) bool) {
		for k := range m {
			if !yield(k) {
				return
			}
		}
	}
}
