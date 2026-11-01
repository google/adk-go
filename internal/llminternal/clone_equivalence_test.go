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

package llminternal

import (
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
)

// referenceClone and referenceDeepCopy are the allocate-per-value reflective
// copy that clone and deepCopy replace, kept verbatim as the oracle for the
// equivalence tests below.
func referenceClone[M any](src M) M {
	val := reflect.ValueOf(src)

	// Handle nil pointers
	if val.Kind() == reflect.Pointer && val.IsNil() {
		var zero M
		return zero
	}

	srcIsPointer := val.Kind() == reflect.Pointer

	// Dereference pointer to get the underlying value
	if srcIsPointer {
		val = val.Elem()
	}

	// Create a new instance of the same type
	newVal := reflect.New(val.Type()).Elem()

	// Recursively copy fields
	referenceDeepCopy(val, newVal)

	// Return as the original type
	if srcIsPointer {
		return newVal.Addr().Interface().(M)
	}
	return newVal.Interface().(M)
}

func referenceDeepCopy(src, dst reflect.Value) {
	switch src.Kind() {
	case reflect.Struct:
		t := src.Type()
		for i := 0; i < src.NumField(); i++ {
			if !t.Field(i).IsExported() {
				panic(fmt.Sprintf("deepCopy: unexported field %q in type %q", t.Field(i).Name, t.Name()))
			}
			// Create a copy of the field and set it on the destination struct
			fieldCopy := reflect.New(src.Field(i).Type()).Elem()
			referenceDeepCopy(src.Field(i), fieldCopy)
			dst.Field(i).Set(fieldCopy)
		}
	case reflect.Slice:
		if src.IsNil() {
			return
		}
		dst.Set(reflect.MakeSlice(src.Type(), src.Len(), src.Cap()))
		for i := 0; i < src.Len(); i++ {
			// Create a copy of each element and set it in the new slice
			elemCopy := reflect.New(src.Index(i).Type()).Elem()
			referenceDeepCopy(src.Index(i), elemCopy)
			dst.Index(i).Set(elemCopy)
		}
	case reflect.Map:
		if src.IsNil() {
			return
		}
		dst.Set(reflect.MakeMap(src.Type()))
		for _, key := range src.MapKeys() {
			// Create copies of the key and value and set them in the new map
			keyCopy := reflect.New(key.Type()).Elem()
			referenceDeepCopy(key, keyCopy)
			valCopy := reflect.New(src.MapIndex(key).Type()).Elem()
			referenceDeepCopy(src.MapIndex(key), valCopy)
			dst.SetMapIndex(keyCopy, valCopy)
		}
	case reflect.Pointer:
		if src.IsNil() {
			return
		}
		// Create a new pointer and deep copy the underlying value
		newPtr := reflect.New(src.Elem().Type())
		referenceDeepCopy(src.Elem(), newPtr.Elem())
		dst.Set(newPtr)
	default:
		// For basic types, direct assignment is sufficient
		dst.Set(src)
	}
}

// filler sets every field reachable from a value to a non-zero value:
// pointers point to filled values, slices get two filled elements and
// spare capacity, maps get two entries, interfaces hold a map (so sharing is
// observable) and funcs are non-nil. Each recursive type is entered at most
// twice on a path. Pointers, slices and maps to values clone panics on are
// left nil and recorded in skipped.
type filler struct {
	n       int
	stack   map[reflect.Type]int
	filled  map[string]bool // "Type.Field" for every struct field set
	skipped map[string]bool // "Type.Field" left nil because clone rejects it
}

func newFiller() *filler {
	return &filler{stack: map[reflect.Type]int{}, filled: map[string]bool{}, skipped: map[string]bool{}}
}

func (f *filler) next() int { f.n++; return f.n }

// fill sets v, which must be settable and zero. It reports false if v was
// left zero.
func (f *filler) fill(v reflect.Value) bool {
	t := v.Type()
	switch t.Kind() {
	case reflect.Bool:
		v.SetBool(true)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(int64(f.next()%100 + 1))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		v.SetUint(uint64(f.next()%100 + 1))
	case reflect.Float32, reflect.Float64:
		v.SetFloat(float64(f.next()) + 0.5)
	case reflect.String:
		v.SetString(fmt.Sprintf("s%d-żółw-✓", f.next()))
	case reflect.Interface:
		if t.NumMethod() != 0 {
			return false
		}
		v.Set(reflect.ValueOf(map[string]any{"k": []any{fmt.Sprint(f.next()), 1.5}}))
	case reflect.Func:
		v.Set(reflect.MakeFunc(t, func(args []reflect.Value) []reflect.Value {
			out := make([]reflect.Value, t.NumOut())
			for i := range out {
				out[i] = reflect.Zero(t.Out(i))
			}
			return out
		}))
	case reflect.Pointer, reflect.Slice, reflect.Map:
		elem := t.Elem()
		if f.stack[elem] >= 2 || clonePanics(elem) || (t.Kind() == reflect.Map && clonePanics(t.Key())) {
			return false
		}
		f.stack[elem]++
		defer func() { f.stack[elem]-- }()
		switch t.Kind() {
		case reflect.Pointer:
			p := reflect.New(elem)
			f.fill(p.Elem())
			v.Set(p)
		case reflect.Slice:
			s := reflect.MakeSlice(t, 2, 3)
			f.fill(s.Index(0))
			f.fill(s.Index(1))
			v.Set(s)
		case reflect.Map:
			m := reflect.MakeMap(t)
			for range 2 {
				k, e := reflect.New(t.Key()).Elem(), reflect.New(elem).Elem()
				f.fill(k)
				f.fill(e)
				m.SetMapIndex(k, e)
			}
			v.Set(m)
		}
	case reflect.Struct:
		for i := range t.NumField() {
			name := t.String() + "." + t.Field(i).Name
			ft := t.Field(i).Type
			if k := ft.Kind(); (k == reflect.Pointer || k == reflect.Slice || k == reflect.Map) && clonePanics(ft.Elem()) {
				f.skipped[name] = true
			} else if f.fill(v.Field(i)) {
				f.filled[name] = true
			}
		}
	default:
		return false
	}
	return true
}

// clonePanics reports whether clone panics on a value of type t: whether t
// holds, itself or in a struct-typed field at any depth, a struct with an
// unexported field. Pointers, slices and maps are only followed when not nil,
// and arrays are assigned rather than walked.
func clonePanics(t reflect.Type) bool {
	if t.Kind() != reflect.Struct {
		return false
	}
	for i := range t.NumField() {
		if !t.Field(i).IsExported() || clonePanics(t.Field(i).Type) {
			return true
		}
	}
	return false
}

// structFields lists "Type.Field" for every field of every struct type
// reachable from t through pointers, slices, arrays, maps and struct fields.
func structFields(t reflect.Type, seen map[reflect.Type]bool, out map[string]bool) {
	if seen[t] {
		return
	}
	seen[t] = true
	switch t.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.Array:
		structFields(t.Elem(), seen, out)
	case reflect.Map:
		structFields(t.Key(), seen, out)
		structFields(t.Elem(), seen, out)
	case reflect.Struct:
		for i := range t.NumField() {
			out[t.String()+"."+t.Field(i).Name] = true
			structFields(t.Field(i).Type, seen, out)
		}
	}
}

// fullGenerateContentConfig returns a config with every field reachable from
// genai.GenerateContentConfig set, including fields a later genai release
// adds, and fails the test if any field is left unset.
func fullGenerateContentConfig(t *testing.T) *genai.GenerateContentConfig {
	t.Helper()
	f := newFiller()
	cfg := &genai.GenerateContentConfig{}
	f.fill(reflect.ValueOf(cfg).Elem())

	// clone panics on genai.Interval (time.Time has unexported fields);
	// TestCloneGenerateContentConfigPanicsLikeReference covers that path.
	wantSkipped := map[string]bool{"genai.GoogleSearch.TimeRangeFilter": true}
	if !reflect.DeepEqual(f.skipped, wantSkipped) {
		t.Fatalf("fields left nil = %v, want %v", f.skipped, wantSkipped)
	}
	all := map[string]bool{}
	structFields(reflect.TypeFor[genai.GenerateContentConfig](), map[reflect.Type]bool{}, all)
	for name := range all {
		if !f.filled[name] && !f.skipped[name] && !strings.HasPrefix(name, "genai.Interval.") && !strings.HasPrefix(name, "time.") {
			t.Errorf("field %s was not set by the filler", name)
		}
	}
	return cfg
}

// representativeGenerateContentConfigs returns hand-written configs: what
// agents set in practice, plus nil and empty values, nil elements, spare
// capacity and values held in interfaces.
func representativeGenerateContentConfigs() map[string]*genai.GenerateContentConfig {
	stops := make([]string, 2, 8)
	copy(stops, []string{"END", "STOP"})
	return map[string]*genai.GenerateContentConfig{
		"nil":   nil,
		"empty": {},
		"sampling, safety, thinking, labels": {
			Temperature:     genai.Ptr[float32](0.2),
			TopP:            genai.Ptr[float32](0.95),
			MaxOutputTokens: 1024,
			StopSequences:   stops,
			SafetySettings: []*genai.SafetySetting{
				{Category: genai.HarmCategoryHarassment, Threshold: genai.HarmBlockThresholdBlockOnlyHigh},
				nil,
				{Category: genai.HarmCategoryDangerousContent, Method: genai.HarmBlockMethodProbability},
			},
			ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: genai.Ptr[int32](0)},
			Labels:         map[string]string{"team": "adk", "unicode": "żółw ✓"},
		},
		"empty non-nil slices and maps": {
			StopSequences:      []string{},
			SafetySettings:     []*genai.SafetySetting{},
			Tools:              make([]*genai.Tool, 0, 4),
			Labels:             map[string]string{},
			ResponseModalities: []string{},
			ContinuationToken:  []byte{},
			SystemInstruction:  &genai.Content{Parts: []*genai.Part{}},
		},
		"response schema and tools": {
			ResponseMIMEType: "application/json",
			ResponseSchema:   benchGenerateContentConfig(1, 0).Tools[0].FunctionDeclarations[0].Parameters,
			Tools: append(benchGenerateContentConfig(3, 0).Tools, &genai.Tool{
				GoogleSearch:  &genai.GoogleSearch{ExcludeDomains: []string{"example.com"}},
				CodeExecution: &genai.ToolCodeExecution{},
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:                 "raw_schema",
					ParametersJsonSchema: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}},
					Parameters: &genai.Schema{
						Type:       genai.TypeObject,
						Properties: map[string]*genai.Schema{"nil_property": nil},
						AnyOf:      []*genai.Schema{{Type: genai.TypeString}, nil},
						Default:    map[string]any{"q": "x"},
						Example:    []any{"a", 1.0},
						Nullable:   genai.Ptr(true),
					},
				}},
			}),
			ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode:                 genai.FunctionCallingConfigModeAny,
				AllowedFunctionNames: []string{"lookup_order_0"},
			}},
		},
		"system instruction with inline data": {
			SystemInstruction: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
				{Text: "Odpowiadaj po polsku. 🙂"},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: make([]byte, 64<<10, 80<<10), DisplayName: "logo"}},
				{FileData: &genai.FileData{FileURI: "gs://bucket/policy.pdf", MIMEType: "application/pdf"}},
				nil,
				{Text: "thought", Thought: true, ThoughtSignature: []byte("sig")},
			}},
		},
		"speech, http options, json schema": {
			SpeechConfig: &genai.SpeechConfig{
				LanguageCode: "pl-PL",
				MultiSpeakerVoiceConfig: &genai.MultiSpeakerVoiceConfig{SpeakerVoiceConfigs: []*genai.SpeakerVoiceConfig{
					{Speaker: "a", VoiceConfig: &genai.VoiceConfig{PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{VoiceName: "Kore"}}},
					{Speaker: "b", VoiceConfig: &genai.VoiceConfig{ReplicatedVoiceConfig: &genai.ReplicatedVoiceConfig{VoiceSampleAudio: []byte{1, 2, 3}}}},
				}},
			},
			HTTPOptions: &genai.HTTPOptions{
				BaseURL:   "https://example.com",
				Headers:   http.Header{"X-Trace": {"1", "2"}, "X-Nil": nil, "X-Empty": {}},
				Timeout:   genai.Ptr(5 * time.Second),
				ExtraBody: map[string]any{"nested": map[string]any{"k": []any{"v"}}},
			},
			ResponseMIMEType:   "application/json",
			ResponseJsonSchema: map[string]any{"type": "object"},
			ResponseModalities: []string{"TEXT", "AUDIO"},
			MediaResolution:    genai.MediaResolutionHigh,
			ContinuationToken:  []byte("token"),
		},
	}
}

// copyChecker walks a source value and two copies of it in step, ref and got,
// and records every difference between the copies: values, nil-ness, lengths
// and capacities and, unless valuesOnly, which parts share memory with the
// source and whether got's spare slice capacity is zeroed.
type copyChecker struct {
	valuesOnly bool
	diffs      []string
}

func (c *copyChecker) errorf(path, format string, args ...any) {
	c.diffs = append(c.diffs, path+": "+fmt.Sprintf(format, args...))
}

// diffCopies lists the differences between ref and got, two copies of src,
// including differences in what they share with src.
func diffCopies(src, ref, got any) []string {
	c := &copyChecker{}
	c.check("", reflect.ValueOf(src), reflect.ValueOf(ref), reflect.ValueOf(got))
	return c.diffs
}

// diffValues lists the differences in value between want and got.
func diffValues(want, got any) []string {
	c := &copyChecker{valuesOnly: true}
	c.check("", reflect.ValueOf(want), reflect.ValueOf(want), reflect.ValueOf(got))
	return c.diffs
}

// sameRef reports whether a and b, two values of the same type, refer to the
// same memory: the same pointer target, slice backing array or map.
func sameRef(a, b reflect.Value) bool {
	switch a.Kind() {
	case reflect.Pointer:
		return a.Pointer() == b.Pointer() && a.Type().Elem().Size() > 0
	case reflect.Slice:
		return a.Pointer() == b.Pointer() && a.Cap() > 0 && a.Type().Elem().Size() > 0
	case reflect.Map, reflect.Func, reflect.Chan, reflect.UnsafePointer:
		return a.Pointer() == b.Pointer()
	}
	return false
}

func (c *copyChecker) check(path string, src, ref, got reflect.Value) {
	if ref.Type() != got.Type() {
		c.errorf(path, "type %v, want %v", got.Type(), ref.Type())
		return
	}
	shared := func(what string) {
		if !c.valuesOnly && sameRef(src, ref) != sameRef(src, got) {
			c.errorf(path, "shares the %s with the source = %v, want %v", what, sameRef(src, got), sameRef(src, ref))
		}
	}
	switch src.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.Map, reflect.Func, reflect.Chan, reflect.Interface:
		if ref.IsNil() != got.IsNil() {
			c.errorf(path, "nil = %v, want %v", got.IsNil(), ref.IsNil())
			return
		}
		if got.IsNil() {
			return
		}
	}
	switch src.Kind() {
	case reflect.Pointer:
		shared("pointer target")
		c.check(path, src.Elem(), ref.Elem(), got.Elem())
	case reflect.Slice:
		if ref.Len() != got.Len() || ref.Cap() != got.Cap() {
			c.errorf(path, "len, cap = %d, %d, want %d, %d", got.Len(), got.Cap(), ref.Len(), ref.Cap())
			return
		}
		shared("backing array")
		for i := range src.Len() {
			c.check(fmt.Sprintf("%s[%d]", path, i), src.Index(i), ref.Index(i), got.Index(i))
		}
		if c.valuesOnly {
			return
		}
		// A fresh copy has zeros in its spare capacity, as MakeSlice leaves.
		full := got.Slice3(0, got.Cap(), got.Cap())
		for i := got.Len(); i < got.Cap(); i++ {
			if !full.Index(i).IsZero() {
				c.errorf(path, "element %d beyond len is %v, want zero", i, full.Index(i))
			}
		}
	case reflect.Map:
		if ref.Len() != got.Len() {
			c.errorf(path, "len = %d, want %d", got.Len(), ref.Len())
			return
		}
		shared("map")
		for _, k := range src.MapKeys() {
			rv, gv := ref.MapIndex(k), got.MapIndex(k)
			if !gv.IsValid() {
				c.errorf(path, "key %v missing", k)
				continue
			}
			c.check(fmt.Sprintf("%s[%v]", path, k), src.MapIndex(k), rv, gv)
		}
	case reflect.Interface:
		// referenceClone assigns interface values, so what they hold is
		// shared with the source.
		s, r, g := src.Elem(), ref.Elem(), got.Elem()
		if r.Type() != g.Type() {
			c.errorf(path, "holds %v, want %v", g.Type(), r.Type())
			return
		}
		if !c.valuesOnly && sameRef(s, r) != sameRef(s, g) {
			c.errorf(path, "shares the held value with the source = %v, want %v", sameRef(s, g), sameRef(s, r))
		}
		if !reflect.DeepEqual(r.Interface(), g.Interface()) {
			c.errorf(path, "holds %#v, want %#v", g.Interface(), r.Interface())
		}
	case reflect.Func:
		if ref.Pointer() != got.Pointer() {
			c.errorf(path, "func differs")
		}
	case reflect.Struct:
		for i := range src.NumField() {
			c.check(path+"."+src.Type().Field(i).Name, src.Field(i), ref.Field(i), got.Field(i))
		}
	case reflect.Array:
		for i := range src.Len() {
			c.check(fmt.Sprintf("%s[%d]", path, i), src.Index(i), ref.Index(i), got.Index(i))
		}
	default:
		if !ref.Equal(got) {
			c.errorf(path, "= %v, want %v", got, ref)
		}
	}
}

// mutate changes every value reachable from v without going through
// interfaces, which both copies share with the source by design.
func mutate(v reflect.Value) {
	switch v.Kind() {
	case reflect.Bool:
		v.SetBool(!v.Bool())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(v.Int() + 1)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		v.SetUint(v.Uint() + 1)
	case reflect.Float32, reflect.Float64:
		v.SetFloat(v.Float() + 1)
	case reflect.String:
		v.SetString(v.String() + "!")
	case reflect.Pointer:
		if !v.IsNil() {
			mutate(v.Elem())
		}
	case reflect.Slice, reflect.Array:
		for i := range v.Len() {
			mutate(v.Index(i))
		}
		if v.Kind() == reflect.Slice && !v.IsNil() && v.Len() < v.Cap() {
			// Write into the spare capacity, as append would.
			mutate(v.Slice(0, v.Len()+1).Index(v.Len()))
		}
	case reflect.Map:
		if v.IsNil() {
			return
		}
		for _, k := range v.MapKeys() {
			e := reflect.New(v.Type().Elem()).Elem()
			e.Set(v.MapIndex(k))
			if e.Kind() == reflect.Pointer {
				mutate(e) // the value is the copy's own pointer
			} else if e.Kind() != reflect.Interface {
				mutate(e)
				v.SetMapIndex(k, e)
			}
		}
		k := reflect.New(v.Type().Key()).Elem()
		if k.Kind() == reflect.String {
			k.SetString("added by mutate")
			v.SetMapIndex(k, reflect.Zero(v.Type().Elem()))
		}
	case reflect.Struct:
		for i := range v.NumField() {
			mutate(v.Field(i))
		}
	}
}

// checkCloneGenerateContentConfig compares CloneGenerateContentConfig with
// referenceClone on src, then checks that mutating the copy leaves src as it
// was and the other way round.
func checkCloneGenerateContentConfig(t *testing.T, src *genai.GenerateContentConfig) {
	t.Helper()
	ref := referenceClone(src)
	got := CloneGenerateContentConfig(src)
	if (got == nil) != (src == nil) || (got != nil && got == src) {
		t.Fatalf("CloneGenerateContentConfig(%p) = %p", src, got)
	}
	for _, d := range diffCopies(src, ref, got) {
		t.Errorf("copy differs from the reference copy at cfg%s", d)
	}
	if src == nil {
		return
	}

	// Mutating every value of the copy (outside interfaces) leaves src as
	// it was, and mutating src leaves the copy as it was.
	snapshot := referenceClone(src)
	mutate(reflect.ValueOf(got))
	for _, d := range diffValues(snapshot, src) {
		t.Errorf("mutating the copy changed the source at cfg%s", d)
	}
	gotSnapshot := referenceClone(got)
	mutate(reflect.ValueOf(src))
	for _, d := range diffValues(gotSnapshot, got) {
		t.Errorf("mutating the source changed the copy at cfg%s", d)
	}
	if len(diffValues(snapshot, src)) == 0 {
		t.Error("mutate changed nothing, so the checks above are vacuous")
	}
}

func TestCloneGenerateContentConfigMatchesReference(t *testing.T) {
	for name, cfg := range representativeGenerateContentConfigs() {
		t.Run(name, func(t *testing.T) {
			checkCloneGenerateContentConfig(t, cfg)
		})
	}
	t.Run("every field set", func(t *testing.T) {
		checkCloneGenerateContentConfig(t, fullGenerateContentConfig(t))
	})
	for _, n := range []int{0, 5, 20} {
		t.Run(fmt.Sprintf("benchmark config fds=%d", n), func(t *testing.T) {
			checkCloneGenerateContentConfig(t, benchGenerateContentConfig(n, 100<<10))
		})
	}
}

// TestBasicRequestProcessorIsolatesConfig checks through the request
// processor that editing req.Config leaves the agent's GenerateContentConfig
// as it was, and the other way round, for a config with every field set.
func TestBasicRequestProcessorIsolatesConfig(t *testing.T) {
	agentCfg := fullGenerateContentConfig(t)
	base, err := agent.New(agent.Config{Name: "isolation_agent"})
	if err != nil {
		t.Fatal(err)
	}
	a := &mockLLMAgent{Agent: base, s: &State{Model: &mockLLM{name: "m"}, GenerateContentConfig: agentCfg}}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})
	requestConfig := func() *genai.GenerateContentConfig {
		req := &model.LLMRequest{}
		for _, err := range basicRequestProcessor(ctx, req, &Flow{}) {
			if err != nil {
				t.Fatal(err)
			}
		}
		return req.Config
	}

	agentSnapshot := referenceClone(agentCfg)
	mutate(reflect.ValueOf(requestConfig()))
	for _, d := range diffValues(agentSnapshot, agentCfg) {
		t.Errorf("editing req.Config changed the agent's config at cfg%s", d)
	}

	reqCfg := requestConfig()
	reqSnapshot := referenceClone(reqCfg)
	mutate(reflect.ValueOf(agentCfg))
	for _, d := range diffValues(reqSnapshot, reqCfg) {
		t.Errorf("editing the agent's config changed req.Config at cfg%s", d)
	}
}

// TestCloneGenerateContentConfigPanicsLikeReference pins the one input the
// copy cannot handle: genai.Interval holds time.Time, whose fields are
// unexported, so a config with GoogleSearch.TimeRangeFilter set panics with
// the same message as the reference.
func TestCloneGenerateContentConfigPanicsLikeReference(t *testing.T) {
	cfg := &genai.GenerateContentConfig{Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{
		TimeRangeFilter: &genai.Interval{StartTime: time.Unix(0, 0), EndTime: time.Unix(60, 0)},
	}}}}
	recovered := func(f func()) (r any) {
		defer func() { r = recover() }()
		f()
		return nil
	}
	want := recovered(func() { referenceClone(cfg) })
	got := recovered(func() { CloneGenerateContentConfig(cfg) })
	if want == nil || got != want {
		t.Errorf("CloneGenerateContentConfig panicked with %v, want %v", got, want)
	}
}

// TestCloneGenericMatchesReference covers clone's other caller, which copies
// history contents.
func TestCloneGenericMatchesReference(t *testing.T) {
	f := newFiller()
	content := &genai.Content{}
	f.fill(reflect.ValueOf(content).Elem())
	for name, src := range map[string]*genai.Content{
		"nil":                    nil,
		"every field set":        content,
		"inline data and a call": {Role: genai.RoleModel, Parts: []*genai.Part{{InlineData: &genai.Blob{Data: []byte("img")}}, {FunctionCall: &genai.FunctionCall{Name: "f", Args: map[string]any{"a": map[string]any{"b": 1}}}}}},
	} {
		t.Run(name, func(t *testing.T) {
			for _, d := range diffCopies(src, referenceClone(src), clone(src)) {
				t.Errorf("copy differs from the reference copy at content%s", d)
			}
		})
	}
}
