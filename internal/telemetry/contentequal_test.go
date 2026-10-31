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

package telemetry

import (
	"math"
	"math/rand/v2"
	"reflect"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// referenceContentEqual is the definition contentEqual implements.
func referenceContentEqual(a, b []contentAttr) bool {
	return reflect.DeepEqual(a, b)
}

// checkContentEqual fails t unless contentEqual agrees with
// referenceContentEqual on (a, b) and on (b, a), and returns the answer.
func checkContentEqual(t *testing.T, name string, a, b []contentAttr) bool {
	t.Helper()
	want := referenceContentEqual(a, b)
	if got := contentEqual(a, b); got != want {
		t.Errorf("%s: contentEqual(a, b) = %v, reflect.DeepEqual = %v", name, got, want)
	}
	if got, want := contentEqual(b, a), referenceContentEqual(b, a); got != want {
		t.Errorf("%s: contentEqual(b, a) = %v, reflect.DeepEqual = %v", name, got, want)
	}
	return want
}

// Values requestContent never builds, which contentEqual leaves to
// reflect.DeepEqual.
type (
	floatPart  float64        // NaN is not equal to itself.
	mapPart    map[string]any // Values of mixed types.
	funcPart   func()         // Equal only when both are nil.
	floatBox   struct{ F float64 }
	structPart struct {
		F float64
		M map[string]any
		P *string
	}
)

func (floatPart) value() attribute.Value  { return attribute.Value{} }
func (mapPart) value() attribute.Value    { return attribute.Value{} }
func (funcPart) value() attribute.Value   { return attribute.Value{} }
func (floatBox) value() attribute.Value   { return attribute.Value{} }
func (structPart) value() attribute.Value { return attribute.Value{} }

func msgs(ms ...chatMessage) []contentAttr {
	return []contentAttr{{genAIInputMessages, messageList(ms)}}
}

func parts(ps ...contentValue) []contentAttr {
	return msgs(chatMessage{Role: roleUser, Parts: partList(ps)})
}

func strPtr(s string) *string { return &s }

func TestContentEqualEdgeCases(t *testing.T) {
	nan := math.NaN()
	sharedText := &textPart{Type: partTypeText, Content: "shared"}
	sharedNaN := &floatBox{F: nan}
	sharedMap := mapPart{"f": nan}
	sameParts := partList{floatPart(nan)}
	sameMessages := messageList{{Role: roleUser, Parts: partList{floatPart(nan)}}}
	sameContent := []contentAttr{{genAIInputMessages, sameMessages}}
	selfA := partList{nil, textPart{Type: partTypeText, Content: "a"}}
	selfA[0] = selfA
	selfB := partList{nil, textPart{Type: partTypeText, Content: "a"}}
	selfB[0] = selfB
	selfC := partList{nil, textPart{Type: partTypeText, Content: "c"}}
	selfC[0] = selfC
	mixed := func() mapPart {
		return mapPart{"s": "x", "i": 1, "f": 1.5, "b": true, "n": nil, "l": []any{"y", 2, nil}, "m": map[string]any{"k": []byte("z")}}
	}
	mixedEdited := mixed()
	mixedEdited["m"] = map[string]any{"k": []byte("Z")}

	tests := []struct {
		name string
		a, b []contentAttr
		want bool
	}{
		{name: "nil and nil", a: nil, b: nil, want: true},
		{name: "nil and empty", a: nil, b: []contentAttr{}, want: false},
		{name: "empty and empty", a: []contentAttr{}, b: []contentAttr{}, want: true},
		{name: "keys differ", a: []contentAttr{{genAIInputMessages, messageList{}}}, b: []contentAttr{{genAIOutputMessages, messageList{}}}, want: false},
		{name: "nil value and nil value", a: []contentAttr{{genAIInputMessages, nil}}, b: []contentAttr{{genAIInputMessages, nil}}, want: true},
		{name: "nil value and nil message list", a: []contentAttr{{genAIInputMessages, nil}}, b: []contentAttr{{genAIInputMessages, messageList(nil)}}, want: false},
		{name: "nil message list and empty", a: []contentAttr{{genAIInputMessages, messageList(nil)}}, b: []contentAttr{{genAIInputMessages, messageList{}}}, want: false},
		{name: "message list and part list", a: []contentAttr{{genAIInputMessages, messageList{}}}, b: []contentAttr{{genAIInputMessages, partList{}}}, want: false},
		{name: "nil parts and empty", a: msgs(chatMessage{Role: roleUser}), b: msgs(chatMessage{Role: roleUser, Parts: partList{}}), want: false},
		{name: "roles differ", a: msgs(chatMessage{Role: roleUser, Parts: partList{}}), b: msgs(chatMessage{Role: roleTool, Parts: partList{}}), want: false},
		{name: "finish reason nil and empty", a: msgs(chatMessage{Parts: partList{}}), b: msgs(chatMessage{Parts: partList{}, FinishReason: strPtr("")}), want: false},
		{name: "finish reasons equal at two addresses", a: msgs(chatMessage{FinishReason: strPtr(finishStop)}), b: msgs(chatMessage{FinishReason: strPtr(finishStop)}), want: true},
		{name: "finish reasons differ", a: msgs(chatMessage{FinishReason: strPtr(finishStop)}), b: msgs(chatMessage{FinishReason: strPtr(finishLength)}), want: false},
		{name: "text and reasoning", a: parts(textPart{partTypeText, "x"}), b: parts(textPart{partTypeReasoning, "x"}), want: false},
		{name: "unicode text", a: parts(textPart{partTypeText, "żółw 🐢"}), b: parts(textPart{partTypeText, strings.Clone("żółw 🐢")}), want: true},
		{name: "invalid UTF-8 differs", a: parts(textPart{partTypeText, "\xff"}), b: parts(textPart{partTypeText, "\xfe"}), want: false},
		{name: "same strings, other part type", a: parts(textPart{"t", "c"}), b: parts(uriPart{Type: "t", URI: "c"}), want: false},
		{name: "part and pointer to it", a: parts(textPart{"t", "c"}), b: parts(&textPart{"t", "c"}), want: false},
		{name: "arguments nil and empty", a: parts(toolCallPart{Type: partTypeToolCall}), b: parts(toolCallPart{Type: partTypeToolCall, Arguments: rawJSON{}}), want: false},
		{name: "arguments equal in two arrays", a: parts(toolCallPart{Name: "f", Arguments: rawJSON(`{"q":1}`)}), b: parts(toolCallPart{Name: "f", Arguments: rawJSON(`{"q":1}`)}), want: true},
		{name: "arguments differ", a: parts(toolCallPart{Name: "f", Arguments: rawJSON(`{"q":1}`)}), b: parts(toolCallPart{Name: "f", Arguments: rawJSON(`{"q":2}`)}), want: false},
		{name: "responses nil and empty", a: parts(toolResponsePart{Response: nil}), b: parts(toolResponsePart{Response: rawJSON{}}), want: false},
		{name: "responses differ in length", a: parts(toolResponsePart{Response: rawJSON("null")}), b: parts(toolResponsePart{Response: rawJSON("nul")}), want: false},
		{name: "blob truncated differs", a: parts(blobPart{Content: "AA=="}), b: parts(blobPart{Content: "AA==", Truncated: true}), want: false},
		{name: "nil part and nil part", a: parts(nil), b: parts(nil), want: true},
		{name: "nil part and part", a: parts(nil), b: parts(textPart{}), want: false},
		{name: "shared pointer", a: parts(sharedText), b: parts(sharedText), want: true},
		{name: "equal pointees", a: parts(&textPart{"t", "c"}), b: parts(&textPart{"t", "c"}), want: true},
		{name: "NaN", a: parts(floatPart(nan)), b: parts(floatPart(nan)), want: false},
		{name: "shared pointer to NaN", a: parts(sharedNaN), b: parts(sharedNaN), want: true},
		{name: "NaN at two addresses", a: parts(&floatBox{F: nan}), b: parts(&floatBox{F: nan}), want: false},
		{name: "same part list holding NaN", a: msgs(chatMessage{Parts: sameParts}), b: msgs(chatMessage{Parts: sameParts}), want: true},
		{name: "same message list holding NaN", a: []contentAttr{{genAIInputMessages, sameMessages}}, b: []contentAttr{{genAIInputMessages, sameMessages}}, want: true},
		{name: "same content holding NaN", a: sameContent, b: sameContent, want: true},
		{name: "shorter view of the same content", a: sameContent, b: sameContent[:0], want: false},
		{name: "same map holding NaN", a: parts(sharedMap), b: parts(sharedMap), want: true},
		{name: "map values of mixed types", a: parts(mixed()), b: parts(mixed()), want: true},
		{name: "map values of mixed types differ deep down", a: parts(mixed()), b: parts(mixedEdited), want: false},
		{name: "nil funcs", a: parts(funcPart(nil)), b: parts(funcPart(nil)), want: true},
		{name: "func", a: parts(funcPart(t.Fail)), b: parts(funcPart(t.Fail)), want: false},
		{name: "struct with a map and a pointer", a: parts(structPart{M: map[string]any{"a": 1}, P: strPtr("p")}), b: parts(structPart{M: map[string]any{"a": 1}, P: strPtr("p")}), want: true},
		{name: "struct pointers differ", a: parts(structPart{P: strPtr("p")}), b: parts(structPart{P: strPtr("q")}), want: false},
		{name: "raw JSON as a part", a: parts(rawJSON(nil)), b: parts(rawJSON{}), want: false},
		{name: "nested part lists", a: parts(partList{textPart{"t", "c"}}), b: parts(partList{textPart{"t", "c"}}), want: true},
		{name: "nested part lists differ", a: parts(partList{textPart{"t", "c"}}), b: parts(partList{textPart{"t", "d"}}), want: false},
		{name: "message as a part", a: parts(chatMessage{Role: "r"}), b: parts(chatMessage{Role: "r"}), want: true},
		{name: "part list holding itself, twice", a: parts(selfA), b: parts(selfB), want: true},
		{name: "part lists holding themselves differ", a: parts(selfA), b: parts(selfC), want: false},
		{name: "system instructions", a: []contentAttr{{genAISystemInstructions, partList{textPart{"t", "c"}}}}, b: []contentAttr{{genAISystemInstructions, partList{textPart{"t", "c"}}}}, want: true},
	}
	for _, tc := range tests {
		if got := checkContentEqual(t, tc.name, tc.a, tc.b); got != tc.want {
			t.Errorf("%s: reflect.DeepEqual = %v, want %v (the test case is wrong)", tc.name, got, tc.want)
		}
	}
}

// contentGen generates the values requestContent builds, from small domains
// so that generated values are often equal, and, when odd is set, values it
// never builds.
type contentGen struct {
	r   *rand.Rand
	odd bool
}

var genStrings = []string{"", "a", "b", "é", "\xff"}

func (g *contentGen) str() string { return genStrings[g.r.IntN(len(genStrings))] }

func (g *contentGen) float() float64 {
	return []float64{0, math.Copysign(0, -1), 1, math.Inf(1), math.NaN()}[g.r.IntN(5)]
}

func (g *contentGen) raw() rawJSON {
	switch g.r.IntN(5) {
	case 0:
		return nil
	case 1:
		return rawJSON{}
	case 2:
		return rawJSON("null")
	case 3:
		return rawJSON(`{"q":1}`)
	default:
		return rawJSON(`{"q":"é"}`)
	}
}

func (g *contentGen) content() []contentAttr {
	switch g.r.IntN(10) {
	case 0:
		return nil
	case 1:
		return []contentAttr{}
	}
	keys := []attribute.Key{genAISystemInstructions, genAIInputMessages, genAIOutputMessages}
	out := make([]contentAttr, g.r.IntN(3)+1)
	for i := range out {
		out[i].key = keys[g.r.IntN(len(keys))]
		switch n := g.r.IntN(6); {
		case n == 0:
			out[i].value = g.parts(0)
		case n == 1 && g.odd:
			out[i].value = g.oddValue(0)
		default:
			out[i].value = g.messages(0)
		}
	}
	return out
}

func (g *contentGen) messages(depth int) messageList {
	switch g.r.IntN(10) {
	case 0:
		return nil
	case 1:
		return messageList{}
	}
	out := make(messageList, g.r.IntN(4)+1)
	for i := range out {
		out[i] = g.message(depth)
	}
	return out
}

func (g *contentGen) message(depth int) chatMessage {
	m := chatMessage{Role: g.str(), Parts: g.parts(depth)}
	if g.odd && g.r.IntN(3) == 0 {
		m.FinishReason = strPtr(g.str())
	}
	return m
}

func (g *contentGen) parts(depth int) partList {
	switch g.r.IntN(10) {
	case 0:
		return nil
	case 1:
		return partList{}
	}
	out := make(partList, g.r.IntN(4)+1)
	for i := range out {
		out[i] = g.part(depth)
	}
	return out
}

func (g *contentGen) part(depth int) contentValue {
	n := 5
	if g.odd {
		n = 7
	}
	switch g.r.IntN(n) {
	case 0:
		return textPart{Type: g.str(), Content: g.str()}
	case 1:
		return toolCallPart{Type: g.str(), ID: g.str(), Name: g.str(), Arguments: g.raw()}
	case 2:
		return toolResponsePart{Type: g.str(), ID: g.str(), Name: g.str(), Response: g.raw()}
	case 3:
		return blobPart{Type: g.str(), MIMEType: g.str(), Modality: g.str(), Content: g.str(), Truncated: g.r.IntN(2) == 0}
	case 4:
		return uriPart{Type: g.str(), MIMEType: g.str(), Modality: g.str(), URI: g.str()}
	default:
		return g.oddValue(depth)
	}
}

// oddValue is a value of a type requestContent never puts where contentEqual
// meets it.
func (g *contentGen) oddValue(depth int) contentValue {
	if depth >= 2 {
		return floatPart(g.float())
	}
	switch g.r.IntN(11) {
	case 0:
		return nil
	case 1:
		return &textPart{Type: g.str(), Content: g.str()}
	case 2:
		return floatPart(g.float())
	case 3:
		return &floatBox{F: g.float()}
	case 4:
		return mapPart{"s": g.str(), "f": g.float(), "n": nil, "l": []any{g.str(), int64(g.r.IntN(2))}}
	case 5:
		if g.r.IntN(2) == 0 {
			return funcPart(nil)
		}
		return funcPart(func() {})
	case 6:
		return structPart{F: g.float(), M: map[string]any{"k": g.str()}, P: strPtr(g.str())}
	case 7:
		return g.raw()
	case 8:
		return g.parts(depth + 1)
	case 9:
		return g.messages(depth + 1)
	default:
		return g.message(depth + 1)
	}
}

// clone copies c into new backing arrays, so equality has to be found element
// by element, except that with share set it keeps some lists as they are.
// Pointers are kept or copied at random.
func (g *contentGen) clone(c []contentAttr, share bool) []contentAttr {
	if c == nil {
		return nil
	}
	out := make([]contentAttr, len(c))
	for i, a := range c {
		out[i] = contentAttr{a.key, g.cloneValue(a.value, share)}
	}
	return out
}

func (g *contentGen) cloneValue(v contentValue, share bool) contentValue {
	switch v := v.(type) {
	case messageList:
		if v == nil || share && g.r.IntN(3) == 0 {
			return v
		}
		out := make(messageList, len(v))
		for i, m := range v {
			out[i] = g.cloneMessage(m, share)
		}
		return out
	case partList:
		return g.cloneParts(v, share)
	case chatMessage:
		return g.cloneMessage(v, share)
	case toolCallPart:
		v.Arguments = cloneRaw(v.Arguments)
		return v
	case toolResponsePart:
		v.Response = cloneRaw(v.Response)
		return v
	case rawJSON:
		return cloneRaw(v)
	case *textPart:
		if g.r.IntN(2) == 0 {
			return v
		}
		p := *v
		return &p
	case *floatBox:
		if g.r.IntN(2) == 0 {
			return v
		}
		p := *v
		return &p
	case mapPart:
		if g.r.IntN(2) == 0 {
			return v
		}
		out := make(mapPart, len(v))
		for k, x := range v {
			out[k] = x
		}
		return out
	case structPart:
		if g.r.IntN(2) == 0 {
			v.P = strPtr(*v.P)
		}
		return v
	default:
		return v
	}
}

func (g *contentGen) cloneMessage(m chatMessage, share bool) chatMessage {
	m.Parts = g.cloneParts(m.Parts, share)
	if m.FinishReason != nil && g.r.IntN(2) == 0 {
		m.FinishReason = strPtr(*m.FinishReason)
	}
	return m
}

func (g *contentGen) cloneParts(p partList, share bool) partList {
	if p == nil || share && g.r.IntN(3) == 0 {
		return p
	}
	out := make(partList, len(p))
	for i, v := range p {
		out[i] = g.cloneValue(v, share)
	}
	return out
}

func cloneRaw(r rawJSON) rawJSON {
	if r == nil {
		return nil
	}
	return append(rawJSON{}, r...)
}

// mutate changes one thing at one level of c, which must come from clone
// without share.
func (g *contentGen) mutate(c []contentAttr) []contentAttr {
	if len(c) == 0 || g.r.IntN(5) == 0 {
		return reshape(g.r, c)
	}
	i := g.r.IntN(len(c))
	if g.r.IntN(4) == 0 {
		c[i].key += "x"
	} else {
		c[i].value = g.mutateValue(c[i].value)
	}
	return c
}

func (g *contentGen) mutateValue(v contentValue) contentValue {
	if g.r.IntN(8) == 0 {
		// Most likely of another type.
		return g.part(0)
	}
	switch v := v.(type) {
	case messageList:
		if len(v) == 0 || g.r.IntN(5) == 0 {
			return reshape(g.r, v)
		}
		i := g.r.IntN(len(v))
		v[i] = g.mutateMessage(v[i])
		return v
	case partList:
		return g.mutateParts(v)
	case chatMessage:
		return g.mutateMessage(v)
	case textPart:
		if g.r.IntN(2) == 0 {
			v.Type += "x"
		} else {
			v.Content += "x"
		}
		return v
	case toolCallPart:
		switch g.r.IntN(4) {
		case 0:
			v.Type += "x"
		case 1:
			v.ID += "x"
		case 2:
			v.Name += "x"
		default:
			v.Arguments = g.mutateRaw(v.Arguments)
		}
		return v
	case toolResponsePart:
		switch g.r.IntN(4) {
		case 0:
			v.Type += "x"
		case 1:
			v.ID += "x"
		case 2:
			v.Name += "x"
		default:
			v.Response = g.mutateRaw(v.Response)
		}
		return v
	case blobPart:
		switch g.r.IntN(5) {
		case 0:
			v.Type += "x"
		case 1:
			v.MIMEType += "x"
		case 2:
			v.Modality += "x"
		case 3:
			v.Content += "x"
		default:
			v.Truncated = !v.Truncated
		}
		return v
	case uriPart:
		switch g.r.IntN(4) {
		case 0:
			v.Type += "x"
		case 1:
			v.MIMEType += "x"
		case 2:
			v.Modality += "x"
		default:
			v.URI += "x"
		}
		return v
	case rawJSON:
		return g.mutateRaw(v)
	case floatPart:
		return floatPart(g.float())
	case *floatBox:
		return &floatBox{F: g.float()}
	case *textPart:
		return &textPart{Type: v.Type + "x", Content: v.Content}
	default:
		return g.part(0)
	}
}

func (g *contentGen) mutateMessage(m chatMessage) chatMessage {
	switch g.r.IntN(3) {
	case 0:
		m.Role += "x"
	case 1:
		m.Parts = g.mutateParts(m.Parts)
	default:
		switch {
		case m.FinishReason == nil:
			m.FinishReason = strPtr("")
		case g.r.IntN(2) == 0:
			m.FinishReason = nil
		default:
			m.FinishReason = strPtr(*m.FinishReason + "x")
		}
	}
	return m
}

func (g *contentGen) mutateParts(p partList) partList {
	if len(p) == 0 || g.r.IntN(5) == 0 {
		return reshape(g.r, p)
	}
	i := g.r.IntN(len(p))
	p[i] = g.mutateValue(p[i])
	return p
}

func (g *contentGen) mutateRaw(r rawJSON) rawJSON {
	switch {
	case r == nil:
		return rawJSON{}
	case len(r) == 0:
		return nil
	case g.r.IntN(2) == 0:
		return r[:len(r)-1]
	default:
		r[len(r)-1] ^= 1
		return r
	}
}

// reshape turns nil into empty and empty into nil, and otherwise drops or
// repeats an element.
func reshape[S ~[]E, E any](r *rand.Rand, s S) S {
	switch {
	case s == nil:
		return S{}
	case len(s) == 0:
		return nil
	case r.IntN(2) == 0:
		return s[:len(s)-1]
	default:
		return append(s[:len(s):len(s)], s[0])
	}
}

// TestContentEqualMatchesDeepEqual checks contentEqual against reflect.DeepEqual
// on generated pairs: independent values, copies, copies sharing some lists,
// copies with one change at any depth, and a value with itself.
func TestContentEqualMatchesDeepEqual(t *testing.T) {
	const pairs = 20000
	for _, odd := range []bool{false, true} {
		seed := uint64(2)
		if odd {
			seed = 3
		}
		g := &contentGen{r: rand.New(rand.NewPCG(1, seed)), odd: odd}
		kinds := []struct {
			name string
			pair func() ([]contentAttr, []contentAttr)
			// minEqual and minUnequal are the least share of pairs, in
			// percent, that must come out equal and unequal.
			minEqual, minUnequal int
		}{
			{"independent", func() ([]contentAttr, []contentAttr) { return g.content(), g.content() }, 1, 50},
			{"copy", func() ([]contentAttr, []contentAttr) { a := g.content(); return a, g.clone(a, false) }, 50, 0},
			{"copy sharing lists", func() ([]contentAttr, []contentAttr) { a := g.content(); return a, g.clone(a, true) }, 50, 0},
			{"copy with one change", func() ([]contentAttr, []contentAttr) { a := g.content(); return a, g.mutate(g.clone(a, false)) }, 0, 80},
			{"itself", func() ([]contentAttr, []contentAttr) { a := g.content(); return a, a }, 100, 0},
		}
		for _, kind := range kinds {
			equal := 0
			for range pairs {
				a, b := kind.pair()
				if checkContentEqual(t, kind.name, a, b) {
					equal++
				}
				if t.Failed() {
					t.Fatalf("a: %#v\nb: %#v", a, b)
				}
			}
			t.Logf("odd=%v %s: %d of %d pairs equal", odd, kind.name, equal, pairs)
			if !odd && strings.HasPrefix(kind.name, "copy") && kind.name != "copy with one change" && equal != pairs {
				t.Errorf("odd=%v %s: %d of %d pairs equal, want all", odd, kind.name, equal, pairs)
			}
			if equal*100 < kind.minEqual*pairs || (pairs-equal)*100 < kind.minUnequal*pairs {
				t.Errorf("odd=%v %s: %d of %d pairs equal, want at least %d%% equal and %d%% unequal", odd, kind.name, equal, pairs, kind.minEqual, kind.minUnequal)
			}
		}
	}
}

// TestContentEqualOnRequestContent checks contentEqual against
// reflect.DeepEqual on what requestContent builds from representative
// requests: each against a second conversion of itself, and against the
// conversion of every other request.
func TestContentEqualOnRequestContent(t *testing.T) {
	big := strings.Repeat("x", 100_000)
	requests := map[string]func() *model.LLMRequest{
		"nil":                func() *model.LLMRequest { return nil },
		"empty":              func() *model.LLMRequest { return &model.LLMRequest{} },
		"no contents":        func() *model.LLMRequest { return &model.LLMRequest{Contents: []*genai.Content{}} },
		"nil content":        func() *model.LLMRequest { return &model.LLMRequest{Contents: []*genai.Content{nil}} },
		"no parts":           func() *model.LLMRequest { return &model.LLMRequest{Contents: []*genai.Content{{Role: genai.RoleUser}}} },
		"text":               func() *model.LLMRequest { return textRequest("hello", genai.RoleUser) },
		"text by model":      func() *model.LLMRequest { return textRequest("hello", genai.RoleModel) },
		"other text":         func() *model.LLMRequest { return textRequest("hellp", genai.RoleUser) },
		"unicode":            func() *model.LLMRequest { return textRequest("żółw 🐢", genai.RoleUser) },
		"invalid UTF-8":      func() *model.LLMRequest { return textRequest("a\xffb", genai.RoleUser) },
		"large text":         func() *model.LLMRequest { return textRequest(big, genai.RoleUser) },
		"large text, edited": func() *model.LLMRequest { return textRequest(big[1:]+"y", genai.RoleUser) },
		"thought": func() *model.LLMRequest {
			return partsRequest(genai.RoleModel, &genai.Part{Text: "hello", Thought: true})
		},
		"unrepresentable parts": func() *model.LLMRequest {
			return partsRequest(genai.RoleModel, &genai.Part{}, nil, &genai.Part{ExecutableCode: &genai.ExecutableCode{Code: "1"}})
		},
		"system instruction": func() *model.LLMRequest {
			req := textRequest("hello", genai.RoleUser)
			req.Config = &genai.GenerateContentConfig{SystemInstruction: genai.NewContentFromText("be terse", genai.RoleUser)}
			return req
		},
		"empty system instruction": func() *model.LLMRequest {
			req := textRequest("hello", genai.RoleUser)
			req.Config = &genai.GenerateContentConfig{SystemInstruction: &genai.Content{}}
			return req
		},
		"tool call": func() *model.LLMRequest {
			return partsRequest(genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "f", Args: map[string]any{"q": "w", "n": 1}}})
		},
		"tool call, other args": func() *model.LLMRequest {
			return partsRequest(genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "f", Args: map[string]any{"q": "w", "n": 2}}})
		},
		"tool call, nil args": func() *model.LLMRequest {
			return partsRequest(genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "f"}})
		},
		"tool call, empty args": func() *model.LLMRequest {
			return partsRequest(genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "f", Args: map[string]any{}}})
		},
		"tool call, unserializable args": func() *model.LLMRequest {
			return partsRequest(genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "f", Args: map[string]any{"x": math.NaN()}}})
		},
		"tool response": func() *model.LLMRequest {
			return partsRequest(genai.RoleUser, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "f", Response: map[string]any{"result": big}}})
		},
		"tool response, nil": func() *model.LLMRequest {
			return partsRequest(genai.RoleUser, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "f"}})
		},
		"inline data": func() *model.LLMRequest {
			return partsRequest(genai.RoleUser, &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}}})
		},
		"inline data, empty": func() *model.LLMRequest {
			return partsRequest(genai.RoleUser, &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png"}})
		},
		"inline data, cut": func() *model.LLMRequest {
			return partsRequest(genai.RoleUser, &genai.Part{InlineData: &genai.Blob{MIMEType: "audio/wav", Data: make([]byte, maxInlineDataBytes+1)}})
		},
		"file data": func() *model.LLMRequest {
			return partsRequest(genai.RoleUser, &genai.Part{FileData: &genai.FileData{MIMEType: "video/mp4", FileURI: "gs://b/o"}})
		},
		"conversation": func() *model.LLMRequest { return benchRequest(40) },
		"longer conversation": func() *model.LLMRequest {
			return benchRequest(41)
		},
	}
	for name, build := range requests {
		a, b := requestContent(build()), requestContent(build())
		if !checkContentEqual(t, name+" with itself", a, b) {
			t.Errorf("%s: two conversions of one request are not deeply equal", name)
		}
		for other, buildOther := range requests {
			if other != name {
				checkContentEqual(t, name+" with "+other, a, requestContent(buildOther()))
			}
		}
	}
}

func textRequest(text, role string) *model.LLMRequest {
	return &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText(text, genai.Role(role))}}
}

func partsRequest(role string, ps ...*genai.Part) *model.LLMRequest {
	return &model.LLMRequest{Contents: []*genai.Content{{Role: role, Parts: ps}}}
}

// TestContentEqualCoversEveryField changes each field of the structs that
// contentEqual compares, in turn, and checks it notices, so that a field added
// to one of them cannot go uncompared.
func TestContentEqualCoversEveryField(t *testing.T) {
	if got := reflect.TypeFor[contentAttr]().NumField(); got != 2 {
		t.Errorf("contentAttr has %d fields; contentEqual compares key and value only", got)
	}
	full := []contentValue{
		textPart{Type: "t", Content: "c"},
		toolCallPart{Type: "t", ID: "i", Name: "n", Arguments: rawJSON(`{}`)},
		toolResponsePart{Type: "t", ID: "i", Name: "n", Response: rawJSON(`{}`)},
		blobPart{Type: "t", MIMEType: "m", Modality: "o", Content: "c", Truncated: true},
		uriPart{Type: "t", MIMEType: "m", Modality: "o", URI: "u"},
		chatMessage{Role: "r", Parts: partList{textPart{}}, FinishReason: strPtr("f")},
	}
	for _, v := range full {
		typ := reflect.TypeOf(v)
		for i := range typ.NumField() {
			f := typ.Field(i)
			changed := reflect.New(typ).Elem()
			changed.Set(reflect.ValueOf(v))
			fv := changed.Field(i)
			switch f.Type.Kind() {
			case reflect.String:
				fv.SetString(fv.String() + "x")
			case reflect.Bool:
				fv.SetBool(!fv.Bool())
			case reflect.Pointer:
				fv.Set(reflect.New(f.Type.Elem()))
			case reflect.Slice:
				fv.Set(reflect.Append(fv, reflect.Zero(f.Type.Elem())))
			default:
				t.Fatalf("%s.%s has kind %s, which this test does not know how to change", typ.Name(), f.Name, f.Type.Kind())
			}
			wrap := func(v contentValue) []contentAttr {
				if m, ok := v.(chatMessage); ok {
					return msgs(m)
				}
				return parts(v)
			}
			a, b := wrap(v), wrap(changed.Interface().(contentValue))
			if contentEqual(a, b) {
				t.Errorf("contentEqual ignores %s.%s", typ.Name(), f.Name)
			}
			checkContentEqual(t, typ.Name()+"."+f.Name, a, b)
		}
	}
}
