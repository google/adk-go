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

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/google/cel-go/cel"
	celast "github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/ext"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
)

// celCostLimit bounds the work one expression may do, so a document cannot
// stall a run with an expensive comprehension.
const celCostLimit = 1_000_000

// newCELEnv declares the variables every cel node expression can read:
//
//	state       session state before the node runs
//	session     {id, app_name, user_id}
//	user        {content}: the text of the user's message
//	invocation  {id, agent_name}
//	input       the node's input: its predecessor's output
//	vars        the node's vars evaluated so far
var newCELEnv = sync.OnceValues(func() (*cel.Env, error) {
	mapType := cel.MapType(cel.StringType, cel.DynType)
	return cel.NewEnv(
		cel.OptionalTypes(),
		ext.Strings(),
		ext.Lists(),
		ext.Math(),
		ext.Sets(),
		ext.Encoders(),
		ext.Bindings(),
		cel.Variable("state", mapType),
		cel.Variable("session", mapType),
		cel.Variable("user", mapType),
		cel.Variable("invocation", mapType),
		cel.Variable("input", cel.DynType),
		cel.Variable("vars", mapType),
	)
})

// expression is a compiled CEL program.
type expression struct {
	program cel.Program
	// varRefs are the names the expression reads from vars.
	varRefs []string
}

// compileCEL type-checks source. A non-nil want is the type its result must
// have; a dyn result passes, and is checked again when evaluated.
func compileCEL(source string, want *cel.Type) (*expression, error) {
	env, err := newCELEnv()
	if err != nil {
		return nil, fmt.Errorf("create CEL environment: %w", err)
	}
	ast, issues := env.Compile(source)
	if issues != nil && issues.Err() != nil {
		return nil, issues.Err()
	}
	if out := ast.OutputType(); want != nil && !out.IsExactType(want) && !out.IsExactType(cel.DynType) {
		return nil, fmt.Errorf("expression has type %s, want %s", out, want)
	}
	refs, err := varRefs(ast.NativeRep().Expr())
	if err != nil {
		return nil, err
	}
	program, err := env.Program(ast, cel.CostLimit(celCostLimit))
	if err != nil {
		return nil, err
	}
	return &expression{program: program, varRefs: refs}, nil
}

// varRefs returns the names read from the vars variable, as vars.x,
// vars.?x, vars['x'] or vars[?'x']. A name computed at run time cannot be
// checked against the declared vars, so it is an error.
func varRefs(e celast.Expr) ([]string, error) {
	var refs []string
	var err error
	isVars := func(e celast.Expr) bool { return e.Kind() == celast.IdentKind && e.AsIdent() == "vars" }
	celast.PreOrderVisit(e, celast.NewExprVisitor(func(e celast.Expr) {
		switch e.Kind() {
		case celast.SelectKind:
			if sel := e.AsSelect(); isVars(sel.Operand()) {
				refs = append(refs, sel.FieldName())
			}
		case celast.CallKind:
			call := e.AsCall()
			switch call.FunctionName() {
			case operators.Index, operators.OptIndex, operators.OptSelect:
			default:
				return
			}
			if args := call.Args(); len(args) == 2 && isVars(args[0]) {
				name, ok := types.String(""), false
				if args[1].Kind() == celast.LiteralKind {
					name, ok = args[1].AsLiteral().(types.String)
				}
				if !ok {
					err = errors.New("vars must be read by a constant name")
					return
				}
				refs = append(refs, string(name))
			}
		}
	}))
	return refs, err
}

// errEval reports a failed evaluation. cel-go's error text can quote the
// values involved, which may be user or model data, so it is not passed on.
var errEval = errors.New("CEL evaluation failed")

func (e *expression) eval(activation map[string]any) (any, error) {
	val, _, err := e.program.Eval(activation)
	if err != nil {
		return nil, errEval
	}
	return nativeValue(val)
}

func (e *expression) evalBool(activation map[string]any) (bool, error) {
	out, err := e.eval(activation)
	if err != nil {
		return false, err
	}
	b, ok := out.(bool)
	if !ok {
		return false, fmt.Errorf("expression returned %T, want bool", out)
	}
	return b, nil
}

func (e *expression) evalString(activation map[string]any) (string, error) {
	out, err := e.eval(activation)
	if err != nil {
		return "", err
	}
	s, ok := out.(string)
	if !ok {
		return "", fmt.Errorf("expression returned %T, want string", out)
	}
	return s, nil
}

// contextVars binds the ADK context to the CEL variables shared by every
// expression. input and vars are bound by the caller.
func contextVars(ctx agent.ReadonlyContext) map[string]any {
	// Convert each key on its own, so one value that cannot be converted does
	// not take the rest of the state with it.
	state := map[string]any{}
	for k, v := range ctx.ReadonlyState().All() {
		v = plainValue(v)
		// plainValue leaves a double alone, but in state a whole one is as
		// ambiguous as the nested numbers it does convert: a session service
		// that stores state as JSON returns 2 as 2.0.
		if f, ok := v.(float64); ok && f == math.Trunc(f) && f >= math.MinInt64 && f < math.MaxInt64 {
			v = int64(f)
		}
		state[k] = v
	}
	return map[string]any{
		"state":      state,
		"session":    map[string]any{"id": ctx.SessionID(), "app_name": ctx.AppName(), "user_id": ctx.UserID()},
		"user":       map[string]any{"content": contentText(ctx.UserContent())},
		"invocation": map[string]any{"id": ctx.InvocationID(), "agent_name": ctx.AgentName()},
	}
}

// plainValue converts a runtime value into JSON-shaped Go values, so CEL sees
// maps, lists, strings, bools, int64 and float64 whatever produced it. A whole
// number nested in a map or list becomes an int64 even if it was a float64,
// because a session service that stores state as JSON cannot tell 2 from 2.0;
// an expression that needs a double writes double(state.x). A value that is
// itself a float64 stays one, so a double one node outputs is a double in the
// next.
func plainValue(v any) any {
	switch t := v.(type) {
	case nil, string, bool, int64, float64:
		return t
	case *genai.Content:
		return contentText(t)
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprint(v)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		return fmt.Sprint(v)
	}
	return normalizeNumbers(out)
}

func normalizeNumbers(v any) any {
	switch t := v.(type) {
	case json.Number:
		if i, err := t.Int64(); err == nil {
			return i
		}
		f, _ := t.Float64()
		return f
	case map[string]any:
		for k, item := range t {
			t[k] = normalizeNumbers(item)
		}
	case []any:
		for i, item := range t {
			t[i] = normalizeNumbers(item)
		}
	}
	return v
}

func contentText(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		if p != nil {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// nativeValue converts a CEL result into plain Go values.
func nativeValue(val ref.Val) (any, error) {
	switch v := val.(type) {
	case types.Null:
		return nil, nil
	case types.String:
		return string(v), nil
	case types.Bool:
		return bool(v), nil
	case types.Int:
		return int64(v), nil
	case types.Uint:
		return uint64(v), nil
	case types.Double:
		return float64(v), nil
	case *types.Optional:
		if !v.HasValue() {
			return nil, nil
		}
		return nativeValue(v.GetValue())
	case traits.Mapper:
		out := map[string]any{}
		for it := v.Iterator(); it.HasNext() == types.True; {
			key := it.Next()
			k, ok := key.(types.String)
			if !ok {
				return nil, errors.New("map key is not a string")
			}
			item, err := nativeValue(v.Get(key))
			if err != nil {
				return nil, err
			}
			out[string(k)] = item
		}
		return out, nil
	case traits.Lister:
		out := []any{}
		for it := v.Iterator(); it.HasNext() == types.True; {
			item, err := nativeValue(it.Next())
			if err != nil {
				return nil, err
			}
			out = append(out, item)
		}
		return out, nil
	}
	return val.Value(), nil
}
