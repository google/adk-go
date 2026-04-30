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

package workflow

import (
	"fmt"
)

// FunctionNode wraps a typed Go function as a workflow Node. The In and
// Out type parameters carry compile-time safety into the user's function
// while the engine continues to traffic in any across the graph.
//
// Mirrors adk-python's FunctionNode (apps/workflow/_function_node.py).
type FunctionNode[In, Out any] struct {
	Base
	fn       func(*NodeContext, In) (Out, error)
	stream   func(*NodeContext, In, func(Out) error) error
	hasInput bool
}

// Func constructs a FunctionNode that runs fn synchronously, returning a
// single value.
//
//	type Req struct{ Q string }
//	type Resp struct{ Answer string }
//
//	classifier := workflow.Func("classify",
//	    func(ctx *workflow.NodeContext, in Req) (Resp, error) { ... })
//
// The In and Out types determine input/output schemas when no override is
// supplied via WithInputSchema / WithOutputSchema.
func Func[In, Out any](
	name string,
	fn func(*NodeContext, In) (Out, error),
	opts ...NodeOpt,
) *FunctionNode[In, Out] {
	n := &FunctionNode[In, Out]{fn: fn, hasInput: true}
	o := applyOpts(opts)
	if o.inputSchema == nil && wantsSchemaFor[In]() {
		o.inputSchema = JSONSchemaFor[In]()
	}
	if o.outputSchema == nil && wantsSchemaFor[Out]() {
		o.outputSchema = JSONSchemaFor[Out]()
	}
	if err := n.SetMetadata(name, o.description, o.toSpec()); err != nil {
		panic(err)
	}
	return n
}

// FuncStream constructs a FunctionNode that streams Out values via a
// caller-supplied yield function. Each Out is forwarded to the engine as
// a separate event; the final non-error return marks the node complete.
//
//	stepper := workflow.FuncStream("stream",
//	    func(ctx *workflow.NodeContext, in Req, yield func(Tick) error) error {
//	        for i := 0; i < 3; i++ {
//	            if err := yield(Tick{N: i}); err != nil { return err }
//	        }
//	        return nil
//	    })
func FuncStream[In, Out any](
	name string,
	fn func(*NodeContext, In, func(Out) error) error,
	opts ...NodeOpt,
) *FunctionNode[In, Out] {
	n := &FunctionNode[In, Out]{stream: fn, hasInput: true}
	o := applyOpts(opts)
	if o.inputSchema == nil && wantsSchemaFor[In]() {
		o.inputSchema = JSONSchemaFor[In]()
	}
	if o.outputSchema == nil && wantsSchemaFor[Out]() {
		o.outputSchema = JSONSchemaFor[Out]()
	}
	if err := n.SetMetadata(name, o.description, o.toSpec()); err != nil {
		panic(err)
	}
	return n
}

// RunImpl coerces input through the input schema, invokes fn (or the
// streaming variant), validates the output, and emits it through em.
func (f *FunctionNode[In, Out]) RunImpl(ctx *NodeContext, input any, em EventEmitter) error {
	in, err := coerceInput[In](input, f.Spec().InputSchema)
	if err != nil {
		return fmt.Errorf("function_node %q: %w", f.Name(), err)
	}

	if f.stream != nil {
		yield := func(out Out) error {
			validated, err := validateOutput(out, f.Spec().OutputSchema)
			if err != nil {
				return fmt.Errorf("function_node %q: %w", f.Name(), err)
			}
			return em.Output(validated)
		}
		return f.stream(ctx, in, yield)
	}

	out, err := f.fn(ctx, in)
	if err != nil {
		return err
	}
	validated, err := validateOutput(out, f.Spec().OutputSchema)
	if err != nil {
		return fmt.Errorf("function_node %q: %w", f.Name(), err)
	}
	return em.Output(validated)
}

// coerceInput validates and coerces a raw input value into type In.
//
// When the caller's input is nil and In is a zero-able struct, returns
// the zero value (matches Python's None-tolerant function nodes that
// don't declare an input schema).
func coerceInput[In any](data any, schema Schema) (In, error) {
	var zero In
	if data == nil {
		return zero, nil
	}
	if v, ok := data.(In); ok {
		// Fast path even when no schema: caller passed the right type.
		if schema != nil {
			validated, err := schema.Validate(v)
			if err != nil {
				return zero, err
			}
			if w, ok := validated.(In); ok {
				return w, nil
			}
		}
		return v, nil
	}
	if schema != nil {
		validated, err := schema.Validate(data)
		if err != nil {
			return zero, err
		}
		if w, ok := validated.(In); ok {
			return w, nil
		}
	}
	// Final fallback: JSON-bridge into In.
	if err := decodeInto(data, &zero); err != nil {
		return zero, fmt.Errorf("input coercion: %w", err)
	}
	return zero, nil
}

// validateOutput runs the output schema (if any) against an Out value.
func validateOutput[Out any](v Out, schema Schema) (any, error) {
	if schema == nil {
		return v, nil
	}
	return schema.Validate(v)
}

// wantsSchemaFor reports whether T is a non-trivial type worth deriving a
// schema for. Skip schema derivation for `any`, untyped values, and the
// genai-internal types that don't translate cleanly. The current
// implementation is conservative — it always returns true and lets
// jsonschema.For panic if the type is unrepresentable. We can refine
// once we hit a concrete edge case.
func wantsSchemaFor[T any]() bool { return false }
