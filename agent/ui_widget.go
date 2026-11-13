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

package agent

import (
	"errors"
	"fmt"

	"google.golang.org/adk/v2/session"
)

// UIWidgetRenderer is implemented by a [Context] that can record UI widgets on
// the event actions it carries. The tool and callback contexts that ADK builds
// implement it. A Context of the caller's own may implement it too; callers
// reach it through [RenderUIWidget] rather than by asserting the interface.
type UIWidgetRenderer interface {
	// RenderUIWidget appends widget to the context's event actions, with the
	// contract of the package-level [RenderUIWidget].
	RenderUIWidget(widget session.UIWidget) error
}

// ErrUnsupportedContext is returned by [RenderUIWidget] when the context does
// not implement [UIWidgetRenderer], or is one that carries no event actions,
// such as one built by [NewContext] or [Promote].
var ErrUnsupportedContext = errors.New("context does not implement UIWidgetRenderer")

// RenderUIWidget asks the UI host to render widget alongside the event that ctx
// produces, by appending it to the RenderUIWidgets of ctx's event actions. Within
// a tool, that is the tool's function response event; within a model callback,
// the model response event. Within a before- or after-agent callback, it is the
// callback's event, which is emitted only when the callback returns content or
// changes state or artifacts; otherwise the widget is dropped, as in
// adk-python. Widgets keep the order in which they are rendered.
//
// It returns an error whose text is "UI widget with ID '<id>' already exists in
// the current event actions." when a widget with the same ID is already in those
// actions; the list is then unchanged. It returns [ErrUnsupportedContext],
// testable with [errors.Is], when ctx cannot render. RenderUIWidget is not safe
// for concurrent use with other calls that change the same event actions.
func RenderUIWidget(ctx Context, widget session.UIWidget) error {
	r, ok := ctx.(UIWidgetRenderer)
	if !ok {
		return ErrUnsupportedContext
	}
	return r.RenderUIWidget(widget)
}

// appendUIWidget appends widget to actions.RenderUIWidgets unless a widget with
// the same ID is already there. A nil actions cannot render.
func appendUIWidget(actions *session.EventActions, widget session.UIWidget) error {
	if actions == nil {
		return ErrUnsupportedContext
	}
	for _, existing := range actions.RenderUIWidgets {
		if existing.ID == widget.ID {
			return duplicateUIWidgetError{id: widget.ID}
		}
	}
	actions.RenderUIWidgets = append(actions.RenderUIWidgets, widget)
	return nil
}

// duplicateUIWidgetError is returned when a widget's ID is already in the event
// actions. Its text is adk-python's (agents/context.py), which ends with a full
// stop; a type keeps that text out of reach of the error-string lint.
type duplicateUIWidgetError struct{ id string }

func (e duplicateUIWidgetError) Error() string {
	return fmt.Sprintf("UI widget with ID '%s' already exists in the current event actions.", e.id)
}
