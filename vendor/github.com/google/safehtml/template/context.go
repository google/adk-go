// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"strings"
)

// context describes the state an HTML parser must be in when it reaches the
// portion of HTML produced by evaluating a particular template node.
//
// The zero value of type Context is the start context for a template that
// produces an HTML fragment as defined at
// http://www.w3.org/TR/html5/syntax.html#the-end
// where the context element is null.
type context struct {
	state   state
	delim   delim
	element element
	attr    attr
	err     *Error
	// scriptType is the lowercase value of the "type" attribute inside the current "script"
	// element (see https://dev.w3.org/html5/spec-preview/the-script-element.html#attr-script-type).
	// This field will be empty if the parser is currently not in a script element,
	// the type attribute has not already been parsed in the current element, or if the
	// value of the type attribute cannot be determined at parse time.
	scriptType string
	// linkRel is the value of the "rel" attribute inside the current "link"
	// element (see https://html.spec.whatwg.org/multipage/semantics.html#attr-link-rel).
	// This value has been normalized to lowercase with exactly one space between tokens
	// and exactly one space at start and end, so that a lookup of any token foo can
	// be performed by searching for the substring " foo ".
	// This field will be empty if the parser is currently not in a link element,
	// the rel attribute has not already been parsed in the current element, or if the
	// value of the rel attribute cannot be determined at parse time.
	linkRel string
}

// eq returns whether Context c is equal to Context d.
func (c context) eq(d context) bool {
	return c.state == d.state &&
		c.delim == d.delim &&
		c.element.eq(d.element) &&
		c.attr.eq(d.attr) &&
		c.err == d.err &&
		c.scriptType == d.scriptType &&
		c.linkRel == d.linkRel
}

// state describes a high-level HTML parser state.
//
// It bounds the top of the element stack, and by extension the HTML insertion
// mode, but also contains state that does not correspond to anything in the
// HTML5 parsing algorithm because a single token production in the HTML
// grammar may contain embedded actions in a template. For instance, the quoted
// HTML attribute produced by
//     <div title="Hello {{.World}}">
// is a single token in HTML's grammar but in a template spans several nodes.
type state uint8

//go:generate stringer -type state

const (
	// stateText is parsed character data. An HTML parser is in
	// this state when its parse position is outside an HTML tag,
	// directive, comment, and special element body.
	stateText state = iota
	// stateSpecialElementBody occurs inside a specal HTML element body.
	stateSpecialElementBody
	// stateTag occurs before an HTML attribute or the end of a tag.
	stateTag
	// stateAttrName occurs inside an attribute name.
	// It occurs between the ^'s in ` ^name^ = value`.
	stateAttrName
	// stateAfterName occurs after an attr name has ended but before any
	// equals sign. It occurs between the ^'s in ` name^ ^= value`.
	stateAfterName
	// stateBeforeValue occurs after the equals sign but before the value.
	// It occurs between the ^'s in ` name =^ ^value`.
	stateBeforeValue
	// stateHTMLCmt occurs inside an <!-- HTML comment -->.
	stateHTMLCmt
	// stateAttr occurs inside an HTML attribute whose content is text.
	stateAttr
	// stateError is an infectious error state outside any valid
	// HTML/CSS/JS construct.
	stateError
)

// isComment reports whether a state contains content meant for template
// authors & maintainers, not for end-users or machines.
func isComment(s state) bool {
	switch s {
	case stateHTMLCmt:
		return true
	}
	return false
}

// isInTag reports whether s occurs solely inside an HTML tag.
func isInTag(s state) bool {
	switch s {
	case stateTag, stateAttrName, stateAfterName, stateBeforeValue, stateAttr:
		return true
	}
	return false
}

// delim is the delimiter that will end the current HTML attribute.
type delim uint8

//go:generate stringer -type delim

const (
	// delimNone occurs outside any attribute.
	delimNone delim = iota
	// delimDoubleQuote occurs when a double quote (") closes the attribute.
	delimDoubleQuote
	// delimSingleQuote occurs when a single quote (') closes the attribute.
	delimSingleQuote
	// delimSpaceOrTagEnd occurs when a space or right angle bracket (>)
	// closes the attribute.
	delimSpaceOrTagEnd
)

type element struct {
	// name is the lowercase name of the element. If context joining has occurred, name
	// will be arbitrarily assigned the element name from one of the joined contexts.
	name string
	// names contains all possible names the element could assume because of context joining.
	// For example, after joining the contexts in the "if" and "else" branches of
	//     {{if .C}}<img{{else}}<audio{{end}} src="/some/path">`,
	// names will contain "img" and "audio".
	// names can also contain empty strings, which represent joined contexts with no element name.
	// names will be empty if no context joining occurred.
	names []string
}

// eq reports whether a and b have the same name. All other fields are ignored.
func (e element) eq(d element) bool {
	return e.name == d.name
}

// String returns the string representation of the element.
func (e element) String() string {
	return "element" + strings.Title(e.name)
}

// attr represents the attribute that the parser is in, that is,
// starting from stateAttrName until stateTag/stateText (exclusive).
type attr struct {
	// name is the lowercase name of the attribute. If context joining has occurred, name
	// will be arbitrarily assigned the attribute name from one of the joined contexts.
	name string
	// value is the value of the attribute. If context joining has occurred, value
	// will be arbitrarily assigned the attribute value from one of the joined contexts.
	// If there are multiple actions in the attribute value, value will contain the
	// concatenation of all values seen so far. For example, in
	//    <a name="foo{{.X}}bar{{.Y}}">
	// value is "foo" at "{{.X}}" and "foobar" at "{{.Y}}".
	value string
	// ambiguousValue indicates whether value contains an ambiguous value due to context-joining.
	ambiguousValue bool
	// names contains all possible names the attribute could assume because of context joining.
	// For example, after joining the contexts in the "if" and "else" branches of
	//     <a {{if .C}}title{{else}}name{{end}}="foo">
	// names will contain "title" and "name".
	// names can also contain empty strings, which represent joined contexts with no attribute name.
	// names will be empty if no context joining occurred.
	names []string
}

// eq reports whether a and b have the same name. All other fields are ignored.
func (a attr) eq(b attr) bool {
	return a.name == b.name
}

// String returns the string representation of the attr.
func (a attr) String() string {
	return "attr" + strings.Title(a.name)
}
