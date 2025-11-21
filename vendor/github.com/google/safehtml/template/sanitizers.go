// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package template

import (
	"fmt"
	"text/template"

	"github.com/google/safehtml/internal/safehtmlutil"
	"github.com/google/safehtml"
)

// sanitizationContext determines what type of sanitization to perform
// on a template action.
type sanitizationContext uint8

const (
	_ = iota
	sanitizationContextAsyncEnum
	sanitizationContextDirEnum
	sanitizationContextHTML
	sanitizationContextHTMLValOnly
	sanitizationContextIdentifier
	sanitizationContextLoadingEnum
	sanitizationContextNone
	sanitizationContextRCDATA
	sanitizationContextScript
	sanitizationContextStyle
	sanitizationContextStyleSheet
	sanitizationContextTargetEnum
	sanitizationContextTrustedResourceURL
	sanitizationContextTrustedResourceURLOrURL
	sanitizationContextURL
	sanitizationContextURLSet
)

// String returns the string representation of sanitizationContext s.
func (s sanitizationContext) String() string {
	if int(s) >= len(sanitizationContextInfo) {
		return fmt.Sprintf("invalid sanitization context %d", s)
	}
	return sanitizationContextInfo[s].name
}

// sanitizerName returns the name of the sanitizer to call in sanitizationContext s.
// It returns an empty string if no sanitization is required in s.
func (s sanitizationContext) sanitizerName() string {
	if int(s) >= len(sanitizationContextInfo) {
		return fmt.Sprintf("invalid sanitization context %d", s)
	}
	return sanitizationContextInfo[s].sanitizerName
}

// isEnum reports reports whether s is a sanitization context for enumerated values.
func (s sanitizationContext) isEnum() bool {
	return s == sanitizationContextAsyncEnum || s == sanitizationContextDirEnum || s == sanitizationContextLoadingEnum || s == sanitizationContextTargetEnum
}

// isURLorTrustedResourceURL reports reports whether s is a sanitization context for URL or TrustedResourceURL values.
func (s sanitizationContext) isURLorTrustedResourceURL() bool {
	return s == sanitizationContextTrustedResourceURL || s == sanitizationContextTrustedResourceURLOrURL || s == sanitizationContextURL
}

// sanitizationContextInfo[x] contains the name for sanitization context x and the
// name of the sanitizer to call in that context.
// If sanitizationContextInfo[x].sanitizerName is empty, then no sanitizer needs
// to be called in x.
var sanitizationContextInfo = [...]struct {
	name, sanitizerName string
}{
	sanitizationContextAsyncEnum:               {"AsyncEnum", sanitizeAsyncEnumFuncName},
	sanitizationContextDirEnum:                 {"DirEnum", sanitizeDirEnumFuncName},
	sanitizationContextHTML:                    {"HTML", sanitizeHTMLFuncName},
	sanitizationContextHTMLValOnly:             {"HTMLValOnly", sanitizeHTMLValOnlyFuncName},
	sanitizationContextIdentifier:              {"Identifier", sanitizeIdentifierFuncName},
	sanitizationContextLoadingEnum:             {"LoadingEnum", sanitizeLoadingEnumFuncName},
	sanitizationContextNone:                    {"None", ""},
	sanitizationContextRCDATA:                  {"RCDATA", sanitizeRCDATAFuncName},
	sanitizationContextScript:                  {"Script", sanitizeScriptFuncName},
	sanitizationContextStyle:                   {"Style", sanitizeStyleFuncName},
	sanitizationContextStyleSheet:              {"StyleSheet", sanitizeStyleSheetFuncName},
	sanitizationContextTargetEnum:              {"TargetEnum", sanitizeTargetEnumFuncName},
	sanitizationContextTrustedResourceURL:      {"TrustedResourceURL", sanitizeTrustedResourceURLFuncName},
	sanitizationContextTrustedResourceURLOrURL: {"TrustedResourceURLOrURL", sanitizeTrustedResourceURLOrURLFuncName},
	sanitizationContextURL:                     {"URL", sanitizeURLFuncName},
	sanitizationContextURLSet:                  {"URLSet", sanitizeURLSetFuncName},
}

var funcs = template.FuncMap{
	queryEscapeURLFuncName:                         safehtmlutil.QueryEscapeURL,
	normalizeURLFuncName:                           safehtmlutil.NormalizeURL,
	validateTrustedResourceURLSubstitutionFuncName: validateTrustedResourceURLSubstitution,
	evalArgsFuncName:                               evalArgs,
	sanitizeHTMLCommentFuncName:                    sanitizeHTMLComment,
	sanitizeAsyncEnumFuncName:                      sanitizeAsyncEnum,
	sanitizeDirEnumFuncName:                        sanitizeDirEnum,
	sanitizeHTMLFuncName:                           sanitizeHTML,
	sanitizeHTMLValOnlyFuncName:                    sanitizeHTMLValOnly,
	sanitizeIdentifierFuncName:                     sanitizeIdentifier,
	sanitizeLoadingEnumFuncName:                    sanitizeLoadingEnum,
	sanitizeRCDATAFuncName:                         sanitizeRCDATA,
	sanitizeScriptFuncName:                         sanitizeScript,
	sanitizeStyleFuncName:                          sanitizeStyle,
	sanitizeStyleSheetFuncName:                     sanitizeStyleSheet,
	sanitizeTargetEnumFuncName:                     sanitizeTargetEnum,
	sanitizeTrustedResourceURLFuncName:             sanitizeTrustedResourceURL,
	sanitizeTrustedResourceURLOrURLFuncName:        sanitizeTrustedResourceURLOrURL,
	sanitizeURLFuncName:                            sanitizeURL,
	sanitizeURLSetFuncName:                         sanitizeURLSet,
}

const (
	queryEscapeURLFuncName                         = "_queryEscapeURL"
	normalizeURLFuncName                           = "_normalizeURL"
	validateTrustedResourceURLSubstitutionFuncName = "_validateTrustedResourceURLSubstitution"
	evalArgsFuncName                               = "_evalArgs"
	sanitizeHTMLCommentFuncName                    = "_sanitizeHTMLComment"
	sanitizeAsyncEnumFuncName                      = "_sanitizeAsyncEnum"
	sanitizeDirEnumFuncName                        = "_sanitizeDirEnum"
	sanitizeHTMLFuncName                           = "_sanitizeHTML"
	sanitizeHTMLValOnlyFuncName                    = "_sanitizeHTMLValOnly"
	sanitizeIdentifierFuncName                     = "_sanitizeIdentifier"
	sanitizeLoadingEnumFuncName                    = "_sanitizeLoadingEnum"
	sanitizeRCDATAFuncName                         = "_sanitizeRCDATA"
	sanitizeScriptFuncName                         = "_sanitizeScript"
	sanitizeStyleFuncName                          = "_sanitizeStyle"
	sanitizeStyleSheetFuncName                     = "_sanitizeStyleSheet"
	sanitizeTargetEnumFuncName                     = "_sanitizeTargetEnum"
	sanitizeTrustedResourceURLFuncName             = "_sanitizeTrustedResourceURL"
	sanitizeTrustedResourceURLOrURLFuncName        = "_sanitizeTrustedResourceURLOrURL"
	sanitizeURLFuncName                            = "_sanitizeURL"
	sanitizeURLSetFuncName                         = "_sanitizeURLSet"
)

// urlLinkRelVals contains values for a link element's rel attribute that indicate that the same link
// element's href attribute may contain a safehtml.URL value.
var urlLinkRelVals = map[string]bool{
	"alternate":    true,
	"author":       true,
	"bookmark":     true,
	"canonical":    true,
	"cite":         true,
	"dns-prefetch": true,
	"help":         true,
	"icon":         true,
	"license":      true,
	"next":         true,
	"preconnect":   true,
	"prefetch":     true,
	"preload":      true,
	"prerender":    true,
	"prev":         true,
	"search":       true,
	"subresource":  true,
}

// elementSpecificAttrValSanitizationContext[x][y] is the sanitization context for
// attribute x when it appears within element y.
var elementSpecificAttrValSanitizationContext = map[string]map[string]sanitizationContext{
	"accept": {
		"input": sanitizationContextNone,
	},
	"action": {
		"form": sanitizationContextURL,
	},
	"defer": {
		"script": sanitizationContextNone,
	},
	"formaction": {
		"button": sanitizationContextURL,
		"input":  sanitizationContextURL,
	},
	"formmethod": {
		"button": sanitizationContextNone,
		"input":  sanitizationContextNone,
	},
	"href": {
		"a":    sanitizationContextTrustedResourceURLOrURL,
		"area": sanitizationContextTrustedResourceURLOrURL,
	},
	"method": {
		"form": sanitizationContextNone,
	},
	"pattern": {
		"input": sanitizationContextNone,
	},
	"readonly": {
		"input":    sanitizationContextNone,
		"textarea": sanitizationContextNone,
	},
	"src": {
		"audio":  sanitizationContextTrustedResourceURLOrURL,
		"img":    sanitizationContextTrustedResourceURLOrURL,
		"input":  sanitizationContextTrustedResourceURLOrURL,
		"source": sanitizationContextTrustedResourceURLOrURL,
		"video":  sanitizationContextTrustedResourceURLOrURL,
	},
	"srcdoc": {
		"iframe": sanitizationContextHTMLValOnly,
	},
}

// globalAttrValSanitizationContext[x] is the sanitization context for attribute x when
// it appears within any element not in the key set of elementSpecificAttrValSanitizationContext[x].
var globalAttrValSanitizationContext = map[string]sanitizationContext{
	"align":                 sanitizationContextNone,
	"alt":                   sanitizationContextNone,
	"aria-activedescendant": sanitizationContextIdentifier,
	"aria-atomic":           sanitizationContextNone,
	"aria-autocomplete":     sanitizationContextNone,
	"aria-busy":             sanitizationContextNone,
	"aria-checked":          sanitizationContextNone,
	"aria-controls":         sanitizationContextIdentifier,
	"aria-current":          sanitizationContextNone,
	"aria-disabled":         sanitizationContextNone,
	"aria-dropeffect":       sanitizationContextNone,
	"aria-expanded":         sanitizationContextNone,
	"aria-haspopup":         sanitizationContextNone,
	"aria-hidden":           sanitizationContextNone,
	"aria-invalid":          sanitizationContextNone,
	"aria-label":            sanitizationContextNone,
	"aria-labelledby":       sanitizationContextIdentifier,
	"aria-level":            sanitizationContextNone,
	"aria-live":             sanitizationContextNone,
	"aria-multiline":        sanitizationContextNone,
	"aria-multiselectable":  sanitizationContextNone,
	"aria-orientation":      sanitizationContextNone,
	"aria-owns":             sanitizationContextIdentifier,
	"aria-posinset":         sanitizationContextNone,
	"aria-pressed":          sanitizationContextNone,
	"aria-readonly":         sanitizationContextNone,
	"aria-relevant":         sanitizationContextNone,
	"aria-required":         sanitizationContextNone,
	"aria-selected":         sanitizationContextNone,
	"aria-setsize":          sanitizationContextNone,
	"aria-sort":             sanitizationContextNone,
	"aria-valuemax":         sanitizationContextNone,
	"aria-valuemin":         sanitizationContextNone,
	"aria-valuenow":         sanitizationContextNone,
	"aria-valuetext":        sanitizationContextNone,
	"async":                 sanitizationContextAsyncEnum,
	"autocapitalize":        sanitizationContextNone,
	"autocomplete":          sanitizationContextNone,
	"autocorrect":           sanitizationContextNone,
	"autofocus":             sanitizationContextNone,
	"autoplay":              sanitizationContextNone,
	"bgcolor":               sanitizationContextNone,
	"border":                sanitizationContextNone,
	"cellpadding":           sanitizationContextNone,
	"cellspacing":           sanitizationContextNone,
	"checked":               sanitizationContextNone,
	"cite":                  sanitizationContextURL,
	"class":                 sanitizationContextNone,
	"color":                 sanitizationContextNone,
	"cols":                  sanitizationContextNone,
	"colspan":               sanitizationContextNone,
	"contenteditable":       sanitizationContextNone,
	"controls":              sanitizationContextNone,
	"datetime":              sanitizationContextNone,
	"dir":                   sanitizationContextDirEnum,
	"disabled":              sanitizationContextNone,
	"download":              sanitizationContextNone,
	"draggable":             sanitizationContextNone,
	"enctype":               sanitizationContextNone,
	"face":                  sanitizationContextNone,
	"for":                   sanitizationContextIdentifier,
	"formenctype":           sanitizationContextNone,
	"frameborder":           sanitizationContextNone,
	"height":                sanitizationContextNone,
	"hidden":                sanitizationContextNone,
	"href":                  sanitizationContextTrustedResourceURL,
	"hreflang":              sanitizationContextNone,
	"id":                    sanitizationContextIdentifier,
	"ismap":                 sanitizationContextNone,
	"itemid":                sanitizationContextNone,
	"itemprop":              sanitizationContextNone,
	"itemref":               sanitizationContextNone,
	"itemscope":             sanitizationContextNone,
	"itemtype":              sanitizationContextNone,
	"label":                 sanitizationContextNone,
	"lang":                  sanitizationContextNone,
	"list":                  sanitizationContextIdentifier,
	"loading":               sanitizationContextLoadingEnum,
	"loop":                  sanitizationContextNone,
	"max":                   sanitizationContextNone,
	"maxlength":             sanitizationContextNone,
	"media":                 sanitizationContextNone,
	"min":                   sanitizationContextNone,
	"minlength":             sanitizationContextNone,
	"multiple":              sanitizationContextNone,
	"muted":                 sanitizationContextNone,
	"name":                  sanitizationContextIdentifier,
	"nonce":                 sanitizationContextNone,
	"open":                  sanitizationContextNone,
	"placeholder":           sanitizationContextNone,
	"poster":                sanitizationContextURL,
	"preload":               sanitizationContextNone,
	"rel":                   sanitizationContextNone,
	"required":              sanitizationContextNone,
	"reversed":              sanitizationContextNone,
	"role":                  sanitizationContextNone,
	"rows":                  sanitizationContextNone,
	"rowspan":               sanitizationContextNone,
	"selected":              sanitizationContextNone,
	"shape":                 sanitizationContextNone,
	"size":                  sanitizationContextNone,
	"sizes":                 sanitizationContextNone,
	"slot":                  sanitizationContextNone,
	"span":                  sanitizationContextNone,
	"spellcheck":            sanitizationContextNone,
	"src":                   sanitizationContextTrustedResourceURL,
	"srcset":                sanitizationContextURLSet,
	"start":                 sanitizationContextNone,
	"step":                  sanitizationContextNone,
	"style":                 sanitizationContextStyle,
	"summary":               sanitizationContextNone,
	"tabindex":              sanitizationContextNone,
	"target":                sanitizationContextTargetEnum,
	"title":                 sanitizationContextNone,
	"translate":             sanitizationContextNone,
	"type":                  sanitizationContextNone,
	"valign":                sanitizationContextNone,
	"value":                 sanitizationContextNone,
	"width":                 sanitizationContextNone,
	"wrap":                  sanitizationContextNone,
}

// elementContentSanitizationContext maps element names to element content sanitization contexts.
var elementContentSanitizationContext = map[string]sanitizationContext{
	"a":          sanitizationContextHTML,
	"abbr":       sanitizationContextHTML,
	"address":    sanitizationContextHTML,
	"article":    sanitizationContextHTML,
	"aside":      sanitizationContextHTML,
	"audio":      sanitizationContextHTML,
	"b":          sanitizationContextHTML,
	"bdi":        sanitizationContextHTML,
	"bdo":        sanitizationContextHTML,
	"blockquote": sanitizationContextHTML,
	"body":       sanitizationContextHTML,
	"button":     sanitizationContextHTML,
	"canvas":     sanitizationContextHTML,
	"caption":    sanitizationContextHTML,
	"center":     sanitizationContextHTML,
	"cite":       sanitizationContextHTML,
	"code":       sanitizationContextHTML,
	"colgroup":   sanitizationContextHTML,
	"command":    sanitizationContextHTML,
	"data":       sanitizationContextHTML,
	"datalist":   sanitizationContextHTML,
	"dd":         sanitizationContextHTML,
	"del":        sanitizationContextHTML,
	"details":    sanitizationContextHTML,
	"dfn":        sanitizationContextHTML,
	"dialog":     sanitizationContextHTML,
	"div":        sanitizationContextHTML,
	"dl":         sanitizationContextHTML,
	"dt":         sanitizationContextHTML,
	"em":         sanitizationContextHTML,
	"fieldset":   sanitizationContextHTML,
	"figcaption": sanitizationContextHTML,
	"figure":     sanitizationContextHTML,
	"font":       sanitizationContextHTML,
	"footer":     sanitizationContextHTML,
	"form":       sanitizationContextHTML,
	"frame":      sanitizationContextHTML,
	"frameset":   sanitizationContextHTML,
	"h1":         sanitizationContextHTML,
	"h2":         sanitizationContextHTML,
	"h3":         sanitizationContextHTML,
	"h4":         sanitizationContextHTML,
	"h5":         sanitizationContextHTML,
	"h6":         sanitizationContextHTML,
	"head":       sanitizationContextHTML,
	"header":     sanitizationContextHTML,
	"html":       sanitizationContextHTML,
	"i":          sanitizationContextHTML,
	"iframe":     sanitizationContextHTML,
	"ins":        sanitizationContextHTML,
	"kbd":        sanitizationContextHTML,
	"label":      sanitizationContextHTML,
	"legend":     sanitizationContextHTML,
	"lh":         sanitizationContextHTML,
	"li":         sanitizationContextHTML,
	"main":       sanitizationContextHTML,
	"map":        sanitizationContextHTML,
	"mark":       sanitizationContextHTML,
	"menu":       sanitizationContextHTML,
	"meter":      sanitizationContextHTML,
	"nav":        sanitizationContextHTML,
	"noscript":   sanitizationContextHTML,
	"ol":         sanitizationContextHTML,
	"optgroup":   sanitizationContextHTML,
	"option":     sanitizationContextHTML,
	"output":     sanitizationContextHTML,
	"p":          sanitizationContextHTML,
	"picture":    sanitizationContextHTML,
	"pre":        sanitizationContextHTML,
	"progress":   sanitizationContextHTML,
	"q":          sanitizationContextHTML,
	"rb":         sanitizationContextHTML,
	"rp":         sanitizationContextHTML,
	"rt":         sanitizationContextHTML,
	"rtc":        sanitizationContextHTML,
	"ruby":       sanitizationContextHTML,
	"s":          sanitizationContextHTML,
	"samp":       sanitizationContextHTML,
	"script":     sanitizationContextScript,
	"section":    sanitizationContextHTML,
	"select":     sanitizationContextHTML,
	"slot":       sanitizationContextHTML,
	"small":      sanitizationContextHTML,
	"span":       sanitizationContextHTML,
	"strong":     sanitizationContextHTML,
	"style":      sanitizationContextStyleSheet,
	"sub":        sanitizationContextHTML,
	"summary":    sanitizationContextHTML,
	"sup":        sanitizationContextHTML,
	"table":      sanitizationContextHTML,
	"tbody":      sanitizationContextHTML,
	"td":         sanitizationContextHTML,
	"textarea":   sanitizationContextRCDATA,
	"tfoot":      sanitizationContextHTML,
	"th":         sanitizationContextHTML,
	"thead":      sanitizationContextHTML,
	"time":       sanitizationContextHTML,
	"title":      sanitizationContextRCDATA,
	"tr":         sanitizationContextHTML,
	"u":          sanitizationContextHTML,
	"ul":         sanitizationContextHTML,
	"var":        sanitizationContextHTML,
	"video":      sanitizationContextHTML,
}

// allowedVoidElements is a set of names of void elements actions may appear in.
var allowedVoidElements = map[string]bool{
	"area":   true,
	"br":     true,
	"col":    true,
	"hr":     true,
	"img":    true,
	"input":  true,
	"link":   true,
	"param":  true,
	"source": true,
	"track":  true,
	"wbr":    true,
}

var sanitizeAsyncEnumValues = map[string]bool{
	"async": true,
}

func sanitizeAsyncEnum(args ...interface{}) (string, error) {
	input := safehtmlutil.Stringify(args...)
	if sanitizeAsyncEnumValues[input] {
		return input, nil
	}
	return "", fmt.Errorf(`expected one of the following strings: ["async"]`)
}

var sanitizeDirEnumValues = map[string]bool{
	"auto": true,
	"ltr":  true,
	"rtl":  true,
}

func sanitizeDirEnum(args ...interface{}) (string, error) {
	input := safehtmlutil.Stringify(args...)
	if sanitizeDirEnumValues[input] {
		return input, nil
	}
	return "", fmt.Errorf(`expected one of the following strings: ["auto" "ltr" "rtl"]`)
}

func sanitizeHTML(args ...interface{}) (string, error) {
	if len(args) > 0 {
		if safeTypeValue, ok := safehtmlutil.Indirect(args[0]).(safehtml.HTML); ok {
			return safeTypeValue.String(), nil
		}
	}
	input := safehtmlutil.Stringify(args...)
	return safehtml.HTMLEscaped(input).String(), nil
}

func sanitizeHTMLValOnly(args ...interface{}) (string, error) {
	if len(args) > 0 {
		if safeTypeValue, ok := safehtmlutil.Indirect(args[0]).(safehtml.HTML); ok {
			return safeTypeValue.String(), nil
		}
	}
	return "", fmt.Errorf(`expected a safehtml.HTML value`)
}

func sanitizeIdentifier(args ...interface{}) (string, error) {
	if len(args) > 0 {
		if safeTypeValue, ok := safehtmlutil.Indirect(args[0]).(safehtml.Identifier); ok {
			return safeTypeValue.String(), nil
		}
	}
	return "", fmt.Errorf(`expected a safehtml.Identifier value`)
}

var sanitizeLoadingEnumValues = map[string]bool{
	"eager": true,
	"lazy":  true,
}

func sanitizeLoadingEnum(args ...interface{}) (string, error) {
	input := safehtmlutil.Stringify(args...)
	if sanitizeLoadingEnumValues[input] {
		return input, nil
	}
	return "", fmt.Errorf(`expected one of the following strings: ["eager" "lazy"]`)
}

func sanitizeRCDATA(args ...interface{}) (string, error) {
	input := safehtmlutil.Stringify(args...)
	return safehtml.HTMLEscaped(input).String(), nil
}

func sanitizeScript(args ...interface{}) (string, error) {
	if len(args) > 0 {
		if safeTypeValue, ok := safehtmlutil.Indirect(args[0]).(safehtml.Script); ok {
			return safeTypeValue.String(), nil
		}
	}
	return "", fmt.Errorf(`expected a safehtml.Script value`)
}

func sanitizeStyle(args ...interface{}) (string, error) {
	if len(args) > 0 {
		if safeTypeValue, ok := safehtmlutil.Indirect(args[0]).(safehtml.Style); ok {
			return safeTypeValue.String(), nil
		}
	}
	return "", fmt.Errorf(`expected a safehtml.Style value`)
}

func sanitizeStyleSheet(args ...interface{}) (string, error) {
	if len(args) > 0 {
		if safeTypeValue, ok := safehtmlutil.Indirect(args[0]).(safehtml.StyleSheet); ok {
			return safeTypeValue.String(), nil
		}
	}
	return "", fmt.Errorf(`expected a safehtml.StyleSheet value`)
}

var sanitizeTargetEnumValues = map[string]bool{
	"_blank": true,
	"_self":  true,
}

func sanitizeTargetEnum(args ...interface{}) (string, error) {
	input := safehtmlutil.Stringify(args...)
	if sanitizeTargetEnumValues[input] {
		return input, nil
	}
	return "", fmt.Errorf(`expected one of the following strings: ["_blank" "_self"]`)
}

func sanitizeTrustedResourceURL(args ...interface{}) (string, error) {
	if len(args) > 0 {
		if safeTypeValue, ok := safehtmlutil.Indirect(args[0]).(safehtml.TrustedResourceURL); ok {
			return safeTypeValue.String(), nil
		}
	}
	return "", fmt.Errorf(`expected a safehtml.TrustedResourceURL value`)
}

func sanitizeTrustedResourceURLOrURL(args ...interface{}) (string, error) {
	if len(args) > 0 {
		switch v := safehtmlutil.Indirect(args[0]).(type) {
		case safehtml.TrustedResourceURL, safehtml.URL:
			return safehtmlutil.Stringify(v), nil
		}
	}
	input := safehtmlutil.Stringify(args...)
	return safehtml.URLSanitized(input).String(), nil
}

func sanitizeURL(args ...interface{}) (string, error) {
	if len(args) > 0 {
		if safeTypeValue, ok := safehtmlutil.Indirect(args[0]).(safehtml.URL); ok {
			return safeTypeValue.String(), nil
		}
	}
	input := safehtmlutil.Stringify(args...)
	return safehtml.URLSanitized(input).String(), nil
}

func sanitizeURLSet(args ...interface{}) (string, error) {
	input := safehtmlutil.Stringify(args...)
	return safehtml.URLSetSanitized(input).String(), nil
}
