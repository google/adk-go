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

// Package mrkdwn converts the Markdown that models write into Slack mrkdwn.
package mrkdwn

import (
	"regexp"
	"strconv"
	"strings"
)

// Render converts the Markdown that models write into Slack mrkdwn. It is
// line-oriented and deliberately not a full Markdown parser: fenced and inline
// code keep their text, Markdown tables become a code block because Slack has
// no tables, and everything else is rewritten with regular expressions. Slack
// reads *bold*, _italic_, ~strike~, <url|text>, and needs &, < and > escaped
// everywhere, code included. A fence's info string (```go) is dropped, since
// Slack would show it as the first line of the block.
//
// Render is not a sanitizer: Slack tokens a model echoes (<@U…>, <#C…>,
// <!here>, <!channel>, existing <url|label> links) and bare URLs pass through
// unescaped, and NUL bytes are dropped from the input. The output is mrkdwn,
// not Markdown, so rendering it a second time escapes it again: callers render
// once.
func Render(md string) string {
	// NUL is the placeholder marker below, so it must not occur in the input.
	md = strings.ReplaceAll(md, "\x00", "")
	in := strings.Split(md, "\n")
	out := make([]string, 0, len(in))
	for i := 0; i < len(in); {
		switch {
		case isFence(in[i]):
			out = append(out, fence(in[i]))
			j := i + 1
			for j < len(in) && !isFence(in[j]) {
				out = append(out, escape(in[j]))
				j++
			}
			if j < len(in) {
				out = append(out, fence(in[j]))
				j++
			}
			i = j
		case isTableRow(in[i]) && i+1 < len(in) && isTableRow(in[i+1]):
			j := i
			out = append(out, "```")
			for j < len(in) && isTableRow(in[j]) {
				out = append(out, escape(in[j]))
				j++
			}
			out = append(out, "```")
			i = j
		default:
			out = append(out, renderLine(in[i]))
			i++
		}
	}
	return strings.Join(out, "\n")
}

func isFence(s string) bool { return strings.HasPrefix(strings.TrimLeft(s, " \t"), "```") }

// fence keeps a fence line's indentation and backticks and drops the rest.
func fence(s string) string {
	rest := strings.TrimLeft(s, " \t")
	indent := s[:len(s)-len(rest)]
	return indent + rest[:len(rest)-len(strings.TrimLeft(rest, "`"))]
}

func isTableRow(s string) bool {
	t := strings.TrimSpace(s)
	return len(t) >= 2 && strings.HasPrefix(t, "|") && strings.HasSuffix(t, "|")
}

var (
	// A closing run of hashes only counts after a space, so "C#" keeps its #.
	headingRE = regexp.MustCompile(`^\s*#{1,6}\s+(.*?)(?:\s+#+)?\s*$`)
	bulletRE  = regexp.MustCompile(`^(\s*)[-*+]\s+`)
	// Double-backtick spans may contain a single backtick.
	inlineCodeRE = regexp.MustCompile("``(?:[^`\n]|`[^`\n])+``|`[^`\n]+`")
	// Slack's own tokens a model may echo back, by their ID shapes, so that
	// <!DOCTYPE html> or an HTML comment is still escaped as plain text.
	slackTokenRE = regexp.MustCompile(`<(?:[@#][A-Z0-9]+|!(?:here|channel|everyone|subteam\^[A-Z0-9]+|date\^[^|>\n]+))(?:\|[^>\n]*)?>|<(?:https?://|mailto:)[^>\n]*>`)
	// Only http(s) targets without < > | become <url|text>. Anything else is
	// left as written, so a fragment like (#setup) never becomes <#setup|…>.
	linkRE = regexp.MustCompile(`\[([^\]\n]+)\]\((https?://[^)\s<>|]+)\)`)
	// A URL must not end on a closing ** or ~~, nor on sentence punctuation,
	// which can sit between the URL and that closing marker: **https://x.io**.
	bareURLRE = regexp.MustCompile(`https?://[^\s<>]*[^\s<>*~.,;:!?)]`)
	strikeRE  = regexp.MustCompile(`~~([^~\s](?:[^~\n]*?[^~\s])?)~~`)
	// The span may contain its own marker (**2 * 3**, __user_id__). Like
	// italic below, __ needs a non-word char before it, so snake__case stays.
	boldRE = regexp.MustCompile(`\*\*([^*\s](?:[^\n]*?[^*\s])?)\*\*|(^|[^\w])__([^_\s](?:[^\n]*?[^_\s])?)__`)
	// A leading non-word char (or line start) keeps 2*3*4 from becoming italic.
	italicRE = regexp.MustCompile(`(^|[^\w*])\*([^*\s](?:[^*\n]*?[^*\s])?)\*`)
)

func renderLine(line string) string {
	quote := false
	if rest, ok := strings.CutPrefix(line, ">"); ok {
		quote = true
		line = strings.TrimPrefix(rest, " ")
	}
	heading := false
	if m := headingRE.FindStringSubmatch(line); m != nil {
		heading = true
		line = m[1]
	} else {
		line = bulletRE.ReplaceAllString(line, "${1}• ")
	}

	var ph placeholders
	// Code is escaped on the way out, so the link label below restores it
	// after escaping its own text.
	line = ph.take(inlineCodeRE, line, escape)
	// Links go before Slack tokens, so a token in a label is still plain text
	// when escape runs and no raw < or > reaches the link.
	line = ph.take(linkRE, line, func(s string) string {
		m := linkRE.FindStringSubmatch(s)
		return "<" + m[2] + "|" + ph.restore(escape(plainLabel(m[1]))) + ">"
	})
	line = ph.take(slackTokenRE, line, func(s string) string { return s })
	line = ph.take(bareURLRE, line, func(s string) string { return s })
	line = escape(line)
	line = strikeRE.ReplaceAllString(line, "~$1~")
	// Bold goes through a placeholder so the italic pass cannot see its stars.
	line = ph.take(boldRE, line, func(s string) string {
		m := boldRE.FindStringSubmatch(s)
		prefix, inner := "", m[1]
		if inner == "" {
			prefix, inner = m[2], m[3]
		}
		// Italic inside the span is converted here, since the placeholder
		// hides it from the italic pass below.
		inner = italicRE.ReplaceAllString(inner, "${1}_${2}_")
		// The whole heading becomes bold, and Slack cannot nest bold.
		if heading {
			return prefix + inner
		}
		return prefix + "*" + inner + "*"
	})
	line = italicRE.ReplaceAllString(line, "${1}_${2}_")
	line = ph.restore(line)
	if heading && line != "" {
		line = "*" + line + "*"
	}
	if quote {
		line = "> " + line
	}
	return line
}

func escape(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	return strings.ReplaceAll(s, ">", "&gt;")
}

// plainLabel drops emphasis markers: Slack shows a link label as plain text.
func plainLabel(s string) string {
	s = strikeRE.ReplaceAllString(s, "$1")
	s = boldRE.ReplaceAllString(s, "${1}${2}${3}")
	return italicRE.ReplaceAllString(s, "${1}${2}")
}

// placeholders swaps spans out of a line while other rewrites run, then puts
// their rendered form back. The marker uses NUL, which Render strips from the
// input first, so no rewrite can match across a span.
type placeholders struct{ vals []string }

func (p *placeholders) take(re *regexp.Regexp, line string, render func(string) string) string {
	return re.ReplaceAllStringFunc(line, func(s string) string {
		p.vals = append(p.vals, render(s))
		return "\x00" + strconv.Itoa(len(p.vals)-1) + "\x00"
	})
}

var markerRE = regexp.MustCompile("\x00(\\d+)\x00")

// A value can contain earlier markers (bold around inline code, say), so it
// expands until none remain. Each pass removes a marker and no value contains
// its own, so the loop ends.
func (p *placeholders) restore(line string) string {
	for markerRE.MatchString(line) {
		line = markerRE.ReplaceAllStringFunc(line, func(m string) string {
			i, _ := strconv.Atoi(m[1 : len(m)-1])
			return p.vals[i]
		})
	}
	return line
}
