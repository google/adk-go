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

package agentanalytics

import (
	"cmp"
	"slices"
	"sort"
	"strings"
)

// Byte spans for the legacy-pattern backstop (see leavesLegacyValue). Each
// pass of the scanners and of the legacy expressions writes its output
// through a spanWriter, which can record the runs of the output that the
// pass copied from its input; a legacy pass also records the spans of its
// input that it removed (a value, a header value or a credential), and the
// rewrite of a URI the bytes of the URI that it does not keep (see
// uriAligner). Composing the runs of every pass maps each byte of a result
// back to the byte of the scrubbed text that it is a copy of, so the
// backstop compares what the scrub keeps with what the expressions remove
// byte by byte, whatever the bytes spell.

// copyRun records that out[out:out+n], the output of one or more passes, is
// a copy of in[in:in+n], their input, byte by byte. In a rewritten URI, a
// byte that the rewrite escapes or unescapes, as the ^ of a path that it
// writes as %5E, maps from the first byte of how the output writes it to the
// first byte of how the input does, and the letters of the scheme, which
// url.Parse writes in lowercase, map whatever their case.
type copyRun struct{ out, in, n int }

// passTrace records what passes over a text did: the runs of their output
// copied from the text, in order, and the spans of the text that they
// removed (recorded by the legacy passes and by the rewrites of URIs).
// unaligned reports that a rewritten URI could not be aligned with the text
// within the budget of its searches (see uriAligner), so that its runs may
// miss bytes that the output keeps.
type passTrace struct {
	runs      []copyRun
	removed   [][2]int
	unaligned bool
}

// spanWriter writes the output of one pass over s and, when tr is not nil,
// records in tr which runs of the output are copies of s.
type spanWriter struct {
	s  string
	b  strings.Builder
	tr *passTrace
}

// copy writes s[from:to].
func (w *spanWriter) copy(from, to int) {
	if from >= to {
		return
	}
	if w.tr != nil {
		w.tr.runs = appendRun(w.tr.runs, copyRun{w.b.Len(), from, to - from})
	}
	w.b.WriteString(w.s[from:to])
}

// lit writes t, which holds no byte of s.
func (w *spanWriter) lit(t string) {
	w.b.WriteString(t)
}

// rewriteURI writes out, the text that sanitizeURIWith returned for the URI
// s[from:to], recording which runs of out are copies of s[from:to] and which
// bytes of s[from:to] out does not keep (see uriAligner).
func (w *spanWriter) rewriteURI(from, to int, out string) {
	if w.tr == nil {
		w.b.WriteString(out)
		return
	}
	a := newURIAligner(w.tr, w.b.Len(), from, w.s[from:to], out)
	w.b.WriteString(out)
	a.alignURI()
}

// placeholderLenAt returns the length of the placeholder of a redacted value
// or text, written as is or escaped in a URI, that starts at s[i], or 0.
func placeholderLenAt(s string, i int) int {
	if i >= len(s) || (s[i] != '[' && s[i] != '%') {
		return 0
	}
	for _, p := range [...]string{redactedValue, redactedURIPlaceholder, redactedSensitiveTextPlaceholder, "%5BREDACTED%5D"} {
		if strings.HasPrefix(s[i:], p) {
			return len(p)
		}
	}
	return 0
}

// uriAligner aligns out, the text that sanitizeURIWith wrote for the URI
// orig, with orig, recording in tr the runs of out that are copies of orig,
// offset by base in the output and by from in the input, and the bytes of
// orig that out does not keep as removed. The rewrite keeps most bytes of a
// URI, writes some escaped or unescaped, as url.URL.String writes the ^ of a
// path as %5E or a %41 of a host as A, and replaces values with placeholders,
// around which the text passes of a component may write spaces, a ":" or
// quotes and drop the spaces they replace. The alignment reads both texts
// once from left to right, part by part (see alignURI), and a search for
// where orig goes on after a placeholder, or for a byte of out that orig does
// not hold next, never leaves the part that it aligns and reads at most
// budget bytes in all. Beyond that, the rest of orig counts as removed and
// the trace as unaligned, so the work stays linear in the length of both
// texts.
type uriAligner struct {
	tr         *passTrace
	base, from int
	orig, out  string
	// ri is where the part of orig that is aligned ends.
	ri     int
	budget int
}

func newURIAligner(tr *passTrace, base, from int, orig, out string) *uriAligner {
	return &uriAligner{tr: tr, base: base, from: from, orig: orig, out: out, budget: 8*(len(orig)+len(out)) + 256}
}

// alignURI aligns out with orig part by part, as url.Parse reads them (see
// uriBounds): the scheme, which url.Parse writes in lowercase, the
// authority, each segment of the path, and the key and the value of each pair
// of the query and of the fragment, so that each byte of a part that the
// rewrite keeps maps to that part, whatever the rewrite does to the others.
// Segments or pairs that the rewrite does not keep as many of, as when it
// unescapes a %2F of a path or scrubs a fragment as text, are aligned as a
// whole part (see alignText). A URI replaced as a whole keeps nothing.
func (a *uriAligner) alignURI() {
	ok := true
	if a.out != redactedURIPlaceholder {
		ob, ub := uriBounds(a.orig), uriBounds(a.out)
		ok = a.alignScheme(ob[1], ub[1]) &&
			a.alignText(ob[1], ob[2], ub[1], ub[2], true) &&
			a.alignSplit(ob[2], ob[3], ub[2], ub[3], '/', false) &&
			a.alignDelimited(ob[3], ob[4], ub[3], ub[4]) &&
			a.alignDelimited(ob[4], ob[5], ub[4], ub[5])
	}
	a.drop(a.ri, len(a.orig))
	if !ok {
		a.tr.unaligned = true
	}
}

// uriBounds returns where the parts of the URI s end, as url.Parse reads
// them: b[1] where its scheme ends at the ":", b[2] where its authority,
// after "//", ends at the path, b[3] where the path ends at the "?" of the
// query, b[4] where the query ends at the "#" of the fragment, and b[5], the
// end of s.
func uriBounds(s string) [6]int {
	frag := strings.IndexByte(s, '#')
	if frag < 0 {
		frag = len(s)
	}
	query := strings.IndexByte(s[:frag], '?')
	if query < 0 {
		query = frag
	}
	scheme := strings.IndexByte(s[:query], ':')
	if scheme < 0 {
		return [6]int{0, 0, 0, query, frag, len(s)}
	}
	auth := scheme + 1
	if strings.HasPrefix(s[auth:query], "//") {
		auth += 2
		if i := strings.IndexByte(s[auth:query], '/'); i >= 0 {
			auth += i
		} else {
			auth = query
		}
	}
	return [6]int{0, scheme, auth, query, frag, len(s)}
}

// alignScheme aligns the scheme orig[:oe] with out[:ue], which url.Parse
// writes in lowercase.
func (a *uriAligner) alignScheme(oe, ue int) bool {
	if oe == ue && strings.EqualFold(a.orig[:oe], a.out[:ue]) {
		a.keep(0, 0, oe)
		return true
	}
	return a.alignText(0, oe, 0, ue, false)
}

// alignDelimited aligns the query or the fragment orig[oa:ob], which starts
// with its "?" or "#", with out[ua:ub], pair by pair (see alignSplit). A part
// that out does not hold, as the empty fragment of a URI that ends in "#",
// which url.URL.String leaves out, is removed.
func (a *uriAligner) alignDelimited(oa, ob, ua, ub int) bool {
	if oa == ob {
		return true
	}
	if ua == ub || a.out[ua] != a.orig[oa] {
		a.drop(oa, ob)
		return true
	}
	a.keep(ua, oa, 1)
	return a.alignSplit(oa+1, ob, ua+1, ub, '&', true)
}

// alignSplit aligns orig[oa:ob] with out[ua:ub] piece by piece between the
// bytes sep, when both hold as many of them, and as a whole otherwise. With
// pairs, each piece is a key and a value (see alignPair).
func (a *uriAligner) alignSplit(oa, ob, ua, ub int, sep byte, pairs bool) bool {
	if a.orig[oa:ob] == a.out[ua:ub] {
		a.keep(ua, oa, ob-oa)
		return true
	}
	if strings.Count(a.orig[oa:ob], string(sep)) != strings.Count(a.out[ua:ub], string(sep)) {
		return a.alignText(oa, ob, ua, ub, true)
	}
	for {
		oe, ue := pieceEnd(a.orig, oa, ob, sep), pieceEnd(a.out, ua, ub, sep)
		var ok bool
		if pairs {
			ok = a.alignPair(oa, oe, ua, ue)
		} else {
			ok = a.alignText(oa, oe, ua, ue, true)
		}
		if !ok {
			return false
		}
		if oe == ob {
			return true
		}
		a.keep(ue, oe, 1)
		oa, ua = oe+1, ue+1
	}
}

// alignPair aligns the pair orig[oa:ob] with out[ua:ub], its key and its
// value apart when both hold an "=": the rewrite escapes each "=" that it
// writes in a key or a value.
func (a *uriAligner) alignPair(oa, ob, ua, ub int) bool {
	oe, ue := pieceEnd(a.orig, oa, ob, '='), pieceEnd(a.out, ua, ub, '=')
	if oe == ob || ue == ub || a.orig[oa:ob] == a.out[ua:ub] {
		return a.alignText(oa, ob, ua, ub, true)
	}
	if !a.alignText(oa, oe, ua, ue, true) {
		return false
	}
	a.keep(ue, oe, 1)
	return a.alignText(oe+1, ob, ue+1, ub, true)
}

// pieceEnd returns where the first sep in s[from:to] is, or to.
func pieceEnd(s string, from, to int, sep byte) int {
	if i := strings.IndexByte(s[from:to], sep); i >= 0 {
		return from + i
	}
	return to
}

// alignText aligns out[ua:ub] with orig[oa:ob], reading the escapes and the
// "+" of a URI as the bytes that they stand for when decode is set (see
// decodeUnits), and reports whether its searches stayed within the budget.
// A byte that both hold next is kept. A placeholder that orig does not hold
// stands for a value of orig, which ends where the text after the
// placeholder goes on in orig, and any other byte that orig does not hold
// next stands for the same byte further on, after bytes that the rewrite
// dropped, or, when orig does not hold it, for none, as the spaces, quotes
// and ":" that a pass writes around a placeholder. What out does not keep of
// orig is removed.
func (a *uriAligner) alignText(oa, ob, ua, ub int, decode bool) bool {
	if a.orig[oa:ob] == a.out[ua:ub] {
		a.keep(ua, oa, ob-oa)
		return true
	}
	o, u := decodeUnits(a.orig, oa, ob, decode), decodeUnits(a.out, ua, ub, decode)
	if o.text == u.text {
		a.keepUnits(o, u, 0, 0, len(o.text))
		return true
	}
	i, k := 0, 0
	for i < len(u.text) {
		if n := placeholderLenAt(u.text, i); n > 0 {
			if s := skipInlineSpaces(o.text, k); strings.HasPrefix(o.text[s:], u.text[i:i+n]) {
				a.dropUnits(o, k, s)
				a.keepUnits(o, u, s, i, n)
				i, k = i+n, s+n
				continue
			}
			i += n
			end := nextPlaceholder(u.text, i)
			if end == i {
				continue
			}
			run := u.text[i:end]
			if p := a.index(o.text, k, run); p >= 0 {
				a.dropUnits(o, k, k+p)
				a.keepUnits(o, u, k+p, i, len(run))
				i, k = end, k+p+len(run)
			}
			// Otherwise orig holds the run, if at all, only with bytes
			// between its bytes, which are aligned one by one.
			continue
		}
		if k < len(o.text) && u.text[i] == o.text[k] {
			a.keepUnits(o, u, k, i, 1)
			i, k = i+1, k+1
			continue
		}
		if p := a.indexByte(o.text, k, u.text[i]); p >= 0 {
			a.dropUnits(o, k, k+p)
			a.keepUnits(o, u, k+p, i, 1)
			k += p + 1
		}
		i++
	}
	a.dropUnits(o, k, len(o.text))
	return a.budget >= 0
}

// index returns where t first holds sub at or after t[k], relative to k,
// charging the bytes that the search reads to the budget, or -1 when t does
// not hold it or the budget is spent.
func (a *uriAligner) index(t string, k int, sub string) int {
	if a.budget < 0 {
		return -1
	}
	p := strings.Index(t[k:], sub)
	if p < 0 {
		a.budget -= len(t) - k
	} else {
		a.budget -= p + len(sub)
	}
	return p
}

// indexByte is index for the byte c.
func (a *uriAligner) indexByte(t string, k int, c byte) int {
	if a.budget < 0 {
		return -1
	}
	p := strings.IndexByte(t[k:], c)
	if p < 0 {
		a.budget -= len(t) - k
	} else {
		a.budget -= p + 1
	}
	return p
}

// keep records out[oi:oi+n] as a copy of orig[ri:ri+n].
func (a *uriAligner) keep(oi, ri, n int) {
	a.tr.runs = appendRun(a.tr.runs, copyRun{a.base + oi, a.from + ri, n})
	a.ri = max(a.ri, ri+n)
}

// drop records orig[ri:rj] as removed.
func (a *uriAligner) drop(ri, rj int) {
	if ri >= rj {
		return
	}
	if k := len(a.tr.removed) - 1; k >= 0 && a.tr.removed[k][1] == a.from+ri {
		a.tr.removed[k][1] = a.from + rj
	} else {
		a.tr.removed = append(a.tr.removed, [2]int{a.from + ri, a.from + rj})
	}
	a.ri = max(a.ri, rj)
}

// keepUnits records u.text[i:i+n] as a copy of o.text[k:k+n], unit by unit:
// a unit that out writes as orig does as a copy of it, and a unit that the
// rewrite escapes or unescapes by its first byte (see copyRun).
func (a *uriAligner) keepUnits(o, u unitText, k, i, n int) {
	if o.pos == nil && u.pos == nil {
		a.keep(u.from+i, o.from+k, n)
		return
	}
	for t := range n {
		r0, r1 := o.at(k+t), o.at(k+t+1)
		w0, w1 := u.at(i+t), u.at(i+t+1)
		if r1-r0 == w1-w0 && a.orig[r0:r1] == a.out[w0:w1] {
			a.keep(w0, r0, r1-r0)
		} else {
			a.keep(w0, r0, 1)
		}
		a.ri = max(a.ri, r1)
	}
}

// dropUnits records the bytes of o.text[k0:k1] as removed.
func (a *uriAligner) dropUnits(o unitText, k0, k1 int) {
	if k0 < k1 {
		a.drop(o.at(k0), o.at(k1))
	}
}

// unitText is the text of s[from:to] for an alignment, with each percent
// escape of two hex digits read as the byte that it stands for and each "+"
// as a space when it is decoded (see decodeUnits); pos[k] is where the unit
// that text[k] stands for starts in s, and pos[len(text)] is the end. pos is
// nil when text is s[from:to] as it is.
type unitText struct {
	text string
	pos  []int
	from int
}

// at returns where the unit that text[k] stands for starts in s.
func (t unitText) at(k int) int {
	if t.pos == nil {
		return t.from + k
	}
	return t.pos[k]
}

// decodeUnits returns s[from:to] as a unitText, decoded when decode is set:
// a byte that the rewrite of a URI escapes or unescapes, or a space that it
// writes as "+" or %20, reads the same either way.
func decodeUnits(s string, from, to int, decode bool) unitText {
	if !decode || !strings.ContainsAny(s[from:to], "%+") {
		return unitText{text: s[from:to], from: from}
	}
	b := make([]byte, 0, to-from)
	pos := make([]int, 0, to-from+1)
	for i := from; i < to; {
		pos = append(pos, i)
		c := s[i]
		switch {
		case c == '%' && i+2 < to && isHexByte(s[i+1]) && isHexByte(s[i+2]):
			c = unhexByte(s[i+1])<<4 | unhexByte(s[i+2])
			i += 3
		case c == '+':
			c = ' '
			i++
		default:
			i++
		}
		b = append(b, c)
	}
	return unitText{text: string(b), pos: append(pos, to), from: from}
}

// unhexByte returns the value of the hex digit c.
func unhexByte(c byte) byte {
	switch {
	case c <= '9':
		return c - '0'
	case c >= 'a':
		return c - 'a' + 10
	default:
		return c - 'A' + 10
	}
}

// nextPlaceholder returns where the first placeholder at or after s[i]
// starts (see placeholderLenAt), or len(s).
func nextPlaceholder(s string, i int) int {
	for i < len(s) {
		j := strings.IndexAny(s[i:], "[%")
		if j < 0 {
			break
		}
		if i += j; placeholderLenAt(s, i) > 0 {
			return i
		}
		i++
	}
	return len(s)
}

// skipInlineSpaces returns where the spaces and tabs from s[i] on end.
func skipInlineSpaces(s string, i int) int {
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	return i
}

// remove records that the pass removed s[from:to].
func (w *spanWriter) remove(from, to int) {
	if w.tr != nil && from < to {
		w.tr.removed = append(w.tr.removed, [2]int{from, to})
	}
}

// done returns the output of the pass, which goes on with s[last:]: s itself
// when the pass wrote nothing.
func (w *spanWriter) done(last int) string {
	if last == 0 && w.b.Len() == 0 {
		if w.tr != nil {
			w.tr.runs = identityRuns(len(w.s))
		}
		return w.s
	}
	w.copy(last, len(w.s))
	return w.b.String()
}

// failClosed returns redactedSensitiveTextPlaceholder for s as a whole,
// which keeps no byte of it.
func (w *spanWriter) failClosed() string {
	if w.tr != nil {
		w.tr.runs = w.tr.runs[:0]
		w.tr.removed = append(w.tr.removed[:0], [2]int{0, len(w.s)})
	}
	return redactedSensitiveTextPlaceholder
}

// appendRun appends r to runs, merging it into the last run when it goes on
// from there in both texts.
func appendRun(runs []copyRun, r copyRun) []copyRun {
	if r.n <= 0 {
		return runs
	}
	if k := len(runs) - 1; k >= 0 && runs[k].out+runs[k].n == r.out && runs[k].in+runs[k].n == r.in {
		runs[k].n += r.n
		return runs
	}
	return append(runs, r)
}

// identityRuns returns the runs of a text of n bytes copied as a whole.
func identityRuns(n int) []copyRun {
	if n == 0 {
		return nil
	}
	return []copyRun{{0, 0, n}}
}

// composeRuns returns the runs that map the output of outer, a pass over
// the output of inner, back to the input of inner. Both are sorted by out.
func composeRuns(outer, inner []copyRun) []copyRun {
	var res []copyRun
	for _, o := range outer {
		j := sort.Search(len(inner), func(j int) bool { return inner[j].out+inner[j].n > o.in })
		for ; j < len(inner) && inner[j].out < o.in+o.n; j++ {
			r := inner[j]
			lo, hi := max(o.in, r.out), min(o.in+o.n, r.out+r.n)
			res = appendRun(res, copyRun{o.out + lo - o.in, r.in + lo - r.out, hi - lo})
		}
	}
	return res
}

// mapSpans returns the spans of the input of runs that the spans of their
// output copy, dropping the bytes that no run copies.
func mapSpans(spans [][2]int, runs []copyRun) [][2]int {
	var res [][2]int
	for _, sp := range spans {
		j := sort.Search(len(runs), func(j int) bool { return runs[j].out+runs[j].n > sp[0] })
		for ; j < len(runs) && runs[j].out < sp[1]; j++ {
			r := runs[j]
			lo, hi := max(sp[0], r.out), min(sp[1], r.out+r.n)
			res = append(res, [2]int{r.in + lo - r.out, r.in + hi - r.out})
		}
	}
	return res
}

// tracedPass is one pass over a text that records its copies and removals
// in tr when tr is not nil.
type tracedPass func(s string, tr *passTrace) string

// passChain runs passes over a text, each over the output of the one before,
// composing their traces when trace is set.
type passChain struct {
	s         string
	trace     bool
	runs      []copyRun
	removed   [][2]int
	unaligned bool
}

// startPasses returns the chain of passes over s that records its trace in
// tr, when tr is not nil, at its end (see passChain.end).
func startPasses(s string, tr *passTrace) passChain {
	c := passChain{s: s, trace: tr != nil}
	if c.trace {
		c.runs = identityRuns(len(s))
	}
	return c
}

// run runs p over the output of the passes so far. A pass that returns its
// input as it is and removes nothing leaves the trace as it is.
func (c *passChain) run(p tracedPass) {
	if !c.trace {
		c.s = p(c.s, nil)
		return
	}
	var pt passTrace
	next := p(c.s, &pt)
	c.unaligned = c.unaligned || pt.unaligned
	if next == c.s && len(pt.removed) == 0 {
		return
	}
	c.removed = append(c.removed, mapSpans(pt.removed, c.runs)...)
	c.runs = composeRuns(pt.runs, c.runs)
	c.s = next
}

// end returns the output of the chain and records in tr, when it is not nil,
// the runs of it copied from the text, the spans of the text removed, and
// whether a pass left a rewritten URI unaligned.
func (c *passChain) end(tr *passTrace) string {
	if tr != nil {
		tr.runs, tr.removed, tr.unaligned = c.runs, c.removed, c.unaligned
	}
	return c.s
}

// isExemptSpace reports whether c is whitespace, which carries nothing of a
// value by itself, so that the backstop does not count it (see
// keepsRemovedByte): the scanners and the expressions may each keep or drop
// the spaces around a delimiter.
func isExemptSpace(c byte) bool {
	return c == ' ' || ('\t' <= c && c <= '\r')
}

// trimRemovedEdges returns s[from:to], a span that a legacy pass removed,
// without the whitespace, control bytes and quotes, possibly escaped, at
// its ends: the quotes around a value, which the passes write again around
// the placeholder, and the spaces around it.
func trimRemovedEdges(s string, from, to int) (int, int) {
	for from < to {
		switch c := s[from]; {
		case isDelimSpace(c) || isQuoteByte(c):
			from++
			continue
		case c == '\\' && from+1 < to && isQuoteByte(s[from+1]):
			from += 2
			continue
		}
		break
	}
	for to > from && (isDelimSpace(s[to-1]) || isQuoteByte(s[to-1])) {
		to--
		if isQuoteByte(s[to]) && to > from && s[to-1] == '\\' {
			to--
		}
	}
	return from, to
}

// redactionPlaceholders are the placeholders that this package writes for a
// redacted value or text, which a text that it scrubs again may hold.
var redactionPlaceholders = [...]string{redactedValue, redactedURIPlaceholder, redactedSensitiveTextPlaceholder}

// wholePlaceholderBytes marks the bytes of s that belong to a whole
// placeholder value (see redactionPlaceholders): a placeholder, possibly in
// one pair of quotes (", ', \" or \'), that a value boundary precedes (the
// start of s, whitespace, a control byte, a delimiter, a value stop or an
// opening bracket) and follows (see isRedactedPlaceholderBoundary). A
// placeholder glued to other bytes, as the [REDACTED] of [REDACTED]SECRET or
// x/[REDACTED], is not a whole value, and its bytes count as any other.
func wholePlaceholderBytes(s string) []bool {
	marked := make([]bool, len(s))
	for p := 0; p < len(s); {
		i := strings.Index(s[p:], "[REDACTED")
		if i < 0 {
			break
		}
		i += p
		p = i + 1
		n := 0
		for _, ph := range redactionPlaceholders {
			if strings.HasPrefix(s[i:], ph) {
				n = len(ph)
				break
			}
		}
		if n == 0 {
			continue
		}
		start, end := i, i+n
		var quote string
		if start > 0 && isQuoteByte(s[start-1]) {
			quote = s[start-1 : start]
			if start > 1 && s[start-2] == '\\' {
				quote = s[start-2 : start]
			}
		}
		if quote != "" && strings.HasPrefix(s[end:], quote) {
			start -= len(quote)
			end += len(quote)
		} else {
			quote = ""
		}
		if start > 0 && !isPlaceholderLeftBoundary(s[start-1]) {
			continue
		}
		if !isRedactedPlaceholderBoundary(s[end:]) {
			continue
		}
		for b := start; b < end; b++ {
			marked[b] = true
		}
		p = end
	}
	return marked
}

// isPlaceholderLeftBoundary reports whether c, the byte before a placeholder
// or its opening quote, ends the text before a value.
func isPlaceholderLeftBoundary(c byte) bool {
	return isDelimSpace(c) || isTextValueStop(c) || strings.IndexByte(":=([{", c) >= 0
}

// spanCheck compares scrubbed, the scrub of raw, with what a legacy scrub
// removes from raw, byte by byte (see leavesLegacyValue).
type spanCheck struct {
	raw, scrubbed string
	// kept holds the runs of scrubbed copied from raw, sorted by in.
	kept []copyRun
	// placeholder holds wholePlaceholderBytes of raw.
	placeholder []bool
	// keyBytes holds redactedKeyBytesWith of scrubbed, for the expressions
	// on main and, at index 1, for every key that isSensitiveKey accepts.
	keyBytes [2][]bool
}

func newSpanCheck(raw, scrubbed string, kept []copyRun) *spanCheck {
	sortCmp := func(a, b copyRun) int { return cmp.Compare(a.in, b.in) }
	sorted := slices.IsSortedFunc(kept, sortCmp)
	if !sorted {
		kept = slices.SortedFunc(slices.Values(kept), sortCmp)
	}
	return &spanCheck{raw: raw, scrubbed: scrubbed, kept: kept}
}

// keepsRemovedByte reports whether scrubbed keeps a byte of raw that a legacy
// scrub removes, as one of the spans removed. The ends of each span are
// trimmed (see trimRemovedEdges). Whitespace does not count (see
// isExemptSpace), nor does a byte of a whole placeholder value already in raw
// (see wholePlaceholderBytes), nor a byte that scrubbed keeps as part of a
// sensitive key whose value it redacts (see redactedKeyBytesWith): the
// expressions redacted such a key with the value of an earlier pair when they
// did not read it as a key. compoundKeys reports that the legacy scrub reads
// every key that isSensitiveKey accepts.
func (c *spanCheck) keepsRemovedByte(removed [][2]int, compoundKeys bool) bool {
	for _, sp := range removed {
		from, to := trimRemovedEdges(c.raw, sp[0], sp[1])
		if from >= to {
			continue
		}
		j := sort.Search(len(c.kept), func(j int) bool { return c.kept[j].in+c.kept[j].n > from })
		for ; j < len(c.kept) && c.kept[j].in < to; j++ {
			r := c.kept[j]
			for p := max(from, r.in); p < min(to, r.in+r.n); p++ {
				if isExemptSpace(c.raw[p]) {
					continue
				}
				if c.placeholder == nil {
					c.placeholder = wholePlaceholderBytes(c.raw)
				}
				if c.placeholder[p] {
					continue
				}
				idx := 0
				if compoundKeys {
					idx = 1
				}
				if c.keyBytes[idx] == nil {
					c.keyBytes[idx] = redactedKeyBytesWith(c.scrubbed, compoundKeys)
				}
				if !c.keyBytes[idx][r.out+p-r.in] {
					return true
				}
			}
		}
	}
	return false
}
