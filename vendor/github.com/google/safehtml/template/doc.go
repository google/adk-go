// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

/*
Package template (safehtml/template) implements data-driven templates for
generating HTML output safe against code injection. It provides an interface
similar to that of package html/template, but produces HTML output that is more
secure. Therefore, it should be used instead of html/template to render HTML.

The documentation here focuses on the security features of the package. For
information about how to program the templates themselves, see the
documentation for text/template.


Basic usage

This package provides an API almost identical to that of text/template and
html/template to parse and execute HTML templates safely.

  tmpl := template.Must(template.New("name").Parse(`<div>Hello {{.}}</div>`))
  err := tmpl.Execute(out, data)

If successful, out will contain code-injection-safe HTML. Otherwise, err's
string representation will describe the error that occurred.

Elements of data might be modified at run time before being included in out, or
rejected completely if such a conversion is not possible. Pass values of
appropriate types from package safehtml to ensure that they are included in the
template's HTML output in their expected form. More details are provided below
in "Contextual autosanitization" and "Sanitization contexts".


Security improvements

safehtml/template produces HTML more resistant to code injection than
html/template because it:
  * Allows values of types only from package safehtml to bypass run-time
    sanitization. These types represent values that are known---by construction
    or by run-time sanitization---to be safe for use in various HTML contexts
    without being processed by certain sanitization functions.
  * Does not attempt to escape CSS or JavaScript. Instead of attempting to
    parse and escape these complex languages, safehtml/template allows values
    of only the appropriate types from package safehtml (e.g. safehtml.Style,
    safehtml.Script) to be used in these contexts, since they are already
    guaranteed to be safe.
  * Emits an error if user data is interpolated in unsafe contexts, such as
    within disallowed elements or unquoted attribute values.
  * Only loads templates from trusted sources. This ensures that the contents
    of the template are always under programmer control. More details are
    provided below in "Trusted template sources".
  * Differentiates between URLs that load code and those that do not. URLs in
    the former category must be supplied to the template as values of type
    safehtml.TrustedResourceURL, whose type contract promises that the URL
    identifies a trustworthy resource. URLs in the latter category can be
    sanitized at run time.


Threat model

safehtml/template assumes that programmers are trustworthy. Therefore, data
fully under programmer control, such as string literals, are considered safe.
The types from package safehtml are designed around this same assumption, so
their type contracts are trusted by this package.

safehtml/template considers all other data values untrustworthy and
conservatively assumes that such values could result in a code-injection
vulnerability if included verbatim in HTML.


Trusted template sources

safehtml/template loads templates only from trusted sources. Therefore, template
text, file paths, and file patterns passed to Parse* functions and methods must
be entirely under programmer control.

This constraint is enforced by using unexported string types for the parameters
of Parse* functions and methods, such as trustedFilePattern for ParseGlob.
The only values that may be assigned to these types (and thus provided as
arguments) are untyped string constants such as string literals, which are
always under programmer control.


Contextual autosanitization

Code injection vulnerabilities, such as cross-site scripting (XSS), occur when
untrusted data values are embedded in a HTML document. For example,

  import "text/template"
  ...
  var t = template.Must(template.New("foo").Parse(`<a href="{{ .X }}">{{ .Y }}</a>`))
  func renderHTML(x, y string) string {
    var out bytes.Buffer
    err := t.Execute(&out, struct{ X, Y string }{x, y})
    // Error checking elided
    return out.String()
  }

If x and y originate from user-provided data, an attacker who controls these
strings could arrange for them to contain the following values:

  x = "javascript:evil()"
  y = "</a><script>alert('pwned')</script><a>"

which will cause renderHTML to return the following unsafe HTML:

  <a href="javascript:evil()"></a><script>alert('pwned')</script><a></a>

To prevent such vulnerabilities, untrusted data must be sanitized before being
included in HTML. A sanitization function takes untrusted data and returns a
string that will not create a code-injection vulnerability in the destination
context. The function might return the input unchanged if it deems it safe,
escape special runes in the input's string representation to prevent them from
triggering undesired state changes in the HTML parser, or entirely replace the
input by an innocuous string (also known as "filtering"). If none of these
conversions are possible, the sanitization function aborts template processing.

safehtml/template contextually autosanitizes untrusted data by adding
appropriate sanitization functions to template actions to ensure that the
action output is safe to include in the HTML context in which the action
appears. For example, in

  import "safehtml/template"
  ...
  var t = template.Must(template.New("foo").Parse(`<a href="{{ .X }}">{{ .Y }}</a>`))
  func renderHTML(x, y string) string {
    var out bytes.Buffer
    err := t.Execute(&out, struct{ X, Y string }{x, y})
    // Error checking elided
    return out.String()
  }

the contextual autosanitizer rewrites the template to

  <a href="{{ .X | _sanitizeTrustedResourceURLOrURL | _sanitizeHTML }}">{{ .Y | _sanitizeHTML }}</a>

so that the template produces the following safe, sanitized HTML output (split
across multiple lines for clarity):

  <a href="about:invalid#zGoSafez">
  &lt;/a&gt;&lt;script&gt;alert(&#39;pwned&#39;)&lt;/script&gt;&lt;a&gt;
  </a>

Similar template systems such as html/template, Soy, and Angular, refer to this
functionality as "contextual autoescaping". safehtml/template uses the term
"autosanitization" instead of "autoescaping" since "sanitization" broadly
captures the operations of escaping and filtering.


Sanitization contexts

The types of sanitization functions inserted into an action depend on the
action's sanitization context, which is determined by its surrounding text.
The following table describes these sanitization contexts.

  +--------------------+----------------------------------+------------------------------+-----------------------+
  | Context            | Examples                         | Safe types                   | Run-time sanitizer    |
  |--------------------+----------------------------------+------------------------------+-----------------------+
  | HTMLContent        | Hello {{.}}                      | safehtml.HTML                | safehtml.HTMLEscaped  |
  |                    | <title>{{.}}</title>             |                              |                       |
  +--------------------------------------------------------------------------------------------------------------+
  | HTMLValOnly        | <iframe srcdoc="{{.}}"></iframe> | safehtml.HTML*               | N/A                   |
  +--------------------------------------------------------------------------------------------------------------+
  | URL                | <q cite="{{.}}">Cite</q>         | safehtml.URL                 | safehtml.URLSanitized |
  +--------------------------------------------------------------------------------------------------------------+
  | URL or             | <a href="{{.}}">Link</a>         | safehtml.URL                 | safehtml.URLSanitized |
  | TrustedResourceURL |                                  | safehtml.TrustedResourceURL  |                       |
  +--------------------------------------------------------------------------------------------------------------+
  | TrustedResourceURL | <script src="{{.}}"></script>    | safehtml.TrustedResourceURL† | N/A                   |
  +--------------------------------------------------------------------------------------------------------------+
  | Script             | <script>{{.}}</script>           | safehtml.Script*             | N/A                   |
  +--------------------------------------------------------------------------------------------------------------+
  | Style              | <p style="{{.}}">Paragraph</p>   | safehtml.Style*              | N/A                   |
  +--------------------------------------------------------------------------------------------------------------+
  | Stylesheet         | <style>{{.}}</style>             | safehtml.StyleSheet*         | N/A                   |
  +--------------------------------------------------------------------------------------------------------------+
  | Identifier         | <h1 id="{{.}}">Hello</h1>        | safehtml.Identifier*         | N/A                   |
  +--------------------------------------------------------------------------------------------------------------+
  | Enumerated value   | <a target="{{.}}">Link</a>       | Allowed string values        | N/A                   |
  |                    |                                  | ("_self" or "_blank" for     |                       |
  |                    |                                  | the given example)           |                       |
  +--------------------------------------------------------------------------------------------------------------+
  | None               | <h1 class="{{.}}">Hello</h1>     | N/A (any type allowed)       | N/A (any type         |
  |                    |                                  |                              |      allowed)         |
  +--------------------+----------------------------------+------------------------------+-----------------------+
   *: Values only of this type are allowed in this context. Other values will trigger a run-time error.
   †: If the action is a prefix of the attribute value, values only of this type are allowed.
      Otherwise, values of any type are allowed. See "Substitutions in URLs" for more details.

For each context, the function named in "Run-time sanitizer" is called to
sanitize the output of the action. However, if the action outputs a value of
any of the types listed in "Safe types", the run-time sanitizer is not called.
For example, in

  <title>{{ .X }}</title>

if X is a string value, a HTML sanitizer that calls safehtml.HTMLEscaped will be
added to the action to sanitize X.

  // _sanitizeHTML calls safehtml.HTMLEscaped.
  <title>{{ .X | _sanitizeHTML }}</title>

However, if X is a safehtml.HTML value, _sanitizeHTML will not change its
value, since safehtml.HTML values are already safe to use in HTML contexts.
Therefore, the string contents of X will bypass context-specific
sanitization (in this case, HTML escaping) and appear unchanged in the
template's HTML output. Note that in attribute value contexts, HTML escaping
will always take place, whether or not context-specific sanitization is
performed. More details can be found at the end of this section.

In certain contexts, the autosanitizer allows values only of that context's
"Safe types". Any other values will trigger an error and abort template
processing. For example, the template

  <style>{{ .X }}</style>

triggers a run-time error if X is not a safehtml.StyleSheet. Otherwise, the
string form of X will appear unchanged in the output. The only exception to
this behavior is in TrustedResourceURL sanitization contexts, where actions may
output data of any type if the action occurs after a safe attribute value prefix.
More details can be found below in "Substitutions in URLs".


Unconditional sanitization

In attribute value contexts, action outputs are always HTML-escaped after
context-specific sanitization to ensure that the attribute values cannot change
change the structure of the surrounding HTML tag. In URL or TrustedResourceURL
sanitization contexts, action outputs are additionally URL-normalized to reduce
the likelihood of downstream URL-parsing bugs. For example, the template

  <a href="{{ .X }}">Link</a>
  <p id="{{ .Y }}">Text</p>

is rewritten by the autosanitizer into

  // _sanitizeHTML calls safehtml.HTMLEscaped.
  <a href="{{ .X | _sanitizeTrustedResourceURLOrURL | _normalizeURL | _sanitizeHTML }}">Link</a>
  <p id="{{ .Y | _sanitizeIdentifier | _sanitizeHTML }}">Text</p>

Even if X is a safehtml.URL or safehtml.TrustedResourceURL value, which
remains unchanged after _sanitizeTrustedResourceURLOrURL, X will still be
URL-normalized and HTML-escaped. Likewise, Y will still be HTML-escaped even if
its string form is left unchanged by _sanitizeIdentifier.


Substitutions in URLs

Values of any type may be substituted into attribute values in URL and
TrustedResourceURL sanitization contexts only if the action is preceded by a
safe URL prefix. For example, in

  <q cite="http://www.foo.com/{{ .PathComponent }}">foo</q>

Since "http://www.foo.com/" is a safe URL prefix, PathComponent can safely be
interpolated into this URL sanitization context after URL normalization.
Similarly, in

  <script src="https://www.bar.com/{{ .PathComponent }}"></script>

Since "https://www.bar.com/" is a safe TrustedResourceURL prefix, PathComponent
can safely be interpolated into this TrustedResourceURL sanitization context
after URL escaping. Substitutions after a safe TrustedResourceURL prefix are
escaped instead of normalized to prevent the injection of any new URL
components, including additional path components. URL escaping also takes place
in URL sanitization contexts where the substitutions occur in the query or
fragment part of the URL, such as in:

  <a href="/foo?q={{ .Query }}&hl={{ .LangCode }}">Link</a>

A URL prefix is considered safe in a URL sanitization context if it does
not end in an incomplete HTML character reference (e.g. https&#1) or incomplete
percent-encoding character triplet (e.g. /fo%6), does not contain whitespace or control
characters, and one of the following is true:
  * The prefix has a safe scheme (i.e. http, https, mailto, or ftp).
  * The prefix has the data scheme with base64 encoding and an allowed audio, image,
    or video MIME type (e.g. data:img/jpeg;base64, data:video/mp4;base64).
  * The prefix has no scheme at all, and cannot be interpreted as a scheme prefix (e.g. /path).

A URL prefix is considered safe in a TrustedResourceURL sanitization context if it does
not end in an incomplete HTML character reference (e.g. https&#1) or incomplete
percent-encoding character triplet (e.g. /fo%6), does not contain white space or control
characters, and one of the following is true:
  * The prefix has the https scheme and contains a domain name (e.g. https://www.foo.com).
  * The prefix is scheme-relative and contains a domain name (e.g. //www.foo.com/).
  * The prefix is path-absolute and contains a path (e.g. /path).
  * The prefix is "about:blank".
*/
package template
