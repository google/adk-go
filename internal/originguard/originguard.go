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

// Package originguard decides whether an HTTP request or WebSocket upgrade may
// be served, from its Origin and Host headers.
//
// It answers two separate questions, in this order:
//
//  1. Was this request pointed at us by rebound DNS? A connection accepted on a
//     loopback address can only have come from this machine, so a Host naming
//     anything but loopback means the name resolved here without belonging
//     here. Origin cannot catch that, because a rebound page looks same-origin
//     to the browser and so may carry no Origin at all.
//  2. Is the Origin one we serve? Absent, matching the request's own origin, or
//     on the configured allowlist.
//
// Both rest on the server being reachable from this machine only. A server
// reachable over a network cannot tell a rebound page from an ordinary remote
// browser: the Host it names really does resolve here, and the Origin it claims
// really is its own. Neither check fires there and the page is served. A
// container with a published port is such a server whatever address the port
// was published on, because inside the container the connection still arrives
// on a routable interface. Nothing here closes that; authentication does.
//
// Ported from adk-python's cli/api_server.py, which applies the same two checks
// as _is_dns_rebinding_request and _is_request_origin_allowed.
package originguard

import (
	"errors"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
)

// ErrHostNotAllowed is returned when the request's Host header names a host the
// server cannot legitimately be reached under, which is what DNS rebinding
// looks like from the server side.
var ErrHostNotAllowed = errors.New("host not allowed")

// ErrOriginNotAllowed is returned when the request carries an Origin the server
// does not serve.
var ErrOriginNotAllowed = errors.New("origin not allowed")

// Config is what a Policy is built from.
type Config struct {
	// BindHost is the address the server will be bound to, and arms the
	// rebinding check when it is a loopback address. Empty means the caller has
	// not said, and the check stays off.
	//
	// A wildcard address ("", ":8080", "0.0.0.0", "[::]") names every
	// interface, so it says no more about who can reach the server than saying
	// nothing does, and [Policy.servesOnlyThisMachine] treats it the same way.
	//
	// Not inferred from the accepted connection, though that would arm it for
	// more callers. A connection accepted on loopback does not mean the server
	// is bound to loopback: a sidecar proxy, an nginx proxy_pass to
	// 127.0.0.1, and the machine's own hostname on Debian, which resolves to
	// 127.0.1.1, all deliver ordinary traffic over the loopback interface under
	// a Host we would then refuse. Those requests need not carry an Origin, so
	// nothing else in this package would rescue them.
	BindHost string

	// AllowedOrigins lists the origins the server serves, as
	// scheme://host[:port]; a bare host or host:port is read as http. A single
	// "*" entry turns both checks off.
	AllowedOrigins []string
}

// Policy is an immutable decision procedure. Build one with [New]; the zero
// value is not usable.
type Policy struct {
	// allowAll disables both checks. Set by an "*" entry in the allowed
	// origins, which is an operator saying the server is meant to be reachable
	// from anywhere.
	allowAll bool
	// bindDeclared records that the caller said what it binds,
	// boundToLoopback that what it named is a loopback address, and
	// bindWildcard that it named every interface rather than one. A declared
	// loopback bind is what arms the rebinding check.
	bindDeclared    bool
	boundToLoopback bool
	bindWildcard    bool
	// origins holds the allowed origins, canonicalized.
	origins map[string]bool
	// hosts holds the host of each allowed origin, lowercased and without a
	// port. Listing an origin vouches for its host: a loopback bind behind a
	// same-machine proxy sees the proxy's name in Host, and that is a legitimate
	// way to reach the server rather than a rebinding attempt.
	hosts map[string]bool
}

// New builds a Policy from cfg.
func New(cfg Config) *Policy {
	p := &Policy{
		bindDeclared:    cfg.BindHost != "",
		boundToLoopback: cfg.BindHost != "" && isLoopbackAddr(cfg.BindHost),
		bindWildcard:    cfg.BindHost != "" && isWildcardAddr(cfg.BindHost),
		origins:         make(map[string]bool),
		hosts:           make(map[string]bool),
	}
	for _, raw := range cfg.AllowedOrigins {
		origin := strings.TrimSpace(raw)
		if origin == "" {
			continue
		}
		if origin == "*" {
			p.allowAll = true
			continue
		}
		origin = NormalizeOrigin(origin)
		p.origins[origin] = true
		if u, err := url.Parse(origin); err == nil && u.Hostname() != "" {
			p.hosts[bareHost(u.Hostname())] = true
		}
	}
	return p
}

// NormalizeOrigin turns an address into an RFC 6454 origin
// (scheme://host[:port]), the only form a browser sends in Origin and accepts
// in Access-Control-Allow-Origin.
//
// A bare host or host:port gets http:// prepended, because the addresses this
// names are local development servers. Any path, query or trailing slash is
// dropped, because an origin has none. The scheme and host are lowercased,
// which is how a browser spells them, so that a configured "http://LocalHost"
// still matches. "*" and the empty string pass through.
func NormalizeOrigin(addr string) string {
	addr = strings.TrimSpace(addr)
	if addr == "" || addr == "*" {
		return addr
	}
	if !strings.Contains(addr, "://") {
		addr = "http://" + addr
	}
	u, err := url.Parse(addr)
	if err != nil || u.Host == "" {
		// Not something we can read as an origin. Pass it through rather than
		// inventing a value: the operator sees their own input echoed back.
		return addr
	}
	return strings.ToLower(u.Scheme) + "://" + strings.ToLower(u.Host)
}

// Check reports whether the request may be served, returning
// [ErrHostNotAllowed] or [ErrOriginNotAllowed] if not.
//
// Both errors are safe to show a caller: they name the check that failed and
// nothing about the server's configuration.
func (p *Policy) Check(r *http.Request) error {
	if p.allowAll {
		return nil
	}
	if p.isDNSRebinding(r) {
		return ErrHostNotAllowed
	}
	// Absent Origin means a non-browser client, which this check was never able
	// to constrain. adk-python skips it on the same grounds.
	if origin := firstHeaderValue(r, "Origin"); origin != "" && !p.originAllowed(r, origin) {
		return ErrOriginNotAllowed
	}
	return nil
}

// CheckOrigin is the [github.com/gorilla/websocket.Upgrader] CheckOrigin hook.
//
// gorilla's default rejects an Origin whose host differs from Host, which both
// accepts a rebound page (it controls the two equally) and rejects a browser at
// an origin the operator did allow. This applies the same rules as [Check]
// instead, so the WebSocket and the REST API agree on who may connect.
func (p *Policy) CheckOrigin(r *http.Request) bool {
	if p.allowAll {
		return true
	}
	origin := firstHeaderValue(r, "Origin")
	return origin == "" || p.originAllowed(r, origin)
}

// maxOriginLen bounds the Origin we are willing to parse.
//
// A real one cannot approach it: a hostname is at most 253 bytes, plus a scheme
// and a port. net/http will hand us a header up to MaxHeaderBytes, and this
// runs before anything has authorized the caller, so the URL parse and the two
// lowercase copies inside NormalizeOrigin should not be reachable at that size.
const maxOriginLen = 2048

// originAllowed reports whether origin is one the server serves.
func (p *Policy) originAllowed(r *http.Request, rawOrigin string) bool {
	if len(rawOrigin) > maxOriginLen {
		return false
	}
	origin := NormalizeOrigin(rawOrigin)
	if p.origins[origin] {
		return true
	}
	// A server reached over loopback serves loopback pages. Without this, a
	// page rebound onto 127.0.0.1 passes the same-origin comparison below,
	// because it controls the Origin it sends and the Host it sends it to
	// alike, and they agree.
	//
	// Applied whether or not origins are configured, which is where this
	// diverges from adk-python. There the rule is skipped once allowed origins
	// are configured, and [Policy.isDNSRebinding] catches the rebound page
	// instead. That check needs a declared loopback BindHost and does not run
	// without one, while a caller that configures origins at all makes this
	// rule's condition false, so gating on it the way Python does can leave
	// both checks off at once. Configured origins are matched above and are
	// unaffected either way.
	if p.servesOnlyThisMachine(r) && !isLoopbackAddr(originHost(origin)) {
		return false
	}
	requestOrigin := effectiveRequestOrigin(r)
	return requestOrigin != "" && origin == requestOrigin
}

// servesOnlyThisMachine reports whether this server can be reached from the
// machine it runs on and nowhere else.
//
// One named address answers it outright, either way. Naming a routable one is
// an operator saying the server is exposed, and the rule above must then not
// fire at all, however the connection happened to arrive.
//
// A wildcard bind names every interface, which says neither, so the accepted
// connection stands in, as it does when nothing was declared at all. Telling
// the guard what you bind must never leave it with less to go on than telling
// it nothing.
//
// That last part diverges from adk-python, which takes a declared bind at face
// value here and so lets a wildcard one turn the rule off
// (_is_request_origin_allowed). Its comment there expects the accepted socket
// to stand in for a wildcard bind, which is what this does. The deployment it
// has in mind, a container serving a real hostname, accepts on a routable
// address and is unaffected.
//
// The accepted connection is enough here, unlike in [Policy.isDNSRebinding]:
// this narrows which Origin a browser may present, and a caller sending no
// Origin is never affected by it.
func (p *Policy) servesOnlyThisMachine(r *http.Request) bool {
	if p.bindDeclared && !p.bindWildcard {
		return p.boundToLoopback
	}
	return isLoopbackAddr(serverHost(r))
}

// isDNSRebinding reports whether the request must be rejected as possible DNS
// rebinding.
//
// This catches what [Policy.originAllowed] cannot: a rebound page's same-origin
// GET carries no Origin header, so only the Host it names gives it away.
func (p *Policy) isDNSRebinding(r *http.Request) bool {
	// Only a declared loopback bind supports this inference. On any other
	// address the server is legitimately reachable under whatever name resolves
	// to it, and rejecting those would break every deliberate LAN or public
	// deployment. See [Config.BindHost] for why the accepted connection does
	// not stand in for a declaration.
	if !p.boundToLoopback {
		return false
	}
	// Only the real Host header will do. It is a forbidden request header, so a
	// page cannot set it, whereas it may set X-Forwarded-Host or Forwarded
	// freely and those must not be able to talk the guard out of firing.
	host := r.Host
	if host == "" {
		// Browsers always send Host, so its absence is not a rebinding vector.
		// Nor is anything else: HTTP/1.1 and HTTP/2 both require it, so an
		// empty one means the request did not arrive over either.
		return false
	}
	// Host is a singleton header, so a list in it is smuggling rather than a
	// client. net/http rejects two Host headers before a handler sees them, but
	// permits a comma inside the one value. Refused outright rather than left
	// to fail the comparisons below, so that the answer does not depend on a
	// list happening to match no allowed host.
	if strings.Contains(host, ",") {
		return true
	}
	if isLoopbackAddr(host) {
		return false
	}
	return !p.hosts[bareHost(host)]
}

// Middleware applies [Policy.Check] to every request before the handler runs.
//
// Every method, not just the unsafe ones. The reads behind this server are
// whole session histories, and a rebound page's requests look same-origin to
// the browser, so neither the method nor the Origin can be relied on to
// distinguish them.
func (p *Policy) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := p.Check(r); err != nil {
			http.Error(w, "Forbidden: "+err.Error(), http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// serverHost returns the address this request was accepted on, or "" when the
// request did not come from an [net/http.Server].
//
// The accepted socket rather than a configured bind address, because it is the
// more precise of the two: under a wildcard bind it distinguishes a connection
// that arrived on loopback — a browser on this machine, possibly a rebound one
// — from one that arrived over the network, where no such inference holds.
func serverHost(r *http.Request) string {
	addr, ok := r.Context().Value(http.LocalAddrContextKey).(net.Addr)
	if !ok || addr == nil {
		return ""
	}
	return addr.String()
}

// originHost returns the host of an origin, or "" if it has none.
func originHost(origin string) string {
	u, err := url.Parse(origin)
	if err != nil {
		return ""
	}
	return u.Hostname()
}

// effectiveRequestOrigin reconstructs the origin a browser would compare
// against when deciding this request is same-origin. The result is lowercased,
// so that it can be compared with an Origin that has been through
// [NormalizeOrigin].
//
// Forwarded and X-Forwarded-* are honoured here, unlike in the rebinding check:
// behind a proxy they are the only record of what the browser actually
// addressed, and a mismatch here costs a legitimate caller a 403.
func effectiveRequestOrigin(r *http.Request) string {
	if proto, host, ok := forwardedProtoHost(r); ok {
		return originScheme(proto) + "://" + strings.ToLower(host)
	}
	host := firstHeaderValue(r, "X-Forwarded-Host")
	if host == "" {
		host = r.Host
	}
	if host == "" {
		return ""
	}
	proto := firstHeaderValue(r, "X-Forwarded-Proto")
	if proto == "" {
		proto = requestScheme(r)
	}
	return originScheme(proto) + "://" + strings.ToLower(host)
}

// forwardedProtoHost reads proto and host from the first element of an RFC 7239
// Forwarded header. Both must be present for the result to be usable.
func forwardedProtoHost(r *http.Request) (proto, host string, ok bool) {
	forwarded := r.Header.Get("Forwarded")
	if forwarded == "" {
		return "", "", false
	}
	first, _, _ := strings.Cut(forwarded, ",")
	// SplitSeq rather than Split: this runs before anything has authorized the
	// caller, and net/http accepts a header up to MaxHeaderBytes, so a large
	// one should not also cost a slice of every element in it.
	for element := range strings.SplitSeq(first, ";") {
		name, value, found := strings.Cut(element, "=")
		if !found {
			continue
		}
		value = unquote(strings.TrimSpace(value))
		switch strings.ToLower(strings.TrimSpace(name)) {
		case "proto":
			proto = value
		case "host":
			host = value
		}
	}
	if proto == "" || host == "" {
		return "", "", false
	}
	return proto, host, true
}

// requestScheme returns the scheme the request arrived over.
func requestScheme(r *http.Request) string {
	if r.TLS != nil {
		return "https"
	}
	if r.URL != nil && r.URL.Scheme != "" {
		return r.URL.Scheme
	}
	return "http"
}

// originScheme maps a request scheme onto the scheme space browsers use in
// Origin, where a WebSocket connection reports the scheme of the page.
func originScheme(scheme string) string {
	switch strings.ToLower(scheme) {
	case "ws":
		return "http"
	case "wss":
		return "https"
	default:
		return strings.ToLower(scheme)
	}
}

// firstHeaderValue returns the first comma-separated element of a header,
// trimmed, or "" when the header is absent.
func firstHeaderValue(r *http.Request, name string) string {
	value := r.Header.Get(name)
	if value == "" {
		return ""
	}
	first, _, _ := strings.Cut(value, ",")
	return strings.TrimSpace(first)
}

// unquote strips one pair of wrapping double quotes, which RFC 7239 allows
// around a Forwarded element's value.
func unquote(value string) string {
	if len(value) >= 2 && value[0] == '"' && value[len(value)-1] == '"' {
		return value[1 : len(value)-1]
	}
	return value
}

// isLoopbackAddr reports whether host, with or without a port, names a loopback
// address.
func isLoopbackAddr(host string) bool {
	bare := bareHost(host)
	if bare == "" {
		return false
	}
	if bare == "localhost" {
		return true
	}
	addr, err := netip.ParseAddr(bare)
	return err == nil && addr.IsLoopback()
}

// isWildcardAddr reports whether host, with or without a port, names every
// interface rather than one: an empty host, "0.0.0.0" or "[::]".
func isWildcardAddr(host string) bool {
	bare := bareHost(host)
	if bare == "" {
		return true
	}
	addr, err := netip.ParseAddr(bare)
	return err == nil && addr.IsUnspecified()
}

// bareHost lowercases host and strips its port and root dot, ready for
// comparison. A host it cannot parse comes back whole, so that a caller never
// reads "127.0.0.1:8080.evil.com" as loopback.
func bareHost(host string) string {
	return strings.TrimRight(strings.ToLower(stripPort(host)), ".")
}

// stripPort returns host without its port, or host unchanged when it has no
// valid one.
func stripPort(host string) string {
	var bare, suffix string
	switch {
	case strings.HasPrefix(host, "["): // [addr] or [addr]:port
		end := strings.Index(host, "]")
		if end < 0 {
			return host
		}
		bare, suffix = host[1:end], host[end+1:]
	case strings.Count(host, ":") == 1: // host:port; a bracketless IPv6 has more
		bare, suffix, _ = strings.Cut(host, ":")
		suffix = ":" + suffix
	default:
		return host
	}
	if suffix == "" {
		return bare
	}
	port, ok := strings.CutPrefix(suffix, ":")
	if !ok || port == "" {
		return host
	}
	for _, c := range []byte(port) {
		if c < '0' || c > '9' {
			return host
		}
	}
	return bare
}
