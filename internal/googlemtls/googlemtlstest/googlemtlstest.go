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

// Package googlemtlstest runs a local TLS server that stands in for the
// Google API hosts one label below googleapis.com and their mTLS names, for
// tests of clients built on googlemtls.
package googlemtlstest

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/hex"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"google.golang.org/adk/v2/internal/googlemtls"
)

// Request is what the server saw of one request.
type Request struct {
	// Host is the request's Host header (HTTP/2 :authority).
	Host string
	// ServerName is the TLS server name the client sent.
	ServerName string
	// PeerCerts is the number of certificates the client presented.
	PeerCerts int
	// Leaf is the Fingerprint of the first certificate the client presented.
	Leaf string
}

// Server is a TLS server with a certificate for *.googleapis.com and
// *.mtls.googleapis.com. It asks for a client certificate without requiring
// one and answers every request with 204.
type Server struct {
	*httptest.Server
	// Requests receives one value per request served.
	Requests chan Request
	// Roots trusts the server's certificate.
	Roots      *x509.CertPool
	clientCert tls.Certificate
}

// NewServer starts a Server that is closed when the test ends.
func NewServer(t testing.TB) *Server {
	t.Helper()
	caKey := newKey(t)
	caTmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "googlemtlstest CA"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign,
	}
	caDER, err := x509.CreateCertificate(rand.Reader, caTmpl, caTmpl, &caKey.PublicKey, caKey)
	if err != nil {
		t.Fatal(err)
	}
	ca, err := x509.ParseCertificate(caDER)
	if err != nil {
		t.Fatal(err)
	}
	leaf := func(serial int64, dns []string, usage x509.ExtKeyUsage) tls.Certificate {
		key := newKey(t)
		tmpl := &x509.Certificate{
			SerialNumber: big.NewInt(serial),
			Subject:      pkix.Name{CommonName: "googlemtlstest"},
			DNSNames:     dns,
			NotBefore:    time.Now().Add(-time.Hour),
			NotAfter:     time.Now().Add(time.Hour),
			KeyUsage:     x509.KeyUsageDigitalSignature,
			ExtKeyUsage:  []x509.ExtKeyUsage{usage},
		}
		der, err := x509.CreateCertificate(rand.Reader, tmpl, ca, &key.PublicKey, caKey)
		if err != nil {
			t.Fatal(err)
		}
		return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}
	}
	s := &Server{
		Requests:   make(chan Request, 16),
		Roots:      x509.NewCertPool(),
		clientCert: leaf(3, nil, x509.ExtKeyUsageClientAuth),
	}
	s.Roots.AddCert(ca)
	s.Server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req := Request{Host: r.Host, ServerName: r.TLS.ServerName, PeerCerts: len(r.TLS.PeerCertificates)}
		if len(r.TLS.PeerCertificates) > 0 {
			req.Leaf = Fingerprint(r.TLS.PeerCertificates[0].Raw)
		}
		s.Requests <- req
		w.WriteHeader(http.StatusNoContent)
	}))
	s.TLS = &tls.Config{
		Certificates: []tls.Certificate{leaf(2, []string{"*.googleapis.com", "*.mtls.googleapis.com"}, x509.ExtKeyUsageServerAuth)},
		ClientAuth:   tls.RequestClientCert,
	}
	s.StartTLS()
	t.Cleanup(s.Close)
	return s
}

func newKey(t testing.TB) *ecdsa.PrivateKey {
	t.Helper()
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// Provider presents the server's client certificate.
func (s *Server) Provider(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
	return &s.clientCert, nil
}

// Configure points an mTLS transport at s whatever host a request names,
// trusts s, and bypasses any proxy from the environment.
func (s *Server) Configure(tr *http.Transport) {
	addr := s.Listener.Addr().String()
	tr.Proxy = nil
	tr.DialContext = func(ctx context.Context, network, _ string) (net.Conn, error) {
		var d net.Dialer
		return d.DialContext(ctx, network, addr)
	}
	tr.TLSClientConfig.RootCAs = s.Roots
}

// Leaf returns the Fingerprint of the certificate Provider presents.
func (s *Server) Leaf() string { return Fingerprint(s.clientCert.Certificate[0]) }

// Fingerprint returns the hex SHA-256 of a DER certificate.
func Fingerprint(der []byte) string {
	sum := sha256.Sum256(der)
	return hex.EncodeToString(sum[:])
}

// Route makes transports that googlemtls.NewTransport creates during the test
// send their mTLS requests to s, while the certificate source stays the one
// in place, the environment's unless a test replaced it. The test must not
// run in parallel.
func (s *Server) Route(t testing.TB) {
	t.Helper()
	t.Cleanup(googlemtls.SetTestHooks(nil, s.Configure))
}

// Install makes transports that googlemtls.NewTransport creates during the
// test present the server's client certificate and send their mTLS requests
// to s. The test must not run in parallel.
func (s *Server) Install(t testing.TB) {
	t.Helper()
	restore := googlemtls.SetTestHooks(func() (func(*tls.CertificateRequestInfo) (*tls.Certificate, error), error) {
		return s.Provider, nil
	}, s.Configure)
	t.Cleanup(restore)
}
