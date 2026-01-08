// Copyright 2025 Google LLC
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

// Package oauth2handler provides OAuth2 flow handling for CLI applications.
package oauth2handler

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"
	"google.golang.org/adk/auth"
)

// FlowType represents the OAuth2 flow type.
type FlowType string

const (
	// FlowTypeAuthCode uses Authorization Code flow with local HTTP server.
	FlowTypeAuthCode FlowType = "auth_code"
	// FlowTypeDevice uses Device Authorization flow.
	FlowTypeDevice FlowType = "device"
)

// Handler handles OAuth2 flows for CLI applications.
type Handler struct {
	flowType     FlowType
	port         int
	callbackPath string
	mu           sync.Mutex
	server       *http.Server
	authCode     string
	authErr      error
	done         chan struct{}
}

// New creates a new OAuth2 handler with the specified port and callback path.
func New(flowType FlowType, port int, callbackPath string) *Handler {
	if callbackPath == "" {
		callbackPath = "/callback"
	}
	if !strings.HasPrefix(callbackPath, "/") {
		callbackPath = "/" + callbackPath
	}
	return &Handler{
		flowType:     flowType,
		port:         port,
		callbackPath: callbackPath,
	}
}

// HandleAuthRequest processes an OAuth2 authorization request.
// It returns the authorization code or an error.
func (h *Handler) HandleAuthRequest(ctx context.Context, authConfig *auth.AuthConfig) (*auth.AuthCredential, error) {
	if authConfig == nil || authConfig.ExchangedAuthCredential == nil {
		return nil, fmt.Errorf("invalid auth config")
	}

	oauth2Cred := authConfig.ExchangedAuthCredential.OAuth2
	if oauth2Cred == nil {
		return nil, fmt.Errorf("not an OAuth2 credential")
	}

	switch h.flowType {
	case FlowTypeAuthCode:
		return h.handleAuthCodeFlow(ctx, authConfig)
	case FlowTypeDevice:
		return h.handleDeviceFlow(ctx, authConfig)
	default:
		return nil, fmt.Errorf("unsupported flow type: %s", h.flowType)
	}
}

// handleAuthCodeFlow implements Authorization Code flow with local HTTP server.
func (h *Handler) handleAuthCodeFlow(ctx context.Context, authConfig *auth.AuthConfig) (*auth.AuthCredential, error) {
	oauth2Cred := authConfig.ExchangedAuthCredential.OAuth2

	// Use configured port
	port := h.port
	var redirectURI string
	if h.callbackPath == "/" {
		redirectURI = fmt.Sprintf("http://localhost:%d/", port)
	} else {
		redirectURI = fmt.Sprintf("http://localhost:%d%s", port, h.callbackPath)
	}
	oauth2Cred.RedirectURI = redirectURI

	// Rebuild auth URI with updated redirect_uri
	authURI, err := url.Parse(oauth2Cred.AuthURI)
	if err != nil {
		return nil, fmt.Errorf("invalid auth URI: %w", err)
	}
	query := authURI.Query()
	query.Set("redirect_uri", redirectURI)
	authURI.RawQuery = query.Encode()

	// Setup HTTP server for callback
	h.done = make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc(h.callbackPath, h.handleCallback)

	h.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: mux,
	}

	// Start server
	go func() {
		if err := h.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			h.mu.Lock()
			h.authErr = err
			h.mu.Unlock()
			close(h.done)
		}
	}()

	// Open browser
	fmt.Printf("\nOpening browser for OAuth2 authorization...\n")
	fmt.Printf("If your browser doesn't open automatically, please visit:\n%s\n\n", authURI.String())
	if err := openBrowser(authURI.String()); err != nil {
		fmt.Printf("Failed to open browser: %v\n", err)
	}

	// Wait for callback or context cancellation
	select {
	case <-h.done:
		h.server.Close()
	case <-ctx.Done():
		h.server.Close()
		return nil, ctx.Err()
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	if h.authErr != nil {
		return nil, h.authErr
	}

	// Get token endpoint from auth scheme
	var tokenEndpoint, clientID, clientSecret string
	switch scheme := authConfig.AuthScheme.(type) {
	case *auth.OAuth2Scheme:
		if scheme.Flows.AuthorizationCode != nil {
			tokenEndpoint = scheme.Flows.AuthorizationCode.TokenURL
		}
		if cred := authConfig.ExchangedAuthCredential; cred != nil && cred.OAuth2 != nil {
			clientID = cred.OAuth2.ClientID
			clientSecret = cred.OAuth2.ClientSecret
		}
	case *auth.OpenIDConnectScheme:
		tokenEndpoint = scheme.TokenEndpoint
		if cred := authConfig.ExchangedAuthCredential; cred != nil && cred.OAuth2 != nil {
			clientID = cred.OAuth2.ClientID
			clientSecret = cred.OAuth2.ClientSecret
		}
	default:
		return nil, fmt.Errorf("unsupported auth scheme type: %T", authConfig.AuthScheme)
	}

	if tokenEndpoint == "" {
		return nil, fmt.Errorf("no token endpoint found in auth scheme")
	}

	// Exchange auth code for access token
	token, err := h.exchangeCodeForToken(ctx, tokenEndpoint, h.authCode, redirectURI, clientID, clientSecret)
	if err != nil {
		return nil, fmt.Errorf("failed to exchange code for token: %w", err)
	}

	// Return credential with access token
	result := authConfig.ExchangedAuthCredential.Copy()
	result.OAuth2.AccessToken = token.AccessToken
	result.OAuth2.RefreshToken = token.RefreshToken
	if token.ExpiresIn > 0 {
		result.OAuth2.ExpiresAt = time.Now().Unix() + int64(token.ExpiresIn)
	}
	return result, nil
}

// tokenResponse represents the OAuth2 token response.
type tokenResponse struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
}

// exchangeCodeForToken exchanges an authorization code for an access token.
func (h *Handler) exchangeCodeForToken(ctx context.Context, tokenEndpoint, code, redirectURI, clientID, clientSecret string) (*tokenResponse, error) {
	data := url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"redirect_uri":  {redirectURI},
		"client_id":     {clientID},
		"client_secret": {clientSecret},
	}

	req, err := http.NewRequestWithContext(ctx, "POST", tokenEndpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	req.URL.RawQuery = data.Encode()

	// For GitHub, we need to use form post body instead of query
	req.URL.RawQuery = ""
	req.Body = http.NoBody
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	// Create POST body
	req.Body = io.NopCloser(strings.NewReader(data.Encode()))
	req.ContentLength = int64(len(data.Encode()))

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("token request failed: %s - %s", resp.Status, string(body))
	}

	var token tokenResponse
	if err := json.Unmarshal(body, &token); err != nil {
		return nil, fmt.Errorf("failed to parse token response: %w", err)
	}

	if token.AccessToken == "" {
		return nil, fmt.Errorf("no access token in response: %s", string(body))
	}

	return &token, nil
}

// handleCallback handles the OAuth2 callback request.
func (h *Handler) handleCallback(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Extract code and state from query parameters
	code := r.URL.Query().Get("code")
	errorParam := r.URL.Query().Get("error")

	if errorParam != "" {
		errorDesc := r.URL.Query().Get("error_description")
		h.authErr = fmt.Errorf("OAuth error: %s - %s", errorParam, errorDesc)
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "<html><body><h1>Authorization Failed</h1><p>%s</p><p>You can close this window.</p></body></html>", h.authErr)

		// Delay before signaling done to ensure response is sent
		go func() {
			time.Sleep(100 * time.Millisecond)
			close(h.done)
		}()
		return
	}

	if code == "" {
		h.authErr = fmt.Errorf("no authorization code received")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "<html><body><h1>Authorization Failed</h1><p>No code received</p><p>You can close this window.</p></body></html>")

		// Delay before signaling done to ensure response is sent
		go func() {
			time.Sleep(100 * time.Millisecond)
			close(h.done)
		}()
		return
	}

	h.authCode = code
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "<html><body><h1>Authorization Successful!</h1><p>You can close this window and return to your application.</p></body></html>")

	// Delay before signaling done to ensure response is sent
	go func() {
		time.Sleep(100 * time.Millisecond)
		close(h.done)
	}()
}

// handleDeviceFlow implements Device Authorization flow (RFC 8628).
func (h *Handler) handleDeviceFlow(ctx context.Context, authConfig *auth.AuthConfig) (*auth.AuthCredential, error) {
	oauth2Cred := authConfig.ExchangedAuthCredential.OAuth2

	// Get endpoints from auth scheme
	var deviceAuthURL, tokenURL string
	var scopes []string
	var clientID, clientSecret string

	switch scheme := authConfig.AuthScheme.(type) {
	case *auth.OAuth2Scheme:
		if scheme.Flows != nil && scheme.Flows.AuthorizationCode != nil {
			// Use authorization code flow endpoints (device flow often shares the same token endpoint)
			tokenURL = scheme.Flows.AuthorizationCode.TokenURL
			// Device auth URL is typically at the same host, often /device/code or similar
			// This can be customized per provider
			for scope := range scheme.Flows.AuthorizationCode.Scopes {
				scopes = append(scopes, scope)
			}
		}
		if cred := authConfig.ExchangedAuthCredential; cred != nil && cred.OAuth2 != nil {
			clientID = cred.OAuth2.ClientID
			clientSecret = cred.OAuth2.ClientSecret
		}
	case *auth.OpenIDConnectScheme:
		tokenURL = scheme.TokenEndpoint
		scopes = scheme.Scopes
		if cred := authConfig.ExchangedAuthCredential; cred != nil && cred.OAuth2 != nil {
			clientID = cred.OAuth2.ClientID
			clientSecret = cred.OAuth2.ClientSecret
		}
	default:
		return nil, fmt.Errorf("unsupported auth scheme type for device flow: %T", authConfig.AuthScheme)
	}

	// Try to get device auth URL from credential if provided
	if oauth2Cred.AuthURI != "" {
		// Try to derive device auth URL from auth URI
		authURI := oauth2Cred.AuthURI

		// GitHub pattern: /login/oauth/authorize -> /login/device/code
		if strings.Contains(authURI, "github.com/login/oauth/authorize") {
			deviceAuthURL = strings.Replace(authURI, "/login/oauth/authorize", "/login/device/code", 1)
			// Remove any query parameters
			if idx := strings.Index(deviceAuthURL, "?"); idx != -1 {
				deviceAuthURL = deviceAuthURL[:idx]
			}
		} else if idx := strings.Index(authURI, "/authorize"); idx != -1 {
			// Generic pattern: replace /authorize with /device/code
			deviceAuthURL = authURI[:idx] + "/device/code"
		} else if idx := strings.Index(authURI, "/oauth2/"); idx != -1 {
			deviceAuthURL = authURI[:idx] + "/oauth2/device/code"
		}
	}

	if deviceAuthURL == "" {
		return nil, fmt.Errorf("device authorization endpoint not configured - please provide device_authorization_endpoint")
	}

	if tokenURL == "" {
		return nil, fmt.Errorf("token endpoint not configured in auth scheme")
	}

	// Create OAuth2 config for device flow
	config := &oauth2.Config{
		ClientID:     clientID,
		ClientSecret: clientSecret,
		Endpoint: oauth2.Endpoint{
			DeviceAuthURL: deviceAuthURL,
			TokenURL:      tokenURL,
		},
		Scopes: scopes,
	}

	// Step 1: Request device authorization
	fmt.Println("\nInitiating OAuth2 Device Authorization Flow...")
	deviceAuth, err := config.DeviceAuth(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get device authorization: %w", err)
	}

	// Step 2: Display user code and verification URI
	fmt.Println()
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  To authorize this application, please:")
	fmt.Println()
	fmt.Printf("  1. Go to: %s\n", deviceAuth.VerificationURI)
	fmt.Printf("  2. Enter code: %s\n", deviceAuth.UserCode)
	fmt.Println()
	if deviceAuth.VerificationURIComplete != "" {
		fmt.Printf("  Or visit: %s\n", deviceAuth.VerificationURIComplete)
		fmt.Println()
	}
	fmt.Println("  Waiting for authorization...")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println()

	// Try to open the verification URI in browser
	if deviceAuth.VerificationURIComplete != "" {
		_ = openBrowser(deviceAuth.VerificationURIComplete)
	} else {
		_ = openBrowser(deviceAuth.VerificationURI)
	}

	// Step 3: Poll for token (DeviceAccessToken handles polling with proper interval)
	token, err := config.DeviceAccessToken(ctx, deviceAuth)
	if err != nil {
		return nil, fmt.Errorf("device authorization failed: %w", err)
	}

	fmt.Println("✓ Authorization successful!")
	fmt.Println()

	// Return credential with access token
	result := authConfig.ExchangedAuthCredential.Copy()
	result.OAuth2.AccessToken = token.AccessToken
	result.OAuth2.RefreshToken = token.RefreshToken
	if !token.Expiry.IsZero() {
		result.OAuth2.ExpiresAt = token.Expiry.Unix()
	}
	return result, nil
}

// openBrowser opens the specified URL in the default browser.
func openBrowser(url string) error {
	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "linux":
		cmd = exec.Command("xdg-open", url)
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		return fmt.Errorf("unsupported platform: %s", runtime.GOOS)
	}

	return cmd.Start()
}

// Close closes the handler and any resources.
func (h *Handler) Close() error {
	if h.server != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return h.server.Shutdown(ctx)
	}
	return nil
}
