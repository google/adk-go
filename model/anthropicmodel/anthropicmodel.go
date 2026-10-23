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

package anthropicmodel

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	"golang.org/x/oauth2/google"

	"google.golang.org/adk/v2/internal/version"
	"google.golang.org/adk/v2/model"
)

const (
	cloudPlatformScope           = "https://www.googleapis.com/auth/cloud-platform"
	defaultResponseHeaderTimeout = 10 * time.Minute
)

var errRequestOptionsProbeDone = errors.New("request options probe done")

type anthropicModel struct {
	name        string
	modelID     string
	backend     Backend
	project     string
	location    string
	maxTokens   int
	effort      Effort
	promptCache *PromptCacheConfig
	cfgLogger   *slog.Logger
	client      anthropic.Client
}

// NewModel creates a new [model.LLM] backed by Anthropic Claude on either the
// first-party Anthropic API ([BackendAnthropic]) or Google Cloud Vertex AI
// ([BackendVertex]).
func NewModel(ctx context.Context, modelName string, cfg ClientConfig) (model.LLM, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	resProject, resLocation, modelID, isResource, err := parseModelName(modelName)
	if err != nil {
		return nil, err
	}

	backend := cfg.Backend
	if backend == "" {
		backend = BackendAnthropic
	}
	if backend != BackendAnthropic && backend != BackendVertex {
		return nil, fmt.Errorf("%w: unsupported Backend %q; valid values are %q, %q", ErrInvalidConfig, backend, BackendAnthropic, BackendVertex)
	}

	if cfg.MaxTokens < 0 {
		return nil, fmt.Errorf("%w: MaxTokens must be non-negative", ErrInvalidConfig)
	}
	maxTokens := cfg.MaxTokens
	if maxTokens == 0 {
		maxTokens = DefaultMaxTokens
	}

	if cfg.MaxRetries != nil && *cfg.MaxRetries < 0 {
		return nil, fmt.Errorf("%w: MaxRetries must be non-negative", ErrInvalidConfig)
	}

	if err := validateEffort(cfg.Effort); err != nil {
		return nil, err
	}
	if err := validatePromptCacheConfig(cfg.PromptCache); err != nil {
		return nil, err
	}

	if cfg.BaseURL != "" {
		if err := validateBaseURL(cfg.BaseURL); err != nil {
			return nil, err
		}
	}

	reqOptsHasBaseURL, err := inspectRequestOptions(ctx, cfg.RequestOptions)
	if err != nil {
		return nil, err
	}
	hasCustomBaseURL := cfg.BaseURL != "" || reqOptsHasBaseURL

	var (
		opts     []option.RequestOption
		project  string
		location string
	)

	// Always disable the SDK's ambient environment autoload so we control
	// credential precedence and the custom BaseURL credential guard.
	opts = append(opts, option.WithoutEnvironmentDefaults())

	switch backend {
	case BackendAnthropic:
		if isResource {
			return nil, fmt.Errorf("%w: Vertex resource model name requires BackendVertex", ErrInvalidConfig)
		}
		apiKey := strings.TrimSpace(cfg.APIKey)
		authToken := strings.TrimSpace(cfg.AuthToken)
		effectiveBaseURL := strings.TrimSpace(cfg.BaseURL)

		if hasCustomBaseURL {
			if apiKey == "" && authToken == "" {
				return nil, fmt.Errorf("%w: custom BaseURL on BackendAnthropic requires an explicit APIKey or AuthToken in ClientConfig; ambient environment credentials are not sent to custom endpoints", ErrInvalidConfig)
			}
		} else {
			if envBaseURL := strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL")); envBaseURL != "" {
				if err := validateBaseURL(envBaseURL); err != nil {
					return nil, err
				}
				effectiveBaseURL = envBaseURL
			}
			if apiKey == "" && authToken == "" {
				if envKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")); envKey != "" {
					apiKey = envKey
				} else if envToken := strings.TrimSpace(os.Getenv("ANTHROPIC_AUTH_TOKEN")); envToken != "" {
					authToken = envToken
				}
			}
		}

		if apiKey == "" && authToken == "" {
			return nil, fmt.Errorf("%w: %w: no Anthropic credential found; set ClientConfig.APIKey, ClientConfig.AuthToken, ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN", ErrInvalidConfig, ErrAuthentication)
		}

		if cfg.HTTPClient != nil {
			opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
		} else {
			opts = append(opts, option.WithHTTPClient(defaultAnthropicHTTPClient()))
		}
		if effectiveBaseURL != "" {
			opts = append(opts, option.WithBaseURL(effectiveBaseURL))
		}
		if apiKey != "" {
			opts = append(opts, option.WithAPIKey(apiKey))
		}
		if authToken != "" {
			opts = append(opts, option.WithAuthToken(authToken))
		}

	case BackendVertex:
		cfgProject := strings.TrimSpace(cfg.Project)
		cfgLocation := strings.TrimSpace(cfg.Location)

		if isResource {
			if cfgProject != "" && cfgProject != resProject {
				return nil, fmt.Errorf("%w: ClientConfig.Project %q does not match resource project %q", ErrInvalidConfig, cfgProject, resProject)
			}
			if cfgLocation != "" && cfgLocation != resLocation {
				return nil, fmt.Errorf("%w: ClientConfig.Location %q does not match resource location %q", ErrInvalidConfig, cfgLocation, resLocation)
			}
		}

		project = firstNonEmpty(
			cfgProject,
			resProject,
			strings.TrimSpace(os.Getenv("GOOGLE_CLOUD_PROJECT")),
			strings.TrimSpace(os.Getenv("CLOUD_ML_PROJECT_ID")),
		)
		location = firstNonEmpty(
			cfgLocation,
			resLocation,
			strings.TrimSpace(os.Getenv("GOOGLE_CLOUD_LOCATION")),
			strings.TrimSpace(os.Getenv("CLOUD_ML_REGION")),
		)

		if project == "" {
			return nil, fmt.Errorf("%w: BackendVertex requires a project via ClientConfig.Project, a full resource model name, GOOGLE_CLOUD_PROJECT, or CLOUD_ML_PROJECT_ID", ErrInvalidConfig)
		}
		if location == "" {
			return nil, fmt.Errorf("%w: BackendVertex requires a location via ClientConfig.Location, a full resource model name, GOOGLE_CLOUD_LOCATION, or CLOUD_ML_REGION", ErrInvalidConfig)
		}
		if err := validateVertexProject(project); err != nil {
			return nil, err
		}
		if err := validateVertexLocation(location); err != nil {
			return nil, err
		}

		if hasCustomBaseURL && cfg.Credentials == nil && cfg.TokenSource == nil {
			return nil, fmt.Errorf("%w: custom BaseURL on BackendVertex requires explicit Credentials or TokenSource in ClientConfig; Application Default Credentials are not sent to custom endpoints", ErrInvalidConfig)
		}

		detachedCtx := context.WithoutCancel(ctx)
		creds, err := resolveVertexCredentials(detachedCtx, project, cfg)
		if err != nil {
			return nil, err
		}

		// WithHTTPClient must be registered before vertex.WithCredentials so the
		// Vertex option wraps the custom HTTPClient's transport with OAuth2.
		if cfg.HTTPClient != nil {
			opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
		}

		vertexOpt, err := buildVertexOption(detachedCtx, location, project, creds)
		if err != nil {
			return nil, err
		}
		opts = append(opts, vertexOpt)

		if cfg.BaseURL != "" {
			opts = append(opts, option.WithBaseURL(strings.TrimSpace(cfg.BaseURL)))
		}

		trackingHeader := fmt.Sprintf("google-adk/%s gl-go/%s", version.Version, strings.TrimPrefix(runtime.Version(), "go"))
		opts = append(opts,
			option.WithHeader("x-goog-api-client", trackingHeader),
			option.WithHeader("user-agent", trackingHeader),
		)
	}

	if cfg.MaxRetries != nil {
		opts = append(opts, option.WithMaxRetries(*cfg.MaxRetries))
	}
	if len(cfg.RequestOptions) > 0 {
		opts = append(opts, cfg.RequestOptions...)
	}

	var promptCache *PromptCacheConfig
	if cfg.PromptCache != nil {
		copied := *cfg.PromptCache
		if copied.TTL == "" {
			copied.TTL = PromptCacheTTL5m
		}
		promptCache = &copied
	}

	return &anthropicModel{
		name:        strings.TrimSpace(modelName),
		modelID:     modelID,
		backend:     backend,
		project:     project,
		location:    location,
		maxTokens:   maxTokens,
		effort:      cfg.Effort,
		promptCache: promptCache,
		cfgLogger:   cfg.Logger,
		client:      anthropic.NewClient(opts...),
	}, nil
}

func defaultAnthropicHTTPClient() *http.Client {
	if t, ok := http.DefaultTransport.(*http.Transport); ok {
		cloned := t.Clone()
		cloned.ResponseHeaderTimeout = defaultResponseHeaderTimeout
		return &http.Client{Transport: cloned}
	}
	return &http.Client{Transport: http.DefaultTransport}
}

func inspectRequestOptions(ctx context.Context, opts []option.RequestOption) (hasCustomBaseURL bool, err error) {
	for i, opt := range opts {
		if opt == nil {
			return false, fmt.Errorf("%w: RequestOptions[%d] must not be nil", ErrInvalidConfig, i)
		}
	}
	if len(opts) == 0 {
		return false, nil
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%w: invalid RequestOptions configuration", ErrInvalidConfig)
		}
	}()

	var capturedURL *url.URL
	probeOpts := make([]option.RequestOption, 0, len(opts)+2)
	probeOpts = append(probeOpts,
		option.WithMiddleware(func(req *http.Request, _ option.MiddlewareNext) (*http.Response, error) {
			capturedURL = req.URL
			return nil, errRequestOptionsProbeDone
		}),
	)
	probeOpts = append(probeOpts, opts...)
	probeOpts = append(probeOpts, option.WithMaxRetries(0))

	var probeClient anthropic.Client
	execErr := probeClient.Execute(ctx, http.MethodPost, "v1/messages", nil, nil, probeOpts...)
	if errors.Is(execErr, errRequestOptionsProbeDone) {
		if capturedURL == nil || capturedURL.Host == "" || (capturedURL.Scheme != "http" && capturedURL.Scheme != "https") {
			return false, fmt.Errorf("%w: BaseURL in RequestOptions must be a valid http or https URL with a host", ErrInvalidConfig)
		}
		return true, nil
	}
	if execErr != nil {
		if strings.Contains(execErr.Error(), "base url is not set") {
			return false, nil
		}
		if errors.Is(execErr, context.Canceled) || errors.Is(execErr, context.DeadlineExceeded) {
			return false, execErr
		}
		if strings.Contains(execErr.Error(), "WithBaseURL") {
			return false, fmt.Errorf("%w: BaseURL in RequestOptions must be a valid http or https URL with a host", ErrInvalidConfig)
		}
		return false, fmt.Errorf("%w: invalid RequestOptions configuration", ErrInvalidConfig)
	}
	return false, nil
}

func resolveVertexCredentials(ctx context.Context, project string, cfg ClientConfig) (*google.Credentials, error) {
	if cfg.Credentials != nil {
		if cfg.Credentials.TokenSource == nil {
			return nil, fmt.Errorf("%w: ClientConfig.Credentials must have a non-nil TokenSource", ErrInvalidConfig)
		}
		return cfg.Credentials, nil
	}
	if cfg.TokenSource != nil {
		return &google.Credentials{
			ProjectID:   project,
			TokenSource: cfg.TokenSource,
		}, nil
	}
	// The token source captures this context and reuses it for every later
	// refresh, so it must outlive the constructor call.
	creds, err := google.FindDefaultCredentials(context.WithoutCancel(ctx), cloudPlatformScope)
	if err != nil {
		return nil, fmt.Errorf("%w: %w: failed to find Google Application Default Credentials for Vertex AI", ErrInvalidConfig, ErrAuthentication)
	}
	return creds, nil
}

func buildVertexOption(ctx context.Context, location, project string, creds *google.Credentials) (opt option.RequestOption, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%w: failed to initialize Vertex AI client option", ErrInvalidConfig)
		}
	}()
	return vertex.WithCredentials(ctx, location, project, creds), nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// Name returns the model name passed to [NewModel].
func (m *anthropicModel) Name() string {
	return m.name
}

func (m *anthropicModel) logger() *slog.Logger {
	if m.cfgLogger != nil {
		return m.cfgLogger
	}
	return slog.Default() //nolint:forbidigo // unsupported content is skipped with a warning rather than failing the request
}

func (m *anthropicModel) resolveModelID(reqModel string) (string, error) {
	trimmed := strings.TrimSpace(reqModel)
	if trimmed == "" {
		return m.modelID, nil
	}
	resProject, resLocation, modelID, isResource, err := parseModelName(trimmed)
	if err != nil {
		return "", err
	}
	if isResource {
		if m.backend != BackendVertex {
			return "", fmt.Errorf("%w: Vertex resource model name requires BackendVertex", ErrInvalidConfig)
		}
		if resProject != m.project {
			return "", fmt.Errorf("%w: request model project %q does not match configured project %q", ErrInvalidConfig, resProject, m.project)
		}
		if resLocation != m.location {
			return "", fmt.Errorf("%w: request model location %q does not match configured location %q", ErrInvalidConfig, resLocation, m.location)
		}
	}
	return modelID, nil
}

// GenerateContent implements [model.LLM].
func (m *anthropicModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}
		if req == nil {
			yield(nil, fmt.Errorf("%w: request must not be nil", ErrNoContent))
			return
		}

		params, reqOpts, unwrapOutput, err := m.buildMessageParams(ctx, req)
		if err != nil {
			yield(nil, err)
			return
		}

		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}

		if stream {
			m.generateStreaming(ctx, params, reqOpts, unwrapOutput, yield)
			return
		}

		msg, err := m.client.Messages.New(ctx, params, slices.Clone(reqOpts)...)
		if err != nil {
			yield(nil, classifyError(err))
			return
		}
		resp, err := convertMessageToLLMResponse(msg, unwrapOutput)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(resp, nil)
	}
}
