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

package llminternal

import (
	"errors"
	"fmt"
	"iter"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/exp/authtool"
	"google.golang.org/adk/v2/internal/authhandler"
	icontext "google.golang.org/adk/v2/internal/context"
	icredentialservice "google.golang.org/adk/v2/internal/credentialservice"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// toolsetAuthIDPrefix starts the functionCallId of a toolset's credential
// request, as adk-python's TOOLSET_AUTH_CREDENTIAL_ID_PREFIX
// (auth/auth_preprocessor.py). No tool call has such an ID, so
// authPreprocessor reruns nothing for the answer.
const toolsetAuthIDPrefix = "_adk_toolset_auth_"

// toolsetAuthProcessor resolves the credential of every toolset of the agent
// that is a tool.AuthenticatedToolset with a config, before the toolsets list
// their tools, as adk-python's toolset_auth request processor
// (flows/llm_flows/tools/_toolset_auth.py) does. A resolved credential goes to
// the run's resolved credentials (agent.ResolvedCredential). When any toolset
// has none, it yields one request event for those toolsets and ends the
// invocation. It uses the step's resolution when listToolsForRerun has already
// made it, so that each step resolves once and yields its request or error
// here, after the rerun's events.
func toolsetAuthProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		step := f.resolveToolsetCredentials(ctx)
		if step.err != nil {
			yield(nil, step.err)
			return
		}
		if len(step.pending) == 0 {
			return
		}
		pending := make([]authRequest, len(step.pending))
		for i, p := range step.pending {
			r, err := authhandler.GenerateAuthRequest(&p.cfg)
			if err != nil {
				yield(nil, err)
				return
			}
			pending[i] = authRequest{functionCallID: p.functionCallID, cfg: *r}
		}
		ev, err := authRequestEvent(ctx, pending, "")
		if err != nil {
			yield(nil, err)
			return
		}
		ctx.EndInvocation()
		yield(ev, nil)
	}
}

// toolsetAuthStep is what resolving the toolsets' credentials found in one
// step: a request for each toolset with no credential, or the error that
// stopped the resolution.
type toolsetAuthStep struct {
	pending []authRequest
	err     error
}

// resolveToolsetCredentials resolves the credential of every authenticated
// toolset of the agent into the run's resolved credentials, once per step:
// the first call of a step resolves and keeps the result in f.toolsetAuth,
// and later calls of that step return it.
func (f *Flow) resolveToolsetCredentials(ctx agent.InvocationContext) *toolsetAuthStep {
	if f.toolsetAuth == nil {
		f.toolsetAuth = resolveToolsetAuth(ctx)
	}
	return f.toolsetAuth
}

// listToolsForRerun lists the agent's tools for a call the user's answer
// reruns, once this step's toolset credentials are resolved, so that a
// toolset lists the tools it offers with its credential. adk-python's rerun
// gets that listing from the invocation's listing cache
// (tools/base_toolset.py, get_tools_with_prefix), filled when the call was
// first made.
func (f *Flow) listToolsForRerun(ctx agent.InvocationContext) error {
	f.resolveToolsetCredentials(ctx)
	return f.listTools(ctx)
}

// resolveToolsetAuth resolves the credential of each authenticated toolset of
// ctx's agent. Two toolsets with one name make one request: the first one's
// place and the last one's config, as adk-python's dict does.
func resolveToolsetAuth(ctx agent.InvocationContext) *toolsetAuthStep {
	step := &toolsetAuthStep{}
	llmAgent := asLLMAgent(ctx.Agent())
	if llmAgent == nil {
		return step
	}
	cctx := icontext.NewCallbackContext(ctx)
	resolved := icredentialservice.ResolvedFrom(ctx)
	index := map[string]int{}
	for _, ts := range Reveal(llmAgent).Toolsets {
		a, ok := ts.(tool.AuthenticatedToolset)
		if !ok {
			continue
		}
		cfg := a.AuthConfig()
		if cfg == nil {
			continue
		}
		cred, err := toolsetCredential(cctx, cfg)
		if err != nil {
			step.err = fmt.Errorf("toolset %q: %w", ts.Name(), err)
			return step
		}
		if cred != nil {
			key, err := cfg.Key()
			if err != nil {
				step.err = fmt.Errorf("toolset %q: credential key: %w", ts.Name(), err)
				return step
			}
			resolved.Set(key, cred)
			continue
		}
		id := toolsetAuthIDPrefix + ts.Name()
		if i, ok := index[id]; ok {
			step.pending[i].cfg = *cfg
			continue
		}
		index[id] = len(step.pending)
		step.pending = append(step.pending, authRequest{functionCallID: id, cfg: *cfg})
	}
	return step
}

// toolsetCredential returns the credential cfg describes, or nil when the user
// has to provide one. A config the manager reports as invalid or unsupported
// with an *authhandler.ConfigError also returns nil, since adk-python catches
// the ValueError its CredentialManager raises and asks for the credential. A
// credential that comes with an error of a failed exchange or refresh is
// returned, as adk-python returns it without raising. Every other error is
// returned, including the manager's refusal of a config with no scheme type,
// which adk-python's AuthConfig cannot hold.
func toolsetCredential(ctx agent.Context, cfg *toolauth.Config) (*toolauth.Credential, error) {
	m, err := authtool.NewCredentialManager(cfg, nil)
	if err != nil {
		return nil, err
	}
	cred, err := m.GetAuthCredential(ctx)
	if cred != nil {
		return cred, nil
	}
	if _, ok := errors.AsType[*authhandler.ConfigError](err); ok {
		return nil, nil
	}
	return nil, err
}
