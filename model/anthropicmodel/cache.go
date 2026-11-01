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

import "github.com/anthropics/anthropic-sdk-go"

func toCacheControlParam(cfg *PromptCacheConfig) anthropic.CacheControlEphemeralParam {
	cc := anthropic.NewCacheControlEphemeralParam()
	if cfg != nil && cfg.TTL == PromptCacheTTL1h {
		cc.TTL = anthropic.CacheControlEphemeralTTLTTL1h
	} else {
		cc.TTL = anthropic.CacheControlEphemeralTTLTTL5m
	}
	return cc
}

// applyCacheBreakpoints attaches ephemeral cache_control breakpoints to the
// last tool definition, the system instruction block, and the last cacheable
// (non-thinking) block in messages, matching adk-python's _apply_cache_breakpoints.
func applyCacheBreakpoints(
	cacheCfg *PromptCacheConfig,
	system []anthropic.TextBlockParam,
	messages []anthropic.MessageParam,
	tools []anthropic.ToolUnionParam,
) []anthropic.TextBlockParam {
	if cacheCfg == nil {
		return system
	}
	cc := toCacheControlParam(cacheCfg)

	if len(tools) > 0 && tools[len(tools)-1].OfTool != nil {
		tools[len(tools)-1].OfTool.CacheControl = cc
	}

	markLastCacheableMessageBlock(messages, cc)

	if len(system) > 0 {
		system[len(system)-1].CacheControl = cc
	}
	return system
}

func markLastCacheableMessageBlock(messages []anthropic.MessageParam, cc anthropic.CacheControlEphemeralParam) {
	for i := len(messages) - 1; i >= 0; i-- {
		blocks := messages[i].Content
		for j := len(blocks) - 1; j >= 0; j-- {
			block := &blocks[j]
			// Claude rejects cache_control on thinking and redacted_thinking blocks.
			if block.OfThinking != nil || block.OfRedactedThinking != nil {
				continue
			}
			if setBlockCacheControl(block, cc) {
				return
			}
		}
	}
}

func setBlockCacheControl(block *anthropic.ContentBlockParamUnion, cc anthropic.CacheControlEphemeralParam) bool {
	switch {
	case block.OfText != nil:
		block.OfText.CacheControl = cc
		return true
	case block.OfImage != nil:
		block.OfImage.CacheControl = cc
		return true
	case block.OfDocument != nil:
		block.OfDocument.CacheControl = cc
		return true
	case block.OfToolUse != nil:
		block.OfToolUse.CacheControl = cc
		return true
	case block.OfToolResult != nil:
		block.OfToolResult.CacheControl = cc
		return true
	default:
		return false
	}
}
