# Changelog

## [2.8.0](https://github.com/google/adk-go/compare/v2.7.1...v2.8.0) (2026-11-07)


### Features

* **artifact:** support artifact:// reference parts in the in-memory service ([#1164](https://github.com/google/adk-go/issues/1164)) ([640eb4f](https://github.com/google/adk-go/commit/640eb4f1ba8756a5086096f97ce4535b2628ebd0))
* **authhandler:** add the server side of the end-user OAuth2 consent flow ([#2143](https://github.com/google/adk-go/issues/2143)) ([249974d](https://github.com/google/adk-go/commit/249974dc15580f08d11629581641ad45a47f646e))
* **cli:** configure Agent Engine deployment environment ([#1981](https://github.com/google/adk-go/issues/1981)) ([c7087e3](https://github.com/google/adk-go/commit/c7087e3e44b23e3adeb2bbe1911ef57e239983cd))
* **mcptoolset:** add ToMCPTool to convert ADK tools into MCP tool definitions ([#2133](https://github.com/google/adk-go/issues/2133)) ([6d4199c](https://github.com/google/adk-go/commit/6d4199ce5bbd332c36170c2ec60d1ee10427d630))
* **mcptoolset:** add tool name prefixes ([#1821](https://github.com/google/adk-go/issues/1821)) ([8360e75](https://github.com/google/adk-go/commit/8360e75d5c4bdd8dbf587f12c057a292f8270d3e))
* **memory:** add Vertex AI RAG memory service ([#2130](https://github.com/google/adk-go/issues/2130)) ([c264bdc](https://github.com/google/adk-go/commit/c264bdc17b42f6ad458136007d8481ae4d0cb8cd))
* **session:** add RequestedAuthConfigs to event actions ([#2114](https://github.com/google/adk-go/issues/2114)) ([5de88a7](https://github.com/google/adk-go/commit/5de88a7f68301388c7a0287a6bc96dc577bab25c))
* **telemetry:** add a MeterProvider with OTLP and Google Cloud export ([#2042](https://github.com/google/adk-go/issues/2042)) ([ba5d42b](https://github.com/google/adk-go/commit/ba5d42b85b62ce99e6431d31fe77a92796c5bd04))


### Bug Fixes

* **adka2a:** convert empty A2A text parts instead of failing ([#2088](https://github.com/google/adk-go/issues/2088)) ([ee8d5ca](https://github.com/google/adk-go/commit/ee8d5ca44fe26532730ba1bbcf599c91247f58cc))
* **adka2a:** keep LLM agent instructions out of agent card skills ([#2110](https://github.com/google/adk-go/issues/2110)) ([1c35470](https://github.com/google/adk-go/commit/1c3547019cb4d661ce5f4b3cbff8af65e11eb6bb))
* **adka2a:** mark a finished A2A task as the final response ([#2102](https://github.com/google/adk-go/issues/2102)) ([d6bb1e8](https://github.com/google/adk-go/commit/d6bb1e8609fa2e3ef826f451b4708ef1a8740130))
* **adkrest:** answer [] from /run when a run yields no events ([#2090](https://github.com/google/adk-go/issues/2090)) ([0159a2e](https://github.com/google/adk-go/commit/0159a2e7ff83cbc07ec52d339d5dd4433fc80d47))
* **adkrest:** answer 500, not 404, when /run cannot read the session ([#2107](https://github.com/google/adk-go/issues/2107)) ([8ce8f01](https://github.com/google/adk-go/commit/8ce8f01ca5a628aa6e37cc4cd7c0525b92f7177f))
* **adkrest:** include timestamp in REST Event model ([#1870](https://github.com/google/adk-go/issues/1870)) ([4db0bd9](https://github.com/google/adk-go/commit/4db0bd991b898861553cb804335012574b471034))
* **adkrest:** let AutoCreateSession reach /run and /run_sse ([#2106](https://github.com/google/adk-go/issues/2106)) ([f6da03a](https://github.com/google/adk-go/commit/f6da03ab78210ab24f4a63c192a37dfe385b2694))
* **adkrest:** preserve debug span attribute types ([#1988](https://github.com/google/adk-go/issues/1988)) ([c377836](https://github.com/google/adk-go/commit/c377836c419a1d1a17e9b054c9aad34de68cf2a7))
* **agent:** list a multi-agent loader's agents in sorted order ([#2164](https://github.com/google/adk-go/issues/2164)) ([2e84254](https://github.com/google/adk-go/commit/2e842548d3ecf9542511534d0dd6aa1d6156b097))
* **agentregistry:** speak A2A 0.3 to 0.3 interfaces ([#2145](https://github.com/google/adk-go/issues/2145)) ([869020a](https://github.com/google/adk-go/commit/869020a1c3ba754790513f0371544e603ea3e238))
* **agent:** reject an empty agent name in agent.New ([#2124](https://github.com/google/adk-go/issues/2124)) ([d0c978b](https://github.com/google/adk-go/commit/d0c978b0ce4fe2068be977eb5fd90a0add2dd462)), closes [#2123](https://github.com/google/adk-go/issues/2123)
* **agenttool:** accept the input schema a workflow agent's declaration advertises ([#2069](https://github.com/google/adk-go/issues/2069)) ([31111f8](https://github.com/google/adk-go/commit/31111f89d1bb760109d29d427f9c7957f8553e10))
* **agenttool:** report a sub-agent paused on an unresolved long-running call ([#1856](https://github.com/google/adk-go/issues/1856)) ([f25d142](https://github.com/google/adk-go/commit/f25d1424b20b0dfe22c582f73d6ef9d270aa63a0))
* **agenttool:** set SkipSummarization only after the input is valid ([#2049](https://github.com/google/adk-go/issues/2049)) ([4cc7ce5](https://github.com/google/adk-go/commit/4cc7ce5f4ba73c239ecabdf23b064036473f3c4f))
* **apigee:** send the model ID, not the apigee/ model string, to the proxy ([#2067](https://github.com/google/adk-go/issues/2067)) ([71b0761](https://github.com/google/adk-go/commit/71b0761f3b466408b11f571aba6882e0eb2c6226))
* **artifact:** copy parts in the in-memory service on save and load ([#1782](https://github.com/google/adk-go/issues/1782)) ([1b4f6ee](https://github.com/google/adk-go/commit/1b4f6ee0d0bf0a50dd429b223da917fa48b68ee4))
* **exampletool:** give Gemini models plain fences, not the tool_code ones ([#2072](https://github.com/google/adk-go/issues/2072)) ([5199d9f](https://github.com/google/adk-go/commit/5199d9fefc557ec4e3cd986764142583e594bcc8))
* **exampletool:** write example call arguments in a stable order ([#2073](https://github.com/google/adk-go/issues/2073)) ([e342d2f](https://github.com/google/adk-go/commit/e342d2f6c7493bdc373190518c726d5d5228359d))
* **functioncallmodifier:** leave content-less responses to the rest of the chain ([#2080](https://github.com/google/adk-go/issues/2080)) ([128135a](https://github.com/google/adk-go/commit/128135a82ba5c9af89e1ed381aa848ab2cbbb9b4))
* **functioncallmodifier:** modify a copy of each tool declaration ([#2081](https://github.com/google/adk-go/issues/2081)) ([f0af46d](https://github.com/google/adk-go/commit/f0af46d1f23711fe53b39a3cb8409fa2f27f5d6b))
* **functiontool:** keep tool panics out of the model's function response ([#2120](https://github.com/google/adk-go/issues/2120)) ([e83ab5b](https://github.com/google/adk-go/commit/e83ab5b0ebbcd0b58cb86bd79a30bb03af952c72))
* **llmagent:** return an error to the model when transfer_to_agent names an unknown agent ([#2161](https://github.com/google/adk-go/issues/2161)) ([c8fff67](https://github.com/google/adk-go/commit/c8fff670009d0f56266b6570d27e60024e4c976e))
* **llmagent:** write finish_task output to OutputKey for task agents ([#1977](https://github.com/google/adk-go/issues/1977)) ([8c2e2e2](https://github.com/google/adk-go/commit/8c2e2e2a4b293d2739f885822b304607ef2c6600))
* **llminternal:** accept a decoded value in the tool confirmation response wrapper ([#2103](https://github.com/google/adk-go/issues/2103)) ([0d95aed](https://github.com/google/adk-go/commit/0d95aed436c2f0b9bf0cefc6e5f694cddabf59d4))
* **llminternal:** do not rerun a confirmed tool on repeated approval ([#1865](https://github.com/google/adk-go/issues/1865)) ([550bb4c](https://github.com/google/adk-go/commit/550bb4c267a03d5306e6cf20260b65b5c26cabbf)), closes [#1752](https://github.com/google/adk-go/issues/1752)
* **llminternal:** do not start the current turn at a posted-back tool result ([#2083](https://github.com/google/adk-go/issues/2083)) ([e847ba7](https://github.com/google/adk-go/commit/e847ba7d5799168f00a35b8669664858dbe622bb))
* **llminternal:** drop internal artifact parts from live history replay ([#1765](https://github.com/google/adk-go/issues/1765)) ([22ff183](https://github.com/google/adk-go/commit/22ff18397dfe8de12858e3b9c8948a8cc3676fff))
* **llminternal:** end the turn when a preprocessed tool response is final ([#2104](https://github.com/google/adk-go/issues/2104)) ([074b087](https://github.com/google/adk-go/commit/074b087948ec7f8d4f6a9708b0c84c3a4981b8c8))
* **llminternal:** inject text artifacts loaded as inline data into instructions ([#2101](https://github.com/google/adk-go/issues/2101)) ([860c7da](https://github.com/google/adk-go/commit/860c7dac9877160bf51094202e0a66c7c15a0133))
* **llminternal:** keep a single_turn sub-agent's transcript out of the root agent's context ([#2128](https://github.com/google/adk-go/issues/2128)) ([ab78c5a](https://github.com/google/adk-go/commit/ab78c5aeda134836596e5c204cf8c0361e403e5f))
* **llminternal:** keep a streamed text chunk's thought signature on the merged part ([#2071](https://github.com/google/adk-go/issues/2071)) ([28201d6](https://github.com/google/adk-go/commit/28201d64d80c52602675851e94c9afd89b709321))
* **llminternal:** keep ModelVersion on the aggregated streaming response ([#2070](https://github.com/google/adk-go/issues/2070)) ([8be9129](https://github.com/google/adk-go/commit/8be9129edf02611366a8c58344046bba5ab25437))
* **llminternal:** keep Partial and UsageMetadata on after-model callback replacements ([#1763](https://github.com/google/adk-go/issues/1763)) ([6ae94e1](https://github.com/google/adk-go/commit/6ae94e12a3787f6c9cbbf75a2978d92263adad46))
* **llminternal:** omit metadata-only foreign parts from context ([#1895](https://github.com/google/adk-go/issues/1895)) ([6caa3b8](https://github.com/google/adk-go/commit/6caa3b853ea4d179124f089daf6f39498dc93425))
* **llminternal:** send live tool responses through the sender goroutine ([#1903](https://github.com/google/adk-go/issues/1903)) ([33a587e](https://github.com/google/adk-go/commit/33a587e28f1b3e7ea31fa9639e8f8007bb72c1cd))
* **llminternal:** skip nil pointers in live realtime input ([#1854](https://github.com/google/adk-go/issues/1854)) ([0d22357](https://github.com/google/adk-go/commit/0d2235729d782db815855da059d7a5fff819be5e))
* **llminternal:** stop sending Gemini 3.x Live audio and images as mediaChunks ([#2118](https://github.com/google/adk-go/issues/2118)) ([d8e4744](https://github.com/google/adk-go/commit/d8e47448e6d3f6ef1c54d42806c31e9ec563558d))
* **mcptoolset:** let a CommandTransport reconnect ([#2115](https://github.com/google/adk-go/issues/2115)) ([1a84f04](https://github.com/google/adk-go/commit/1a84f04f7306889a9ad7a5c929b0374b9a8257fc))
* **model/gemini:** report a blocked prompt instead of failing with "empty response" ([#2068](https://github.com/google/adk-go/issues/2068)) ([15e9d4c](https://github.com/google/adk-go/commit/15e9d4cd406b240f1272720e397d15fd4a296976))
* **openaimodel:** refuse non-function Tool fields genai adds ([#1983](https://github.com/google/adk-go/issues/1983)) ([275e190](https://github.com/google/adk-go/commit/275e190e8c501b83dce6d77c22f69341cda44aa9)), closes [#1982](https://github.com/google/adk-go/issues/1982)
* **openaimodel:** retain terminal text after call ID fallback ([#1897](https://github.com/google/adk-go/issues/1897)) ([a1603c7](https://github.com/google/adk-go/commit/a1603c7f179af285b08ef4f03e757a6deb0630b9))
* **openaimodel:** send executable code parts as text ([#1956](https://github.com/google/adk-go/issues/1956)) ([ded357e](https://github.com/google/adk-go/commit/ded357e351dee44c023398a49d314aa326a59c94))
* **openaimodel:** trace recovered terminal conversion failures ([#2056](https://github.com/google/adk-go/issues/2056)) ([450b4a5](https://github.com/google/adk-go/commit/450b4a5424f3eaca5f022775c80593488eb41a45))
* **parallelagent:** keep the parent's isolation scope in each branch ([#2054](https://github.com/google/adk-go/issues/2054)) ([543edd3](https://github.com/google/adk-go/commit/543edd3d37c205f4a931489e3db1bbf3e5d76d0d))
* **remoteagent:** resume tasks with mixed tool responses ([#2148](https://github.com/google/adk-go/issues/2148)) ([7c87fdf](https://github.com/google/adk-go/commit/7c87fdff09af9311e05d488190dedc1aaaa7bc30))
* **retryandreflect:** drop invocation failure counts when the run ends ([#2024](https://github.com/google/adk-go/issues/2024)) ([7fafcbb](https://github.com/google/adk-go/commit/7fafcbbca3fa9bbc729ce7b080006f296f2930ab))
* **retryandreflect:** populate MaxRetries in retry-exceeded guidance template ([#2174](https://github.com/google/adk-go/issues/2174)) ([4417f2d](https://github.com/google/adk-go/commit/4417f2d4f924e04599b6a03f1475c1e642962840))
* **runner:** only auto-create a session when Get reports session.ErrNotFound ([#1899](https://github.com/google/adk-go/issues/1899)) ([ca7db16](https://github.com/google/adk-go/commit/ca7db168e51f05b7ab6b9930010250fcc27dff70))
* **runner:** propagate parent plugins to agenttool sub-runners ([#1777](https://github.com/google/adk-go/issues/1777)) ([24d33b3](https://github.com/google/adk-go/commit/24d33b3575044c1626a2506faa3d1acbf81d36b0))
* **runner:** route a function response to the calling sub-agent under a workflow root ([#1808](https://github.com/google/adk-go/issues/1808)) ([dce89da](https://github.com/google/adk-go/commit/dce89da583637716a26a52712bc408e174a32173))
* **runner:** run OnEvent and AfterRun for a BeforeRun reply in RunLive ([#2105](https://github.com/google/adk-go/issues/2105)) ([dbaca47](https://github.com/google/adk-go/commit/dbaca475d108ac5da198b7fa80fce8b8aa7dd294))
* **runner:** stamp a timestamp on events agents yield without one ([#2091](https://github.com/google/adk-go/issues/2091)) ([2649b4a](https://github.com/google/adk-go/commit/2649b4ada76e025d84483df8cd0d05c068b1c152))
* **runner:** treat the inbound message as a user turn ([#1877](https://github.com/google/adk-go/issues/1877)) ([de7088b](https://github.com/google/adk-go/commit/de7088bf4aa5c9445adbc4e8b362c5c1927dde95))
* **sequentialagent:** append the live completion instruction to an InstructionProvider ([#2079](https://github.com/google/adk-go/issues/2079)) ([dd39c80](https://github.com/google/adk-go/commit/dd39c80a1e3db243823cc71fbb78682a8d12690a))
* **sequentialagent:** stop RunLive on a sub-agent error, as Run does ([#2151](https://github.com/google/adk-go/issues/2151)) ([82b64f4](https://github.com/google/adk-go/commit/82b64f452060f3bdd0f6034521d4fa82a503f6fe))
* **session/database:** delete a session's events explicitly in Delete ([#1774](https://github.com/google/adk-go/issues/1774)) ([7af0f5f](https://github.com/google/adk-go/commit/7af0f5f06a545df1d59d0ee4e0c59063689dee9d))
* **session/database:** lock app and user state rows in Create ([#1930](https://github.com/google/adk-go/issues/1930)) ([2f3b2c2](https://github.com/google/adk-go/commit/2f3b2c2305e8e9ec62cafc13cd1c5f2d4959a7f6))
* **session/vertexai:** drop temp: keys from the initial state on Create ([#2100](https://github.com/google/adk-go/issues/2100)) ([0c109f7](https://github.com/google/adk-go/commit/0c109f7dad33815bcee4f101b7b724a1293d8823))
* **session:** define List order as last update time ascending ([#1919](https://github.com/google/adk-go/issues/1919)) ([8be7e25](https://github.com/google/adk-go/commit/8be7e250858a4f1ddd925d7d166c2695707e72b6))
* **session:** keep in-memory Get from returning events before After ([#2089](https://github.com/google/adk-go/issues/2089)) ([10395f0](https://github.com/google/adk-go/commit/10395f010ba1f992278ca368954f2e212ea362c2))
* **telemetry:** leave gcp.project_id off the resource when no project is known ([#2082](https://github.com/google/adk-go/issues/2082)) ([9679009](https://github.com/google/adk-go/commit/967900997525ac42f580eb43436ef8148d69b764))
* **triggers:** use timer-and-select for context-aware retry backoff ([#2052](https://github.com/google/adk-go/issues/2052)) ([123fea9](https://github.com/google/adk-go/commit/123fea94f49d7387c915602e4386a1b7ecd27aca))
* **workflow:** honor UUID provider for tool node call IDs ([#2030](https://github.com/google/adk-go/issues/2030)) ([010b91e](https://github.com/google/adk-go/commit/010b91e9807ad6c5c8f73da84c0187a50fde0164))
* **workflow:** return ctx.Err() instead of a partial ParallelWorker result ([#2146](https://github.com/google/adk-go/issues/2146)) ([dc44b8c](https://github.com/google/adk-go/commit/dc44b8c96d3c54b613e06f1f746b920500d363ef))


### Performance Improvements

* **llminternal:** use strings.Builder in streaming response aggregator ([#2051](https://github.com/google/adk-go/issues/2051)) ([ca21a26](https://github.com/google/adk-go/commit/ca21a26fb0021c922a4489e22ba7b380299ae64e))
* **remoteagent:** track builder per text part in artifact aggregation ([734a5a2](https://github.com/google/adk-go/commit/734a5a2ad63d2f58c41f05515db7147045dc4486))
* **remoteagent:** use strings.Builder for artifact aggregation ([#2050](https://github.com/google/adk-go/issues/2050)) ([734a5a2](https://github.com/google/adk-go/commit/734a5a2ad63d2f58c41f05515db7147045dc4486))


### Documentation

* explain how a merged commit links an issue in the release notes ([#2141](https://github.com/google/adk-go/issues/2141)) ([dd8330f](https://github.com/google/adk-go/commit/dd8330ffd3c0d08bcd56f56eb8569762995de2c8))

## [2.7.1](https://github.com/google/adk-go/compare/v2.7.0...v2.7.1) (2026-11-04)


### Bug Fixes

* **agentanalytics:** classify Write API errors and split oversized batches ([#2016](https://github.com/google/adk-go/issues/2016)) ([2466e97](https://github.com/google/adk-go/commit/2466e976ac371b8e2074315c70cbbdaf9b1f2498))
* **model/openaimodel:** refuse media in system and developer Responses messages ([#2025](https://github.com/google/adk-go/issues/2025)) ([40f132d](https://github.com/google/adk-go/commit/40f132d697752596c79d307c4ef2bfb70c55ad9c))
* **model/openaimodel:** refuse non-text payloads beside system instruction text ([#2113](https://github.com/google/adk-go/issues/2113)) ([fe99252](https://github.com/google/adk-go/commit/fe992528e7d1c7e9f648158d47957c241c569ff3)), closes [#2112](https://github.com/google/adk-go/issues/2112)
