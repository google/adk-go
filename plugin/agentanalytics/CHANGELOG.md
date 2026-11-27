# Changelog

## [0.2.0](https://github.com/google/adk-go/compare/plugin/agentanalytics/v0.1.0...plugin/agentanalytics/v0.2.0) (2026-11-27)


### ⚠ BREAKING CHANGES

* **agentanalytics:** ArrowSchema, SerializedArrowSchema, BatchProcessor, NewBatchProcessor, FormatContentParts and SmartTruncate are no longer exported, and DropStats, RetryConfig, GCSUploadRequest, ViewDefinition and Config can no longer be built with unkeyed literals outside the package.

### Features

* **agentanalytics:** add event provenance, context walkers, and callback stack helpers ([#2380](https://github.com/google/adk-go/issues/2380)) ([0fe0615](https://github.com/google/adk-go/commit/0fe061559cb0101d2969da12d327b0424bee40d2))
* **agentanalytics:** add pre-built agent evaluation views (v_eval_session, v_eval_turn) ([#2175](https://github.com/google/adk-go/issues/2175)) ([e15287f](https://github.com/google/adk-go/commit/e15287fae12bed0c9a561f78b6a3e62f27138c51))
* **agentanalytics:** add ToolResultClassifier hook and schema lifecycle callbacks ([#2180](https://github.com/google/adk-go/issues/2180)) ([98275d2](https://github.com/google/adk-go/commit/98275d2f97cabfc3b71d4ef63050b6ac29c659fd))
* **model:** add Capabilities() to model.LLM for output schema with tools ([#2492](https://github.com/google/adk-go/issues/2492)) ([eb2f354](https://github.com/google/adk-go/commit/eb2f3543f45debd5d8fb4b1e6ce2ae32824bb720))


### Bug Fixes

* **agentanalytics:** add retired-activation ledger and credit bounds ([#2389](https://github.com/google/adk-go/issues/2389)) ([b4383d8](https://github.com/google/adk-go/commit/b4383d8e7670f67816a3badc5d589ee0bf5144cc))
* **agentanalytics:** align checkpoint gating, MCP isError tool results, and empty error messages ([#2047](https://github.com/google/adk-go/issues/2047)) ([#2178](https://github.com/google/adk-go/issues/2178)) ([f423f02](https://github.com/google/adk-go/commit/f423f0299b5d64be0b3ff171c245cb2a1180839a))
* **agentanalytics:** assign possible owners to nearest lookalike member ([#2517](https://github.com/google/adk-go/issues/2517)) ([c04a021](https://github.com/google/adk-go/commit/c04a0214091345c467e3353d416c0dc1f0ee394c))
* **agentanalytics:** cap response summary and function name summary length ([#2246](https://github.com/google/adk-go/issues/2246)) ([f55660a](https://github.com/google/adk-go/commit/f55660a66749dd7ffc80a25f4ac6c64bdb890817))
* **agentanalytics:** harden Close/afterRun lifecycle and defer late-settled retirement ([#2375](https://github.com/google/adk-go/issues/2375)) ([14daf2c](https://github.com/google/adk-go/commit/14daf2ccb7ba66170bc01bdbc3aa3fe8ccf514ca))
* **agentanalytics:** isolate BigQuery RowErrors and track RowErrorDrops ([#2376](https://github.com/google/adk-go/issues/2376)) ([b019ecc](https://github.com/google/adk-go/commit/b019ecc3f628a97c5b64f930a0c5a43f6182a8ce))
* **agentanalytics:** match an error-ended tool call to its own attempt ([#2377](https://github.com/google/adk-go/issues/2377)) ([2924008](https://github.com/google/adk-go/commit/2924008fb6269ec37419ce31a182f16e8c043b90))
* **agentanalytics:** preserve retired late parent cancel cause and custom metadata truncation flag ([#2020](https://github.com/google/adk-go/issues/2020)) ([ca197aa](https://github.com/google/adk-go/commit/ca197aaaa582e9c6d85da4559729a099a7c98907))
* **agentanalytics:** redact and bound code part labels in summaries ([#2332](https://github.com/google/adk-go/issues/2332)) ([cf55159](https://github.com/google/adk-go/commit/cf55159d2d7894e682fb006267910a8ded36ccb6)), closes [#2259](https://github.com/google/adk-go/issues/2259)
* **agentanalytics:** redact compound and state-prefixed sensitive keys ([#2250](https://github.com/google/adk-go/issues/2250)) ([6e73acc](https://github.com/google/adk-go/commit/6e73accaf9b3b333d050b7a6889a3c7a2bc8431f))
* **agentanalytics:** redact credential-named types and guard omitzero in JSONFormatter ([#2249](https://github.com/google/adk-go/issues/2249)) ([5b3d897](https://github.com/google/adk-go/commit/5b3d897b19898a31df19ab786ee85670e69e9a61))
* **agentanalytics:** require adk/v2 v2.8.0 and patched klauspost/compress ([#2477](https://github.com/google/adk-go/issues/2477)) ([f779980](https://github.com/google/adk-go/commit/f77998059feaaa053b64c3a650106067b452d162))
* **agentanalytics:** sanitize unsafe GCSPathPrefix values and cap GCS object names at 1,024 bytes ([#2018](https://github.com/google/adk-go/issues/2018)) ([11f4e29](https://github.com/google/adk-go/commit/11f4e29f588ee0f08d2e372f76737412075460bf))
* **agentanalytics:** scope context walks to stdlib, ADK, and thin wrappers ([#2516](https://github.com/google/adk-go/issues/2516)) ([b675bc9](https://github.com/google/adk-go/commit/b675bc97c7a4b62a424a0aad415ba79a770ab888))
* **agentanalytics:** scope lookalike parent resolution to active goroutine ([#2379](https://github.com/google/adk-go/issues/2379)) ([6730502](https://github.com/google/adk-go/commit/673050266d9bf329f8cc46a93720ab27ba14ca46))
* **agentanalytics:** suppress recovered tool completions and log retryandreflect errors ([#2381](https://github.com/google/adk-go/issues/2381)) ([b758093](https://github.com/google/adk-go/commit/b758093d84f3c9097f9e0d3d21d5d1461d764b4e))
* **deps:** bump gorilla/websocket past GO-2026-6278 ([#2567](https://github.com/google/adk-go/issues/2567)) ([881ee33](https://github.com/google/adk-go/commit/881ee33ad2c8edf91987f2268dcddd240b3059bc))
* **runner:** start invocation spans before plugin callbacks ([#1898](https://github.com/google/adk-go/issues/1898)) ([3f5d74d](https://github.com/google/adk-go/commit/3f5d74d36e3f5c734904516bc83fd1b8556722dc))
* **workflow:** do not carry a consumed answer over to a node's next pause ([#2330](https://github.com/google/adk-go/issues/2330)) ([faab8f1](https://github.com/google/adk-go/commit/faab8f18001c35b8b4728bf0940e24114946ee5a))


### Code Refactoring

* **agentanalytics:** unexport internal transport/truncation helpers and guard struct literals for v1.0.0 ([#2181](https://github.com/google/adk-go/issues/2181)) ([e09d5b8](https://github.com/google/adk-go/commit/e09d5b8804a10da3c7ba92510614a2ec92324639))


### Documentation

* **agentanalytics:** add package godoc documenting Go design and parity behavior ([#2176](https://github.com/google/adk-go/issues/2176)) ([1edb057](https://github.com/google/adk-go/commit/1edb0572264d71b3ba3ab4e9bd24a979e267a96e))
