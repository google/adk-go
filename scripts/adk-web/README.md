# ADK Web bundle

This directory rebuilds the prebuilt Angular UI that the ADK Go server serves.

`update-adk-web.sh` builds [google/adk-web](https://github.com/google/adk-web)
in a Docker container and copies the result into
`cmd/launcher/web/webui/distr/`. That directory is committed, and
`cmd/launcher/web/webui/webui.go` embeds it with `//go:embed distr/*`, so the UI
ships inside the server binary. It is about 8MB of minified JavaScript.

## Why the revision is pinned

The `Dockerfile` pins the upstream revision in the `ADK_WEB_REF` build argument.
Please do not change it back to a branch name.

The script used to run `git clone -b main`, with a cache-buster in front of it
that forced the newest commit on every run. A refresh therefore built whatever
upstream happened to have that day, and nothing recorded which commit that was.

On 2026-06-30 such a refresh landed in commit 893e4a4, "V2 release (#1109)". It
changed 105 files under `distr/` inside a 547-file commit whose message never
mentions the UI. The bundle it pulled in carried an upstream breaking change:
ADK v2 had moved 30 developer API endpoints under a `/dev/apps/{app_name}/`
prefix. The Go server still served the old paths, so most of the UI returned 404
in the browser. Nobody noticed for two months.

A pin does not stop upstream from making breaking changes. It makes a refresh
reviewable. The diff is 8MB of minified JavaScript that no reviewer can read, so
the upstream revision range is the only part anyone can actually check.

### What the pin does not cover

`ADK_WEB_REF` pins the source, not the toolchain. The container installs from
`package-lock.json` with `npm ci`, so the dependency tree is pinned too, but the
`node:iron-trixie` base image floats: it tracks the newest Node 20 on Debian
trixie. Two builds of the same `ADK_WEB_REF` weeks apart can therefore use
different Node and npm versions and need not produce identical bytes. Pin the
base image by digest if you ever need a build to be reproducible bit for bit.

## How to refresh the bundle

You need Docker. From the repository root:

```
./scripts/adk-web/update-adk-web.sh
```

The script prints the previous upstream commit, the new one, and a GitHub
compare URL for the range between them. Put all three in the commit message.

To build a different revision once, without moving the pin, set `ADK_WEB_REF`:

```
ADK_WEB_REF=v1.0.6 ./scripts/adk-web/update-adk-web.sh
```

That is for trying a candidate. It does not change what anyone else builds.

## How to bump the pin

1.  Choose a revision. Pick a release tag from
    [the tag list](https://github.com/google/adk-web/tags), then pin its commit
    SHA rather than the tag name, because upstream can move a tag. Name the tag
    in the comment above `ADK_WEB_REF` so the SHA stays readable.
2.  Read the upstream changes between the current pin and your candidate, at
    `https://github.com/google/adk-web/compare/<old>...<new>`. Look for changed
    API paths, methods and request bodies.
3.  Edit the `ADK_WEB_REF` default in the `Dockerfile`.
4.  Run `./scripts/adk-web/update-adk-web.sh`.
5.  Run the route contract test (see below) and fix anything it reports.
6.  Commit the `Dockerfile` change and the new `distr/` contents together.

Bumping the pin in a commit that does nothing else keeps the change reviewable.

## The provenance file

Each build writes `distr/adk-web-version.json`, for example:

```json
{
  "upstream_repo": "https://github.com/google/adk-web",
  "upstream_ref": "v1.0.5",
  "upstream_sha": "33c3568021d129339f52b75352dfc35deaf1bf0d",
  "built_at": "2026-08-26T09:14:02Z"
}
```

`upstream_ref` is what the build asked for and `upstream_sha` is what it got,
read from `git rev-parse HEAD` in the checkout. The two differ when the ref is a
tag or a branch. The container copies the file into the dist tree after
`npm run build`, so the Angular build cannot clean it away.

`built_at` is when the container wrote the file. Docker reuses that layer while
`ADK_WEB_REF` is unchanged, so rebuilding the same revision keeps the original
date. Pass `--no-cache` to `docker build` if you need a fresh one.

The bundle currently in the tree was built before this file existed, so it does
not have one and its upstream commit is unknown. The first refresh reports the
previous commit as unknown and fills the gap. After that, `update-adk-web.sh`
fails if the file is missing, so the provenance cannot silently disappear again.

The file is served at `/ui/adk-web-version.json` alongside the rest of the
bundle, which is a convenient way to check what a running server is serving.

## Reviewer checklist for a refresh commit

-   The commit changes `distr/` and little else. A refresh buried inside a large
    release commit is how the 2026-06-30 breakage went unnoticed.
-   The commit message names the old and new upstream commits and links the
    compare URL.
-   `distr/adk-web-version.json` is present, and its `upstream_sha` matches the
    new commit in the message.
-   `ADK_WEB_REF` in the `Dockerfile` matches `upstream_ref` in that file, and
    is a tag or a SHA rather than a branch name.
-   The upstream compare URL has been read for changes to API paths, HTTP
    methods or request bodies, and any found are called out in the message.
-   The route contract test passes:

    ```
    go test ./server/adkrest/ -run TestUIContract
    ```

    `TestUIContractGoldenListMatchesBundle` reads the new bundle and reports
    endpoints the UI calls that the golden list does not know about.
    `TestUIContractEveryUIEndpointRoutes` then proves the server routes each
    one. A refresh that changes which endpoints the UI calls will fail the first
    test, and the golden list in `server/adkrest/uicontract_test.go` needs
    updating in the same commit.
-   Someone has started the server and loaded the UI. The contract test checks
    routing, not rendering.
