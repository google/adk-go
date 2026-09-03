# Stage 1: Clone the repository
FROM node:iron-trixie AS downloader
RUN apt-get update && apt-get install -y git

# The upstream ref to build. Pass --build-arg ADK_WEB_REF=<sha-or-tag> to move
# it forward; see scripts/adk-web/README.md for how to bump the pin.
#
# This is pinned, and must stay pinned, because the bundle it produces is
# committed and embedded into the server binary. When this cloned `main`
# instead, a refresh took whatever upstream had that day. On 2026-06-30 one such
# refresh (commit 893e4a4, 105 files under distr/ inside a 547-file release)
# pulled in an upstream change that moved 30 developer API endpoints under a
# /dev/apps/{app}/ prefix. Nothing in the commit said so, and most of the web UI
# returned 404 for two months. A floating ref makes a refresh unreviewable: the
# diff is 8MB of minified JavaScript, so the ref is the only thing a reviewer
# can actually read.
#
# Pinning also makes the clone layer cacheable: the same ref must yield the same
# checkout, so there is deliberately no cache-buster here.
#
# The default is a SHA rather than a tag name, because upstream can move a tag.
# 33c3568 is upstream tag v1.0.5, dated 2026-08-26.
# NOTE: the bundle committed in cmd/launcher/web/webui/distr/ was built on
# 2026-06-30, before this pin existed, and carries no provenance file. The pin
# below therefore does not describe what is committed today: building at it
# moves the UI forward rather than reproducing the current bundle. The first
# refresh is a deliberate version bump; after it, the provenance file written
# below records exactly what shipped.
ARG ADK_WEB_REF=33c3568021d129339f52b75352dfc35deaf1bf0d

# `git clone -b` accepts a branch or tag but not a SHA, so clone the repository
# and then detach onto the requested ref, which may be either.
RUN git clone https://github.com/google/adk-web /adk-web \
    && git -C /adk-web checkout --detach "${ADK_WEB_REF}"

# Record what was actually built. The SHA is resolved from the checkout rather
# than assumed, so it stays correct when ADK_WEB_REF names a tag or a branch.
RUN printf '{\n  "upstream_repo": "https://github.com/google/adk-web",\n  "upstream_ref": "%s",\n  "upstream_sha": "%s",\n  "built_at": "%s"\n}\n' \
    "${ADK_WEB_REF}" \
    "$(git -C /adk-web rev-parse HEAD)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > /adk-web-version.json

# Stage 2: Build the application
FROM node:iron-trixie AS builder

WORKDIR /adk-web

# Copy the package files first to leverage the Docker cache for the install
# step. This ensures the install is only re-run when package.json or
# package-lock.json changes, and not when other source files change.
#
# `npm ci` rather than `npm install`, because install re-resolves semver ranges
# and can produce a different bundle from the same pinned ref, which undoes the
# reproducibility the pin above exists to give. ci installs exactly what
# package-lock.json names, and fails if the lockfile is missing or has drifted
# from package.json.
COPY --from=downloader /adk-web/package*.json ./
RUN npm ci

# Copy the rest of the source code
COPY --from=downloader /adk-web ./

RUN npm run build

# Copy the provenance file in after the build, so the build cannot overwrite or
# clean it away. update-adk-web.sh copies the whole browser directory out, so
# this lands in cmd/launcher/web/webui/distr/ alongside the bundle.
COPY --from=downloader /adk-web-version.json ./dist/agent_framework_web/browser/adk-web-version.json
