# Stage 1: Clone the repository
FROM node:iron-trixie AS downloader
RUN apt-get update && apt-get install -y git

# The upstream revision to build. A SHA rather than a tag, because upstream can
# move a tag. 33c3568 is upstream tag v1.0.5, dated 2026-08-26. Pass
# --build-arg ADK_WEB_REF=<sha-or-tag> for a one-off build; see
# scripts/adk-web/README.md for how to move the pin for everyone.
#
# Keep it pinned. This used to clone `main` behind a cache-buster, so a refresh
# built whatever upstream had that day and recorded nothing, which is how a
# breaking UI change shipped unnoticed and left most of the UI returning 404 for
# two months.
#
# This pin does NOT describe the bundle committed in
# cmd/launcher/web/webui/distr/, which predates it. Building here moves the UI
# forward rather than reproducing what is in the tree. README.md in this
# directory explains why, and what changes after the first refresh.
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
