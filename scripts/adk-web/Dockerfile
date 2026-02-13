# Stage 1: Clone the repository
FROM node:iron-trixie AS downloader
RUN apt-get update && apt-get install -y git

# Trigger re-clone when the main branch changes.
ADD https://api.github.com/repos/google/adk-web/git/refs/heads/main version.json
RUN git clone -b main https://github.com/google/adk-web /adk-web && rm version.json

# Stage 2: Build the application
FROM node:iron-trixie AS builder

WORKDIR /adk-web

# Copy package files first to leverage Docker cache for npm install.
# This ensures that npm install is only re-run if package.json or package-lock.json changes,
# and not when other source files change.
COPY --from=downloader /adk-web/package*.json ./
RUN npm install

# Copy the rest of the source code
COPY --from=downloader /adk-web ./

RUN npm run build
