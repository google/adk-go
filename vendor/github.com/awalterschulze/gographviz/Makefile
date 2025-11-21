.PHONY: help regenerate test dependencies build checkers action

# Prefer tools that we've installed
export PATH := $(HOME)/go/bin:$(PATH)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

regenerate: ## Re-generate lexers and parsers and pass through goimports
	go get github.com/goccmack/gocc
	go install github.com/goccmack/gocc
	gocc -zip -o ./internal/ dot.bnf
	find . -type f -name '*.go' | xargs goimports -w

test: ## Perform package tests
	go test ./...

dependencies: ## Grab necessary dependencies for checkers
	go version
	go get golang.org/x/tools/cmd/goimports
	go get github.com/kisielk/errcheck
	go get -u golang.org/x/lint/golint

build: ## Perform build process
	go build .

checkers: ## Run all checkers (errcheck, gofmt and golint)
	errcheck -ignore 'fmt:[FS]?[Pp]rint*' ./...
	gofmt -l -s -w .
	golint -set_exit_status
	git diff --exit-code

action: dependencies regenerate build test checkers ## Run steps of github action
