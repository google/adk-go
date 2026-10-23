# OpenAPI toolset

Generate ADK tools from an OpenAPI 3.x document without writing one function
wrapper per endpoint.

The example starts a local `httptest` server, loads its JSON document through
`openapitoolset.NewFromURL`, and prints the generated operation. It needs no
credentials and makes no external network calls.

## Run

```bash
go run ./tool/openapitoolset/examples/openapi
```

Output:

```text
get_widget: Get a widget by ID
```

The same toolset can load YAML or JSON from bytes, a local file, or an HTTP URL.
Pass it to an ADK agent as a `tool.Toolset`. Supply an `auth.CredentialProvider`
in the config when operation requests need credentials.
