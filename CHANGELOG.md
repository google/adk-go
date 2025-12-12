# Changelog

## Unreleased

- functiontool: support non-struct inputs from LLM function calls
  - Automatically wraps function tools whose input type is a non-struct (e.g. `string`, `int`) so they accept LLM function-call arguments of the form `{ "input": <value> }`.
  - Adds `nonStructInputWrapper` implementation, unit + integration tests, and documentation.
