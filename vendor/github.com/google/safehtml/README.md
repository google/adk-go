# Safe HTML for Go

`safehtml` provides immutable string-like types that wrap web types such as
HTML, JavaScript and CSS. These wrappers are safe by construction against XSS
and similar web vulnerabilities, and they can only be interpolated in safe ways.
You can read more about our approach to web security in our
[whitepaper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42934.pdf),
or this [OWASP talk](https://www.youtube.com/watch?v=ccfEu-Jj0as).

Additional subpackages provide APIs for managing exceptions to the
safety rules, and a template engine with a syntax and interface that closely
matches [`html/template`](https://golang.org/pkg/html/template/). You can refer
to the [godoc](https://pkg.go.dev/github.com/google/safehtml?tab=doc)
for each (sub)package for the API documentation and code examples.
More end-to-end demos are available in `example_test.go`.

This is not an officially supported Google product.
