// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file deals with preparing a schema for validation, including various checks,
// optimizations, and the resolution of cross-schema references.

package jsonschema

import (
	"errors"
	"fmt"
	"net/url"
	"reflect"
	"regexp"
	"strings"
)

// A Resolved consists of a [Schema] along with associated information needed to
// validate documents against it.
// A Resolved has been validated against its meta-schema, and all its references
// (the $ref and $dynamicRef keywords) have been resolved to their referenced Schemas.
// Call [Schema.Resolve] to obtain a Resolved from a Schema.
type Resolved struct {
	root *Schema
	// map from $ids to their schemas
	resolvedURIs map[string]*Schema
}

// Schema returns the schema that was resolved.
// It must not be modified.
func (r *Resolved) Schema() *Schema { return r.root }

// A Loader reads and unmarshals the schema at uri, if any.
type Loader func(uri *url.URL) (*Schema, error)

// ResolveOptions are options for [Schema.Resolve].
type ResolveOptions struct {
	// BaseURI is the URI relative to which the root schema should be resolved.
	// If non-empty, must be an absolute URI (one that starts with a scheme).
	// It is resolved (in the URI sense; see [url.ResolveReference]) with root's
	// $id property.
	// If the resulting URI is not absolute, then the schema cannot contain
	// relative URI references.
	BaseURI string
	// Loader loads schemas that are referred to by a $ref but are not under the
	// root schema (remote references).
	// If nil, resolving a remote reference will return an error.
	Loader Loader
	// ValidateDefaults determines whether to validate values of "default" keywords
	// against their schemas.
	// The [JSON Schema specification] does not require this, but it is
	// recommended if defaults will be used.
	//
	// [JSON Schema specification]: https://json-schema.org/understanding-json-schema/reference/annotations
	ValidateDefaults bool
}

// Resolve resolves all references within the schema and performs other tasks that
// prepare the schema for validation.
// If opts is nil, the default values are used.
func (root *Schema) Resolve(opts *ResolveOptions) (*Resolved, error) {
	// There are up to five steps required to prepare a schema to validate.
	// 1. Load: read the schema from somewhere and unmarshal it.
	//    This schema (root) may have been loaded or created in memory, but other schemas that
	//    come into the picture in step 4 will be loaded by the given loader.
	// 2. Check: validate the schema against a meta-schema, and perform other well-formedness checks.
	//    Precompute some values along the way.
	// 3. Resolve URIs: determine the base URI of the root and all its subschemas, and
	//    resolve (in the URI sense) all identifiers and anchors with their bases. This step results
	//    in a map from URIs to schemas within root.
	// 4. Resolve references: all refs in the schemas are replaced with the schema they refer to.
	// 5. (Optional.) If opts.ValidateDefaults is true, validate the defaults.
	if root.path != "" {
		return nil, fmt.Errorf("jsonschema: Resolve: %s already resolved", root)
	}
	r := &resolver{loaded: map[string]*Resolved{}}
	if opts != nil {
		r.opts = *opts
	}
	var base *url.URL
	if r.opts.BaseURI == "" {
		base = &url.URL{} // so we can call ResolveReference on it
	} else {
		var err error
		base, err = url.Parse(r.opts.BaseURI)
		if err != nil {
			return nil, fmt.Errorf("parsing base URI: %w", err)
		}
	}

	if r.opts.Loader == nil {
		r.opts.Loader = func(uri *url.URL) (*Schema, error) {
			return nil, errors.New("cannot resolve remote schemas: no loader passed to Schema.Resolve")
		}
	}

	resolved, err := r.resolve(root, base)
	if err != nil {
		return nil, err
	}
	if r.opts.ValidateDefaults {
		if err := resolved.validateDefaults(); err != nil {
			return nil, err
		}
	}
	// TODO: before we return, throw away anything we don't need for validation.
	return resolved, nil
}

// A resolver holds the state for resolution.
type resolver struct {
	opts ResolveOptions
	// A cache of loaded and partly resolved schemas. (They may not have had their
	// refs resolved.) The cache ensures that the loader will never be called more
	// than once with the same URI, and that reference cycles are handled properly.
	loaded map[string]*Resolved
}

func (r *resolver) resolve(s *Schema, baseURI *url.URL) (*Resolved, error) {
	if baseURI.Fragment != "" {
		return nil, fmt.Errorf("base URI %s must not have a fragment", baseURI)
	}
	if err := s.check(); err != nil {
		return nil, err
	}

	m, err := resolveURIs(s, baseURI)
	if err != nil {
		return nil, err
	}
	rs := &Resolved{root: s, resolvedURIs: m}
	// Remember the schema by both the URI we loaded it from and its canonical name,
	// which may differ if the schema has an $id.
	// We must set the map before calling resolveRefs, or ref cycles will cause unbounded recursion.
	r.loaded[baseURI.String()] = rs
	r.loaded[s.uri.String()] = rs

	if err := r.resolveRefs(rs); err != nil {
		return nil, err
	}
	return rs, nil
}

func (root *Schema) check() error {
	// Check for structural validity. Do this first and fail fast:
	// bad structure will cause other code to panic.
	if err := root.checkStructure(); err != nil {
		return err
	}

	var errs []error
	report := func(err error) { errs = append(errs, err) }

	for ss := range root.all() {
		ss.checkLocal(report)
	}
	return errors.Join(errs...)
}

// checkStructure verifies that root and its subschemas form a tree.
// It also assigns each schema a unique path, to improve error messages.
func (root *Schema) checkStructure() error {
	var check func(reflect.Value, []byte) error
	check = func(v reflect.Value, path []byte) error {
		// For the purpose of error messages, the root schema has path "root"
		// and other schemas' paths are their JSON Pointer from the root.
		p := "root"
		if len(path) > 0 {
			p = string(path)
		}
		s := v.Interface().(*Schema)
		if s == nil {
			return fmt.Errorf("jsonschema: schema at %s is nil", p)
		}
		if s.path != "" {
			// We've seen s before.
			// The schema graph at root is not a tree, but it needs to
			// be because we assume a unique parent when we store a schema's base
			// in the Schema. A cycle would also put Schema.all into an infinite
			// recursion.
			return fmt.Errorf("jsonschema: schemas at %s do not form a tree; %s appears more than once (also at %s)",
				root, s.path, p)
		}
		s.path = p

		for _, info := range schemaFieldInfos {
			fv := v.Elem().FieldByIndex(info.sf.Index)
			switch info.sf.Type {
			case schemaType:
				// A field that contains an individual schema.
				// A nil is valid: it just means the field isn't present.
				if !fv.IsNil() {
					if err := check(fv, fmt.Appendf(path, "/%s", info.jsonName)); err != nil {
						return err
					}
				}

			case schemaSliceType:
				for i := range fv.Len() {
					if err := check(fv.Index(i), fmt.Appendf(path, "/%s/%d", info.jsonName, i)); err != nil {
						return err
					}
				}

			case schemaMapType:
				iter := fv.MapRange()
				for iter.Next() {
					key := escapeJSONPointerSegment(iter.Key().String())
					if err := check(iter.Value(), fmt.Appendf(path, "/%s/%s", info.jsonName, key)); err != nil {
						return err
					}
				}
			}

		}
		return nil
	}

	return check(reflect.ValueOf(root), make([]byte, 0, 256))
}

// checkLocal checks s for validity, independently of other schemas it may refer to.
// Since checking a regexp involves compiling it, checkLocal saves those compiled regexps
// in the schema for later use.
// It appends the errors it finds to errs.
func (s *Schema) checkLocal(report func(error)) {
	addf := func(format string, args ...any) {
		msg := fmt.Sprintf(format, args...)
		report(fmt.Errorf("jsonschema.Schema: %s: %s", s, msg))
	}

	if s == nil {
		addf("nil subschema")
		return
	}
	if err := s.basicChecks(); err != nil {
		report(err)
		return
	}

	// TODO: validate the schema's properties,
	// ideally by jsonschema-validating it against the meta-schema.

	// Some properties are present so that Schemas can round-trip, but we do not
	// validate them.
	// Currently, it's just the $vocabulary property.
	// As a special case, we can validate the 2020-12 meta-schema.
	if s.Vocabulary != nil && s.Schema != draft202012 {
		addf("cannot validate a schema with $vocabulary")
	}

	// Check and compile regexps.
	if s.Pattern != "" {
		re, err := regexp.Compile(s.Pattern)
		if err != nil {
			addf("pattern: %v", err)
		} else {
			s.pattern = re
		}
	}
	if len(s.PatternProperties) > 0 {
		s.patternProperties = map[*regexp.Regexp]*Schema{}
		for reString, subschema := range s.PatternProperties {
			re, err := regexp.Compile(reString)
			if err != nil {
				addf("patternProperties[%q]: %v", reString, err)
				continue
			}
			s.patternProperties[re] = subschema
		}
	}

	// Build a set of required properties, to avoid quadratic behavior when validating
	// a struct.
	if len(s.Required) > 0 {
		s.isRequired = map[string]bool{}
		for _, r := range s.Required {
			s.isRequired[r] = true
		}
	}
}

// resolveURIs resolves the ids and anchors in all the schemas of root, relative
// to baseURI.
// See https://json-schema.org/draft/2020-12/json-schema-core#section-8.2, section
// 8.2.1.

// TODO(jba): dynamicAnchors (§8.2.2)
//
// Every schema has a base URI and a parent base URI.
//
// The parent base URI is the base URI of the lexically enclosing schema, or for
// a root schema, the URI it was loaded from or the one supplied to [Schema.Resolve].
//
// If the schema has no $id property, the base URI of a schema is that of its parent.
// If the schema does have an $id, it must be a URI, possibly relative. The schema's
// base URI is the $id resolved (in the sense of [url.URL.ResolveReference]) against
// the parent base.
//
// As an example, consider this schema loaded from http://a.com/root.json (quotes omitted):
//
//	{
//	    allOf: [
//	        {$id: "sub1.json", minLength: 5},
//	        {$id: "http://b.com", minimum: 10},
//	        {not: {maximum: 20}}
//	    ]
//	}
//
// The base URIs are as follows. Schema locations are expressed in the JSON Pointer notation.
//
//	schema         base URI
//	root           http://a.com/root.json
//	allOf/0        http://a.com/sub1.json
//	allOf/1        http://b.com (absolute $id; doesn't matter that it's not under the loaded URI)
//	allOf/2        http://a.com/root.json (inherited from parent)
//	allOf/2/not    http://a.com/root.json (inherited from parent)
func resolveURIs(root *Schema, baseURI *url.URL) (map[string]*Schema, error) {
	resolvedURIs := map[string]*Schema{}

	var resolve func(s, base *Schema) error
	resolve = func(s, base *Schema) error {
		// ids are scoped to the root.
		if s.ID != "" {
			// A non-empty ID establishes a new base.
			idURI, err := url.Parse(s.ID)
			if err != nil {
				return err
			}
			if idURI.Fragment != "" {
				return fmt.Errorf("$id %s must not have a fragment", s.ID)
			}
			// The base URI for this schema is its $id resolved against the parent base.
			s.uri = base.uri.ResolveReference(idURI)
			if !s.uri.IsAbs() {
				return fmt.Errorf("$id %s does not resolve to an absolute URI (base is %s)", s.ID, s.base.uri)
			}
			resolvedURIs[s.uri.String()] = s
			base = s // needed for anchors
		}
		s.base = base

		// Anchors and dynamic anchors are URI fragments that are scoped to their base.
		// We treat them as keys in a map stored within the schema.
		setAnchor := func(anchor string, dynamic bool) error {
			if anchor != "" {
				if _, ok := base.anchors[anchor]; ok {
					return fmt.Errorf("duplicate anchor %q in %s", anchor, base.uri)
				}
				if base.anchors == nil {
					base.anchors = map[string]anchorInfo{}
				}
				base.anchors[anchor] = anchorInfo{s, dynamic}
			}
			return nil
		}

		setAnchor(s.Anchor, false)
		setAnchor(s.DynamicAnchor, true)

		for c := range s.children() {
			if err := resolve(c, base); err != nil {
				return err
			}
		}
		return nil
	}

	// Set the root URI to the base for now. If the root has an $id, this will change.
	root.uri = baseURI
	// The original base, even if changed, is still a valid way to refer to the root.
	resolvedURIs[baseURI.String()] = root
	if err := resolve(root, root); err != nil {
		return nil, err
	}
	return resolvedURIs, nil
}

// resolveRefs replaces every ref in the schemas with the schema it refers to.
// A reference that doesn't resolve within the schema may refer to some other schema
// that needs to be loaded.
func (r *resolver) resolveRefs(rs *Resolved) error {
	for s := range rs.root.all() {
		if s.Ref != "" {
			refSchema, _, err := r.resolveRef(rs, s, s.Ref)
			if err != nil {
				return err
			}
			// Whether or not the anchor referred to by $ref fragment is dynamic,
			// the ref still treats it lexically.
			s.resolvedRef = refSchema
		}
		if s.DynamicRef != "" {
			refSchema, frag, err := r.resolveRef(rs, s, s.DynamicRef)
			if err != nil {
				return err
			}
			if frag != "" {
				// The dynamic ref's fragment points to a dynamic anchor.
				// We must resolve the fragment at validation time.
				s.dynamicRefAnchor = frag
			} else {
				// There is no dynamic anchor in the lexically referenced schema,
				// so the dynamic ref behaves like a lexical ref.
				s.resolvedDynamicRef = refSchema
			}
		}
	}
	return nil
}

// resolveRef resolves the reference ref, which is either s.Ref or s.DynamicRef.
func (r *resolver) resolveRef(rs *Resolved, s *Schema, ref string) (_ *Schema, dynamicFragment string, err error) {
	refURI, err := url.Parse(ref)
	if err != nil {
		return nil, "", err
	}
	// URI-resolve the ref against the current base URI to get a complete URI.
	refURI = s.base.uri.ResolveReference(refURI)
	// The non-fragment part of a ref URI refers to the base URI of some schema.
	// This part is the same for dynamic refs too: their non-fragment part resolves
	// lexically.
	u := *refURI
	u.Fragment = ""
	fraglessRefURI := &u
	// Look it up locally.
	referencedSchema := rs.resolvedURIs[fraglessRefURI.String()]
	if referencedSchema == nil {
		// The schema is remote. Maybe we've already loaded it.
		// We assume that the non-fragment part of refURI refers to a top-level schema
		// document. That is, we don't support the case exemplified by
		// http://foo.com/bar.json/baz, where the document is in bar.json and
		// the reference points to a subschema within it.
		// TODO: support that case.
		if lrs := r.loaded[fraglessRefURI.String()]; lrs != nil {
			referencedSchema = lrs.root
		} else {
			// Try to load the schema.
			ls, err := r.opts.Loader(fraglessRefURI)
			if err != nil {
				return nil, "", fmt.Errorf("loading %s: %w", fraglessRefURI, err)
			}
			lrs, err := r.resolve(ls, fraglessRefURI)
			if err != nil {
				return nil, "", err
			}
			referencedSchema = lrs.root
			assert(referencedSchema != nil, "nil referenced schema")
		}
	}

	frag := refURI.Fragment
	// Look up frag in refSchema.
	// frag is either a JSON Pointer or the name of an anchor.
	// A JSON Pointer is either the empty string or begins with a '/',
	// whereas anchors are always non-empty strings that don't contain slashes.
	if frag != "" && !strings.HasPrefix(frag, "/") {
		info, found := referencedSchema.anchors[frag]
		if !found {
			return nil, "", fmt.Errorf("no anchor %q in %s", frag, s)
		}
		if info.dynamic {
			dynamicFragment = frag
		}
		return info.schema, dynamicFragment, nil
	}
	// frag is a JSON Pointer.
	s, err = dereferenceJSONPointer(referencedSchema, frag)
	return s, "", err
}
