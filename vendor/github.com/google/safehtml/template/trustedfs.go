// Copyright (c) 2021 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

//go:build go1.16
// +build go1.16

package template

import (
	"embed"
	"fmt"
	"io/fs"
	"os"
	"path"
)

// A TrustedFS is an immutable type referencing a filesystem (fs.FS)
// under application control.
//
// In order to ensure that an attacker cannot influence the TrustedFS value, a
// TrustedFS can be instantiated in only two ways. One way is from an embed.FS
// with TrustedFSFromEmbed. It is assumed that embedded filesystems are under
// the programmer's control. The other way is from a TrustedSource using
// TrustedFSFromTrustedSource, in which case the guarantees and caveats of
// TrustedSource apply.
type TrustedFS struct {
	fsys fs.FS
}

// TrustedFSFromEmbed constructs a TrustedFS from an embed.FS.
func TrustedFSFromEmbed(fsys embed.FS) TrustedFS {
	return TrustedFS{fsys: fsys}
}

// TrustedFSFromTrustedSource constructs a TrustedFS from the string in the
// TrustedSource, which should refer to a directory.
func TrustedFSFromTrustedSource(ts TrustedSource) TrustedFS {
	return TrustedFS{fsys: os.DirFS(ts.src)}
}

// Sub returns a TrustedFS at a subdirectory of the receiver.
// It works by calling fs.Sub on the receiver's fs.FS.
func (tf TrustedFS) Sub(dir TrustedSource) (TrustedFS, error) {
	subfs, err := fs.Sub(tf.fsys, dir.String())
	return TrustedFS{fsys: subfs}, err
}

// ParseFS is like ParseFiles or ParseGlob but reads from the TrustedFS
// instead of the host operating system's file system.
// It accepts a list of glob patterns.
// (Note that most file names serve as glob patterns matching only themselves.)
//
// The same behaviors listed for ParseFiles() apply to ParseFS too (e.g. using the base name
// of the file as the template name).
func ParseFS(tfs TrustedFS, patterns ...string) (*Template, error) {
	return parseFS(nil, tfs.fsys, patterns)
}

// ParseFS is like ParseFiles or ParseGlob but reads from the TrustedFS
// instead of the host operating system's file system.
// It accepts a list of glob patterns.
// (Note that most file names serve as glob patterns matching only themselves.)
//
// The same behaviors listed for ParseFiles() apply to ParseFS too (e.g. using the base name
// of the file as the template name).
func (t *Template) ParseFS(tfs TrustedFS, patterns ...string) (*Template, error) {
	return parseFS(t, tfs.fsys, patterns)
}

// Copied from
// https://go.googlesource.com/go/+/refs/tags/go1.17.1/src/text/template/helper.go.
func parseFS(t *Template, fsys fs.FS, patterns []string) (*Template, error) {
	var filenames []string
	for _, pattern := range patterns {
		list, err := fs.Glob(fsys, pattern)
		if err != nil {
			return nil, err
		}
		if len(list) == 0 {
			return nil, fmt.Errorf("template: pattern matches no files: %#q", pattern)
		}
		filenames = append(filenames, list...)
	}
	return parseFiles(t, readFileFS(fsys), filenames...)
}

// Copied with minor changes from
// https://go.googlesource.com/go/+/refs/tags/go1.17.1/src/text/template/helper.go.
func readFileFS(fsys fs.FS) func(string) (string, []byte, error) {
	return func(file string) (string, []byte, error) {
		name := path.Base(file)
		b, err := fs.ReadFile(fsys, file)
		return name, b, err
	}
}
