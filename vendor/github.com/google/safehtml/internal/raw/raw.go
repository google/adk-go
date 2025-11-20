// Copyright (c) 2017 The Go Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

// Package raw provides a coordination point for package safehtml, package
// uncheckedconversions, package legacyconversions, and package testconversions.
// raw must only be imported by these four packages.
package raw

// HTML is the raw constructor for a safehtml.HTML.
var HTML interface{}

// Script is the raw constructor for a safehtml.Script.
var Script interface{}

// Style is the raw constructor for a safehtml.Style.
var Style interface{}

// StyleSheet is the raw constructor for a safehtml.StyleSheet.
var StyleSheet interface{}

// URL is the raw constructor for a safehtml.URL.
var URL interface{}

// TrustedResourceURL is the raw constructor for a safehtml.TrustedResourceURL.
var TrustedResourceURL interface{}

// Identifier is the raw constructor for a safehtml.Identifier.
var Identifier interface{}
