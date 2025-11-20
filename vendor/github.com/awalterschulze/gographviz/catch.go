//Copyright 2017 GoGraphviz Authors
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

package gographviz

import (
	"fmt"
	"strings"
)

type errInterface interface {
	SetStrict(strict bool)
	SetDir(directed bool)
	SetName(name string)
	AddPortEdge(src, srcPort, dst, dstPort string, directed bool, attrs map[string]string)
	AddEdge(src, dst string, directed bool, attrs map[string]string)
	AddNode(parentGraph string, name string, attrs map[string]string)
	AddAttr(parentGraph string, field, value string)
	AddSubGraph(parentGraph string, name string, attrs map[string]string)
	String() string
	getError() error
}

func newErrCatcher(g Interface) errInterface {
	return &errCatcher{g, nil}
}

type errCatcher struct {
	Interface
	errs []error
}

func (e *errCatcher) SetStrict(strict bool) {
	if err := e.Interface.SetStrict(strict); err != nil {
		e.errs = append(e.errs, err)
	}
}

func (e *errCatcher) SetDir(directed bool) {
	if err := e.Interface.SetDir(directed); err != nil {
		e.errs = append(e.errs, err)
	}
}

func (e *errCatcher) SetName(name string) {
	if err := e.Interface.SetName(name); err != nil {
		e.errs = append(e.errs, err)
	}
}

func (e *errCatcher) AddPortEdge(src, srcPort, dst, dstPort string, directed bool, attrs map[string]string) {
	if err := e.Interface.AddPortEdge(src, srcPort, dst, dstPort, directed, attrs); err != nil {
		e.errs = append(e.errs, err)
	}
}

func (e *errCatcher) AddEdge(src, dst string, directed bool, attrs map[string]string) {
	if err := e.Interface.AddEdge(src, dst, directed, attrs); err != nil {
		e.errs = append(e.errs, err)
	}
}

func (e *errCatcher) AddAttr(parentGraph string, field, value string) {
	if err := e.Interface.AddAttr(parentGraph, field, value); err != nil {
		e.errs = append(e.errs, err)
	}
}

func (e *errCatcher) AddSubGraph(parentGraph string, name string, attrs map[string]string) {
	if err := e.Interface.AddSubGraph(parentGraph, name, attrs); err != nil {
		e.errs = append(e.errs, err)
	}
}

func (e *errCatcher) AddNode(parentGraph string, name string, attrs map[string]string) {
	if err := e.Interface.AddNode(parentGraph, name, attrs); err != nil {
		e.errs = append(e.errs, err)
	}
}

func (e *errCatcher) getError() error {
	if len(e.errs) == 0 {
		return nil
	}
	ss := make([]string, len(e.errs))
	for i, err := range e.errs {
		ss[i] = err.Error()
	}
	return fmt.Errorf("errors: [%s]", strings.Join(ss, ","))
}
