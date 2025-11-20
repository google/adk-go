//Copyright 2013 GoGraphviz Authors
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

//Abstract Syntax Tree representing the DOT grammar
package ast

import (
	"errors"
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"sync"

	"github.com/awalterschulze/gographviz/internal/token"
)

var (
	r        = rand.New(rand.NewSource(1234))
	randLock sync.Mutex
)

type Visitor interface {
	Visit(e Elem) Visitor
}

type Elem interface {
	String() string
}

type Walkable interface {
	Walk(v Visitor)
}

type Attrib interface{}

type Bool bool

const (
	FALSE = Bool(false)
	TRUE  = Bool(true)
)

func (this Bool) String() string {
	if this {
		return "true"
	}
	return "false"
}

func (this Bool) Walk(v Visitor) {
	if v == nil {
		return
	}
	v.Visit(this)
}

type GraphType bool

const (
	GRAPH   = GraphType(false)
	DIGRAPH = GraphType(true)
)

func (this GraphType) String() string {
	if this {
		return "digraph"
	}
	return "graph"
}

func (this GraphType) Walk(v Visitor) {
	if v == nil {
		return
	}
	v.Visit(this)
}

type Graph struct {
	Type     GraphType
	Strict   bool
	ID       ID
	StmtList StmtList
}

func NewGraph(t, strict, id, l Attrib) (*Graph, error) {
	g := &Graph{Type: t.(GraphType), Strict: bool(strict.(Bool)), ID: ID("")}
	if id != nil {
		g.ID = id.(ID)
	}
	if l != nil {
		g.StmtList = l.(StmtList)
	}
	return g, nil
}

func (this *Graph) String() string {
	var s string
	if this.Strict {
		s += "strict "
	}
	s += this.Type.String() + " " + this.ID.String() + " {\n"
	if this.StmtList != nil {
		s += this.StmtList.String()
	}
	s += "\n}\n"
	return s
}

func (this *Graph) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	this.Type.Walk(v)
	this.ID.Walk(v)
	this.StmtList.Walk(v)
}

type StmtList []Stmt

func NewStmtList(s Attrib) (StmtList, error) {
	ss := make(StmtList, 1)
	ss[0] = s.(Stmt)
	return ss, nil
}

func AppendStmtList(ss, s Attrib) (StmtList, error) {
	this := ss.(StmtList)
	this = append(this, s.(Stmt))
	return this, nil
}

func (this StmtList) String() string {
	if len(this) == 0 {
		return ""
	}
	s := ""
	for i := 0; i < len(this); i++ {
		ss := this[i].String()
		if len(ss) > 0 {
			s += "\t" + ss + ";\n"
		}
	}
	return s
}

func (this StmtList) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	for i := range this {
		this[i].Walk(v)
	}
}

type Stmt interface {
	Elem
	Walkable
	isStmt()
}

func (this NodeStmt) isStmt()   {}
func (this EdgeStmt) isStmt()   {}
func (this EdgeAttrs) isStmt()  {}
func (this NodeAttrs) isStmt()  {}
func (this GraphAttrs) isStmt() {}
func (this *SubGraph) isStmt()  {}
func (this *Attr) isStmt()      {}

type SubGraph struct {
	ID       ID
	StmtList StmtList
}

func NewSubGraph(maybeId, l Attrib) (*SubGraph, error) {
	g := &SubGraph{}
	if id, ok := maybeId.(ID); maybeId == nil || (ok && len(id) == 0) {
		g.ID = ID(fmt.Sprintf("anon%d", randInt63()))
	} else if ok && (len(id) > 0) {
		g.ID = id
	} else {
		return nil, fmt.Errorf("expected maybeId.(ID) got=%v", maybeId)
	}
	if l != nil {
		g.StmtList = l.(StmtList)
	}
	return g, nil
}

func randInt63() int64 {
	randLock.Lock()
	result := r.Int63()
	randLock.Unlock()
	return result
}

func (this *SubGraph) GetID() ID {
	return this.ID
}

func (this *SubGraph) GetPort() Port {
	return NewPort(nil, nil)
}

func (this *SubGraph) String() string {
	gName := this.ID.String()
	if strings.HasPrefix(gName, "anon") {
		gName = ""
	}
	s := "subgraph " + this.ID.String() + " {\n"
	if this.StmtList != nil {
		s += this.StmtList.String()
	}
	s += "\n}\n"
	return s
}

func (this *SubGraph) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	this.ID.Walk(v)
	this.StmtList.Walk(v)
}

type EdgeAttrs AttrList

func NewEdgeAttrs(a Attrib) (EdgeAttrs, error) {
	return EdgeAttrs(a.(AttrList)), nil
}

func (this EdgeAttrs) String() string {
	s := AttrList(this).String()
	if len(s) == 0 {
		return ""
	}
	return `edge ` + s
}

func (this EdgeAttrs) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	for i := range this {
		this[i].Walk(v)
	}
}

type NodeAttrs AttrList

func NewNodeAttrs(a Attrib) (NodeAttrs, error) {
	return NodeAttrs(a.(AttrList)), nil
}

func (this NodeAttrs) String() string {
	s := AttrList(this).String()
	if len(s) == 0 {
		return ""
	}
	return `node ` + s
}

func (this NodeAttrs) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	for i := range this {
		this[i].Walk(v)
	}
}

type GraphAttrs AttrList

func NewGraphAttrs(a Attrib) (GraphAttrs, error) {
	return GraphAttrs(a.(AttrList)), nil
}

func (this GraphAttrs) String() string {
	s := AttrList(this).String()
	if len(s) == 0 {
		return ""
	}
	return `graph ` + s
}

func (this GraphAttrs) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	for i := range this {
		this[i].Walk(v)
	}
}

type AttrList []AList

func NewAttrList(a Attrib) (AttrList, error) {
	as := make(AttrList, 0)
	if a != nil {
		as = append(as, a.(AList))
	}
	return as, nil
}

func AppendAttrList(as, a Attrib) (AttrList, error) {
	this := as.(AttrList)
	if a == nil {
		return this, nil
	}
	this = append(this, a.(AList))
	return this, nil
}

func (this AttrList) String() string {
	s := ""
	for _, alist := range this {
		ss := alist.String()
		if len(ss) > 0 {
			s += "[ " + ss + " ] "
		}
	}
	if len(s) == 0 {
		return ""
	}
	return s
}

func (this AttrList) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	for i := range this {
		this[i].Walk(v)
	}
}

func PutMap(attrmap map[string]string) AttrList {
	attrlist := make(AttrList, 1)
	attrlist[0] = make(AList, 0)
	keys := make([]string, 0, len(attrmap))
	for key := range attrmap {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, name := range keys {
		value := attrmap[name]
		attrlist[0] = append(attrlist[0], &Attr{ID(name), ID(value)})
	}
	return attrlist
}

func (this AttrList) GetMap() map[string]string {
	attrs := make(map[string]string)
	for _, alist := range this {
		for _, attr := range alist {
			attrs[attr.Field.String()] = attr.Value.String()
		}
	}
	return attrs
}

type AList []*Attr

func NewAList(a Attrib) (AList, error) {
	as := make(AList, 1)
	as[0] = a.(*Attr)
	return as, nil
}

func AppendAList(as, a Attrib) (AList, error) {
	this := as.(AList)
	attr := a.(*Attr)
	this = append(this, attr)
	return this, nil
}

func (this AList) String() string {
	if len(this) == 0 {
		return ""
	}
	str := this[0].String()
	for i := 1; i < len(this); i++ {
		str += `, ` + this[i].String()
	}
	return str
}

func (this AList) Walk(v Visitor) {
	v = v.Visit(this)
	for i := range this {
		this[i].Walk(v)
	}
}

type Attr struct {
	Field ID
	Value ID
}

func NewAttr(f, v Attrib) (*Attr, error) {
	a := &Attr{Field: f.(ID)}
	a.Value = ID("true")
	if v != nil {
		ok := false
		a.Value, ok = v.(ID)
		if !ok {
			return nil, errors.New(fmt.Sprintf("value = %v", v))
		}
	}
	return a, nil
}

func (this *Attr) String() string {
	return this.Field.String() + `=` + this.Value.String()
}

func (this *Attr) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	this.Field.Walk(v)
	this.Value.Walk(v)
}

type Location interface {
	Elem
	Walkable
	isLocation()
	GetID() ID
	GetPort() Port
	IsNode() bool
}

func (this *NodeID) isLocation()    {}
func (this *NodeID) IsNode() bool   { return true }
func (this *SubGraph) isLocation()  {}
func (this *SubGraph) IsNode() bool { return false }

type EdgeStmt struct {
	Source  Location
	EdgeRHS EdgeRHS
	Attrs   AttrList
}

func NewEdgeStmt(id, e, attrs Attrib) (*EdgeStmt, error) {
	var a AttrList = nil
	var err error = nil
	if attrs == nil {
		a, err = NewAttrList(nil)
		if err != nil {
			return nil, err
		}
	} else {
		a = attrs.(AttrList)
	}
	return &EdgeStmt{id.(Location), e.(EdgeRHS), a}, nil
}

func (this EdgeStmt) String() string {
	return strings.TrimSpace(this.Source.String() + this.EdgeRHS.String() + this.Attrs.String())
}

func (this EdgeStmt) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	this.Source.Walk(v)
	this.EdgeRHS.Walk(v)
	this.Attrs.Walk(v)
}

type EdgeRHS []*EdgeRH

func NewEdgeRHS(op, id Attrib) (EdgeRHS, error) {
	return EdgeRHS{&EdgeRH{op.(EdgeOp), id.(Location)}}, nil
}

func AppendEdgeRHS(e, op, id Attrib) (EdgeRHS, error) {
	erhs := e.(EdgeRHS)
	erhs = append(erhs, &EdgeRH{op.(EdgeOp), id.(Location)})
	return erhs, nil
}

func (this EdgeRHS) String() string {
	s := ""
	for i := range this {
		s += this[i].String()
	}
	return strings.TrimSpace(s)
}

func (this EdgeRHS) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	for i := range this {
		this[i].Walk(v)
	}
}

type EdgeRH struct {
	Op          EdgeOp
	Destination Location
}

func (this *EdgeRH) String() string {
	return strings.TrimSpace(this.Op.String() + this.Destination.String())
}

func (this *EdgeRH) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	this.Op.Walk(v)
	this.Destination.Walk(v)
}

type NodeStmt struct {
	NodeID *NodeID
	Attrs  AttrList
}

func NewNodeStmt(id, attrs Attrib) (*NodeStmt, error) {
	nid := id.(*NodeID)
	var a AttrList = nil
	var err error = nil
	if attrs == nil {
		a, err = NewAttrList(nil)
		if err != nil {
			return nil, err
		}
	} else {
		a = attrs.(AttrList)
	}
	return &NodeStmt{nid, a}, nil
}

func (this NodeStmt) String() string {
	return strings.TrimSpace(this.NodeID.String() + ` ` + this.Attrs.String())
}

func (this NodeStmt) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	this.NodeID.Walk(v)
	this.Attrs.Walk(v)
}

type EdgeOp bool

const (
	DIRECTED   EdgeOp = true
	UNDIRECTED EdgeOp = false
)

func (this EdgeOp) String() string {
	if this == DIRECTED {
		return "->"
	}
	return "--"
}

func (this EdgeOp) Walk(v Visitor) {
	if v == nil {
		return
	}
	v.Visit(this)
}

type NodeID struct {
	ID   ID
	Port Port
}

func NewNodeID(id, port Attrib) (*NodeID, error) {
	if port == nil {
		return &NodeID{id.(ID), Port{"", ""}}, nil
	}
	return &NodeID{id.(ID), port.(Port)}, nil
}

func MakeNodeID(id string, port string) *NodeID {
	p := Port{"", ""}
	if len(port) > 0 {
		ps := strings.Split(port, ":")
		p.ID1 = ID(ps[0])
		if len(ps) > 1 {
			p.ID2 = ID(ps[1])
		}
	}
	return &NodeID{ID(id), p}
}

func (this *NodeID) String() string {
	return this.ID.String() + this.Port.String()
}

func (this *NodeID) GetID() ID {
	return this.ID
}

func (this *NodeID) GetPort() Port {
	return this.Port
}

func (this *NodeID) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	this.ID.Walk(v)
	this.Port.Walk(v)
}

//TODO semantic analysis should decide which ID is an ID and which is a Compass Point
type Port struct {
	ID1 ID
	ID2 ID
}

func NewPort(id1, id2 Attrib) Port {
	port := Port{ID(""), ID("")}
	if id1 != nil {
		port.ID1 = id1.(ID)
	}
	if id2 != nil {
		port.ID2 = id2.(ID)
	}
	return port
}

func (this Port) String() string {
	if len(this.ID1) == 0 {
		return ""
	}
	s := ":" + this.ID1.String()
	if len(this.ID2) > 0 {
		s += ":" + this.ID2.String()
	}
	return s
}

func (this Port) Walk(v Visitor) {
	if v == nil {
		return
	}
	v = v.Visit(this)
	this.ID1.Walk(v)
	this.ID2.Walk(v)
}

type ID string

func NewID(id Attrib) (ID, error) {
	if id == nil {
		return ID(""), nil
	}
	id_lit := string(id.(*token.Token).Lit)
	return ID(id_lit), nil
}

func (this ID) String() string {
	return string(this)
}

func (this ID) Walk(v Visitor) {
	if v == nil {
		return
	}
	v.Visit(this)
}
