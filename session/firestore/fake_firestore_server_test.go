// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package firestore

import (
	"bytes"
	"cmp"
	"context"
	"fmt"
	"net"
	"slices"
	"strings"
	"sync"
	"testing"

	fs "cloud.google.com/go/firestore"
	pb "cloud.google.com/go/firestore/apiv1/firestorepb"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// fakeFirestoreServer is an in-process, thread-safe implementation of
// pb.FirestoreServer over bufconn for offline testing of *firestore.Client.
type fakeFirestoreServer struct {
	pb.UnimplementedFirestoreServer

	mu              sync.Mutex
	docs            map[string]*pb.Document
	docVersions     map[string]int64
	txReads         map[string]map[string]int64
	nextTxID        int
	batchCommits    int
	txAborts        int
	failNextWrite   error
	failNextRead    error
	failNextQuery   error
	failBatchWrites error
	onBeforeCommit  func(txID string)
}

func newFakeFirestoreServer() *fakeFirestoreServer {
	return &fakeFirestoreServer{
		docs:        make(map[string]*pb.Document),
		docVersions: make(map[string]int64),
		txReads:     make(map[string]map[string]int64),
	}
}

func newTestFirestoreClient(t *testing.T) (*fs.Client, *fakeFirestoreServer) {
	t.Helper()

	srv := newFakeFirestoreServer()
	lis := bufconn.Listen(1 << 20)
	grpcServer := grpc.NewServer()
	pb.RegisterFirestoreServer(grpcServer, srv)

	go func() {
		_ = grpcServer.Serve(lis)
	}()
	t.Cleanup(func() {
		grpcServer.Stop()
		_ = lis.Close()
	})

	dialer := func(ctx context.Context, _ string) (net.Conn, error) {
		return lis.DialContext(ctx)
	}
	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(dialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("grpc.NewClient() failed: %v", err)
	}
	t.Cleanup(func() {
		_ = conn.Close()
	})

	client, err := fs.NewClient(
		t.Context(),
		"test-project",
		option.WithGRPCConn(conn),
		option.WithoutAuthentication(),
	)
	if err != nil {
		t.Fatalf("firestore.NewClient() failed: %v", err)
	}
	t.Cleanup(func() {
		_ = client.Close()
	})

	return client, srv
}

func truncateTimestampsInDoc(doc *pb.Document) {
	if doc == nil {
		return
	}
	for k, v := range doc.Fields {
		if ts, ok := v.GetValueType().(*pb.Value_TimestampValue); ok && ts.TimestampValue != nil {
			doc.Fields[k] = &pb.Value{
				ValueType: &pb.Value_TimestampValue{
					TimestampValue: timestamppb.New(ts.TimestampValue.AsTime().Truncate(1000)),
				},
			}
		}
	}
}

func (f *fakeFirestoreServer) putDoc(docPath string, fields map[string]*pb.Value) {
	f.mu.Lock()
	defer f.mu.Unlock()

	now := timestamppb.New(timestamppb.Now().AsTime().Truncate(1000))
	fullName := "projects/test-project/databases/(default)/documents/" + strings.TrimPrefix(docPath, "/")
	doc := &pb.Document{
		Name:       fullName,
		Fields:     fields,
		CreateTime: now,
		UpdateTime: now,
	}
	truncateTimestampsInDoc(doc)
	f.docs[fullName] = doc
	f.docVersions[fullName]++
}

func (f *fakeFirestoreServer) getDoc(docPath string) (*pb.Document, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()

	fullName := "projects/test-project/databases/(default)/documents/" + strings.TrimPrefix(docPath, "/")
	doc, ok := f.docs[fullName]
	if !ok {
		return nil, false
	}
	return proto.Clone(doc).(*pb.Document), true
}

func (f *fakeFirestoreServer) BeginTransaction(_ context.Context, _ *pb.BeginTransactionRequest) (*pb.BeginTransactionResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.nextTxID++
	txID := fmt.Sprintf("tx-%d", f.nextTxID)
	f.txReads[txID] = make(map[string]int64)
	return &pb.BeginTransactionResponse{
		Transaction: []byte(txID),
	}, nil
}

func (f *fakeFirestoreServer) Rollback(_ context.Context, req *pb.RollbackRequest) (*emptypb.Empty, error) {
	f.mu.Lock()
	delete(f.txReads, string(req.GetTransaction()))
	f.mu.Unlock()
	return &emptypb.Empty{}, nil
}

func (f *fakeFirestoreServer) BatchGetDocuments(req *pb.BatchGetDocumentsRequest, stream pb.Firestore_BatchGetDocumentsServer) error {
	f.mu.Lock()
	if f.failNextRead != nil {
		err := f.failNextRead
		f.failNextRead = nil
		f.mu.Unlock()
		return err
	}
	txID := string(req.GetTransaction())
	now := timestamppb.Now()
	var responses []*pb.BatchGetDocumentsResponse
	for _, docName := range req.GetDocuments() {
		if txID != "" {
			if reads, ok := f.txReads[txID]; ok {
				if _, seen := reads[docName]; !seen {
					reads[docName] = f.docVersions[docName]
				}
			}
		}
		if doc, ok := f.docs[docName]; ok {
			responses = append(responses, &pb.BatchGetDocumentsResponse{
				Result:   &pb.BatchGetDocumentsResponse_Found{Found: proto.Clone(doc).(*pb.Document)},
				ReadTime: now,
			})
		} else {
			responses = append(responses, &pb.BatchGetDocumentsResponse{
				Result:   &pb.BatchGetDocumentsResponse_Missing{Missing: docName},
				ReadTime: now,
			})
		}
	}
	f.mu.Unlock()

	for _, resp := range responses {
		if err := stream.Send(resp); err != nil {
			return err
		}
	}
	return nil
}

func (f *fakeFirestoreServer) Commit(_ context.Context, req *pb.CommitRequest) (*pb.CommitResponse, error) {
	txID := string(req.GetTransaction())
	if txID != "" {
		f.mu.Lock()
		hook := f.onBeforeCommit
		f.mu.Unlock()
		if hook != nil {
			hook(txID)
		}
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	if f.failNextWrite != nil {
		err := f.failNextWrite
		f.failNextWrite = nil
		return nil, err
	}

	if txID != "" {
		if reads, ok := f.txReads[txID]; ok {
			for docName, readVer := range reads {
				if f.docVersions[docName] != readVer {
					delete(f.txReads, txID)
					f.txAborts++
					return nil, status.Errorf(codes.Aborted, "transaction %s aborted: concurrent modification of %s", txID, docName)
				}
			}
			delete(f.txReads, txID)
		}
	} else if len(req.GetWrites()) > 0 {
		f.batchCommits++
	}

	// A commit is atomic: check every write before applying any, so a
	// rejected write leaves the earlier writes of the same commit unapplied.
	for _, w := range req.GetWrites() {
		if err := f.validateWrite(w); err != nil {
			return nil, err
		}
	}

	now := timestamppb.Now()
	results := make([]*pb.WriteResult, 0, len(req.GetWrites()))

	for _, w := range req.GetWrites() {
		switch op := w.GetOperation().(type) {
		case *pb.Write_Update:
			incoming := proto.Clone(op.Update).(*pb.Document)
			docName := incoming.GetName()
			existing := f.docs[docName]

			var target *pb.Document
			if mask := w.GetUpdateMask(); mask != nil && existing != nil {
				target = proto.Clone(existing).(*pb.Document)
				if target.Fields == nil {
					target.Fields = make(map[string]*pb.Value)
				}
				for _, path := range mask.GetFieldPaths() {
					if val, ok := incoming.GetFields()[path]; ok {
						target.Fields[path] = val
					} else {
						delete(target.Fields, path)
					}
				}
				target.UpdateTime = now
			} else {
				target = incoming
				if target.Fields == nil {
					target.Fields = make(map[string]*pb.Value)
				}
				if existing != nil {
					target.CreateTime = existing.CreateTime
				} else {
					target.CreateTime = now
				}
				target.UpdateTime = now
			}

			for _, ft := range w.GetUpdateTransforms() {
				if ft.GetSetToServerValue() == pb.DocumentTransform_FieldTransform_REQUEST_TIME {
					target.Fields[ft.GetFieldPath()] = &pb.Value{
						ValueType: &pb.Value_TimestampValue{TimestampValue: now},
					}
				}
			}
			truncateTimestampsInDoc(target)
			f.docs[docName] = target
			f.docVersions[docName]++

		case *pb.Write_Delete:
			delete(f.docs, op.Delete)
			f.docVersions[op.Delete]++

		case *pb.Write_Transform:
			docName := op.Transform.GetDocument()
			target, ok := f.docs[docName]
			if !ok {
				target = &pb.Document{
					Name:       docName,
					Fields:     make(map[string]*pb.Value),
					CreateTime: now,
				}
				f.docs[docName] = target
			}
			for _, ft := range op.Transform.GetFieldTransforms() {
				if ft.GetSetToServerValue() == pb.DocumentTransform_FieldTransform_REQUEST_TIME {
					target.Fields[ft.GetFieldPath()] = &pb.Value{
						ValueType: &pb.Value_TimestampValue{TimestampValue: now},
					}
				}
			}
			target.UpdateTime = now
			f.docVersions[docName]++
		}

		results = append(results, &pb.WriteResult{UpdateTime: now})
	}

	return &pb.CommitResponse{
		WriteResults: results,
		CommitTime:   now,
	}, nil
}

func (f *fakeFirestoreServer) BatchWrite(ctx context.Context, req *pb.BatchWriteRequest) (*pb.BatchWriteResponse, error) {
	f.mu.Lock()
	failErr := f.failBatchWrites
	f.mu.Unlock()
	if failErr != nil {
		// BatchWrite reports each write's outcome in the response.
		resp := &pb.BatchWriteResponse{}
		for range req.GetWrites() {
			resp.WriteResults = append(resp.WriteResults, &pb.WriteResult{})
			resp.Status = append(resp.Status, status.Convert(failErr).Proto())
		}
		return resp, nil
	}
	commitResp, err := f.Commit(ctx, &pb.CommitRequest{
		Database: req.GetDatabase(),
		Writes:   req.GetWrites(),
	})
	if err != nil {
		return nil, err
	}
	resp := &pb.BatchWriteResponse{
		WriteResults: commitResp.GetWriteResults(),
	}
	for range commitResp.GetWriteResults() {
		resp.Status = append(resp.Status, status.New(codes.OK, "").Proto())
	}
	return resp, nil
}

func (f *fakeFirestoreServer) RunQuery(req *pb.RunQueryRequest, stream pb.Firestore_RunQueryServer) error {
	f.mu.Lock()
	if f.failNextQuery != nil {
		err := f.failNextQuery
		f.failNextQuery = nil
		f.mu.Unlock()
		return err
	}

	sq := req.GetStructuredQuery()
	parent := req.GetParent()
	var collectionID string
	var allDescendants bool
	if len(sq.GetFrom()) > 0 {
		collectionID = sq.GetFrom()[0].GetCollectionId()
		allDescendants = sq.GetFrom()[0].GetAllDescendants()
	}

	var matched []*pb.Document
	for name, doc := range f.docs {
		if !matchesCollection(name, parent, collectionID, allDescendants) {
			continue
		}
		if sq.GetWhere() != nil && !matchesFilter(doc, sq.GetWhere()) {
			continue
		}
		matched = append(matched, proto.Clone(doc).(*pb.Document))
	}
	f.mu.Unlock()

	sortDocuments(matched, sq.GetOrderBy())

	if lim := sq.GetLimit(); lim != nil {
		n := int(lim.GetValue())
		if n >= 0 && n < len(matched) {
			matched = matched[:n]
		}
	}

	now := timestamppb.Now()
	for _, doc := range matched {
		if err := stream.Send(&pb.RunQueryResponse{
			Document: doc,
			ReadTime: now,
		}); err != nil {
			return err
		}
	}
	return nil
}

func matchesCollection(docName, parent, collectionID string, allDescendants bool) bool {
	if !strings.HasPrefix(docName, parent+"/") {
		return false
	}
	rel := strings.TrimPrefix(docName, parent+"/")
	segments := strings.Split(rel, "/")
	if len(segments)%2 != 0 {
		return false
	}
	lastCollection := segments[len(segments)-2]
	if lastCollection != collectionID {
		return false
	}
	if !allDescendants && len(segments) != 2 {
		return false
	}
	return true
}

func matchesFilter(doc *pb.Document, filter *pb.StructuredQuery_Filter) bool {
	switch flt := filter.GetFilterType().(type) {
	case *pb.StructuredQuery_Filter_FieldFilter:
		ff := flt.FieldFilter
		fieldPath := ff.GetField().GetFieldPath()
		val, ok := doc.GetFields()[fieldPath]
		if !ok {
			return false
		}
		c := compareValues(val, ff.GetValue())
		switch ff.GetOp() {
		case pb.StructuredQuery_FieldFilter_EQUAL:
			return c == 0
		case pb.StructuredQuery_FieldFilter_GREATER_THAN_OR_EQUAL:
			return c >= 0
		case pb.StructuredQuery_FieldFilter_GREATER_THAN:
			return c > 0
		case pb.StructuredQuery_FieldFilter_LESS_THAN_OR_EQUAL:
			return c <= 0
		case pb.StructuredQuery_FieldFilter_LESS_THAN:
			return c < 0
		}
	case *pb.StructuredQuery_Filter_CompositeFilter:
		for _, sub := range flt.CompositeFilter.GetFilters() {
			if !matchesFilter(doc, sub) {
				return false
			}
		}
		return true
	}
	return true
}

func sortDocuments(docs []*pb.Document, orders []*pb.StructuredQuery_Order) {
	slices.SortFunc(docs, func(a, b *pb.Document) int {
		lastDir := pb.StructuredQuery_ASCENDING
		for _, ord := range orders {
			field := ord.GetField().GetFieldPath()
			lastDir = ord.GetDirection()
			var c int
			if field == "__name__" {
				c = cmp.Compare(a.GetName(), b.GetName())
			} else {
				c = compareValues(a.GetFields()[field], b.GetFields()[field])
			}
			if c != 0 {
				if lastDir == pb.StructuredQuery_DESCENDING {
					return -c
				}
				return c
			}
		}
		c := cmp.Compare(a.GetName(), b.GetName())
		if lastDir == pb.StructuredQuery_DESCENDING {
			return -c
		}
		return c
	})
}

func compareValues(a, b *pb.Value) int {
	if a == nil && b == nil {
		return 0
	}
	if a == nil {
		return -1
	}
	if b == nil {
		return 1
	}
	switch av := a.GetValueType().(type) {
	case *pb.Value_StringValue:
		if bv, ok := b.GetValueType().(*pb.Value_StringValue); ok {
			return cmp.Compare(av.StringValue, bv.StringValue)
		}
	case *pb.Value_IntegerValue:
		if bv, ok := b.GetValueType().(*pb.Value_IntegerValue); ok {
			return cmp.Compare(av.IntegerValue, bv.IntegerValue)
		}
	case *pb.Value_DoubleValue:
		if bv, ok := b.GetValueType().(*pb.Value_DoubleValue); ok {
			return cmp.Compare(av.DoubleValue, bv.DoubleValue)
		}
	case *pb.Value_TimestampValue:
		if bv, ok := b.GetValueType().(*pb.Value_TimestampValue); ok {
			at := av.TimestampValue.AsTime()
			bt := bv.TimestampValue.AsTime()
			return at.Compare(bt)
		}
	case *pb.Value_BytesValue:
		if bv, ok := b.GetValueType().(*pb.Value_BytesValue); ok {
			return bytes.Compare(av.BytesValue, bv.BytesValue)
		}
	}
	return 0
}

// validateWrite reports the error Firestore returns for w against the current
// documents. The caller holds f.mu.
func (f *fakeFirestoreServer) validateWrite(w *pb.Write) error {
	if pre := w.GetCurrentDocument(); pre != nil {
		if cond, ok := pre.GetConditionType().(*pb.Precondition_Exists); ok {
			var docName string
			if u := w.GetUpdate(); u != nil {
				docName = u.GetName()
			} else if d := w.GetDelete(); d != "" {
				docName = d
			}
			_, exists := f.docs[docName]
			if cond.Exists && !exists {
				return status.Errorf(codes.NotFound, "document %s does not exist", docName)
			}
			if !cond.Exists && exists {
				return status.Errorf(codes.AlreadyExists, "document %s already exists", docName)
			}
		}
	}
	if u := w.GetUpdate(); u != nil {
		return validateFields(u.GetFields())
	}
	return nil
}

// validateFields reports the error Firestore returns for a document it does
// not store: an array directly inside an array, a field name that is empty or
// reserved (__ at both ends of a longer name), a value nested more than 20
// levels deep, or more scalar values than its index entry limit allows. The
// limits were measured on a Firestore database, except for the array inside
// an array, which the Firestore emulator rejects and some databases accept.
func validateFields(fields map[string]*pb.Value) error {
	if err := validateMapFields(fields, 1); err != nil {
		return err
	}
	// Firestore indexes each scalar value twice and allows 40,000 entries.
	if n := countScalarValues(&pb.Value{ValueType: &pb.Value_MapValue{MapValue: &pb.MapValue{Fields: fields}}}); n > 20000 {
		return status.Errorf(codes.InvalidArgument, "too many index entries for entity")
	}
	return nil
}

func countScalarValues(val *pb.Value) int {
	n := 0
	switch v := val.GetValueType().(type) {
	case *pb.Value_ArrayValue:
		for _, item := range v.ArrayValue.GetValues() {
			n += countScalarValues(item)
		}
	case *pb.Value_MapValue:
		for _, item := range v.MapValue.GetFields() {
			n += countScalarValues(item)
		}
	default:
		n = 1
	}
	return n
}

func validateMapFields(fields map[string]*pb.Value, depth int) error {
	for name, val := range fields {
		if name == "" {
			return status.Errorf(codes.InvalidArgument, "document contains an empty field name")
		}
		if len(name) > 4 && strings.HasPrefix(name, "__") && strings.HasSuffix(name, "__") {
			return status.Errorf(codes.InvalidArgument, "field name %q is reserved", name)
		}
		if err := validateValue(val, depth, false); err != nil {
			return err
		}
	}
	return nil
}

func validateValue(val *pb.Value, depth int, inArray bool) error {
	switch v := val.GetValueType().(type) {
	case *pb.Value_ArrayValue:
		if inArray {
			return status.Errorf(codes.InvalidArgument, "nested arrays are not supported in Firestore")
		}
		if depth > 20 {
			return status.Errorf(codes.InvalidArgument, "value is nested too deeply")
		}
		for _, item := range v.ArrayValue.GetValues() {
			if err := validateValue(item, depth+1, true); err != nil {
				return err
			}
		}
	case *pb.Value_MapValue:
		if depth > 20 {
			return status.Errorf(codes.InvalidArgument, "value is nested too deeply")
		}
		return validateMapFields(v.MapValue.GetFields(), depth+1)
	}
	return nil
}
