// Copyright 2025 Google LLC
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

package main

// ListUnlabeledIssuesArgs represents the arguments for list_unlabeled_issues.
type ListUnlabeledIssuesArgs struct {
	IssueCount int `json:"issue_count"`
}

// ListUnlabeledIssuesResult represents the result of list_unlabeled_issues.
type ListUnlabeledIssuesResult struct {
	Status  string                   `json:"status"`
	Issues  []map[string]interface{} `json:"issues,omitempty"`
	Message string                   `json:"message,omitempty"`
}

// AddLabelAndOwnerArgs represents the arguments for add_label_and_owner_to_issue.
type AddLabelAndOwnerArgs struct {
	IssueNumber int    `json:"issue_number"`
	Label       string `json:"label"`
}

// AddLabelAndOwnerResult represents the result of add_label_and_owner_to_issue.
type AddLabelAndOwnerResult struct {
	Status        string `json:"status"`
	Message       string `json:"message,omitempty"`
	AppliedLabel  string `json:"applied_label,omitempty"`
	AssignedOwner string `json:"assigned_owner,omitempty"`
}

// ChangeIssueTypeArgs represents the arguments for change_issue_type.
type ChangeIssueTypeArgs struct {
	IssueNumber int    `json:"issue_number"`
	IssueType   string `json:"issue_type"`
}

// ChangeIssueTypeResult represents the result of change_issue_type.
type ChangeIssueTypeResult struct {
	Status    string `json:"status"`
	Message   string `json:"message,omitempty"`
	IssueType string `json:"issue_type,omitempty"`
}
