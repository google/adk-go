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

// Package run handles command line parameters for "run"
package run

import (
	"github.com/spf13/cobra"
	"google.golang.org/adk/cmd/adkgo/root"
)

// deployCmd represents the deploy command
var RunCmd = &cobra.Command{
	Use:   "run",
	Short: "Runs components",
	Long:  `Please see subcommands for details`,
	RunE: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 {
			return cmd.Help()
		}
		return nil
	},
}

func init() {
	root.RootCmd.AddCommand(RunCmd)
}
