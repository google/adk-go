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

// package webui handles command line parameters and execution logic for build webui
package webui

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/spf13/cobra"
	"google.golang.org/adk/cmd/adkgo/root/build"
	"google.golang.org/adk/internal/cli/util"
)

type buildFlags struct {
	targetDir string // command line param
}

type sourceFlags struct {
	webuiDir string // command line param
}

type runLocalFlags struct {
	build  buildFlags
	source sourceFlags
}

var flags runLocalFlags

// webuiCmd represents the build webui command
var webuiCmd = &cobra.Command{
	Use:   "webui",
	Short: "Build static ADK Web UI from sources.",
	Long: `
	Builds static ADK Web UI files from sources. 
	WARNINIG: deletes the whole build directory and recreates it anew!
	You need: 
	  - a downloaded version of adk-web (available at https://github.com/google/adk-web)
	  - an ability to build adk-web (prerequisites on https://github.com/google/adk-web):
	  	npm (node js: see https://nodejs.org/en/download)
		ng (angular cli: see https://angular.dev/tools/cli/setup-local)		
	  - go

	Building the adk-web takes a while, and sometimes presents some warnings.
	`,
	RunE: func(cmd *cobra.Command, args []string) error {
		err := flags.buildWebui()
		return err
	},
}

func init() {
	build.BuildCmd.AddCommand(webuiCmd)

	webuiCmd.PersistentFlags().StringVarP(&flags.build.targetDir, "targetDir", "t", "", "Target directory for build output")
	webuiCmd.PersistentFlags().StringVarP(&flags.source.webuiDir, "sourceDir", "s", "", "Directory containing ADK Web UI (from https://github.com/google/adk-web)")
}

func (f *runLocalFlags) cleanTemp() error {
	return util.LogStartStop("Cleaning target directory",
		func(p util.Printer) error {
			p("Clean target directory starting with", f.build.targetDir)
			err := os.RemoveAll(f.build.targetDir)
			if err != nil {
				return fmt.Errorf("failed to clean target directory %v: %w", f.build.targetDir, err)
			}
			err = os.MkdirAll(f.build.targetDir, os.ModeDir|0700)
			if err != nil {
				return fmt.Errorf("failed to create the target directory %v: %w", f.build.targetDir, err)
			}
			return nil
		})
}

func (f *runLocalFlags) ngBuildADKWebUI() error {
	return util.LogStartStop("Building ADK Web UI",
		func(p util.Printer) error {
			cmd := exec.Command("ng", "build", "--output-path="+f.build.targetDir)
			cmd.Dir = f.source.webuiDir
			return util.LogCommand(cmd, p)
		})
}

func (f *runLocalFlags) buildWebui() error {
	err := f.cleanTemp()
	if err != nil {
		return err
	}
	return f.ngBuildADKWebUI()
}
