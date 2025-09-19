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

package local

import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"strconv"

	"github.com/spf13/cobra"
	"google.golang.org/adk/cmd/cli/root/run"
	"google.golang.org/adk/cmd/util"
)

type localServerFlags struct {
	port int
}

type buildFlags struct {
	skipCleaning bool
	tempDir      string
	uiBuildDir   string
	// uiDistDir    string
	execPath string
}

type sourceFlags struct {
	uiDir          string
	srcBasePath    string
	entryPointPath string
}

type webUIDeployFlags struct {
	backendUri string
}

type runLocalFlags struct {
	server localServerFlags
	build  buildFlags
	source sourceFlags
	webUI  webUIDeployFlags
}

var Flags runLocalFlags

// localCmd represents the cloudrun command
var localCmd = &cobra.Command{
	Use:   "local",
	Short: "Runs a local server with WebUI.",
	Long: `
	`,
	RunE: func(cmd *cobra.Command, args []string) error {
		err := Flags.runLocal()
		return err
	},
}

func init() {
	run.RunCmd.AddCommand(localCmd)

	localCmd.PersistentFlags().StringVarP(&Flags.build.tempDir, "tempDir", "t", "", "Temp dir for build")
	localCmd.PersistentFlags().BoolVarP(&Flags.build.skipCleaning, "skipCleaning", "c", true, "Set to true in order to clean build")
	localCmd.PersistentFlags().IntVarP(&Flags.server.port, "serverPort", "s", 8080, "Local proxy port")
	localCmd.PersistentFlags().StringVarP(&Flags.source.uiDir, "webUIDir", "u", "", "ADK Web UI base dir")
	localCmd.PersistentFlags().StringVarP(&Flags.webUI.backendUri, "backendUri", "b", "", "ADK REST API uri")
	localCmd.PersistentFlags().StringVarP(&Flags.source.entryPointPath, "entryPoint", "e", "", "Path to an entry point (go 'main')")
	localCmd.PersistentFlags().StringVarP(&Flags.source.srcBasePath, "srcPath", "p", "", "Path to an entry point (go 'main')")

}

func (f *runLocalFlags) computeFlags() error {
	fmt.Println("Compute flags starting")
	f.build.uiBuildDir = path.Join(f.build.tempDir, "ui")
	// f.build.uiDistDir = path.Join(f.source.uiDir, "/dist/agent_framework_web/browser")
	f.build.execPath = path.Join(f.build.tempDir, "server")

	fmt.Println("Compute flags finished")
	return nil
}

func (f *runLocalFlags) cleanTemp() error {
	err := util.LogStartStop("Cleaning temp",
		func(p util.Printer) error {
			if f.build.skipCleaning {
				p("Cleaning skipped", f.build.tempDir)
				return nil
			}
			p("Clean temp starting with", f.build.tempDir)
			err := os.RemoveAll(f.build.tempDir)
			if err != nil {
				return err
			}
			err = os.MkdirAll(f.build.tempDir, os.ModeDir|0700)
			return err
		})
	return err
}

func (f *runLocalFlags) makeDirs() error {
	err := util.LogStartStop("Make build dirs",
		func(p util.Printer) error {
			p("Making", f.build.uiBuildDir)
			err := os.MkdirAll(f.build.uiBuildDir, os.ModeDir|0700)
			if err != nil {
				return err
			}
			return nil
		})
	return err
}

func (f *runLocalFlags) setBackendForAdkWebUI() error {

	err := util.LogStartStop("Setting backend for Adk Web UI",
		func(p util.Printer) error {
			cmd := exec.Command("npm", "run", "inject-backend", "--backend="+f.webUI.backendUri)
			cmd.Dir = f.source.uiDir
			err := util.LogCommand(cmd, p)
			return err
		})
	return err
}

func (f *runLocalFlags) makeDistForAdkWebUI() error {
	err := util.LogStartStop("Making dist for Adk Web UI",
		func(p util.Printer) error {

			fileToCheck := path.Join(f.build.uiBuildDir, "browser/index.html")
			_, err := os.Stat(fileToCheck)
			if err != nil {
				p("File", fileToCheck, "not found, building from scratch")
				cmd := exec.Command("ng", "build", "--output-path="+f.build.uiBuildDir)
				cmd.Dir = f.source.uiDir
				err = util.LogCommand(cmd, p)
			} else {
				p("File", fileToCheck, "found, building skipped")
			}
			return err
		})
	return err
}

func (f *runLocalFlags) compileEntryPoint() error {
	err := util.LogStartStop("Compiling server",
		func(p util.Printer) error {
			p("Using", f.source.entryPointPath, "as entry point")
			cmd := exec.Command("go", "build", "-o", f.build.execPath, f.source.entryPointPath)

			cmd.Dir = f.source.srcBasePath
			cmd.Env = append(os.Environ(), "CGO_ENABLED=0", "GOOS=linux")
			err := util.LogCommand(cmd, p)
			return err
		})
	return err
}

func (f *runLocalFlags) runLocalServer() error {
	err := util.LogStartStop("Running local server",
		func(p util.Printer) error {
			cmd := exec.Command(f.build.execPath,
				"--port", strconv.Itoa(f.server.port),
				"--front_address", "localhost:"+strconv.Itoa(f.server.port),
				"--start_restapi=true",
				"--start_webui=true",
				"--webui_path", "./ui/browser/",
			)

			cmd.Dir = f.build.tempDir

			p("--------------------------------------------------------------------------")
			p("    Running ADK Web UI on http://localhost:" + strconv.Itoa(f.server.port) + "/ui/                       ")
			p("          ADK REST API on http://localhost:" + strconv.Itoa(f.server.port) + "/api/                      ")
			p("                                                                          ")
			p("                          Press Ctrl-C to stop                            ")
			p("--------------------------------------------------------------------------")
			err := util.LogCommand(cmd, p)
			return err
		})
	return err
}

func (f *runLocalFlags) runLocal() error {
	fmt.Println(Flags)
	var err error

	err = f.computeFlags()
	if err != nil {
		return err
	}
	err = f.cleanTemp()
	if err != nil {
		return err
	}
	err = f.makeDirs()
	if err != nil {
		return err
	}
	err = f.setBackendForAdkWebUI()
	if err != nil {
		return err
	}
	err = f.makeDistForAdkWebUI()
	if err != nil {
		return err
	}
	err = f.compileEntryPoint()
	if err != nil {
		return err
	}
	err = f.runLocalServer()
	if err != nil {
		return err
	}

	return nil
}
