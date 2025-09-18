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

package cloudrun

import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"strconv"

	"github.com/spf13/cobra"
	"google.golang.org/adk/cmd/cli/root/deploy"
	"google.golang.org/adk/cmd/util"
)

type gCloudFlags struct {
	region      string
	projectName string
}

type cloudRunServiceFlags struct {
	serviceName string
}

type localProxyFlags struct {
	port int
}

type buildFlags struct {
	tempDir             string
	uiBuildDir          string
	uiDistDir           string
	execPath            string
	dockerfileBuildPath string
}

type sourceFlags struct {
	uiDir          string
	srcBasePath    string
	entryPointPath string
}

type webUIDeployFlags struct {
	backendUri string
}

type deployCloudRunFlags struct {
	gcloud   gCloudFlags
	cloudRun cloudRunServiceFlags
	proxy    localProxyFlags
	build    buildFlags
	source   sourceFlags
	webUI    webUIDeployFlags
}

var Flags deployCloudRunFlags

// cloudrunCmd represents the cloudrun command
var cloudrunCmd = &cobra.Command{
	Use:   "cloudrun",
	Short: "Deploys the application to cloudrun.",
	Long: `Deployment prepares a Dockerfile which is fed with locally compiled server executable and Web UI static files.
	Service on Cloud run is created using this information. 
	Local proxy adding authentication is optionally started. 
	`,
	RunE: func(cmd *cobra.Command, args []string) error {
		err := Flags.deployOnCloudRun()
		return err
	},
}

func init() {
	deploy.DeployCmd.AddCommand(cloudrunCmd)

	cloudrunCmd.PersistentFlags().StringVarP(&Flags.gcloud.region, "region", "r", "", "GCP Region")
	cloudrunCmd.PersistentFlags().StringVarP(&Flags.gcloud.projectName, "projectName", "p", "", "GCP Project Name")
	cloudrunCmd.PersistentFlags().StringVarP(&Flags.cloudRun.serviceName, "serviceName", "s", "", "Cloud Run Service name")
	cloudrunCmd.PersistentFlags().StringVarP(&Flags.build.tempDir, "tempDir", "t", "", "Temp dir for build")
	cloudrunCmd.PersistentFlags().IntVar(&Flags.proxy.port, "proxyPort", 8081, "Local proxy port")
	cloudrunCmd.PersistentFlags().StringVarP(&Flags.source.uiDir, "webUIDir", "a", "", "ADK Web UI base dir")
	cloudrunCmd.PersistentFlags().StringVarP(&Flags.webUI.backendUri, "backendUri", "b", "", "ADK REST API uri")
	cloudrunCmd.PersistentFlags().StringVarP(&Flags.source.entryPointPath, "entryPoint", "e", "", "Path to an entry point (go 'main')")
	cloudrunCmd.PersistentFlags().StringVarP(&Flags.source.srcBasePath, "srcPath", "", "", "Path to an entry point (go 'main')")

}

func (f *deployCloudRunFlags) computeFlags() error {
	fmt.Println("Compute flags starting")
	f.build.uiBuildDir = path.Join(f.build.tempDir, "ui")
	f.build.uiDistDir = path.Join(f.source.uiDir, "/dist/agent_framework_web/browser")
	f.build.execPath = path.Join(f.build.tempDir, "server")
	f.build.dockerfileBuildPath = path.Join(f.build.tempDir, "Dockerfile")

	fmt.Println("Compute flags finished")
	return nil
}

func (f *deployCloudRunFlags) makeDirs() error {
	fmt.Println("Make dirs starting")
	fmt.Println("  making", f.build.uiBuildDir)
	err := os.MkdirAll(f.build.uiBuildDir, os.ModeDir|0700)
	if err != nil {
		return err
	}
	fmt.Println("Make dirs finished")
	return nil
}

func (f *deployCloudRunFlags) cleanTemp() error {
	files := path.Join(f.build.tempDir, "*")
	fmt.Println("Clean temp starting for ", files)
	// fmt.Println(files)
	err := os.RemoveAll(files)
	if err != nil {
		return err
	}
	fmt.Println("Clean temp finished")
	return nil
}

func (f *deployCloudRunFlags) setBackendForAdkWebUI() error {
	// wd, _ := os.Getwd()
	fmt.Println("Setting backend for Adk Web UI starting")
	cmd := exec.Command("npm", "run", "inject-backend", "--backend="+f.webUI.backendUri)

	cmd.Dir = f.source.uiDir
	// fmt.Println("  Build ADK Web UI dist from Dir: ", cmd.Dir, "Cmd: ", cmd)

	err := cmd.Run()
	fmt.Println("Setting backend for Adk Web UI finished")
	return err
}

func (f *deployCloudRunFlags) makeDistForAdkWebUI() error {
	// wd, _ := os.Getwd()
	fmt.Println("Making dist for Adk Web UI starting")
	cmd := exec.Command("ng", "build", "--output-path="+f.build.uiBuildDir)

	cmd.Dir = f.source.uiDir

	// fmt.Println("  Build ADK Web UI dist from Dir: ", cmd.Dir, "Cmd: ", cmd)

	// err := cmd.Run()
	var err error = nil
	fmt.Println("Making dist for Adk Web UI finished")
	return err
}

func (f *deployCloudRunFlags) compileEntryPoint() error {
	// wd, _ := os.Getwd()
	fmt.Println("Compiling entry point starting: " + f.source.entryPointPath)
	cmd := exec.Command("go", "build", "-o", f.build.execPath, f.source.entryPointPath)

	cmd.Dir = f.source.srcBasePath
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0", "GOOS=linux")
	// cmd.Stdout = os.Stdout
	// cmd.Stderr = os.Stderr

	fmt.Println("  Build Dir: ", cmd.Dir, "Cmd: ", cmd) //, "Env: ", cmd.Env)

	err := cmd.Run()
	fmt.Println("Compiling entry point finished: ")
	return err
}

func (f *deployCloudRunFlags) prepareDockerfile() error {
	// wd, _ := os.Getwd()
	fmt.Println("Preparing Dockerfile starting: ")

	c := `
FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY server  /app/server
COPY ui  /app/ui

FROM gcr.io/distroless/static-debian11

# Set the working directory
WORKDIR /app

# Copy the built executable from the builder stage
COPY --from=builder /app/server /app/server
COPY --from=builder /app/ui /app/ui

EXPOSE 8080

# Command to run the executable when the container starts
CMD ["/app/server", "--port", "8080", "--front_address", "` + f.webUI.backendUri + `"]
`
	// _ = c
	err := os.WriteFile(f.build.dockerfileBuildPath, []byte(c), 0600)

	// cmd := exec.Command("go", "build", "-o", f.execPath, f.entryPointPath)

	// cmd.Dir = f.srcBasePath
	// fmt.Println("  Build Dir: ", cmd.Dir, "Cmd: ", cmd)

	// err := cmd.Run()
	fmt.Println("Preparing Dockerfile finished")
	return err
}

func (f *deployCloudRunFlags) gcloudDeployToCloudRun() error {
	// wd, _ := os.Getwd()
	fmt.Println("Deploy to cloud run starting: " + f.source.entryPointPath)
	cmd := exec.Command("gcloud", "run", "deploy", f.cloudRun.serviceName,
		"--source", ".",
		"--set-secrets=GOOGLE_API_KEY=ADK_KEY:latest",
		"--region", f.gcloud.region,
		"--project", f.gcloud.projectName)

	cmd.Dir = f.build.tempDir
	// cmd.Stdout = os.Stdout
	// cmd.Stderr = os.Stderr

	fmt.Println("  Deploy: ", cmd.Dir, "Cmd: ", cmd)

	err := cmd.Run()
	fmt.Println("Deploy to cloud run finished")
	return err
}

func (f *deployCloudRunFlags) runGcloudProxy() error {
	// wd, _ := os.Getwd()
	fmt.Println("Running gcloud proxy starting: " + f.source.entryPointPath)
	cmd := exec.Command("gcloud", "run", "services", "proxy", f.cloudRun.serviceName, "--project", f.gcloud.projectName, "--port", strconv.Itoa(f.proxy.port))

	cmd.Dir = f.build.tempDir
	// cmd.Stdout = os.Stdout
	// cmd.Stderr = os.Stderr

	fmt.Println("  Run proxy: ", cmd.Dir, "Cmd: ", cmd)

	err := cmd.Run()
	fmt.Println("Running gcloud proxy finished")
	return err
}

func (f *deployCloudRunFlags) xxx() error {
	// wd, _ := os.Getwd()

	err := util.LogStartStop("Deploy to cloud run",
		func(p util.Printer) error {
			cmd := exec.Command("find", "/usr/local/google/home/kdroste/Projects/adk-go/adk-go-cli/adk-go/cmd", "/asdfasdf")

			// cmd := exec.Command("echo", "gcloud", "run", "deploy", f.cloudRun.serviceName,
			// 	"--source", ".",
			// 	"--set-secrets=GOOGLE_API_KEY=ADK_KEY:latest",
			// 	"--region", f.gcloud.region,
			// 	"--project", f.gcloud.projectName)
			cmd.Dir = f.build.tempDir
			//cmd.Stdout = os.Stdout
			// cmd.Stderr = os.Stderr
			p("  Deploy: ", cmd.Dir, "Cmd: ", cmd)

			err := util.LogCommand(cmd, p)
			// err := cmd.Run()
			return err
		})
	return err
}

func (f *deployCloudRunFlags) deployOnCloudRun() error {
	fmt.Println(Flags)
	var err error
	// err := f.xxx()
	// if err != nil {
	// 	return err
	// }

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
	err = f.prepareDockerfile()
	if err != nil {
		return err
	}
	err = f.gcloudDeployToCloudRun()
	if err != nil {
		return err
	}
	err = f.runGcloudProxy()
	if err != nil {
		return err
	}

	return nil
}
