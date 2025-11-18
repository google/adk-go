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

package util

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
)

// Printer is a function printing its arguments
type Printer func(a ...any)

var (
	Reset   = "\033[0m"
	Red     = "\033[31m"
	Green   = "\033[32m"
	Yellow  = "\033[33m"
	Blue    = "\033[34m"
	Magenta = "\033[35m"
	Cyan    = "\033[36m"
	Gray    = "\033[37m"
	White   = "\033[97m"
)

// LogStartStop is a helper function which executes a particular command with logging
func LogStartStop(msg string, command func(p Printer) error) error {
	fmt.Println(msg, ": "+Green+"Starting"+Reset)
	err := command(func(a ...any) { fmt.Println("    "+Green+"> "+Reset, a) })
	fmt.Println()
	if err == nil {
		fmt.Println(msg, ": "+Green+"Finished successfully"+Reset)
	} else {
		fmt.Println(msg, ": "+Red+"Finished with error"+Reset)
		fmt.Println("Error:", err)
	}

	return err
}

type reprintableStream struct {
	prefix []byte
	clean  bool
	stream io.Writer
}

// function Write is an interceptor of a stream adding some decorations
func (s *reprintableStream) Write(p []byte) (total int, err error) {
	start := 0
	err = nil
	if s.clean {
		_, err = s.stream.Write(s.prefix)
		if err != nil {
			return total, err
		}
		s.clean = false
	}
	for i, c := range p {
		if c == '\n' {
			_, err = s.stream.Write(p[start:i])
			if err != nil {
				return len(p), err
			}
			_, err = s.stream.Write(s.prefix)
			if err != nil {
				return len(p), err
			}
			start = i + 1
		}
	}
	if start < len(p) {
		_, err = s.stream.Write(p[start:])
	}

	return len(p), err
}

func newReprintableStream(s io.Writer, prefix, color string) io.Writer {
	return &reprintableStream{prefix: []byte("\n       " + color + prefix + " > " + Reset), stream: s, clean: true}
}

// function LogCommand runs a command pretty-printing its stdout and stderr
func LogCommand(c *exec.Cmd, p Printer) error {
	p("Running : ", Yellow, c.Dir, Reset, " ", c)
	c.Stdout = newReprintableStream(os.Stdout, "  ", Yellow)
	c.Stderr = newReprintableStream(os.Stdout, "  ", Yellow)
	return c.Run()
}

func StripExtension(p, expected string) (string, error) {
	ex := path.Ext(p)
	if ex == "" {
		return "", errors.New("Cannot find extension in '" + p + "'")
	}
	if ex != expected {
		return "", errors.New("Unexpected extension. Found '" + ex + "' instead of '" + expected + "'")
	}
	return p[:len(p)-len(ex)], nil
}
