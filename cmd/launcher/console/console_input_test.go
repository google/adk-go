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

package console

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/ergochat/readline"
)

func TestReadConsoleLineScripted(t *testing.T) {
	reader := bufio.NewReader(strings.NewReader("你好世\nsecond turn\n"))
	var output bytes.Buffer

	for _, want := range []string{"你好世\n", "second turn\n"} {
		got, err := readConsoleLine(context.Background(), reader, &output, false, "\nUser -> ")
		if err != nil {
			t.Fatalf("readConsoleLine() error = %v", err)
		}
		if got != want {
			t.Errorf("readConsoleLine() = %q, want %q", got, want)
		}
	}
	if got, want := output.String(), "\nUser -> \nUser -> "; got != want {
		t.Errorf("prompts = %q, want %q", got, want)
	}
}

func TestReadConsoleLineCanceled(t *testing.T) {
	input, writer := io.Pipe()
	defer input.Close()
	defer writer.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := readConsoleLine(ctx, bufio.NewReader(input), io.Discard, false, "User -> ")
	if !errors.Is(err, context.Canceled) {
		t.Errorf("readConsoleLine() error = %v, want context.Canceled", err)
	}
}

func TestInputInterruptIsNormalExit(t *testing.T) {
	if !isInputInterrupt(fmt.Errorf("read input: %w", readline.ErrInterrupt)) {
		t.Fatal("wrapped readline.ErrInterrupt should end the console normally")
	}
	if isInputInterrupt(io.EOF) {
		t.Fatal("io.EOF should follow the EOF path")
	}
}
