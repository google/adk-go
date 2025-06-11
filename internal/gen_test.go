package internal

//go:generate go test github.com/google/adk-go/internal -update

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var update = flag.Bool("update", false, "Update the JSON schema and util directories from golang.org/x/tools@master")

// TestUpdateJSONSchema updates the JSON schema and util directories
// from the latest version of golang.org/x/tools@master.
func TestUpdateJSONSchema(t *testing.T) {
	if !*update {
		t.Skip("Skipping update; use -update flag to run this test")
	}

	// Task 1: Check if we are in the "internal" directory.
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get current working directory: %v", err)
	}

	if filepath.Base(cwd) != "internal" {
		log.Fatalf("This program must be run from the 'internal' directory.")
	}

	fmt.Println("Successfully verified current directory is 'internal'.")

	// Task 2: Delete existing jsonschema and util directories.
	packagesToUpdate := []string{"jsonschema", "util"}
	for _, dir := range packagesToUpdate {
		fmt.Printf("Removing directory: %s\n", dir)
		if err := os.RemoveAll(dir); err != nil {
			log.Fatalf("Failed to remove directory %s: %v", dir, err)
		}
	}

	// Task 3: Download the latest version of golang.org/x/tools.
	fmt.Println("Downloading latest version of golang.org/x/tools@master")
	cmd := exec.Command("go", "mod", "download", "-json", "golang.org/x/tools@master")
	output, err := cmd.Output()
	if err != nil {
		log.Fatalf("Failed to download golang.org/x/tools: %v", err)
	}

	var modInfo struct {
		Dir string
	}
	if err := json.Unmarshal(output, &modInfo); err != nil {
		log.Fatalf("Failed to parse JSON output: %v", err)
	}

	fmt.Printf("Downloaded to: %s\n", modInfo.Dir)

	// Task 4: Copy directories from the downloaded module.
	sourceDirs := map[string]string{
		"jsonschema": filepath.Join(modInfo.Dir, "internal", "mcp", "jsonschema"),
		"util":       filepath.Join(modInfo.Dir, "internal", "mcp", "internal", "util"),
	}

	for dest, src := range sourceDirs {
		fmt.Printf("Copying %s to %s\n", src, dest)
		if err := copyDir(src, dest); err != nil {
			log.Fatalf("Failed to copy directory from %s to %s: %v", src, dest, err)
		}
	}

	// Task 5: Replace import paths in the new directories.
	oldToNew := map[string]string{
		"golang.org/x/tools/internal/mcp/jsonschema": "github.com/google/adk-go/internal/jsonschema",
		"golang.org/x/tools/internal/mcp/internal/util": "github.com/google/adk-go/internal/util",
	}

	fmt.Printf("Replacing import paths in 'jsonschema' and 'util' directories...\n")
	for _, dir := range packagesToUpdate {
		err := filepath.Walk(dir, func(path string, info fs.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if !info.IsDir() && strings.HasSuffix(info.Name(), ".go") {
				content, err := os.ReadFile(path)
				if err != nil {
					return fmt.Errorf("failed to read file %s: %v", path, err)
				}
				// Create a copy of the content to modify
				newContent := make([]byte, len(content))
				copy(newContent, content)

				for oldImport, newImport := range oldToNew {
					if strings.Contains(string(content), oldImport) {
						fmt.Printf("Found import %s in %s, replacing with %s\n", oldImport, path, newImport)
						newContent = bytes.ReplaceAll(newContent, []byte(oldImport), []byte(newImport))
					}
				}

				if !bytes.Equal(content, newContent) {
					fmt.Printf("Updating imports in: %s\n", path)
					if err := os.WriteFile(path, newContent, 0644); err != nil {
						return fmt.Errorf("failed to write to file %s: %v", path, err)
					}
				}
			}
			return nil
		})

		if err != nil {
			log.Fatalf("Failed to replace import paths: %v", err)
		}
	}

	fmt.Println("Successfully completed all tasks.")
}

// copyDir recursively copies a directory from src to dest.
func copyDir(src, dest string) error {
	return filepath.Walk(src, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}

		destPath := filepath.Join(dest, relPath)

		if info.IsDir() {
			return os.MkdirAll(destPath, 0755)
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(destPath, data, 0644)
	})
}
