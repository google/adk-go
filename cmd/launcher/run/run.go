package run

import (
	"context"
	"log"
	"os"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/launcher/console"
	"google.golang.org/adk/cmd/launcher/web"
)

func Run(ctx context.Context, config *adk.Config) {
	l, _, err := BuildLauncher()
	if err != nil {
		log.Fatalf("cannot build launcher: %v", err)
	}
	err = (*l).Run(ctx, config)
	if err != nil {
		log.Fatalf("run failed: %v", err)
	}
}

func BuildLauncher() (*launcher.Launcher, []string, error) {
	args := os.Args[1:] // skip file name

	if len(args) > 0 && args[0] == "web" {
		return web.BuildLauncher(args[1:])
	}
	return console.BuildLauncher(args[1:])
}
