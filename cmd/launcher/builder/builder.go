package builder

import (
	"os"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/web"
)

func BuildLauncher() (*launcher.Launcher, []string, error) {
	args := os.Args[1:] // skip file name

	if len(args) > 0 && args[0] == "web" {
		return web.BuildLauncher(args[1:])
	}
	return nil, nil, nil
}
