// Command zzrepro reproduces the comment-1 finding against a real Skill
// Registry: list skills via NewSkillRegistrySource, then load each by the name
// the listing returned. THROWAWAY - delete after use.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
)

func main() {
	ctx := context.Background()
	const (
		project  = "kdroste-adk-2025-12"
		location = "global"
	)

	// Low-level client first, so we can see the RAW resource name (sk.Name)
	// that ListSkills returns, next to the listed frontmatter name.
	client, err := skillregistry.NewClient(ctx, skillregistry.Config{ProjectID: project, Location: location})
	if err != nil {
		fmt.Printf("NewClient: %v\n", err)
		os.Exit(1)
	}
	rawSkills, err := client.ListSkills()
	if err != nil {
		fmt.Printf("ListSkills: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("ListSkills returned %d skill(s). Raw resource names vs frontmatter name:\n", len(rawSkills))
	for i, sk := range rawSkills {
		fmName := "<nil frontmatter>"
		if sk.Frontmatter != nil {
			fmName = sk.Frontmatter.Name
		}
		fmt.Printf("  [%d] sk.Name=%q  frontmatter.Name=%q  defaultRevision=%q\n", i, sk.Name, fmName, sk.DefaultRevision)
	}

	// Now the high-level source, exactly as an agent uses it.
	source, err := skillregistry.NewSkillRegistrySource(ctx, skillregistry.SkillRegistrySourceConfig{
		ProjectID: project,
		Location:  location,
	})
	if err != nil {
		fmt.Printf("NewSkillRegistrySource: %v\n", err)
		os.Exit(1)
	}

	fms, err := source.ListFrontmatters(ctx)
	if err != nil {
		fmt.Printf("ListFrontmatters: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("\nListFrontmatters returned %d skill(s). Loading each by the listed name:\n", len(fms))

	fail := 0
	limit := len(fms)
	if limit > 5 {
		limit = 5 // first few, like the reviewer
	}
	for i := 0; i < limit; i++ {
		name := fms[i].Name
		fmt.Printf("\n[%d] listed name %q\n", i, name)

		_, lfErr := source.LoadFrontmatter(ctx, name)
		fmt.Printf("    LoadFrontmatter:  %s\n", status(lfErr))

		_, liErr := source.LoadInstructions(ctx, name)
		fmt.Printf("    LoadInstructions: %s\n", status(liErr))

		if lfErr != nil || liErr != nil {
			fail++
		}
	}
	fmt.Printf("\nSUMMARY: %d of %d first skills failed to load by their listed name.\n", fail, limit)
}

func status(err error) string {
	if err == nil {
		return "OK"
	}
	tag := ""
	if errors.Is(err, skill.ErrSkillNotFound) {
		tag = " [ErrSkillNotFound]"
	}
	return "FAIL" + tag + ": " + err.Error()
}
