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

// Package provides an example of using skills via skill toolset.
package main

import (
	"archive/zip"
	"context"
	"fmt"
	"log"
	"os"
	"time"

	agentskillregistry "google.golang.org/api/agentregistry/v1alpha"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/skilltoolset"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry/simplecache"
)

type faultyClient struct {
	c skillregistry.Client
}

// FindFrontmatters implements [skillregistry.Client].
func (f *faultyClient) FindFrontmatters(query string) ([]*agentskillregistry.Frontmatter, error) {
	panic("unimplemented")
}

// GetRevision implements [skillregistry.Client].
func (f *faultyClient) GetRevision(rev string) (*agentskillregistry.SkillRevision, error) {
	panic("unimplemented")
}

// GetSkill implements [skillregistry.Client].
func (f *faultyClient) GetSkill(name string) (*agentskillregistry.Skill, error) {
	return f.c.GetSkill(name)
}

// GetZip implements [skillregistry.Client].
func (f *faultyClient) GetZip(rev string) (*zip.Reader, error) {
	panic("unimplemented")
}

// ListFrontmatters implements [skillregistry.Client].
func (f *faultyClient) ListFrontmatters() ([]*agentskillregistry.Frontmatter, error) {
	panic("unimplemented")
}

// ListSkills implements [skillregistry.Client].
func (f *faultyClient) ListSkills() ([]*agentskillregistry.Skill, error) {
	return f.c.ListSkills()
}

// ParseSkillName implements [skillregistry.Client].
func (f *faultyClient) ParseSkillName(name string) (projectID, location, skillName string, resErr error) {
	panic("unimplemented")
}

// ResourceID implements [skillregistry.Client].
func (f *faultyClient) ResourceID(name string) string {
	panic("unimplemented")
}

var _ skillregistry.Client = (*faultyClient)(nil)

func NewFaultyClient(c skillregistry.Client) skillregistry.Client {
	return &faultyClient{c: c}
}

func main() {
	ctx := context.Background()

	projectID := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if projectID == "" {
		log.Fatalf("Env var GOOGLE_CLOUD_PROJECT is not set")
	}
	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		log.Fatalf("Env var GOOGLE_CLOUD_LOCATION is not set")
	}

	err := Process10(ctx, projectID, location)
	if err != nil {
		panic(err)
	}

	// private-kdroste-test-skill
}

func Process10(ctx context.Context, projectID, location string) error {
	c, err := skillregistry.NewClient(ctx, skillregistry.Config{ProjectID: projectID, Location: location})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewClient: %w", err)
	}

	cache, err := simplecache.NewCache(simplecache.CacheConfig{Client: c, UpdateInterval: 10 * time.Second})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewCache: %w", err)
	}

	type AvgTime struct {
		activate   bool
		deactivate bool
		workerID   int
		generation int
		nSamples   int
		sum        time.Duration
		max        time.Duration
	}

	rampUpDurationInMillisec := 30000
	nWorkers := 40
	nRepInGen := 100000
	maxGen := 100
	startTime := time.Now()

	partialRes := make(chan AvgTime)
	for i := 0; i < nWorkers; i++ {
		go func(workerID int) {
			// rampup
			sd := time.Duration(i*rampUpDurationInMillisec/nWorkers) * time.Millisecond
			time.Sleep(sd)
			// time.Sleep(time.Duration( float64(workerID)/float64(nWorkers)) * rampUpDuration)

			partialRes <- AvgTime{
				workerID: workerID,
				activate: true,
			}

			for gen := 0; gen < maxGen; gen++ {
				totalDuration := time.Duration(0)
				maxDuration := time.Duration(0)
				for i := 0; i < nRepInGen; i++ {
					lastTime := time.Now()
					_, err := cache.LoadInstructions("private-kdroste-dice-thrower-04")
					// _, err := cache.LoadFrontmatter("private-kdroste-dice-thrower-04")
					if err != nil {
						log.Printf("cannot LoadFrontmatter: %v", err)
					}
					n := time.Now()
					d := n.Sub(lastTime)
					if d > maxDuration {
						maxDuration = d
					}
					totalDuration += d
				}
				// got generation data
				partialRes <- AvgTime{
					workerID:   workerID,
					generation: gen,
					nSamples:   nRepInGen,
					sum:        totalDuration,
					max:        maxDuration,
				}
			}
			partialRes <- AvgTime{
				workerID:   workerID,
				deactivate: true,
			}
		}(i)
	}

	totalSum := time.Duration(0)
	nSamples := 0
	maxDuration := time.Duration(0)
	lastPrint := time.Now()
	workersActive := map[int]bool{}
	for res := range partialRes {
		if res.activate {
			workersActive[res.workerID] = true
			continue
		}
		if res.deactivate {
			delete(workersActive, res.workerID)
			continue
		}
		totalSum += res.sum
		nSamples += res.nSamples
		print := false
		if res.max > maxDuration {
			maxDuration = res.max
			print = true
		}
		if time.Since(lastPrint) > 1*time.Second {
			print = true
		}
		if !print {
			continue
		}
		lastPrint = time.Now()

		log.Printf("res: {workerID: %5v generation: %5v avg: %.8f max: %.8f}, res totals: {sum: %20v nSamples: %10v avg: %.8f [s] max: %.8f [s]}  active: %4v totalTime: %v",
			res.workerID, res.generation, res.sum.Seconds()/float64(res.nSamples), res.max.Seconds(),
			totalSum, nSamples, totalSum.Seconds()/float64(nSamples), maxDuration.Seconds(),
			len(workersActive),
			time.Since(startTime),
		)
	}
	return nil
}

func Process9(ctx context.Context, projectID, location string) error {
	c, err := skillregistry.NewClient(ctx, skillregistry.Config{ProjectID: projectID, Location: location})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewClient: %w", err)
	}

	cache, err := simplecache.NewCache(simplecache.CacheConfig{Client: c, UpdateInterval: 10 * time.Second})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewCache: %w", err)
	}

	type AvgTime struct {
		activate   bool
		deactivate bool
		workerID   int
		generation int
		nSamples   int
		sum        time.Duration
		max        time.Duration
	}

	rampUpDurationInMillisec := 30000
	nWorkers := 40
	nRepInGen := 100000
	maxGen := 100
	startTime := time.Now()

	partialRes := make(chan AvgTime)
	for i := 0; i < nWorkers; i++ {
		go func(workerID int) {
			// rampup
			sd := time.Duration(i*rampUpDurationInMillisec/nWorkers) * time.Millisecond
			time.Sleep(sd)
			// time.Sleep(time.Duration( float64(workerID)/float64(nWorkers)) * rampUpDuration)

			partialRes <- AvgTime{
				workerID: workerID,
				activate: true,
			}

			for gen := 0; gen < maxGen; gen++ {
				totalDuration := time.Duration(0)
				maxDuration := time.Duration(0)
				for i := 0; i < nRepInGen; i++ {
					lastTime := time.Now()
					_, err := cache.LoadInstructions("private-kdroste-dice-thrower-04")
					// _, err := cache.LoadFrontmatter("private-kdroste-dice-thrower-04")
					if err != nil {
						log.Printf("cannot LoadFrontmatter: %v", err)
					}
					n := time.Now()
					d := n.Sub(lastTime)
					if d > maxDuration {
						maxDuration = d
					}
					totalDuration += d
				}
				// got generation data
				partialRes <- AvgTime{
					workerID:   workerID,
					generation: gen,
					nSamples:   nRepInGen,
					sum:        totalDuration,
					max:        maxDuration,
				}
			}
			partialRes <- AvgTime{
				workerID:   workerID,
				deactivate: true,
			}
		}(i)
	}

	totalSum := time.Duration(0)
	nSamples := 0
	maxDuration := time.Duration(0)
	lastPrint := time.Now()
	workersActive := map[int]bool{}
	for res := range partialRes {
		if res.activate {
			workersActive[res.workerID] = true
			continue
		}
		if res.deactivate {
			delete(workersActive, res.workerID)
			continue
		}
		totalSum += res.sum
		nSamples += res.nSamples
		print := false
		if res.max > maxDuration {
			maxDuration = res.max
			print = true
		}
		if time.Since(lastPrint) > 1*time.Second {
			print = true
		}
		if !print {
			continue
		}
		lastPrint = time.Now()

		log.Printf("res: {workerID: %5v generation: %5v avg: %.8f max: %.8f}, res totals: {sum: %20v nSamples: %10v avg: %.8f [s] max: %.8f [s]}  active: %4v totalTime: %v",
			res.workerID, res.generation, res.sum.Seconds()/float64(res.nSamples), res.max.Seconds(),
			totalSum, nSamples, totalSum.Seconds()/float64(nSamples), maxDuration.Seconds(),
			len(workersActive),
			time.Since(startTime),
		)
	}
	return nil
}

func Process8(ctx context.Context, projectID, location string) error {
	c, err := skillregistry.NewClient(ctx, skillregistry.Config{ProjectID: projectID, Location: location})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewClient: %w", err)
	}

	cache, err := simplecache.NewCache(simplecache.CacheConfig{Client: c, UpdateInterval: 15 * time.Second})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewCache: %w", err)
	}
	lastTime := time.Now()
	for {
		_, err := cache.LoadInstructions("private-kdroste-dice-thrower-04")
		// _, err := cache.LoadFrontmatter("private-kdroste-dice-thrower-04")
		n := time.Now()
		d := n.Sub(lastTime)
		lastTime = n
		if err != nil {
			log.Printf("cannot LoadFrontmatter: %v", err)
		} else {
			log.Printf("got fm after: %v", d)
		}
		time.Sleep(1 * time.Second)
	}
	// time.Sleep(5 * time.Minute)
}

func Process7(ctx context.Context, projectID, location string) error {
	c, err := skillregistry.NewClient(ctx, skillregistry.Config{ProjectID: projectID, Location: location})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewClient: %w", err)
	}

	cache, err := simplecache.NewCache(simplecache.CacheConfig{Client: c})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewCache: %w", err)
	}
	err = cache.WarmUp()
	if err != nil {
		return fmt.Errorf("cannot cache.WarmUp: %w", err)
	}

	source, err := simplecache.NewCachedSkillRegistrySource(ctx, cache)
	if err != nil {
		return fmt.Errorf("cannot NewCachedSkillRegistrySource: %w", err)
	}

	model, err := gemini.NewModel(ctx, "gemini-3.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	skillToolset, err := skilltoolset.New(ctx, skilltoolset.Config{Source: source})
	if err != nil {
		log.Fatalf("Failed to create skill toolset: %v", err)
	}

	ss := session.InMemoryService()

	a, err := llmagent.New(llmagent.Config{
		Name:        "skills_agent",
		Model:       model,
		Description: "Agent to demonstrate using skills.",
		Instruction: "You are a helpful assistant.",
		Toolsets:    []tool.Toolset{skillToolset},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(a),
		SessionService: ss,
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
	return nil
}

func Process6(ctx context.Context, projectID, location string) error {
	c, err := skillregistry.NewClient(ctx, skillregistry.Config{ProjectID: projectID, Location: location})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewClient: %w", err)
	}

	cache, err := simplecache.NewCache(simplecache.CacheConfig{Client: c})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewCache: %w", err)
	}
	err = cache.WarmUp()
	if err != nil {
		return fmt.Errorf("cannot cache.WarmUp: %w", err)
	}

	cs, err := simplecache.NewCachedSkillRegistrySource(ctx, cache)
	if err != nil {
		return fmt.Errorf("cannot NewCachedSkillRegistrySource: %w", err)
	}

	fms, err := cs.ListFrontmatters(ctx)
	if err != nil {
		return fmt.Errorf("cannot ListFrontmatters: %w", err)
	}
	log.Printf("got Frontmatters: %+v", len(fms))
	// for _, fm := range fms {
	// 	log.Printf("fm: %+v", fm)
	// }

	skillPath := "projects/kdroste-adk-2025-12/locations/global/skills/private-kdroste-dice-thrower-04"

	f, err := cs.LoadFrontmatter(ctx, skillPath)
	// f, err := cs.LoadFrontmatter(ctx, "private-kdroste-dice-thrower-04")
	if err != nil {
		return fmt.Errorf("cannot LoadFrontmatter: %w", err)
	}
	log.Printf("f: %+v", f)

	i, err := cs.LoadInstructions(ctx, skillPath)
	if err != nil {
		return fmt.Errorf("cannot LoadInstructions: %w", err)
	}
	log.Printf("i: %+v", i)

	return nil
}

func Process5(ctx context.Context, projectID, location string) error {
	c, err := skillregistry.NewClient(ctx, skillregistry.Config{ProjectID: projectID, Location: location})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewClient: %w", err)
	}

	tlc, err := simplecache.NewCache(simplecache.CacheConfig{Client: c})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewCache: %w", err)
	}
	log.Printf("tlc: %+v", tlc)

	err = tlc.WarmUp()
	if err != nil {
		return fmt.Errorf("cannot cache.WarmUp: %w", err)
	}

	log.Printf("tlc after the initial read")

	return nil
}

func Process(ctx context.Context, projectID, location string) error {
	cfg := skillregistry.SkillRegistrySourceConfig{ProjectID: projectID, Location: location}
	source, err := skillregistry.NewSkillRegistrySource(ctx, cfg)
	if err != nil {
		return fmt.Errorf("cannot NewSkillRegistrySource: %w", err)
	}

	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	skillToolset, err := skilltoolset.New(ctx, skilltoolset.Config{Source: source})
	if err != nil {
		log.Fatalf("Failed to create skill toolset: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "skills_agent",
		Model:       model,
		Description: "Agent to demonstrate using skills.",
		Instruction: "You are a helpful assistant.",
		Toolsets:    []tool.Toolset{skillToolset},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
	return nil
}

func Process4(ctx context.Context, projectID, location string) error {
	c, err := skillregistry.NewClient(ctx, skillregistry.Config{ProjectID: projectID, Location: location})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewClient: %w", err)
	}
	s, err := c.GetSkill("projects/kdroste-adk-2025-12/locations/global/skills/private-kdroste-dice-thrower-03")
	if err != nil {
		return fmt.Errorf("cannot GetSkill: %w", err)
	}
	log.Printf("s: %+v", s)
	log.Printf("s.Frontmatter: %+v", s.Frontmatter)

	rev := s.DefaultRevision
	zip, err := c.GetZip(rev)
	if err != nil {
		return fmt.Errorf("cannot GetZip: %w", err)
	}
	for _, f := range zip.File {
		log.Printf("File in zip: %v", f.Name)
	}

	// r, err := c.GetRevision(rev)
	// if err != nil {
	// 	return fmt.Errorf("cannot GetRevision: %w", err)
	// }
	// log.Printf("r: %+v", r)

	return nil
}

// func Process3(ctx context.Context, projectID string) error {
// 	cfg := skillregistry.SkillRegistrySourceConfig{ProjectID: projectID, Location: "us"}
// 	src, err := skillregistry.NewSkillRegistrySource(ctx, cfg)
// 	if err != nil {
// 		return fmt.Errorf("cannot NewSkillRegistrySource: %w", err)
// 	}

// 	f, err := src.FindFrontmatters(ctx, "aa")
// 	if err != nil {
// 		return fmt.Errorf("cannot LoadFrontmatter: %w", err)
// 	}
// 	log.Printf("F: %+v", f)
// 	return nil
// }

func Process2(ctx context.Context, projectID, location string) error {
	cfg := skillregistry.SkillRegistrySourceConfig{ProjectID: projectID, Location: location}
	src, err := skillregistry.NewSkillRegistrySource(ctx, cfg)
	if err != nil {
		return fmt.Errorf("cannot NewSkillRegistrySource: %w", err)
	}

	// skills, err := src.ListSkills()
	// if err != nil {
	// 	panic(err)
	// }

	// for _, s := range skills {
	// 	log.Printf("State: %v", s.State)
	// }

	frs, err := src.ListFrontmatters(ctx)
	if err != nil {
		return fmt.Errorf("cannot ListFrontmatters: %w", err)
	}

	for _, f := range frs {
		log.Printf("f: %+v", f)
		break
	}

	// s, err := src.Load(ctx, "discoveryengine.googleapis.com-report-writing")
	// if err == nil {
	// 	return fmt.Errorf("cannot LoadFrontmatter: %w", err)
	// }
	// log.Printf("s: %+v", s)

	// s, err := src.LoadFrontmatter(ctx, "discoveryengine.googleapis.com-report-writing")
	// if err == nil {
	// 	return fmt.Errorf("cannot LoadFrontmatter: %w", err)
	// }
	// log.Printf("s: %+v", s)

	return nil
}

// func main2() {
// 	ctx := context.Background()
// 	c, err := skillregistry.NewClient(ctx, skillregistry.Config{ProjectID: "kdroste-adk-2025-12", Location: "us"})
// 	if err != nil {
// 		panic(err)
// 	}
// 	skills, err := c.ListSkills()
// 	if err != nil {
// 		panic(err)
// 	}

// 	for _, s := range skills {
// 		log.Printf("State: %v", s.State)
// 	}
// 	_ = skills
// }

// func main2() {
// 	ctx := context.Background()
// 	cfg := skillregistry.SkillRegistryClientConfig{
// 		ProjectID: "kdroste-adk-2025-12",
// 		Location:  "us-central1",
// 	}
// 	c, err := skillregistry.NewSkillRegistryClient(ctx, cfg)
// 	if err != nil {
// 		panic(err)
// 	}
// 	skills := c.AllSkills(ctx)
// 	for sk, err := range skills {
// 		if err != nil {
// 			panic(err)
// 		}
// 		log.Printf("Skill: %+v", sk)
// 		// log.Printf("Zip: %+v", sk.ZippedFilesystem)

// 		s, err := c.GetSkill(ctx, sk.Name)
// 		if err != nil {
// 			panic(err)
// 		}
// 		_ = s
// 		// log.Printf("Skill: %+v", s)

// 		// log.Printf("Zip: %+v", s.ZippedFilesystem)

// 		// rootPath := "/usr/local/google/home/kdroste/tmp/skills"
// 		// p := path.Join(rootPath, sk.DisplayName+".zip")

// 		// log.Printf("Will write to %s", p)
// 		// err = os.WriteFile(p, []byte(s.ZippedFilesystem), 0644)
// 		// if err != nil {
// 		// 	panic(err)
// 		// }

// 	}

// }
