// Copyright 2025 Google LLC
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/internal/artifact/tests"
)

func TestInMemoryArtifactService(t *testing.T) {
	factory := func(t *testing.T) (artifact.Service, error) {
		return artifact.InMemoryService(), nil
	}
	tests.TestArtifactService(t, "InMemory", factory)
}
