package adkrest

import (
	"testing"
)

func TestNewServer_NilConfig(t *testing.T) {
	tc := []struct {
		name string
		cfg  ServerConfig
	}{
		{
			name: "zero value config",
			cfg:  ServerConfig{},
		},
		{
			name: "nil DebugConfig",
			cfg:  ServerConfig{DebugConfig: nil},
		},
		{
			name: "DebugConfig with zero TraceCapacity",
			cfg: ServerConfig{
				DebugConfig: &DebugTelemetryConfig{TraceCapacity: 0},
			},
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			srv, err := NewServer(tt.cfg)
			if err != nil {
				t.Fatalf("NewServer() returned error: %v", err)
			}
			if srv == nil {
				t.Fatal("NewServer() returned nil server")
			}
		})
	}
}
