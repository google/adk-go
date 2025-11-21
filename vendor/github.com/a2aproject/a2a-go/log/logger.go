// Copyright 2025 The A2A Authors
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

// Package log provides utilities for attaching an [slog.Logger] configured with request-specific
// attributes to [context.Context]. The logger can later be retrieved or used indirectly through package-level
// logging function calls.
//
// Server and client extension point implementations should use this package for generating logs
// instead of using other loggers or slog directly.
package log

import (
	"context"
	"log/slog"
)

type loggerKey struct{}

// WithLogger creates a new Context with the provided Logger attached.
func WithLogger(ctx context.Context, logger *slog.Logger) context.Context {
	return context.WithValue(ctx, loggerKey{}, logger)
}

// LoggerFrom returns the Logger associated with the context, or slog.Default() if no context-scoped logger is available.
func LoggerFrom(ctx context.Context) *slog.Logger {
	if logger, ok := ctx.Value(loggerKey{}).(*slog.Logger); ok {
		return logger
	}
	return slog.Default()
}

// Log invokes Log on the [slog.Logger] associated with the provided Context or slog.Default() if no context-scoped logger is available.
func Log(ctx context.Context, level slog.Level, msg string, keyValArgs ...any) {
	logger := LoggerFrom(ctx)
	if logger.Enabled(ctx, level) {
		logger.Log(ctx, level, msg, keyValArgs...)
	}
}

// Info invokes InfoContext on the [slog.Logger] associated with the provided Context or slog.Default() if no context-scoped logger is available.
func Info(ctx context.Context, msg string, keyValArgs ...any) {
	LoggerFrom(ctx).InfoContext(ctx, msg, keyValArgs...)
}

// Warn invokes WarnContext on the [slog.Logger] associated with the provided Context or slog.Default() if no context-scoped logger is available.
func Warn(ctx context.Context, msg string, keyValArgs ...any) {
	LoggerFrom(ctx).WarnContext(ctx, msg, keyValArgs...)
}

// Error invokes ErrorContext on the [slog.Logger] associated with the provided Context or slog.Default() if no context-scoped logger is available.
func Error(ctx context.Context, msg string, err error, keyValArgs ...any) {
	LoggerFrom(ctx).With("error", err).ErrorContext(ctx, msg, keyValArgs...)
}
