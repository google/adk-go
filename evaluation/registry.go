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

package evaluation

import (
	"fmt"
	"sync"
)

// Registry manages available evaluators for different metrics.
type Registry struct {
	mu        sync.RWMutex
	factories map[MetricType]EvaluatorFactory
}

// NewRegistry creates a new evaluator registry.
func NewRegistry() *Registry {
	return &Registry{
		factories: make(map[MetricType]EvaluatorFactory),
	}
}

// Register registers an evaluator factory for a specific metric type.
func (r *Registry) Register(metricType MetricType, factory EvaluatorFactory) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.factories[metricType]; exists {
		return fmt.Errorf("evaluator already registered for metric %s", metricType)
	}

	r.factories[metricType] = factory
	return nil
}

// Get retrieves an evaluator factory for a specific metric type.
func (r *Registry) Get(metricType MetricType) (EvaluatorFactory, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	factory, exists := r.factories[metricType]
	if !exists {
		return nil, fmt.Errorf("no evaluator registered for metric %s", metricType)
	}

	return factory, nil
}

// CreateEvaluator creates an evaluator instance for a specific metric.
func (r *Registry) CreateEvaluator(metricType MetricType, config EvaluatorConfig) (Evaluator, error) {
	factory, err := r.Get(metricType)
	if err != nil {
		return nil, err
	}

	return factory(config)
}

// ListMetrics returns all registered metric types.
func (r *Registry) ListMetrics() []MetricType {
	r.mu.RLock()
	defer r.mu.RUnlock()

	metrics := make([]MetricType, 0, len(r.factories))
	for metricType := range r.factories {
		metrics = append(metrics, metricType)
	}

	return metrics
}

// IsRegistered checks if an evaluator is registered for a metric type.
func (r *Registry) IsRegistered(metricType MetricType) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	_, exists := r.factories[metricType]
	return exists
}

// DefaultRegistry is the global registry instance.
var DefaultRegistry = NewRegistry()

// Register registers an evaluator factory in the default registry.
func Register(metricType MetricType, factory EvaluatorFactory) error {
	return DefaultRegistry.Register(metricType, factory)
}

// CreateEvaluator creates an evaluator using the default registry.
func CreateEvaluator(metricType MetricType, config EvaluatorConfig) (Evaluator, error) {
	return DefaultRegistry.CreateEvaluator(metricType, config)
}
