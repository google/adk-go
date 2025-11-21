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

// RegisterDefaultEvaluators registers all built-in evaluators with the default registry.
// This must be called with the evaluator factory functions from the evaluators package.
//
// Example usage:
//
//	import "google.golang.org/adk/evaluation/evaluators"
//
//	evaluation.RegisterDefaultEvaluators(map[evaluation.MetricType]evaluation.EvaluatorFactory{
//	    evaluation.MetricResponseMatch:          evaluators.NewResponseMatchEvaluator,
//	    evaluation.MetricSemanticResponseMatch:  evaluators.NewSemanticResponseMatchEvaluator,
//	    evaluation.MetricResponseEvaluationScore: evaluators.NewResponseEvaluationScoreEvaluator,
//	    evaluation.MetricToolTrajectoryAvgScore:  evaluators.NewToolTrajectoryEvaluator,
//	    evaluation.MetricToolUseQuality:          evaluators.NewToolUseQualityEvaluator,
//	    evaluation.MetricResponseQuality:         evaluators.NewResponseQualityEvaluator,
//	    evaluation.MetricSafety:                  evaluators.NewSafetyEvaluator,
//	    evaluation.MetricHallucinations:          evaluators.NewHallucinationsEvaluator,
//	})
func RegisterDefaultEvaluators(factories map[MetricType]EvaluatorFactory) error {
	for metricType, factory := range factories {
		if err := Register(metricType, factory); err != nil {
			return err
		}
	}
	return nil
}
