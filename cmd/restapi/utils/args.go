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

// package utils provides utility functions for the REST API server.
package utils

import "flag"

// AdkAPIArgs contains the arguments for the ADK API server.
type AdkAPIArgs struct {
	Port         int
	FrontAddress string
}

// ParseArgs parses the arguments for the ADK API server.
func ParseArgs() AdkAPIArgs {
	portFlag := flag.Int("port", 8080, "Port to listen on")
	frontAddressFlag := flag.String("front_address", "localhost:8001", "Front address to allow CORS requests from")
	flag.Parse()
	if !flag.Parsed() {
		flag.Usage()
		panic("Failed to parse flags")
	}
	return AdkAPIArgs{
		Port:         *portFlag,
		FrontAddress: *frontAddressFlag,
	}
}
