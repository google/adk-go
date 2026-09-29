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

package database_test

import (
	"context"
	"fmt"
	"log"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
)

func ExampleNewSessionService() {
	ctx := context.Background()

	// A file path such as "sessions.db" keeps sessions across restarts.
	svc, err := database.NewSessionService(sqlite.Open("file:example?mode=memory&cache=shared"), &gorm.Config{})
	if err != nil {
		log.Fatal(err)
	}
	// Run on every startup, so the schema picks up columns added by upgrades.
	if err := database.AutoMigrate(svc); err != nil {
		log.Fatal(err)
	}

	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "my-app", UserID: "user"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Session.AppName())
	// Output: my-app
}
