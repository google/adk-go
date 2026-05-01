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

// Travel concierge: a coordinator Gemini agent delegates flight + hotel
// research to specialist sub-agents via the Task API. The base_flow
// mesh runtime auto-routes RequestTask entries to the named sub-agent
// and synthesizes the FunctionResponse from FinishTask.
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/llmagent/task"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const coordinatorInstruction = `You are a travel concierge. The user gives you a trip
brief (origin, destination, dates, traveler count). You do NOT call
flight or hotel APIs directly. Instead:

  1. Call the "flight_agent" tool with the flight criteria.
  2. Call the "hotel_agent" tool with the hotel criteria.
  3. Compose a clear itinerary from the two results.

Be conservative with budgets. If the user didn't specify a budget, ask
once before delegating.`

const flightAgentInstruction = `You research a single flight option for the
coordinator. Use the search_flights tool to look up candidates, then
call finish_task with one chosen flight as a JSON object containing
fields {airline, flight_number, price_usd, depart_iso, arrive_iso}.
Pick the cheapest non-stop within budget when one exists.`

const hotelAgentInstruction = `You research a single hotel option for the
coordinator. Use the search_hotels tool to look up candidates, then
call finish_task with one chosen hotel as a JSON object containing
fields {name, neighborhood, nightly_price_usd, total_usd, nights}.
Prefer well-located properties under budget.`

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Specialist sub-agents. Each gets its own finish_task tool — the
	// Task API mesh expects the sub-agent to signal completion that way.
	searchFlights, err := buildSearchFlightsTool()
	if err != nil {
		log.Fatal(err)
	}
	flightAgent, err := llmagent.New(llmagent.Config{
		Name:        "flight_agent",
		Description: "Researches flights and returns the chosen option via finish_task.",
		Model:       model,
		Instruction: flightAgentInstruction,
		Tools:       []tool.Tool{searchFlights},
	})
	if err != nil {
		log.Fatal(err)
	}
	finishFlight, err := task.NewFinishTaskTool(flightAgent)
	if err != nil {
		log.Fatal(err)
	}
	flightAgent, err = llmagent.New(llmagent.Config{
		Name:        "flight_agent",
		Description: "Researches flights and returns the chosen option via finish_task.",
		Model:       model,
		Instruction: flightAgentInstruction,
		Tools:       []tool.Tool{searchFlights, finishFlight},
	})
	if err != nil {
		log.Fatal(err)
	}

	searchHotels, err := buildSearchHotelsTool()
	if err != nil {
		log.Fatal(err)
	}
	hotelAgent, err := llmagent.New(llmagent.Config{
		Name:        "hotel_agent",
		Description: "Researches hotels and returns the chosen option via finish_task.",
		Model:       model,
		Instruction: hotelAgentInstruction,
		Tools:       []tool.Tool{searchHotels},
	})
	if err != nil {
		log.Fatal(err)
	}
	finishHotel, err := task.NewFinishTaskTool(hotelAgent)
	if err != nil {
		log.Fatal(err)
	}
	hotelAgent, err = llmagent.New(llmagent.Config{
		Name:        "hotel_agent",
		Description: "Researches hotels and returns the chosen option via finish_task.",
		Model:       model,
		Instruction: hotelAgentInstruction,
		Tools:       []tool.Tool{searchHotels, finishHotel},
	})
	if err != nil {
		log.Fatal(err)
	}

	// Coordinator delegates via RequestTaskTool — one tool per task agent.
	requestFlight, err := task.NewRequestTaskTool(flightAgent)
	if err != nil {
		log.Fatal(err)
	}
	requestHotel, err := task.NewRequestTaskTool(hotelAgent)
	if err != nil {
		log.Fatal(err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "concierge",
		Description: "A travel concierge that delegates flight and hotel research to specialists.",
		Model:       model,
		Instruction: coordinatorInstruction,
		Tools:       []tool.Tool{requestFlight, requestHotel},
		SubAgents:   []agent.Agent{flightAgent, hotelAgent},
	})
	if err != nil {
		log.Fatal(err)
	}

	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(rootAgent)}
	l := full.NewLauncher()
	if err = l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

func modelName() string {
	if v := os.Getenv("GOOGLE_GENAI_MODEL"); v != "" {
		return v
	}
	return "gemini-2.5-flash"
}

func buildSearchFlightsTool() (tool.Tool, error) {
	type args struct {
		Origin      string `json:"origin"`
		Destination string `json:"destination"`
		DepartDate  string `json:"depart_date"`
		ReturnDate  string `json:"return_date,omitempty"`
		MaxPriceUSD int    `json:"max_price_usd,omitempty"`
	}
	type flight struct {
		Airline      string `json:"airline"`
		FlightNumber string `json:"flight_number"`
		PriceUSD     int    `json:"price_usd"`
		DepartISO    string `json:"depart_iso"`
		ArriveISO    string `json:"arrive_iso"`
		Stops        int    `json:"stops"`
	}
	type result struct {
		Flights []flight `json:"flights"`
	}
	return functiontool.New[args, result](
		functiontool.Config{
			Name:        "search_flights",
			Description: "Search flights between two cities for given dates.",
		},
		func(_ tool.Context, a args) (result, error) {
			// Stubbed catalog. Replace with a real flight-search API call.
			return result{Flights: []flight{
				{"ANA", "NH7", 1198, a.DepartDate + "T11:55", a.DepartDate + "T15:30", 0},
				{"United", "UA837", 1340, a.DepartDate + "T13:25", a.DepartDate + "T17:00", 0},
				{"JAL", "JL1", 1290, a.DepartDate + "T13:00", a.DepartDate + "T16:35", 0},
			}}, nil
		},
	)
}

func buildSearchHotelsTool() (tool.Tool, error) {
	type args struct {
		City              string `json:"city"`
		CheckIn           string `json:"check_in"`
		Nights            int    `json:"nights"`
		MaxNightlyUSD     int    `json:"max_nightly_usd,omitempty"`
		PreferredArea     string `json:"preferred_area,omitempty"`
	}
	type hotel struct {
		Name         string `json:"name"`
		Neighborhood string `json:"neighborhood"`
		NightlyUSD   int    `json:"nightly_usd"`
		Rating       string `json:"rating"`
	}
	type result struct {
		Hotels []hotel `json:"hotels"`
	}
	return functiontool.New[args, result](
		functiontool.Config{
			Name:        "search_hotels",
			Description: "Search hotels in a city for the given dates.",
		},
		func(_ tool.Context, a args) (result, error) {
			// Stubbed catalog. Replace with a real hotel-search API call.
			return result{Hotels: []hotel{
				{"Park Hyatt " + a.City, "Shinjuku", 480, "5/5"},
				{"Citadines " + a.City, "Shibuya", 220, "4.2/5"},
				{"Hotel Niwa " + a.City, "Kanda", 180, "4.5/5"},
			}}, nil
		},
	)
}
