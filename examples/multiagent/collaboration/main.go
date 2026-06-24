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

// Package main demonstrates a collaborative agent team that uses all three
// LLM agent modes together: the root [llmagent.ModeChat] coordinator
// delegates to a [llmagent.ModeSingleTurn] sub-agent for autonomous lookups
// and to a [llmagent.ModeTask] sub-agent for multi-turn data collection.
//
// Behavior summary:
//
//   - travel_planner (chat, root): talks to the user, decides which
//     sub-agent to delegate to, then summarizes the result.
//   - weather_checker (single_turn): chains user_info, geocode_address
//     and get_weather to answer in one autonomous run, then returns
//     to parent automatically with the result.
//   - flight_booker (task): receives a structured FlightInput, may ask the
//     user clarifying questions, uses search_flights + book_flight, then
//     returns a structured FlightResult to parent.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// ---------------------------------------------------------------------------
// Tools for weather_checker (single_turn).
// ---------------------------------------------------------------------------

// UserInfoInput is the (empty) input for the user_info tool.
type UserInfoInput struct{}

// UserInfoOutput is the output of the user_info tool.
type UserInfoOutput struct {
	Name    string `json:"name"`
	City    string `json:"city"`
	Country string `json:"country"`
}

// userInfo returns the current user's home location. In a real application
// this would consult session state or a profile service.
func userInfo(_ agent.Context, _ UserInfoInput) (UserInfoOutput, error) {
	return UserInfoOutput{
		Name:    "Alex",
		City:    "Zurich",
		Country: "Switzerland",
	}, nil
}

// GeocodeAddressInput is the input for the geocode_address tool.
type GeocodeAddressInput struct {
	Address string `json:"address" jsonschema:"Human-readable address or city name"`
}

// GeocodeAddressOutput is the output of the geocode_address tool.
type GeocodeAddressOutput struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

// geocodeAddress is a mock geocoder for a handful of well-known cities.
func geocodeAddress(_ agent.Context, in GeocodeAddressInput) (GeocodeAddressOutput, error) {
	cities := map[string]GeocodeAddressOutput{
		"zurich":    {Latitude: 47.3769, Longitude: 8.5417},
		"london":    {Latitude: 51.5074, Longitude: -0.1278},
		"paris":     {Latitude: 48.8566, Longitude: 2.3522},
		"tokyo":     {Latitude: 35.6762, Longitude: 139.6503},
		"new york":  {Latitude: 40.7128, Longitude: -74.0060},
		"san jose":  {Latitude: 37.3382, Longitude: -121.8863},
		"barcelona": {Latitude: 41.3851, Longitude: 2.1734},
	}
	for key, coords := range cities {
		if strings.Contains(strings.ToLower(in.Address), key) {
			return coords, nil
		}
	}
	// Fallback so the demo always succeeds.
	return GeocodeAddressOutput{Latitude: 0, Longitude: 0}, nil
}

// GetWeatherInput is the input for the get_weather tool.
type GetWeatherInput struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

// GetWeatherOutput is the output of the get_weather tool.
type GetWeatherOutput struct {
	ConditionsSummary string  `json:"conditions_summary"`
	TempCelsius       float64 `json:"temp_celsius"`
}

// getWeather is a mock weather service. The result is deterministic per
// (lat, lon) so the example is reproducible offline.
func getWeather(_ agent.Context, in GetWeatherInput) (GetWeatherOutput, error) {
	conditions := []string{"sunny", "partly cloudy", "rainy", "overcast", "snowy"}
	idx := (int(in.Latitude*10) + int(in.Longitude*10)) % len(conditions)
	if idx < 0 {
		idx += len(conditions)
	}
	temp := 10 + (in.Latitude/2 - in.Longitude/3)
	return GetWeatherOutput{
		ConditionsSummary: conditions[idx],
		TempCelsius:       temp,
	}, nil
}

// ---------------------------------------------------------------------------
// Tools for flight_booker (task).
// ---------------------------------------------------------------------------

// SearchFlightsInput is the input for the search_flights tool.
type SearchFlightsInput struct {
	Origin      string `json:"origin"`
	Destination string `json:"destination"`
	Date        string `json:"date" jsonschema:"Date of departure in YYYY-MM-DD"`
}

// FlightOption describes a single flight returned by search_flights.
type FlightOption struct {
	FlightNumber string  `json:"flight_number"`
	Airline      string  `json:"airline"`
	DepartTime   string  `json:"depart_time"`
	ArriveTime   string  `json:"arrive_time"`
	PriceUSD     float64 `json:"price_usd"`
}

// SearchFlightsOutput is the output of the search_flights tool.
type SearchFlightsOutput struct {
	Options []FlightOption `json:"options"`
}

// searchFlights returns a mocked list of flights for the requested route.
func searchFlights(_ agent.Context, in SearchFlightsInput) (SearchFlightsOutput, error) {
	return SearchFlightsOutput{
		Options: []FlightOption{
			{
				FlightNumber: "LX38",
				Airline:      "SWISS",
				DepartTime:   in.Date + "T08:15",
				ArriveTime:   in.Date + "T10:45",
				PriceUSD:     420.00,
			},
			{
				FlightNumber: "BA713",
				Airline:      "British Airways",
				DepartTime:   in.Date + "T12:00",
				ArriveTime:   in.Date + "T14:25",
				PriceUSD:     380.00,
			},
			{
				FlightNumber: "AF1815",
				Airline:      "Air France",
				DepartTime:   in.Date + "T17:30",
				ArriveTime:   in.Date + "T19:55",
				PriceUSD:     355.00,
			},
		},
	}, nil
}

// BookFlightInput is the input for the book_flight tool.
type BookFlightInput struct {
	FlightNumber   string `json:"flight_number"`
	PassengerName  string `json:"passenger_name"`
	SeatPreference string `json:"seat_preference" jsonschema:"window | aisle | any"`
	Date           string `json:"date"`
}

// BookFlightOutput is the output of the book_flight tool.
type BookFlightOutput struct {
	BookingReference string `json:"booking_reference"`
	Confirmation     string `json:"confirmation"`
}

// bookFlight mocks a flight reservation and returns a confirmation code.
func bookFlight(_ agent.Context, in BookFlightInput) (BookFlightOutput, error) {
	ref := fmt.Sprintf("ADK-%s-%03d", in.FlightNumber, len(in.PassengerName)*7%1000)
	return BookFlightOutput{
		BookingReference: ref,
		Confirmation: fmt.Sprintf(
			"Booked %s on %s for %s (%s seat).",
			in.FlightNumber, in.Date, in.PassengerName, in.SeatPreference,
		),
	}, nil
}

// ---------------------------------------------------------------------------
// Wiring.
// ---------------------------------------------------------------------------

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-3.1-flash-lite", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// ----- weather_checker tools -----
	userInfoTool, err := functiontool.New(functiontool.Config{
		Name:        "user_info",
		Description: "Return the current user's name and home city.",
	}, userInfo)
	if err != nil {
		log.Fatalf("Failed to create user_info tool: %v", err)
	}
	geocodeAddressTool, err := functiontool.New(functiontool.Config{
		Name:        "geocode_address",
		Description: "Geocode a city or address into latitude/longitude.",
	}, geocodeAddress)
	if err != nil {
		log.Fatalf("Failed to create geocode_address tool: %v", err)
	}
	getWeatherTool, err := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "Return current weather conditions at a lat/lon.",
	}, getWeather)
	if err != nil {
		log.Fatalf("Failed to create get_weather tool: %v", err)
	}

	// ----- flight_booker tools -----
	searchFlightsTool, err := functiontool.New(functiontool.Config{
		Name:        "search_flights",
		Description: "Search for available flights for an origin/destination/date.",
	}, searchFlights)
	if err != nil {
		log.Fatalf("Failed to create search_flights tool: %v", err)
	}
	bookFlightTool, err := functiontool.New(functiontool.Config{
		Name:        "book_flight",
		Description: "Reserve a specific flight for a passenger.",
	}, bookFlight)
	if err != nil {
		log.Fatalf("Failed to create book_flight tool: %v", err)
	}

	// ----- flight_booker schemas -----
	// FlightInput: structured request the coordinator hands to flight_booker.
	flightInputSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"origin": {
				Type:        genai.TypeString,
				Description: "Origin city or airport",
			},
			"destination": {
				Type:        genai.TypeString,
				Description: "Destination city or airport",
			},
			"date": {
				Type:        genai.TypeString,
				Description: "Departure date in YYYY-MM-DD",
			},
			"passenger_name": {
				Type:        genai.TypeString,
				Description: "Full name of the passenger",
			},
		},
		Required: []string{"origin", "destination", "date"},
	}
	// FlightResult: structured output returned to the coordinator.
	flightResultSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"booking_reference": {Type: genai.TypeString},
			"flight_number":     {Type: genai.TypeString},
			"airline":           {Type: genai.TypeString},
			"depart_time":       {Type: genai.TypeString},
			"arrive_time":       {Type: genai.TypeString},
			"price_usd":         {Type: genai.TypeNumber},
		},
		Required: []string{"booking_reference", "flight_number", "price_usd"},
	}

	weatherCheckerInstruction := `You report current weather conditions.
If no location is given, call ` + "`user_info`" + ` to find the user's home city.
Always call ` + "`geocode_address`" + ` to convert the location into coordinates, then ` + "`get_weather`" + `.
Finish with a single sentence summarising the conditions and temperature.
`

	flightBookerInstruction := `You book exactly one flight for the user.
You are given a structured FlightInput. If the passenger name or seat preference is missing,
ask the user once to provide them; otherwise do not chat unnecessarily.
Call ` + "`search_flights`" + ` to list options, pick the cheapest, then call ` + "`book_flight`" + `.
Finish your task by returning the FlightResult.
`

	travelPlannerInstruction := `You are a friendly travel planning assistant.
For weather questions, delegate to ` + "`weather_checker`" + ` and present its single-sentence
answer to the user.
For booking a flight, gather origin, destination and date from the user, then delegate to ` + "`flight_booker`" + `
with a structured FlightInput. When ` + "`flight_booker`" + ` returns a FlightResult, confirm the booking
to the user with the price, times and booking reference.
Do not call the sub-agents' tools yourself.
`

	// ----- sub-agents -----
	weatherChecker, err := llmagent.New(llmagent.Config{
		Name:        "weather_checker",
		Model:       model,
		Description: "Looks up current weather for the user or a named place.",
		Mode:        llmagent.ModeSingleTurn,
		Tools: []tool.Tool{
			getWeatherTool,
			userInfoTool,
			geocodeAddressTool,
		},
		Instruction: weatherCheckerInstruction,
	})
	if err != nil {
		log.Fatalf("Failed to create weather_checker: %v", err)
	}

	flightBooker, err := llmagent.New(llmagent.Config{
		Name:         "flight_booker",
		Model:        model,
		Description:  "Books a flight for the user given an origin, destination and date.",
		Mode:         llmagent.ModeTask,
		InputSchema:  flightInputSchema,
		OutputSchema: flightResultSchema,
		Tools: []tool.Tool{
			searchFlightsTool,
			bookFlightTool,
		},
		Instruction: flightBookerInstruction,
	})
	if err != nil {
		log.Fatalf("Failed to create flight_booker: %v", err)
	}

	travelPlanner, err := llmagent.New(llmagent.Config{
		Name:        "travel_planner",
		Model:       model,
		Description: "Helps the user plan a trip by delegating to specialist sub-agents.",
		SubAgents: []agent.Agent{
			weatherChecker,
			flightBooker,
		},
		Instruction: travelPlannerInstruction,
	})
	if err != nil {
		log.Fatalf("Failed to create travel_planner: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(travelPlanner),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
