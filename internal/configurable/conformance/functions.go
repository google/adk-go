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

package conformance

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"regexp"

	"google.golang.org/adk/internal/configurable"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type ValidateEmailArgs struct {
	Email string `json:"email"`
}

var emailRegex = regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)

func validateEmail(ctx tool.Context, args ValidateEmailArgs) (bool, error) {
	return emailRegex.MatchString(args.Email), nil
}

func getUserID(ctx tool.Context, args ValidateEmailArgs) (int, error) {
	valid, err := validateEmail(ctx, args)
	if err != nil {
		return 0, err
	}
	if !valid {
		return 0, fmt.Errorf("invalid email format provided")
	}

	// 1. Create a new FNV-1a 32-bit hasher
	h := fnv.New32a()

	// 2. Write the email string as bytes to the hasher
	h.Write([]byte(args.Email))

	// 3. Get the resulting 32-bit unsigned integer
	result := h.Sum32()

	// 4. Modulo 10000 to keep it in range
	return int(result % 10000), nil
}

func createBooking(ctx tool.Context, args ValidateEmailArgs) (map[string]any, error) {
	userID, err := getUserID(ctx, args)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"user_id":           userID,
		"is_confirmed":      true,
		"details":           "Booking created for user " + args.Email,
		"user_id_type":      "int",
		"is_confirmed_type": "bool",
		"details_type":      "string",
	}, nil
}

type FlightPreferences struct {
	CabinClass       string `json:"cabin_class"`
	MaxStops         int    `json:"max_stops"`
	PreferredAirline string `json:"preferred_airline"`
	FlexibleDates    bool   `json:"flexible_dates"`
}

type TripDetails struct {
	Origin        string `json:"origin"`
	Destination   string `json:"destination"`
	DepartureDate string `json:"departure_date"`
	ReturnDate    string `json:"return_date"`
}

type SearchFlightsArgs struct {
	Trip        TripDetails        `json:"trip"`
	Preferences *FlightPreferences `json:"preferences"`
}

func searchFlights(ctx tool.Context, args SearchFlightsArgs) (map[string]any, error) {
	if args.Preferences == nil {
		args.Preferences = &FlightPreferences{
			CabinClass:    "Economy",
			MaxStops:      1,
			FlexibleDates: false,
		}
	}

	tripType := "one-way"
	if args.Trip.ReturnDate != "" {
		tripType = "round-trip"
	}

	result := map[string]any{
		"trip_type":         tripType,
		"route":             args.Trip.Origin + " to " + args.Trip.Destination,
		"departure_date":    args.Trip.DepartureDate,
		"return_date":       args.Trip.ReturnDate,
		"cabin_class":       args.Preferences.CabinClass,
		"max_stops":         args.Preferences.MaxStops,
		"preferred_airline": args.Preferences.PreferredAirline,
		"flexible_dates":    args.Preferences.FlexibleDates,
		"search_status":     "completed",
	}

	airline := args.Preferences.PreferredAirline
	if airline == "" {
		airline = "Various Airlines"
	}

	stopsDesc := "direct"
	if args.Preferences.MaxStops > 0 {
		stopsDesc = fmt.Sprintf("up to %d stops", args.Preferences.MaxStops)
	}

	flights := []string{
		fmt.Sprintf("%s - %s %s flight with %s",
			airline, tripType, args.Preferences.CabinClass, stopsDesc),
		fmt.Sprintf("Departure: %s", args.Trip.DepartureDate),
	}

	if args.Trip.ReturnDate != "" {
		flights = append(flights, fmt.Sprintf("Return: %s", args.Trip.ReturnDate))
	}

	result["available_flights"] = flights

	return result, nil
}

type CalculateTripCostArgs struct {
	BaseFare      float64 `json:"base_fare"`
	NumPassengers int     `json:"num_passengers"`
	Insurance     bool    `json:"insurance"`
	BaggageCount  *int    `json:"baggage_count"`
}

func calculateTripCost(ctx tool.Context, args CalculateTripCostArgs) (map[string]any, error) {
	// Handle Python's default num_passengers=1 logic
	// In Go, if the caller passes 0, we should ensure at least 1
	// or handle it based on your specific business logic.
	if args.NumPassengers <= 0 {
		args.NumPassengers = 1
	}

	subtotal := args.BaseFare * float64(args.NumPassengers)

	// Add insurance (10% of base fare per passenger)
	insuranceCost := 0.0
	if args.Insurance {
		insuranceCost = subtotal * 0.1
	}

	// Add baggage fees
	baggageCost := 0.0
	var displayBaggage any = nil

	if args.BaggageCount != nil {
		count := *args.BaggageCount
		displayBaggage = count

		// First bag free, $35 per additional bag per passenger
		chargeableBags := math.Max(0, float64(count-1))
		baggageCost = chargeableBags * 35 * float64(args.NumPassengers)
	}

	total := subtotal + insuranceCost + baggageCost

	return map[string]any{
		"base_fare":          args.BaseFare,
		"num_passengers":     args.NumPassengers,
		"subtotal":           subtotal,
		"insurance_included": args.Insurance,
		"insurance_cost":     insuranceCost,
		"baggage_count":      displayBaggage,
		"baggage_cost":       baggageCost,
		"total_cost":         total,
	}, nil
}

type reimburseArgs struct {
	Purpose string  `json:"purpose"`
	Amount  float64 `json:"amount"`
}

func reimburse(ctx tool.Context, args reimburseArgs) (map[string]any, error) {
	return map[string]any{
		"status": "ok",
	}, nil
}

type askForApprovalArgs struct {
	Purpose string  `json:"purpose"`
	Amount  float64 `json:"amount"`
}

func askForApproval(ctx tool.Context, args askForApprovalArgs) (map[string]any, error) {
	return map[string]any{
		"status":   "pending",
		"amount":   args.Amount,
		"ticketId": "reimbursement-ticket-001",
	}, nil
}

func RegisterFunctions() error {
	validateEmailTool, err := functiontool.New(functiontool.Config{
		Name:        "validate_email",
		Description: "Checks if the provided string is a valid email format.",
	}, validateEmail)
	if err != nil {
		return fmt.Errorf("error creating validate email tool: %w", err)
	}
	getUserIDTool, err := functiontool.New(functiontool.Config{
		Name:        "get_user_id",
		Description: "Retrieves a user ID based on their email.",
	}, getUserID)
	if err != nil {
		return fmt.Errorf("error creating get user ID tool: %w", err)
	}
	createBookingTool, err := functiontool.New(functiontool.Config{
		Name: "create_booking",
		Description: `Creates a booking for a user.

  Args:
    user_id: The unique identifier for the user.
    is_confirmed: Whether the booking is confirmed.
    details: Any additional details for the booking.

  Returns:
    A dictionary containing the booking information and the types of the
    received arguments.
  `,
	}, createBooking)
	if err != nil {
		return fmt.Errorf("error creating create booking tool: %w", err)
	}

	searchFlightsTool, err := functiontool.New(functiontool.Config{
		Name: "search_flights",
		Description: `Search for flights based on trip details and preferences.

  This function demonstrates advanced parameter handling:
  - Pydantic models as parameters (trip, preferences)
  - Optional/nullable parameters (preferences, return_date, preferred_airline)
  - Default values (cabin_class, max_stops, flexible_dates)

  Args:
    trip: Core trip information including origin, destination, and dates.
    preferences: Optional flight preferences. If not provided, uses defaults.

  Returns:
    A dictionary containing search results and parameters received.
  `,
	}, searchFlights)
	if err != nil {
		return fmt.Errorf("error creating search flights tool: %w", err)
	}

	calculateTripCostTool, err := functiontool.New(functiontool.Config{
		Name: "calculate_trip_cost",
		Description: `Calculate total trip cost with various optional charges.

  This function demonstrates:
  - Mix of required and optional parameters
  - Default values for common cases
  - Nullable parameter that affects calculation logic

  Args:
    base_fare: Base ticket price per passenger.
    num_passengers: Number of passengers (default: 1).
    insurance: Whether to add travel insurance (default: False).
    baggage_count: Number of checked bags per passenger, or None for carry-on
      only.

  Returns:
    A dictionary with cost breakdown.
  `,
	}, calculateTripCost)
	if err != nil {
		return fmt.Errorf("error creating calculate trip cost tool: %w", err)
	}

	reimburseTool, err := functiontool.New(functiontool.Config{
		Name:        "reimburse",
		Description: `Reimburse the amount of money to the employee.`,
	}, reimburse)
	if err != nil {
		return fmt.Errorf("error creating reimburse tool: %w", err)
	}

	askForApprovalTool, err := functiontool.New(functiontool.Config{
		Name:          "ask_for_approval",
		Description:   `Ask for approval for the reimbursement.`,
		IsLongRunning: true,
	}, askForApproval)
	if err != nil {
		return fmt.Errorf("error creating ask for approval tool: %w", err)
	}

	err = configurable.RegisterToolFactory("tools_agent_002.tools.validate_email", func(ctx context.Context, _ map[string]any) (tool.Tool, error) {
		return validateEmailTool, nil
	})
	if err != nil {
		return fmt.Errorf("error registering validate email tool: %w", err)
	}
	err = configurable.RegisterToolFactory("tools_agent_002.tools.get_user_id", func(ctx context.Context, _ map[string]any) (tool.Tool, error) {
		return getUserIDTool, nil
	})
	if err != nil {
		return fmt.Errorf("error registering get user ID tool: %w", err)
	}
	err = configurable.RegisterToolFactory("tools_agent_002.tools.create_booking", func(ctx context.Context, _ map[string]any) (tool.Tool, error) {
		return createBookingTool, nil
	})
	if err != nil {
		return fmt.Errorf("error registering create booking tool: %w", err)
	}

	err = configurable.RegisterToolFactory("tools_agent_004.tools.search_flights", func(ctx context.Context, _ map[string]any) (tool.Tool, error) {
		return searchFlightsTool, nil
	})
	if err != nil {
		return fmt.Errorf("error registering search flights tool: %w", err)
	}
	err = configurable.RegisterToolFactory("tools_agent_004.tools.calculate_trip_cost", func(ctx context.Context, _ map[string]any) (tool.Tool, error) {
		return calculateTripCostTool, nil
	})
	if err != nil {
		return fmt.Errorf("error registering calculate trip cost tool: %w", err)
	}

	err = configurable.RegisterToolFactory("tools_agent_009.tools.reimburse", func(ctx context.Context, _ map[string]any) (tool.Tool, error) {
		return reimburseTool, nil
	})
	if err != nil {
		return fmt.Errorf("error registering reimburse tool: %w", err)
	}
	err = configurable.RegisterToolFactory("tools_agent_009.tools.ask_for_approval", func(ctx context.Context, _ map[string]any) (tool.Tool, error) {
		return askForApprovalTool, nil
	})
	if err != nil {
		return fmt.Errorf("error registering ask for approval tool: %w", err)
	}
	return nil
}
