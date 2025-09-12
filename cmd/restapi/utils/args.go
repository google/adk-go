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
