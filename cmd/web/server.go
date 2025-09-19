package web

import (
	"flag"
	"log"
	"net/http"
	"strconv"

	"github.com/gorilla/mux"
	"github.com/rs/cors"
	"google.golang.org/adk/cmd/restapi/config"
	restapiweb "google.golang.org/adk/cmd/restapi/web"
)

type WebConfig struct {
	LocalPort      int
	UIDistPath     string
	FrontEndServer string
	StartRestApi   bool
	StartWebUI     bool
}

// func corsWithArgs(serverConfig *config.ADKAPIRouterConfigs) func(next http.Handler) http.Handler {
// 	return func(next http.Handler) http.Handler {
// 		return serverConfig.Cors.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
// 			// w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
// 			// w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
// 			if r.Method == "OPTIONS" {
// 				w.WriteHeader(http.StatusOK)
// 				return
// 			}
// 			next.ServeHTTP(w, r)
// 		}))
// 	}
// }

// ParseArgs parses the arguments for the ADK API server.

func ParseArgs() *WebConfig {
	localPortFlag := flag.Int("port", 8080, "Port to listen on")
	frontendServerFlag := flag.String("front_address", "localhost:8001", "Front address to allow CORS requests from")
	startRespApi := flag.Bool("start_restapi", true, "Set to start a rest api endpoint '/api'")
	startWebUI := flag.Bool("start_webui", true, "Set to start a web ui endpoint '/ui'")
	webuiDist := flag.String("webui_path", "", "Points to a static web ui dist path with the built version of ADK Web UI")

	flag.Parse()
	if flag.Parsed() == false {
		flag.Usage()
		panic("Failed to parse flags")
	}
	return &(WebConfig{
		LocalPort:      *localPortFlag,
		FrontEndServer: *frontendServerFlag,
		StartRestApi:   *startRespApi,
		StartWebUI:     *startWebUI,
		UIDistPath:     *webuiDist,
	})
}

// func logRequestHandler(h http.Handler) http.Handler {
// 	fn := func(w http.ResponseWriter, r *http.Request) {
// 		fmt.Println(r)
// 	}
// 	return http.HandlerFunc(fn)
// }

func Serve(c *WebConfig) {
	var serverConfig config.ADKAPIRouterConfigs
	serverConfig.Cors = *cors.New(cors.Options{
		AllowedOrigins:   []string{c.FrontEndServer},
		AllowedMethods:   []string{http.MethodGet, http.MethodPost, http.MethodOptions, http.MethodDelete, http.MethodPut},
		AllowCredentials: true})

	rBase := mux.NewRouter().StrictSlash(true)
	// rBase.Use(logRequestHandler)

	if c.StartWebUI {
		rUi := rBase.Methods("GET").PathPrefix("/ui/").Subrouter()
		rUi.Methods("GET").Handler(http.StripPrefix("/ui/", http.FileServer(http.Dir(c.UIDistPath))))
	}

	if c.StartRestApi {
		rApi := rBase.Methods("GET", "POST", "DELETE").PathPrefix("/api/").Subrouter()
		restapiweb.SetupRouter(rApi, &serverConfig)
	}

	log.Fatal(http.ListenAndServe(":"+strconv.Itoa(c.LocalPort), rBase))
}
