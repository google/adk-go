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

package a2a

// AgentCapabilities define optional capabilities supported by an agent.
type AgentCapabilities struct {
	// Extensions is a list of protocol extensions supported by the agent.
	Extensions []AgentExtension `json:"extensions,omitempty" yaml:"extensions,omitempty" mapstructure:"extensions,omitempty"`

	// PushNotifications indicates if the agent supports sending push notifications for asynchronous task updates.
	PushNotifications bool `json:"pushNotifications,omitempty" yaml:"pushNotifications,omitempty" mapstructure:"pushNotifications,omitempty"`

	// StateTransitionHistory indicates if the agent provides a history of state transitions for a task.
	StateTransitionHistory bool `json:"stateTransitionHistory,omitempty" yaml:"stateTransitionHistory,omitempty" mapstructure:"stateTransitionHistory,omitempty"`

	// Streaming indicates if the agent supports streaming responses.
	Streaming bool `json:"streaming,omitempty" yaml:"streaming,omitempty" mapstructure:"streaming,omitempty"`
}

// SecurityRequirements describes a set of security requirements that must be present on a request.
// For example, to specify that mutual TLS AND an oauth2 token for specific scopes is required, the
// following requirements object needs to be created:
//
//	SecurityRequirements{
//		SecuritySchemeName("oauth2"): SecuritySchemeScopes{"read", "write"},
//		SecuritySchemeName("mTLS"): {}
//	}
type SecurityRequirements map[SecuritySchemeName]SecuritySchemeScopes

// AgentCard is a self-describing manifest for an agent. It provides essential
// metadata including the agent's identity, capabilities, skills, supported
// communication methods, and security requirements.
type AgentCard struct {
	// AdditionalInterfaces is a list of additional supported transport and URL combinations.
	// This allows agents to expose multiple transports, potentially at different
	// URLs.
	//
	// Best practices:
	// - SHOULD include all supported transports for completeness
	// - SHOULD include an entry matching the main 'url' and 'preferredTransport'
	// - MAY reuse URLs if multiple transports are available at the same endpoint
	// - MUST accurately declare the transport available at each URL
	//
	// Clients can select any interface from this list based on their transport capabilities
	// and preferences. This enables transport negotiation and fallback scenarios.
	AdditionalInterfaces []AgentInterface `json:"additionalInterfaces,omitempty" yaml:"additionalInterfaces,omitempty" mapstructure:"additionalInterfaces,omitempty"`

	// Capabilities is a declaration of optional capabilities supported by the agent.
	Capabilities AgentCapabilities `json:"capabilities" yaml:"capabilities" mapstructure:"capabilities"`

	// DefaultInputModes a default set of supported input MIME types for all skills, which can be
	// overridden on a per-skill basis.
	DefaultInputModes []string `json:"defaultInputModes" yaml:"defaultInputModes" mapstructure:"defaultInputModes"`

	// DefaultOutputModes is a default set of supported output MIME types for all skills, which can be
	// overridden on a per-skill basis.
	DefaultOutputModes []string `json:"defaultOutputModes" yaml:"defaultOutputModes" mapstructure:"defaultOutputModes"`

	// Description is a human-readable description of the agent, assisting users and other agents
	// in understanding its purpose.
	Description string `json:"description" yaml:"description" mapstructure:"description"`

	// DocumentationURL is an optional URL to the agent's documentation.
	DocumentationURL string `json:"documentationUrl,omitempty" yaml:"documentationUrl,omitempty" mapstructure:"documentationUrl,omitempty"`

	// IconURL is an optional URL to an icon for the agent.
	IconURL string `json:"iconUrl,omitempty" yaml:"iconUrl,omitempty" mapstructure:"iconUrl,omitempty"`

	// Name is a human-readable name for the agent.
	Name string `json:"name" yaml:"name" mapstructure:"name"`

	// PreferredTransport is the transport protocol for the preferred endpoint (the main 'url' field).
	// If not specified, defaults to 'JSONRPC'.
	//
	// IMPORTANT: The transport specified here MUST be available at the main 'url'.
	// This creates a binding between the main URL and its supported transport protocol.
	// Clients should prefer this transport and URL combination when both are supported.
	PreferredTransport TransportProtocol `json:"preferredTransport,omitempty" yaml:"preferredTransport,omitempty" mapstructure:"preferredTransport,omitempty"`

	// ProtocolVersion is the version of the A2A protocol this agent supports.
	ProtocolVersion string `json:"protocolVersion" yaml:"protocolVersion" mapstructure:"protocolVersion"`

	// Provider contains information about the agent's service provider.
	Provider *AgentProvider `json:"provider,omitempty" yaml:"provider,omitempty" mapstructure:"provider,omitempty"`

	// Security is a list of security requirement objects that apply to all agent interactions.
	// Each object lists security schemes that can be used.
	// Follows the OpenAPI 3.0 Security Requirement Object.
	// This list can be seen as an OR of ANDs. Each object in the list describes one
	// possible set of security requirements that must be present on a request.
	// This allows specifying, for example, "callers must either use OAuth OR an API Key AND mTLS.":
	//
	// Security: []SecurityRequirements{
	//		{"oauth2": SecuritySchemeScopes{"read"}},
	// 		{"mTLS": SecuritySchemeScopes{}, "apiKey": SecuritySchemeScopes{"read"}}
	// }
	Security []SecurityRequirements `json:"security,omitempty" yaml:"security,omitempty" mapstructure:"security,omitempty"`

	// SecuritySchemes is a declaration of the security schemes available to authorize requests. The key
	// is the scheme name. Follows the OpenAPI 3.0 Security Scheme Object.
	SecuritySchemes NamedSecuritySchemes `json:"securitySchemes,omitempty" yaml:"securitySchemes,omitempty" mapstructure:"securitySchemes,omitempty"`

	// Signatures is a list of JSON Web Signatures computed for this AgentCard.
	Signatures []AgentCardSignature `json:"signatures,omitempty" yaml:"signatures,omitempty" mapstructure:"signatures,omitempty"`

	// Skills is the set of skills, or distinct capabilities, that the agent can perform.
	Skills []AgentSkill `json:"skills" yaml:"skills" mapstructure:"skills"`

	// SupportsAuthenticatedExtendedCard indicates if the agent can provide an extended agent card with additional details
	// to authenticated users. Defaults to false.
	SupportsAuthenticatedExtendedCard bool `json:"supportsAuthenticatedExtendedCard,omitempty" yaml:"supportsAuthenticatedExtendedCard,omitempty" mapstructure:"supportsAuthenticatedExtendedCard,omitempty"`

	// URL is the preferred endpoint URL for interacting with the agent.
	// This URL MUST support the transport specified by 'preferredTransport'.
	URL string `json:"url" yaml:"url" mapstructure:"url"`

	// Version is the agent's own version number. The format is defined by the provider.
	Version string `json:"version" yaml:"version" mapstructure:"version"`
}

// AgentCardSignature represents a JWS signature of an AgentCard.
// This follows the JSON format of an RFC 7515 JSON Web Signature (JWS).
type AgentCardSignature struct {
	// Header is the unprotected JWS header values.
	Header map[string]any `json:"header,omitempty" yaml:"header,omitempty" mapstructure:"header,omitempty"`

	// Protected is a JWS header for the signature. This is a Base64url-encoded
	// JSON object, as per RFC 7515.
	Protected string `json:"protected" yaml:"protected" mapstructure:"protected"`

	// Signature is the computed signature, Base64url-encoded.
	Signature string `json:"signature" yaml:"signature" mapstructure:"signature"`
}

// AgentExtension is a declaration of a protocol extension supported by an Agent.
type AgentExtension struct {
	// Description is an optional human-readable description of how this agent uses the extension.
	Description string `json:"description,omitempty" yaml:"description,omitempty" mapstructure:"description,omitempty"`

	// Params are optional, extension-specific configuration parameters.
	Params map[string]any `json:"params,omitempty" yaml:"params,omitempty" mapstructure:"params,omitempty"`

	// Required indicates if the client must understand and comply with the extension's
	// requirements to interact with the agent.
	Required bool `json:"required,omitempty" yaml:"required,omitempty" mapstructure:"required,omitempty"`

	// URI is the unique URI identifying the extension.
	URI string `json:"uri" yaml:"uri" mapstructure:"uri"`
}

// AgentInterface declares a combination of a target URL and a transport protocol for interacting
// with the agent.
// This allows agents to expose the same functionality over multiple transport mechanisms.
type AgentInterface struct {
	// Transport is the transport protocol supported at this URL.
	Transport TransportProtocol `json:"transport" yaml:"transport" mapstructure:"transport"`

	// URL is the URL where this interface is available.
	// Must be a valid absolute HTTPS URL in production.
	URL string `json:"url" yaml:"url" mapstructure:"url"`
}

// AgentProvider represents the service provider of an agent.
type AgentProvider struct {
	// Org is the name of the agent provider's organization.
	Org string `json:"organization" yaml:"organization" mapstructure:"organization"`

	// URL is a URL for the agent provider's website or relevant documentation.
	URL string `json:"url" yaml:"url" mapstructure:"url"`
}

// AgentSkill represents a distinct capability or function that an agent can perform.
type AgentSkill struct {
	// Description is a detailed description of the skill, intended to help clients or users
	// understand its purpose and functionality.
	Description string `json:"description" yaml:"description" mapstructure:"description"`

	// Examples are prompts or scenarios that this skill can handle. Provides a hint to
	// the client on how to use the skill.
	Examples []string `json:"examples,omitempty" yaml:"examples,omitempty" mapstructure:"examples,omitempty"`

	// ID is a unique identifier for the agent's skill.
	ID string `json:"id" yaml:"id" mapstructure:"id"`

	// InputModes is the set of supported input MIME types for this skill, overriding the agent's defaults.
	InputModes []string `json:"inputModes,omitempty" yaml:"inputModes,omitempty" mapstructure:"inputModes,omitempty"`

	// Name is a human-readable name for the skill.
	Name string `json:"name" yaml:"name" mapstructure:"name"`

	// OutputModes is the set of supported output MIME types for this skill, overriding the agent's defaults.
	OutputModes []string `json:"outputModes,omitempty" yaml:"outputModes,omitempty" mapstructure:"outputModes,omitempty"`

	// Security is a map of schemes necessary for the agent to leverage this skill.
	// As in the overall AgentCard.security, this list represents a logical OR of
	// security requirement objects.
	// Each object is a set of security schemes that must be used together (a logical AND).
	Security []SecurityRequirements `json:"security,omitempty" yaml:"security,omitempty" mapstructure:"security,omitempty"`

	// Tags is a set of keywords describing the skill's capabilities.
	Tags []string `json:"tags" yaml:"tags" mapstructure:"tags"`
}

// TransportProtocol represents a transport protocol which a client and an agent can use
// for communication. Custom protocols are allowed and the type MUST NOT be treated as an enum.
type TransportProtocol string

const (
	TransportProtocolJSONRPC  TransportProtocol = "JSONRPC"
	TransportProtocolGRPC     TransportProtocol = "GRPC"
	TransportProtocolHTTPJSON TransportProtocol = "HTTP+JSON"
)
