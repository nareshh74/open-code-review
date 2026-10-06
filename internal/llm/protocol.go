// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"fmt"
	"strings"
)

// Canonical protocol identifiers understood by the LLM client factory and
// resolver. These are the only values produced by NormalizeProtocol for known
// protocols; downstream code (NewLLMClient switch, resolver branches) compares
// against these constants exclusively.
//
// Naming convention: <vendor>-<flavor>. New built-in protocols should add a
// constant and protocol definition here, then add a case to NewLLMClient.
const (
	// ProtocolAnthropic is the Anthropic Messages API spoken directly to
	// api.anthropic.com (or a compatible gateway).
	ProtocolAnthropic = "anthropic"
	// ProtocolOpenAIChatCompletions is the OpenAI Chat Completions API
	// (/v1/chat/completions). The value "openai" is kept for full backward
	// compatibility with existing config files.
	ProtocolOpenAIChatCompletions = "openai"
	// ProtocolOpenAIResponses is the OpenAI Responses API (/v1/responses),
	// used by GPT-5.x / o-series models.
	ProtocolOpenAIResponses = "openai-responses"
	// ProtocolAnthropicBedrock is the Anthropic Messages API served by AWS
	// Bedrock. The request body is the same as ProtocolAnthropic — the
	// difference is transport: requests are SigV4-signed from the ambient AWS
	// credential chain rather than carrying an API key, the model moves from
	// the body into the URL path, and the region determines the host. The
	// official SDK's bedrock middleware performs that rewriting, so this
	// shares the Anthropic client rather than reimplementing the protocol.
	ProtocolAnthropicBedrock = "anthropic-bedrock"
	// ProtocolCopilotACP is the Agent Client Protocol served by the installed
	// GitHub Copilot CLI over a subprocess's standard input and output.
	ProtocolCopilotACP = "copilot-acp"
	// ProtocolCopilotAPI is the experimental, undocumented GitHub Copilot chat
	// completions service. A GitHub credential is exchanged for a short-lived
	// Copilot token before each session.
	ProtocolCopilotAPI = "copilot-api"
)

// CredentialSource identifies who owns authentication for a protocol.
type CredentialSource string

const (
	CredentialAPIKey     CredentialSource = "api-key"
	CredentialAWS        CredentialSource = "aws"
	CredentialCopilotCLI CredentialSource = "copilot-cli"
)

type protocolDefinition struct {
	name              string
	credentials       CredentialSource
	requiresURL       bool
	builtInOnly       bool
	acceptsAWSOptions bool
}

var protocolDefinitions = []protocolDefinition{
	{name: ProtocolAnthropic, credentials: CredentialAPIKey, requiresURL: true},
	{name: ProtocolOpenAIChatCompletions, credentials: CredentialAPIKey, requiresURL: true},
	{name: ProtocolOpenAIResponses, credentials: CredentialAPIKey, requiresURL: true},
	{name: ProtocolAnthropicBedrock, credentials: CredentialAWS, acceptsAWSOptions: true},
	{name: ProtocolCopilotACP, credentials: CredentialCopilotCLI, builtInOnly: true},
	{name: ProtocolCopilotAPI, credentials: CredentialAPIKey, builtInOnly: true},
}

func lookupProtocol(name string) (protocolDefinition, bool) {
	for _, definition := range protocolDefinitions {
		if definition.name == name {
			return definition, true
		}
	}
	return protocolDefinition{}, false
}

// CredentialSourceForProtocol returns the authentication owner for a canonical
// protocol. Unknown protocols use API-key requirements until validation rejects
// them.
func CredentialSourceForProtocol(protocol string) CredentialSource {
	if definition, ok := lookupProtocol(protocol); ok {
		return definition.credentials
	}
	return CredentialAPIKey
}

// ProtocolRequiresURL reports whether a protocol addresses an HTTP endpoint.
func ProtocolRequiresURL(protocol string) bool {
	definition, ok := lookupProtocol(protocol)
	return !ok || definition.requiresURL
}

// ProtocolAcceptsAWSOptions reports whether aws_profile and aws_region apply.
func ProtocolAcceptsAWSOptions(protocol string) bool {
	definition, ok := lookupProtocol(protocol)
	return ok && definition.acceptsAWSOptions
}

// ProtocolIsBuiltInOnly reports whether custom and legacy endpoint
// configuration must reject the protocol.
func ProtocolIsBuiltInOnly(protocol string) bool {
	definition, ok := lookupProtocol(protocol)
	return ok && definition.builtInOnly
}

func supportedProtocols() []string {
	protocols := make([]string, len(protocolDefinitions))
	for i, definition := range protocolDefinitions {
		protocols[i] = definition.name
	}
	return protocols
}

// NormalizeProtocol canonicalizes protocol names. It is case-insensitive and
// trims whitespace. Empty string is returned as-is (the caller decides the
// default). Known protocol names are mapped to their canonical constants;
// unknown values are lowercased and trimmed so that ValidateProtocol can
// surface a precise error message rather than silently swallowing a typo.
func NormalizeProtocol(raw string) string {
	normalized := strings.ToLower(strings.TrimSpace(raw))
	switch normalized {
	case "":
		return ""
	case ProtocolAnthropic:
		return ProtocolAnthropic
	case ProtocolOpenAIChatCompletions:
		return ProtocolOpenAIChatCompletions
	case ProtocolOpenAIResponses:
		return ProtocolOpenAIResponses
	case ProtocolAnthropicBedrock:
		return ProtocolAnthropicBedrock
	case ProtocolCopilotACP:
		return ProtocolCopilotACP
	case ProtocolCopilotAPI:
		return ProtocolCopilotAPI
	default:
		return normalized
	}
}

// ValidateProtocol accepts canonical protocol names and rejects everything else.
func ValidateProtocol(p string) error {
	if _, ok := lookupProtocol(p); ok {
		return nil
	}
	protocols := supportedProtocols()
	for i := range protocols {
		protocols[i] = fmt.Sprintf("%q", protocols[i])
	}
	return fmt.Errorf("unsupported protocol %q; supported protocols are %s", p, strings.Join(protocols, ", "))
}
