// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"strings"
	"testing"
)

func TestNormalizeProtocol(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{"empty stays empty", "", ""},
		{"canonical anthropic is idempotent", ProtocolAnthropic, ProtocolAnthropic},
		{"canonical openai is idempotent", ProtocolOpenAIChatCompletions, ProtocolOpenAIChatCompletions},
		{"canonical openai-responses is idempotent", ProtocolOpenAIResponses, ProtocolOpenAIResponses},
		{"canonical copilot-acp is idempotent", ProtocolCopilotACP, ProtocolCopilotACP},
		{"anthropic case-insensitive", "ANTHROPIC", ProtocolAnthropic},
		{"openai-responses case-insensitive", "OpenAI-Responses", ProtocolOpenAIResponses},
		{"unknown passthrough lowercased", "gRPC", "grpc"},
		{"unknown anthropic-vertex preserved", "anthropic-vertex", "anthropic-vertex"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NormalizeProtocol(tt.raw); got != tt.want {
				t.Errorf("NormalizeProtocol(%q) = %q, want %q", tt.raw, got, tt.want)
			}
		})
	}
}

func TestValidateProtocol(t *testing.T) {
	tests := []struct {
		name    string
		p       string
		wantErr bool
		errSub  string
	}{
		{"anthropic ok", ProtocolAnthropic, false, ""},
		{"openai ok", ProtocolOpenAIChatCompletions, false, ""},
		{"openai-responses ok", ProtocolOpenAIResponses, false, ""},
		{"copilot-acp ok", ProtocolCopilotACP, false, ""},
		{"empty rejected", "", true, "unsupported protocol"},
		{"grpc rejected", "grpc", true, "unsupported protocol"},
		{"anthropic-vertex rejected", "anthropic-vertex", true, "unsupported protocol"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateProtocol(tt.p)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ValidateProtocol(%q) returned nil, want error", tt.p)
				}
				if tt.errSub != "" && !strings.Contains(err.Error(), tt.errSub) {
					t.Errorf("ValidateProtocol(%q) error = %q, want substring %q", tt.p, err.Error(), tt.errSub)
				}
				return
			}
			if err != nil {
				t.Errorf("ValidateProtocol(%q) returned unexpected error: %v", tt.p, err)
			}
		})
	}
}

// TestValidateProtocol_ErrorMessageListsAllProtocols makes sure the error
// message enumerates every canonical name so users discover openai-responses
// from any typo.
func TestValidateProtocol_ErrorMessageListsAllProtocols(t *testing.T) {
	err := ValidateProtocol("grpc")
	if err == nil {
		t.Fatal("expected error")
	}
	for _, sub := range []string{ProtocolAnthropic, ProtocolOpenAIChatCompletions, ProtocolOpenAIResponses, ProtocolAnthropicBedrock, ProtocolCopilotACP} {
		if !strings.Contains(err.Error(), sub) {
			t.Errorf("error %q should mention %q", err.Error(), sub)
		}
	}
}

func TestProtocolCredentialSources(t *testing.T) {
	tests := []struct {
		protocol    string
		credentials CredentialSource
		requiresURL bool
		acceptsAWS  bool
		builtInOnly bool
	}{
		{ProtocolAnthropic, CredentialAPIKey, true, false, false},
		{ProtocolOpenAIChatCompletions, CredentialAPIKey, true, false, false},
		{ProtocolOpenAIResponses, CredentialAPIKey, true, false, false},
		{ProtocolAnthropicBedrock, CredentialAWS, false, true, false},
		{ProtocolCopilotACP, CredentialCopilotCLI, false, false, true},
	}
	for _, test := range tests {
		t.Run(test.protocol, func(t *testing.T) {
			if got := CredentialSourceForProtocol(test.protocol); got != test.credentials {
				t.Errorf("CredentialSourceForProtocol = %q, want %q", got, test.credentials)
			}
			if got := ProtocolRequiresURL(test.protocol); got != test.requiresURL {
				t.Errorf("ProtocolRequiresURL = %t, want %t", got, test.requiresURL)
			}
			if got := ProtocolAcceptsAWSOptions(test.protocol); got != test.acceptsAWS {
				t.Errorf("ProtocolAcceptsAWSOptions = %t, want %t", got, test.acceptsAWS)
			}
			if got := ProtocolIsBuiltInOnly(test.protocol); got != test.builtInOnly {
				t.Errorf("ProtocolIsBuiltInOnly = %t, want %t", got, test.builtInOnly)
			}
		})
	}
}
