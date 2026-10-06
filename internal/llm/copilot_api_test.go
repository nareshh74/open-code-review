// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

type copilotAPIFake struct {
	server      *httptest.Server
	discoveries atomic.Int32
	bodies      []map[string]any
	chatAuth    []string
	chatHdr     http.Header
	chatCode    int
	exStatus    int
	exBody      string
}

func newCopilotAPIFake(t *testing.T) *copilotAPIFake {
	t.Helper()
	f := &copilotAPIFake{chatCode: http.StatusOK, exStatus: http.StatusOK}
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/copilot_internal/user":
			f.discoveries.Add(1)
			if got := r.Header.Get("Authorization"); got != "token gho_raw" {
				t.Errorf("discovery Authorization = %q", got)
			}
			w.WriteHeader(f.exStatus)
			if f.exBody != "" {
				_, _ = io.WriteString(w, f.exBody)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"chat_enabled": true,
				"endpoints":    map[string]string{"api": "http://" + r.Host},
			})
		case "/chat/completions":
			f.chatAuth = append(f.chatAuth, r.Header.Get("Authorization"))
			f.chatHdr = r.Header.Clone()
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			f.bodies = append(f.bodies, body)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(f.chatCode)
			if f.chatCode != http.StatusOK {
				_, _ = io.WriteString(w, `{"error":{"message":"bad token"}}`)
				return
			}
			if len(f.bodies) == 1 {
				_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"add","arguments":"{ \"n\": 2.00 }"}}]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
				return
			}
			_, _ = io.WriteString(w, `{"id":"2","object":"chat.completion","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"done"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		default:
			t.Errorf("unexpected path %q", r.URL.Path)
		}
	}))
	t.Cleanup(f.server.Close)
	return f
}

func (f *copilotAPIFake) client(t *testing.T) *copilotAPIClient {
	t.Helper()
	ep := ResolvedEndpoint{Protocol: ProtocolCopilotAPI, Token: " gho_raw\n", Model: "m",
		ExtraHeaders: map[string]string{"X-Trace": "1", "Copilot-Integration-Id": "user"}}
	c, ok := NewLLMClient(ep, nil, nil).(*copilotAPIClient)
	if !ok {
		t.Fatal("factory did not return copilotAPIClient")
	}
	c.githubURL = f.server.URL
	c.checkBase = func(string, string) error { return nil }
	return c
}

func TestCopilotAPIDiscoversAndReplaysToolResults(t *testing.T) {
	f := newCopilotAPIFake(t)
	c := f.client(t)
	if c.cfg.APIKey != "" {
		t.Fatal("raw GitHub credential retained in generic config")
	}
	ctx := context.Background()
	tools := []ToolDef{{Type: "function", Function: FunctionDef{Name: "add", Parameters: map[string]any{"type": "object"}}}}
	first, err := c.CompletionsWithCtx(ctx, ChatRequest{Messages: []Message{{Role: "user", Content: "hi"}}, Tools: tools})
	if err != nil {
		t.Fatal(err)
	}
	calls := first.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Function.Arguments != `{ "n": 2.00 }` {
		t.Fatalf("tool calls = %+v", calls)
	}
	second, err := c.CompletionsWithCtx(ctx, ChatRequest{Tools: tools, Messages: []Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: first.VisibleContent(), ToolCalls: calls},
		{Role: "tool", ToolCallID: "call_1", Content: "4"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if second.VisibleContent() != "done" {
		t.Fatalf("content = %q", second.VisibleContent())
	}
	if n := f.discoveries.Load(); n != 1 {
		t.Fatalf("discoveries = %d, want 1 (cached)", n)
	}
	for _, auth := range f.chatAuth {
		if auth != "Bearer gho_raw" {
			t.Fatalf("completion Authorization = %q", auth)
		}
	}
	if f.chatHdr.Get("Copilot-Integration-Id") != "vscode-chat" || f.chatHdr.Get("X-Trace") != "1" {
		t.Fatalf("completion headers = %v", f.chatHdr)
	}
	msgs, _ := json.Marshal(f.bodies[1]["messages"])
	if !strings.Contains(string(msgs), `"tool_call_id":"call_1"`) || !strings.Contains(string(msgs), `{ \"n\": 2.00 }`) {
		t.Fatalf("replayed messages = %s", msgs)
	}
	if got := c.EndpointURL(); got != f.server.URL {
		t.Fatalf("EndpointURL = %q", got)
	}
}

func TestCopilotAPICompletion401DropsSessionWithoutReplay(t *testing.T) {
	f := newCopilotAPIFake(t)
	f.chatCode = http.StatusUnauthorized
	c := f.client(t)
	_, err := c.CompletionsWithCtx(context.Background(), ChatRequest{Messages: []Message{{Role: "user", Content: "hi"}}})
	if err == nil || !strings.Contains(err.Error(), "HTTP 401") {
		t.Fatalf("err = %v", err)
	}
	if len(f.chatAuth) != 1 {
		t.Fatalf("completion attempts = %d, want 1", len(f.chatAuth))
	}
	if c.chat != nil {
		t.Fatal("client not discarded after 401")
	}
}

func TestCopilotAPIDiscoveryFailures(t *testing.T) {
	tests := []struct {
		name, body, want string
		status           int
	}{
		{"unauthorized", "", "HTTP 401", http.StatusUnauthorized},
		{"forbidden", "", "HTTP 403", http.StatusForbidden},
		{"not found", "", "HTTP 404", http.StatusNotFound},
		{"server", "", "HTTP 502", http.StatusBadGateway},
		{"redirect not followed", "", "HTTP 302", http.StatusFound},
		{"malformed", "{", "malformed JSON", http.StatusOK},
		{"chat disabled", `{"chat_enabled":false,"endpoints":{"api":"https://api.githubcopilot.com"}}`, "not enabled", http.StatusOK},
		{"no endpoint", `{"chat_enabled":true}`, "no Copilot API endpoint", http.StatusOK},
		{"too large", `{"x":"` + strings.Repeat("a", copilotAPIDiscoverMaxBody) + `"}`, "too large", http.StatusOK},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newCopilotAPIFake(t)
			f.exStatus, f.exBody = tt.status, tt.body
			if tt.status != http.StatusOK && tt.body == "" {
				f.exBody = `{"message":"secret-bearing body"}`
			}
			_, err := f.client(t).CompletionsWithCtx(context.Background(), ChatRequest{})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
			if strings.Contains(err.Error(), "gho_raw") || strings.Contains(err.Error(), "secret-bearing") {
				t.Fatalf("error leaks credential or body: %v", err)
			}
			if len(f.chatAuth) != 0 {
				t.Fatal("completion sent after failed discovery")
			}
		})
	}
}

func TestCopilotAPIRejectsUnsafeDiscoveredEndpoint(t *testing.T) {
	f := newCopilotAPIFake(t)
	c := f.client(t)
	c.checkBase = validateCopilotAPIBase // the fake reports an http:// loopback endpoint
	_, err := c.CompletionsWithCtx(context.Background(), ChatRequest{})
	if err == nil || !strings.Contains(err.Error(), "unusable Copilot endpoint") {
		t.Fatalf("err = %v", err)
	}
	if len(f.chatAuth) != 0 {
		t.Fatal("bearer sent to unvalidated endpoint")
	}
}

func TestCopilotAPIRequiresCredential(t *testing.T) {
	c := newCopilotAPIClient(ClientConfig{APIKey: "  "})
	_, err := c.CompletionsWithCtx(context.Background(), ChatRequest{})
	if err == nil || !strings.Contains(err.Error(), "COPILOT_GITHUB_TOKEN") {
		t.Fatalf("err = %v", err)
	}
	if c.EndpointURL() != copilotAPIDefaultGitHubURL {
		t.Fatalf("EndpointURL = %q", c.EndpointURL())
	}
}

func TestValidateCopilotAPIBase(t *testing.T) {
	const ghe = "https://api.contoso.ghe.com"
	ok := []string{"https://api.githubcopilot.com", "https://api.individual.githubcopilot.com/", "https://API.Business.GitHubCopilot.com", "https://copilot-api.contoso.ghe.com"}
	bad := []string{
		"http://api.githubcopilot.com", "https://api.githubcopilot.com:8443", "https://u@api.githubcopilot.com",
		"https://api.githubcopilot.com/v1", "https://api.githubcopilot.com?x=1", "https://api.githubcopilot.com#f",
		"https://evil.com", "https://api.githubcopilot.com.evil.com", "https://apigithubcopilot.com",
		"https://proxy.githubcopilot.com", "https://1.2.3.4", "::bad",
		"https://copilot-api.other.ghe.com", "https://copilot-api.contoso.ghe.com.evil.com", "https://contoso.ghe.com",
	}
	for _, u := range ok {
		if err := validateCopilotAPIBase(u, ghe); err != nil {
			t.Errorf("%q rejected: %v", u, err)
		}
	}
	for _, u := range bad {
		if err := validateCopilotAPIBase(u, ghe); err == nil {
			t.Errorf("%q accepted", u)
		}
	}
}

func TestValidateCopilotGitHubURL(t *testing.T) {
	for _, u := range []string{"https://api.github.com", "https://api.github.com/", "https://api.contoso.ghe.com"} {
		if err := ValidateCopilotGitHubURL(u); err != nil {
			t.Errorf("%q rejected: %v", u, err)
		}
	}
	for _, u := range []string{"http://api.github.com", "https://evil.com", "https://contoso.ghe.com", "https://api.a.b.ghe.com", "https://api.github.com/x", "https://api.github.com:444"} {
		if err := ValidateCopilotGitHubURL(u); err == nil {
			t.Errorf("%q accepted", u)
		}
	}
}
