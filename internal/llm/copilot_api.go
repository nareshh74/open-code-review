// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go/v3"
)

const (
	copilotAPIDefaultGitHubURL = "https://api.github.com"
	copilotAPIDiscoverTimeout  = 15 * time.Second
	copilotAPIDiscoverMaxBody  = 64 << 10
)

// copilotAPIHeaders are the integration headers the private Copilot completion
// endpoint expects. Users cannot override them through extra_headers.
var copilotAPIHeaders = map[string]string{
	"Editor-Version":         "vscode/1.104.1",
	"Copilot-Integration-Id": "vscode-chat",
	"Openai-Intent":          "conversation-edits",
	"X-Initiator":            "agent",
}

// copilotAPIClient asks the GitHub API (github.com or a GHE.com tenant) for the
// account's Copilot endpoint, then delegates chat completions to OpenAIClient
// using the GitHub OAuth token as the bearer.
type copilotAPIClient struct {
	cfg       ClientConfig
	github    string
	githubURL string
	http      *http.Client
	checkBase func(base, githubURL string) error

	mu   sync.Mutex
	base string
	chat *OpenAIClient
}

type copilotUserResponse struct {
	ChatEnabled bool `json:"chat_enabled"`
	Endpoints   struct {
		API string `json:"api"`
	} `json:"endpoints"`
}

func newCopilotAPIClient(cfg ClientConfig) *copilotAPIClient {
	github := strings.TrimSpace(cfg.APIKey)
	githubURL := strings.TrimRight(cfg.URL, "/")
	if githubURL == "" {
		githubURL = copilotAPIDefaultGitHubURL
	}
	cfg.APIKey = ""
	return &copilotAPIClient{
		cfg:       cfg,
		github:    github,
		githubURL: githubURL,
		http: &http.Client{
			Timeout: copilotAPIDiscoverTimeout,
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		checkBase: validateCopilotAPIBase,
	}
}

func (c *copilotAPIClient) CompletionsWithCtx(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	chat, err := c.currentChat(ctx)
	if err != nil {
		finalizeRequest(ctx, c.cfg.retryCollector, err)
		return nil, err
	}
	resp, err := chat.CompletionsWithCtx(ctx, req)
	var apiErr *openai.Error
	if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusUnauthorized {
		// Rediscover on the next call; the request itself is not replayed.
		c.mu.Lock()
		if c.chat == chat {
			c.chat = nil
		}
		c.mu.Unlock()
		return nil, fmt.Errorf("copilot-api: Copilot rejected the GitHub credential (HTTP 401); refresh it and rerun: %w", err)
	}
	return resp, err
}

// EndpointURL reports the discovered Copilot base URL, or the GitHub API URL
// before discovery.
func (c *copilotAPIClient) EndpointURL() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.chat != nil {
		return c.base
	}
	return c.githubURL
}

// ponytail: one mutex serializes discovery; it runs once per client.
func (c *copilotAPIClient) currentChat(ctx context.Context) (*OpenAIClient, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.chat != nil {
		return c.chat, nil
	}
	if c.github == "" {
		return nil, errors.New("copilot-api: no GitHub credential; set providers.copilot-api.api_key_cmd, api_key, or COPILOT_GITHUB_TOKEN")
	}
	user, err := c.discover(ctx)
	if err != nil {
		return nil, err
	}
	base := strings.TrimRight(user.Endpoints.API, "/")
	if err := c.checkBase(base, c.githubURL); err != nil {
		return nil, fmt.Errorf("copilot-api: GitHub returned an unusable Copilot endpoint: %w", err)
	}
	chatCfg := c.cfg
	chatCfg.URL = base
	chatCfg.APIKey = c.github
	chatCfg.ExtraHeaders = make(map[string]string, len(c.cfg.ExtraHeaders)+len(copilotAPIHeaders))
	for k, v := range c.cfg.ExtraHeaders {
		chatCfg.ExtraHeaders[k] = v
	}
	for k, v := range copilotAPIHeaders {
		chatCfg.ExtraHeaders[k] = v
	}
	c.base = base
	c.chat = NewOpenAIClient(chatCfg)
	return c.chat, nil
}

func (c *copilotAPIClient) discover(ctx context.Context) (copilotUserResponse, error) {
	var user copilotUserResponse
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.githubURL+"/copilot_internal/user", nil)
	if err != nil {
		return user, fmt.Errorf("copilot-api: endpoint discovery: %w", err)
	}
	req.Header.Set("Authorization", "token "+c.github)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", userAgent("copilot-api"))
	resp, err := c.http.Do(req)
	if err != nil {
		return user, fmt.Errorf("copilot-api: endpoint discovery: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, copilotAPIDiscoverMaxBody+1))
	if err != nil {
		return user, fmt.Errorf("copilot-api: endpoint discovery: read response: %w", err)
	}
	switch {
	case resp.StatusCode == http.StatusUnauthorized:
		return user, errors.New("copilot-api: GitHub rejected the credential (HTTP 401); refresh it and check that providers.copilot-api.url names the credential's GitHub host")
	case resp.StatusCode == http.StatusForbidden:
		return user, errors.New("copilot-api: GitHub denied Copilot access (HTTP 403); check the account's Copilot entitlement and organization policy")
	case resp.StatusCode == http.StatusNotFound:
		return user, errors.New("copilot-api: no Copilot access found (HTTP 404); check the account and providers.copilot-api.url")
	case resp.StatusCode != http.StatusOK:
		return user, fmt.Errorf("copilot-api: endpoint discovery failed with HTTP %d", resp.StatusCode)
	case len(body) > copilotAPIDiscoverMaxBody:
		return user, errors.New("copilot-api: endpoint discovery response too large")
	}
	if err := json.Unmarshal(body, &user); err != nil {
		return user, errors.New("copilot-api: endpoint discovery returned malformed JSON")
	}
	if !user.ChatEnabled {
		return user, errors.New("copilot-api: Copilot Chat is not enabled for this account")
	}
	if user.Endpoints.API == "" {
		return user, errors.New("copilot-api: endpoint discovery returned no Copilot API endpoint")
	}
	return user, nil
}

// validateCopilotAPIBase accepts only HTTPS roots on Copilot hosts: an
// api*.githubcopilot.com host, or a host inside the configured GHE.com tenant.
// The GitHub credential is sent there, so nothing else passes.
func validateCopilotAPIBase(raw, githubURL string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return err
	}
	host := strings.ToLower(u.Hostname())
	switch {
	case u.Scheme != "https":
		return fmt.Errorf("scheme %q is not https", u.Scheme)
	case u.User != nil, u.RawQuery != "", u.Fragment != "", u.Port() != "":
		return errors.New("endpoint must not carry userinfo, query, fragment, or port")
	case u.Path != "" && u.Path != "/":
		return errors.New("endpoint must be a root URL")
	case net.ParseIP(host) != nil:
		return errors.New("endpoint must not be an IP address")
	case host == "api.githubcopilot.com",
		strings.HasPrefix(host, "api.") && strings.HasSuffix(host, ".githubcopilot.com"):
		return nil
	}
	if tenant := gheTenantDomain(githubURL); tenant != "" && strings.HasSuffix(host, "."+tenant) {
		return nil
	}
	return fmt.Errorf("host %q is not a Copilot API host for this GitHub instance", host)
}

// gheTenantDomain returns "<tenant>.ghe.com" for https://api.<tenant>.ghe.com,
// or "" otherwise.
func gheTenantDomain(githubURL string) string {
	u, err := url.Parse(githubURL)
	if err != nil {
		return ""
	}
	host := strings.ToLower(u.Hostname())
	tenant := strings.TrimPrefix(host, "api.")
	if tenant == host || !strings.HasSuffix(tenant, ".ghe.com") || strings.Count(tenant, ".") != 2 {
		return ""
	}
	return tenant
}

// ValidateCopilotGitHubURL checks providers.copilot-api.url, which names the
// GitHub API the credential belongs to: github.com or a GHE.com tenant.
func ValidateCopilotGitHubURL(raw string) error {
	u, err := url.Parse(strings.TrimRight(raw, "/"))
	ok := err == nil && u.Scheme == "https" && u.User == nil && u.Port() == "" &&
		u.Path == "" && u.RawQuery == "" && u.Fragment == "" &&
		(strings.EqualFold(u.Host, "api.github.com") || gheTenantDomain(raw) != "")
	if !ok {
		return fmt.Errorf("url must be %s or https://api.<tenant>.ghe.com", copilotAPIDefaultGitHubURL)
	}
	return nil
}
