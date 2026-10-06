// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

const (
	copilotACPMaxFrameBytes  = 1 << 20
	copilotACPMaxOutputBytes = 4 << 20
	copilotACPMaxStderrBytes = 64 << 10
	copilotACPShutdownGrace  = 500 * time.Millisecond
)

type acpPhase string

const (
	acpPhaseStart      acpPhase = "start"
	acpPhaseInitialize acpPhase = "initialize"
	acpPhaseSessionNew acpPhase = "session/new"
	acpPhaseModel      acpPhase = "session/set_config_option"
	acpPhasePrompt     acpPhase = "session/prompt"
	acpPhaseClose      acpPhase = "session/close"
	acpPhaseReply      acpPhase = "reply"
	acpPhaseReap       acpPhase = "reap"
)

type acpErrorKind string

const (
	acpErrorProcess  acpErrorKind = "process"
	acpErrorProtocol acpErrorKind = "protocol"
	acpErrorAuth     acpErrorKind = "authentication"
	acpErrorSession  acpErrorKind = "session"
	acpErrorExit     acpErrorKind = "exit"
	acpErrorContext  acpErrorKind = "context"
)

type copilotACPError struct {
	Phase    acpPhase
	Kind     acpErrorKind
	Detail   string
	Stderr   string
	ExitCode *int
	Cause    error
}

func (e *copilotACPError) Error() string {
	var message strings.Builder
	fmt.Fprintf(&message, "copilot-acp %s %s error", e.Phase, e.Kind)
	if e.Detail != "" {
		message.WriteString(": ")
		message.WriteString(e.Detail)
	}
	if e.ExitCode != nil {
		fmt.Fprintf(&message, " (exit code %d)", *e.ExitCode)
	}
	if e.Stderr != "" {
		message.WriteString("; stderr: ")
		message.WriteString(e.Stderr)
	}
	return message.String()
}

func (e *copilotACPError) Unwrap() error {
	return e.Cause
}

type copilotACPConfig struct {
	command string
	args    []string
	model   string
	env     []string
}

type copilotACPOutput struct {
	text      string
	reasoning string
	model     string
	stderr    string
}

type acpProtocolState struct {
	phase     acpPhase
	sessionID string
	nextID    int64
	text      strings.Builder
	reasoning strings.Builder
}

func (s *acpProtocolState) enter(next acpPhase) {
	s.phase = next
}

func (s *acpProtocolState) appendText(value string) error {
	if s.text.Len()+s.reasoning.Len()+len(value) > copilotACPMaxOutputBytes {
		return fmt.Errorf("assistant output exceeds %d bytes", copilotACPMaxOutputBytes)
	}
	s.text.WriteString(value)
	return nil
}

func (s *acpProtocolState) appendReasoning(value string) error {
	if s.text.Len()+s.reasoning.Len()+len(value) > copilotACPMaxOutputBytes {
		return fmt.Errorf("assistant output exceeds %d bytes", copilotACPMaxOutputBytes)
	}
	s.reasoning.WriteString(value)
	return nil
}

type acpWireEnvelope struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *acpWireError   `json:"error,omitempty"`
}

type acpWireError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

type acpInitializeResult struct {
	ProtocolVersion int             `json:"protocolVersion"`
	AgentInfo       acpAgentInfo    `json:"agentInfo"`
	AuthMethods     []acpAuthMethod `json:"authMethods"`
}

type acpAgentInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type acpAuthMethod struct {
	ID          string `json:"id"`
	Description string `json:"description"`
}

type acpSessionResult struct {
	SessionID     string            `json:"sessionId"`
	ConfigOptions []acpConfigOption `json:"configOptions"`
}

type acpConfigOption struct {
	ID           string            `json:"id"`
	Category     string            `json:"category"`
	CurrentValue string            `json:"currentValue"`
	Options      []acpConfigChoice `json:"options"`
}

type acpConfigChoice struct {
	Value string        `json:"value"`
	Meta  acpChoiceMeta `json:"_meta"`
}

type acpChoiceMeta struct {
	Enablement string `json:"copilotEnablement"`
}

type acpSessionUpdateParams struct {
	SessionID string           `json:"sessionId"`
	Update    acpSessionUpdate `json:"update"`
}

type acpSessionUpdate struct {
	Kind    string          `json:"sessionUpdate"`
	Content json.RawMessage `json:"content"`
}

type acpTextContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func runCopilotACP(ctx context.Context, cfg copilotACPConfig, prompt string) (out copilotACPOutput, err error) {
	if err := ctx.Err(); err != nil {
		return out, &copilotACPError{Phase: acpPhaseStart, Kind: acpErrorContext, Detail: err.Error(), Cause: err}
	}
	workingDir, err := os.MkdirTemp("", "ocr-copilot-acp-")
	if err != nil {
		return out, &copilotACPError{Phase: acpPhaseStart, Kind: acpErrorProcess, Detail: err.Error(), Cause: err}
	}
	defer os.RemoveAll(workingDir)

	process, err := startACPProcess(cfg, workingDir)
	if err != nil {
		return out, err
	}
	defer func() {
		if finishErr := process.finish(); err == nil && finishErr != nil {
			err = finishErr
		}
		out.stderr = process.stderr.String()
	}()

	state := acpProtocolState{phase: acpPhaseStart}
	var initialized acpInitializeResult
	state.enter(acpPhaseInitialize)
	if err = process.call(ctx, &state, "initialize", map[string]any{
		"protocolVersion":    1,
		"clientCapabilities": map[string]any{},
		"clientInfo": map[string]string{
			"name":    "open-code-review",
			"version": AppVersion,
		},
	}, &initialized); err != nil {
		return out, err
	}
	if initialized.ProtocolVersion != 1 {
		return out, process.failure(state.phase, acpErrorProtocol, fmt.Sprintf("protocol version is %d, want 1", initialized.ProtocolVersion), nil)
	}

	state.enter(acpPhaseSessionNew)
	var session acpSessionResult
	if err = process.call(ctx, &state, "session/new", map[string]any{
		"cwd":        workingDir,
		"mcpServers": []any{},
	}, &session); err != nil {
		if rpcErrorSignalsAuthentication(err) {
			return out, process.authFailure(state.phase, initialized.AuthMethods, err)
		}
		return out, err
	}
	state.sessionID = strings.TrimSpace(session.SessionID)
	if state.sessionID == "" {
		return out, process.failure(state.phase, acpErrorSession, "session/new returned an empty sessionId", nil)
	}

	selectedModel, selection, err := selectACPModel(session, state.sessionID, cfg.model)
	if err != nil {
		return out, process.failure(acpPhaseModel, acpErrorSession, err.Error(), err)
	}
	if selection != nil {
		state.enter(acpPhaseModel)
		var ignored map[string]any
		if err = process.call(ctx, &state, "session/set_config_option", selection, &ignored); err != nil {
			return out, err
		}
	}

	state.enter(acpPhasePrompt)
	var promptResult struct {
		StopReason string `json:"stopReason"`
	}
	if err = process.call(ctx, &state, "session/prompt", map[string]any{
		"sessionId": state.sessionID,
		"prompt": []map[string]string{{
			"type": "text",
			"text": prompt,
		}},
	}, &promptResult); err != nil {
		return out, err
	}

	state.enter(acpPhaseClose)
	var closeResult map[string]any
	if err = process.call(ctx, &state, "session/close", map[string]string{
		"sessionId": state.sessionID,
	}, &closeResult); err != nil {
		return out, err
	}

	return copilotACPOutput{
		text:      state.text.String(),
		reasoning: state.reasoning.String(),
		model:     selectedModel,
	}, nil
}

func selectACPModel(session acpSessionResult, sessionID, requested string) (string, map[string]string, error) {
	for _, option := range session.ConfigOptions {
		if option.ID != "model" && option.Category != "model" {
			continue
		}
		available := make([]string, 0, len(option.Options))
		seen := make(map[string]struct{}, len(option.Options))
		for _, choice := range option.Options {
			if choice.Value == "" || strings.EqualFold(choice.Meta.Enablement, "disabled") {
				continue
			}
			if _, exists := seen[choice.Value]; exists {
				continue
			}
			seen[choice.Value] = struct{}{}
			available = append(available, choice.Value)
		}
		if _, exists := seen[requested]; !exists {
			return "", nil, fmt.Errorf("requested model %q is not offered by this Copilot session; offered models: %s", requested, strings.Join(available, ", "))
		}
		if option.CurrentValue == requested {
			return reportedACPModel(requested), nil, nil
		}
		configID := option.ID
		if configID == "" {
			configID = "model"
		}
		return reportedACPModel(requested), map[string]string{
			"sessionId": sessionID,
			"configId":  configID,
			"value":     requested,
		}, nil
	}
	return "", nil, errors.New("session/new did not advertise a model configuration option")
}

func reportedACPModel(requested string) string {
	if requested == "auto" {
		return ""
	}
	return requested
}

type acpReadResult struct {
	data []byte
	err  error
}

type acpProcess struct {
	cmd         *exec.Cmd
	stdin       io.WriteCloser
	frames      <-chan acpReadResult
	stderr      *boundedTail
	waited      chan struct{}
	waitMu      sync.Mutex
	waitErr     error
	processTree *acpProcessTree
	finishOnce  sync.Once
	finishErr   error
	readerStop  chan struct{}
	readerDone  chan struct{}
	stderrDone  chan struct{}
}

func startACPProcess(cfg copilotACPConfig, workingDir string) (*acpProcess, error) {
	cmd := exec.Command(cfg.command, cfg.args...)
	cmd.Dir = workingDir
	cmd.Env = cfg.env
	configureACPCommand(cmd)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, &copilotACPError{Phase: acpPhaseStart, Kind: acpErrorProcess, Detail: err.Error(), Cause: err}
	}
	stdout, stdoutWriter, err := os.Pipe()
	if err != nil {
		_ = stdin.Close()
		return nil, &copilotACPError{Phase: acpPhaseStart, Kind: acpErrorProcess, Detail: err.Error(), Cause: err}
	}
	stderrPipe, stderrWriter, err := os.Pipe()
	if err != nil {
		_ = stdin.Close()
		_ = stdout.Close()
		_ = stdoutWriter.Close()
		return nil, &copilotACPError{Phase: acpPhaseStart, Kind: acpErrorProcess, Detail: err.Error(), Cause: err}
	}
	cmd.Stdout = stdoutWriter
	cmd.Stderr = stderrWriter
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		_ = stdout.Close()
		_ = stdoutWriter.Close()
		_ = stderrPipe.Close()
		_ = stderrWriter.Close()
		return nil, &copilotACPError{Phase: acpPhaseStart, Kind: acpErrorProcess, Detail: err.Error(), Cause: err}
	}
	_ = stdoutWriter.Close()
	_ = stderrWriter.Close()
	processTree, err := attachACPProcess(cmd)
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		_ = stdin.Close()
		_ = stdout.Close()
		_ = stderrPipe.Close()
		return nil, &copilotACPError{Phase: acpPhaseStart, Kind: acpErrorProcess, Detail: fmt.Sprintf("own process tree: %v", err), Cause: err}
	}

	frameChannel := make(chan acpReadResult, 1)
	readerStop := make(chan struct{})
	readerDone := make(chan struct{})
	go readACPFrames(stdout, frameChannel, readerStop, readerDone)
	stderr := newBoundedTail(copilotACPMaxStderrBytes)
	stderrDone := make(chan struct{})
	go func() {
		defer close(stderrDone)
		_, _ = io.Copy(stderr, stderrPipe)
	}()

	process := &acpProcess{
		cmd:         cmd,
		stdin:       stdin,
		frames:      frameChannel,
		stderr:      stderr,
		waited:      make(chan struct{}),
		processTree: processTree,
		readerStop:  readerStop,
		readerDone:  readerDone,
		stderrDone:  stderrDone,
	}
	go func() {
		err := cmd.Wait()
		process.waitMu.Lock()
		process.waitErr = err
		process.waitMu.Unlock()
		close(process.waited)
	}()
	return process, nil
}

func readACPFrames(reader io.Reader, sink chan<- acpReadResult, stop <-chan struct{}, done chan<- struct{}) {
	defer close(done)
	defer close(sink)
	send := func(result acpReadResult) bool {
		select {
		case sink <- result:
			return true
		case <-stop:
			return false
		}
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64<<10), copilotACPMaxFrameBytes)
	for scanner.Scan() {
		line := append([]byte(nil), scanner.Bytes()...)
		if !send(acpReadResult{data: line}) {
			return
		}
	}
	if err := scanner.Err(); err != nil {
		send(acpReadResult{err: fmt.Errorf("read stdout JSONL: %w", err)})
		return
	}
	send(acpReadResult{err: io.EOF})
}

func (p *acpProcess) call(ctx context.Context, state *acpProtocolState, method string, params any, result any) error {
	state.nextID++
	request := struct {
		JSONRPC string `json:"jsonrpc"`
		ID      int64  `json:"id"`
		Method  string `json:"method"`
		Params  any    `json:"params"`
	}{
		JSONRPC: "2.0",
		ID:      state.nextID,
		Method:  method,
		Params:  params,
	}
	data, err := json.Marshal(request)
	if err != nil {
		return p.failure(state.phase, acpErrorProtocol, fmt.Sprintf("encode %s request: %v", method, err), err)
	}
	data = append(data, '\n')
	if _, err := p.stdin.Write(data); err != nil {
		return p.failure(state.phase, acpErrorProcess, fmt.Sprintf("write %s request: %v", method, err), err)
	}

	for {
		select {
		case <-ctx.Done():
			return p.failure(state.phase, acpErrorContext, ctx.Err().Error(), ctx.Err())
		case frame, ok := <-p.frames:
			if !ok {
				return p.exitFailure(state.phase, "stdout closed before the response")
			}
			if frame.err != nil {
				if errors.Is(frame.err, io.EOF) {
					return p.exitFailure(state.phase, "stdout closed before the response")
				}
				return p.failure(state.phase, acpErrorProtocol, frame.err.Error(), frame.err)
			}
			envelope, err := decodeACPEnvelope(frame.data)
			if err != nil {
				return p.failure(state.phase, acpErrorProtocol, err.Error(), err)
			}
			if envelope.Method != "" {
				if err := p.handleServerMessage(state, envelope); err != nil {
					return p.failure(state.phase, acpErrorProtocol, err.Error(), err)
				}
				continue
			}
			responseID, err := decodeACPResponseID(envelope.ID)
			if err != nil {
				return p.failure(state.phase, acpErrorProtocol, err.Error(), err)
			}
			if responseID != state.nextID {
				return p.failure(state.phase, acpErrorProtocol, fmt.Sprintf("unexpected response ID %d, want %d", responseID, state.nextID), nil)
			}
			if envelope.Error != nil {
				return p.failure(state.phase, acpErrorSession, formatACPRPCError(envelope.Error), nil)
			}
			if len(envelope.Result) == 0 {
				return p.failure(state.phase, acpErrorProtocol, "response has neither result nor error", nil)
			}
			if err := json.Unmarshal(envelope.Result, result); err != nil {
				return p.failure(state.phase, acpErrorProtocol, fmt.Sprintf("decode %s result: %v", method, err), err)
			}
			return nil
		}
	}
}

func decodeACPEnvelope(data []byte) (acpWireEnvelope, error) {
	if len(data) > copilotACPMaxFrameBytes {
		return acpWireEnvelope{}, fmt.Errorf("stdout JSONL frame exceeds %d bytes", copilotACPMaxFrameBytes)
	}
	if err := validateUniqueJSON(data); err != nil {
		return acpWireEnvelope{}, fmt.Errorf("malformed stdout JSONL frame: %w", err)
	}
	var envelope acpWireEnvelope
	if err := json.Unmarshal(data, &envelope); err != nil {
		return acpWireEnvelope{}, fmt.Errorf("decode stdout JSONL frame: %w", err)
	}
	if envelope.JSONRPC != "2.0" {
		return acpWireEnvelope{}, fmt.Errorf("JSON-RPC version is %q, want %q", envelope.JSONRPC, "2.0")
	}
	return envelope, nil
}

func decodeACPResponseID(raw json.RawMessage) (int64, error) {
	if len(raw) == 0 {
		return 0, errors.New("response is missing an ID")
	}
	var id int64
	if err := json.Unmarshal(raw, &id); err != nil {
		return 0, fmt.Errorf("response ID is not an integer: %w", err)
	}
	return id, nil
}

func (p *acpProcess) handleServerMessage(state *acpProtocolState, envelope acpWireEnvelope) error {
	if envelope.Method == "session/update" {
		var params acpSessionUpdateParams
		if err := json.Unmarshal(envelope.Params, &params); err != nil {
			return fmt.Errorf("decode session/update: %w", err)
		}
		if state.phase != acpPhasePrompt {
			switch params.Update.Kind {
			case "agent_message_chunk", "agent_thought_chunk", "tool_call", "tool_call_update":
				return fmt.Errorf("%s received during %s", params.Update.Kind, state.phase)
			default:
				return nil
			}
		}
		if params.SessionID != state.sessionID {
			return fmt.Errorf("session/update used sessionId %q, want %q", params.SessionID, state.sessionID)
		}
		switch params.Update.Kind {
		case "agent_message_chunk", "agent_thought_chunk":
			var content acpTextContent
			if err := json.Unmarshal(params.Update.Content, &content); err != nil {
				return fmt.Errorf("decode %s content: %w", params.Update.Kind, err)
			}
			if content.Type != "text" {
				return fmt.Errorf("%s content type is %q, want text", params.Update.Kind, content.Type)
			}
			if params.Update.Kind == "agent_message_chunk" {
				return state.appendText(content.Text)
			}
			return state.appendReasoning(content.Text)
		case "tool_call", "tool_call_update":
			return fmt.Errorf("Copilot attempted native agent tool activity %q", params.Update.Kind)
		default:
			return nil
		}
	}

	if len(envelope.ID) == 0 {
		return nil
	}
	if envelope.Method == "session/request_permission" {
		return p.writeServerResponse(envelope.ID, map[string]any{
			"outcome": map[string]string{"outcome": "cancelled"},
		}, nil)
	}
	return p.writeServerResponse(envelope.ID, nil, &acpWireError{
		Code:    -32601,
		Message: "method not supported",
	})
}

func (p *acpProcess) writeServerResponse(id json.RawMessage, result any, rpcError *acpWireError) error {
	response := map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
	}
	if rpcError != nil {
		response["error"] = rpcError
	} else {
		response["result"] = result
	}
	data, err := json.Marshal(response)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = p.stdin.Write(data)
	return err
}

func (p *acpProcess) finish() error {
	p.finishOnce.Do(func() {
		_ = p.stdin.Close()
		forced := false
		timer := time.NewTimer(copilotACPShutdownGrace)
		defer timer.Stop()
		select {
		case <-p.waited:
		case <-timer.C:
			forced = true
			if err := p.processTree.terminate(); err != nil {
				p.finishErr = p.failure(acpPhaseReap, acpErrorProcess, fmt.Sprintf("terminate process tree: %v", err), err)
			}
			<-p.waited
		}
		if err := p.processTree.close(); p.finishErr == nil && err != nil {
			p.finishErr = p.failure(acpPhaseReap, acpErrorProcess, fmt.Sprintf("close process tree: %v", err), err)
		}
		close(p.readerStop)
		<-p.readerDone
		<-p.stderrDone
		p.waitMu.Lock()
		waitErr := p.waitErr
		p.waitMu.Unlock()
		if p.finishErr == nil && waitErr != nil && !forced {
			p.finishErr = p.exitFailure(acpPhaseReap, "process did not exit cleanly")
		}
	})
	return p.finishErr
}

func (p *acpProcess) failure(phase acpPhase, kind acpErrorKind, detail string, cause error) *copilotACPError {
	failure := &copilotACPError{
		Phase:  phase,
		Kind:   kind,
		Detail: detail,
		Stderr: p.stderr.String(),
		Cause:  cause,
	}
	select {
	case <-p.waited:
		if code, ok := p.exitCode(); ok {
			failure.ExitCode = &code
		}
	default:
	}
	return failure
}

func (p *acpProcess) exitFailure(phase acpPhase, detail string) *copilotACPError {
	select {
	case <-p.waited:
	case <-time.After(100 * time.Millisecond):
	}
	failure := p.failure(phase, acpErrorExit, detail, nil)
	if code, ok := p.exitCode(); ok {
		failure.ExitCode = &code
	}
	return failure
}

func (p *acpProcess) exitCode() (int, bool) {
	if p.cmd.ProcessState == nil {
		return 0, false
	}
	return p.cmd.ProcessState.ExitCode(), true
}

func (p *acpProcess) authFailure(phase acpPhase, methods []acpAuthMethod, cause error) *copilotACPError {
	detail := "Copilot CLI authentication failed"
	for _, method := range methods {
		if method.ID == "copilot-login" && method.Description != "" {
			detail += "; " + method.Description
			break
		}
	}
	return p.failure(phase, acpErrorAuth, detail+": "+cause.Error(), cause)
}

func rpcErrorSignalsAuthentication(err error) bool {
	message := strings.ToLower(err.Error())
	for _, marker := range []string{
		"authentication required",
		"authentication failed",
		"not authenticated",
		"unauthorized",
		"login required",
		"log in first",
		"sign in first",
		"copilot login",
	} {
		if strings.Contains(message, marker) {
			return true
		}
	}
	return false
}

func formatACPRPCError(rpcError *acpWireError) string {
	if len(rpcError.Data) == 0 {
		return fmt.Sprintf("JSON-RPC error %d: %s", rpcError.Code, rpcError.Message)
	}
	return fmt.Sprintf("JSON-RPC error %d: %s (%s)", rpcError.Code, rpcError.Message, bytes.TrimSpace(rpcError.Data))
}

type boundedTail struct {
	mu    sync.Mutex
	limit int
	data  []byte
}

func newBoundedTail(limit int) *boundedTail {
	return &boundedTail{limit: limit}
}

func (b *boundedTail) Write(data []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	originalLength := len(data)
	if len(data) >= b.limit {
		b.data = append(b.data[:0], data[len(data)-b.limit:]...)
		return originalLength, nil
	}
	if excess := len(b.data) + len(data) - b.limit; excess > 0 {
		copy(b.data, b.data[excess:])
		b.data = b.data[:len(b.data)-excess]
	}
	b.data = append(b.data, data...)
	return originalLength, nil
}

func (b *boundedTail) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return strings.TrimSpace(string(b.data))
}
