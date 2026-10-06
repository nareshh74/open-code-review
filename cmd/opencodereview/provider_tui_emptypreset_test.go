// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package main

import (
	"path/filepath"
	"testing"

	tea "charm.land/bubbletea/v2"
)

// A preset without suggested models (copilot-api) must still let the user
// type a model and confirm it with Enter.
func TestProviderTUI_EmptyPresetModelCanBeEnteredAndConfirmed(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.json")
	m := newProviderTUI(&Config{}, configPath)
	found := false
	for i, p := range m.providers {
		if p.Name == "copilot-api" {
			m.officialIdx = i
			found = true
		}
	}
	if !found {
		t.Fatal("copilot-api preset not found")
	}
	if len(m.providers[m.officialIdx].Models) != 0 {
		t.Fatal("test assumes copilot-api has no preset models")
	}

	var model tea.Model = m
	press := func(msg tea.KeyPressMsg) providerTUIModel {
		model, _ = model.Update(msg)
		return model.(providerTUIModel)
	}
	press(enterKey())
	if got := press(enterKey()); !got.customModel {
		t.Fatal("Enter on the empty list should open the custom model input")
	}
	for _, r := range "gpt-4.1" {
		press(tea.KeyPressMsg{Code: r, Text: string(r)})
	}
	got := press(enterKey())
	if got.selectedModelFromState() != "gpt-4.1" {
		t.Fatalf("cursor model = %q, want gpt-4.1", got.selectedModelFromState())
	}
	got = press(enterKey())
	if got.customModel {
		t.Fatal("Enter on the added model reopened the custom input")
	}
	if got.step != stepAPIKey {
		t.Fatalf("model step did not advance; step=%d formError=%q", got.step, got.formError)
	}
	if r := got.result(); r.model != "gpt-4.1" {
		t.Fatalf("result model = %q, want gpt-4.1", r.model)
	}
}
