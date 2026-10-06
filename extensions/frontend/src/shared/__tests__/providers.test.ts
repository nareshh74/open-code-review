// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

import { PROVIDER_PRESETS as generatedPresets } from '../providers.generated';
import { isPresetProvider, lookupPreset, mergeModelLists, PROVIDER_PRESETS, requiresApiKey } from '../providers';
import { buildOfficialSaveEntries, detectInitialTab, isConfigReady } from '../configUtils';
import { OcrConfig } from '../types';

describe('generated provider presets', () => {
  it('keeps providers sorted by name with the existing initial provider', () => {
    const names = PROVIDER_PRESETS.map((preset) => preset.name);
    expect(names).toEqual([...names].sort());
    expect(PROVIDER_PRESETS[0].name).toBe('anthropic');
  });

  it.each(generatedPresets)('preserves metadata, model order and the default for $name', (preset) => {
    const found = lookupPreset(`  ${preset.name.toUpperCase()}  `);
    expect(found).toEqual(preset);
    expect(isPresetProvider(`  ${preset.name.toUpperCase()}  `)).toBe(true);
    const userModels = preset.models.length > 0 ? ['user-model', preset.models[0]] : ['user-model'];
    const models = mergeModelLists(found?.models ?? [], userModels);
    expect(models).toEqual([...preset.models, 'user-model']);
    if (preset.models.length > 0) {
      expect(models[0]).toBe(preset.models[0]);
    } else {
      expect(models).toEqual(['user-model']);
    }
  });

  it.each(['bedrock', 'copilot-acp', 'openai-responses'])('uses the built-in configuration path for %s', (name) => {
    const preset = lookupPreset(name);
    expect(preset).toBeDefined();
    const model = preset?.models[0] ?? '';
    const config: OcrConfig = {
      provider: name,
      model: '',
      providers: { [name]: { model } },
      customProviders: {},
      llm: { url: '', authToken: '', model: '', useAnthropic: false },
      language: 'English',
    };
    expect(detectInitialTab(config)).toBe('official');
    expect(isConfigReady(config)).toBe(true);
    expect(buildOfficialSaveEntries(name, model, '', false)).toEqual([
      { key: 'provider', value: name },
      { key: `providers.${name}.model`, value: model },
    ]);
  });

  it('preserves credential ownership and Responses protocol metadata', () => {
    expect(lookupPreset('bedrock')).toMatchObject({
      protocol: 'anthropic-bedrock', credentials: 'aws', envVar: '', baseUrl: '',
    });
    expect(lookupPreset('copilot-acp')).toMatchObject({
      protocol: 'copilot-acp', credentials: 'copilot-cli', envVar: '', baseUrl: '',
    });
    expect(lookupPreset('openai-responses')?.protocol).toBe('openai-responses');
    expect(lookupPreset('anthropic')?.authHeader).toBe('x-api-key');
  });

  it('derives API key requirements from the effective protocol', () => {
    const bedrock = lookupPreset('bedrock');
    const copilot = lookupPreset('copilot-acp');
    const openai = lookupPreset('openai');
    expect(bedrock).toBeDefined();
    expect(copilot).toBeDefined();
    expect(openai).toBeDefined();
    expect(requiresApiKey(bedrock!)).toBe(false);
    expect(requiresApiKey(copilot!)).toBe(false);
    expect(requiresApiKey(bedrock!, 'openai')).toBe(true);
    expect(requiresApiKey(openai!, 'anthropic-bedrock')).toBe(false);
    expect(requiresApiKey(openai!, ' OPENAI ')).toBe(true);
  });

  it('keeps unknown providers outside the built-in preset path', () => {
    expect(lookupPreset('my-custom-provider')).toBeUndefined();
    expect(isPresetProvider('my-custom-provider')).toBe(false);
  });
});
