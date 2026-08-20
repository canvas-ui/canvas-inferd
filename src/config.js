'use strict';

import { DEFAULT_RULES, DEFAULT_SPACES } from './router.js';
import { CAPTION_TASKS as CAPTION_TASK_NAMES } from './providers/caption-task.js';

/**
 * inferd configuration — providers, per-space backends and routing as DATA.
 *
 * Three parts, deliberately separated by who owns them:
 *
 *   providers{} — connection details for an inference backend (`{ type, ...opts }`
 *                 under a caller-chosen id). Built-ins `onnx`/`clip`/`blip` (local, in-process)
 *                 and `ollama`/`vllm` (OpenAI-compatible, loopback presets)
 *                 always exist; re-declaring an id merges over its options.
 *   spaces{}    — which provider+model fills each modality. THE configurable
 *                 surface: server defaults, overridable per user.
 *   rules[]     — routing (`{ space, match }`): which content lands in which
 *                 space. Structural and near-constant — a user picking an
 *                 embedding backend should never restate that photos are images.
 *
 * ```json
 * {
 *   "providers": { "gpu": { "type": "openai", "baseUrl": "http://gpu.local:8000/v1" } },
 *   // ...or just reference the built-in `ollama` / `vllm` preset by id.
 *   "spaces": {
 *     "text":  { "provider": "gpu", "model": "Qwen/Qwen3-Embedding-0.6B", "dim": 1024, "chunk": true },
 *     "image": { "provider": "clip", "model": "Xenova/siglip-base-patch16-224", "dim": 768 }
 *   }
 * }
 * ```
 *
 * Misconfiguration throws. A typo'd provider id would otherwise degrade dense
 * search silently, which is far worse than a loud failure; every part is
 * optional, so nothing throws unless someone actually wrote config.
 */

/**
 * Provider implementations the factory knows how to build.
 *
 * Every remote backend worth pointing inferd at speaks the OpenAI embeddings
 * dialect — Ollama, vLLM, TEI, infinity, LM Studio and OpenAI itself — so there
 * is ONE remote implementation (`openai`) and the differences between them are
 * data: a base URL, and how images are spelled. `ollama` and `vllm` are
 * therefore built-in provider IDS (see builtinProviders) rather than types.
 */
export const PROVIDER_TYPES = new Set(['onnx', 'clip', 'blip', 'openai']);

/**
 * Types that are no longer implementations but must keep loading. `ollama` was
 * its own class speaking Ollama's native /api/embed; it now resolves to the
 * openai provider against Ollama's OpenAI-compatible surface. A stored config
 * saying `{ type: 'ollama', host }` keeps working — the host becomes the base
 * URL, and OpenAIProvider appends /v1 to it.
 */
export const LEGACY_PROVIDER_TYPES = { ollama: 'openai' };

/**
 * Rewrite a legacy spec to its modern equivalent BEFORE it merges over a
 * built-in's defaults — otherwise an explicitly configured `host` would lose to
 * the preset `baseUrl` it is supposed to replace.
 */
function aliasLegacyProvider(raw) {
    if (!raw || LEGACY_PROVIDER_TYPES[raw.type] === undefined) { return raw; }
    const { host, type, ...rest } = raw;
    return { ...rest, type: LEGACY_PROVIDER_TYPES[type], ...(host ? { baseUrl: host } : {}) };
}

/** Keys that describe a space's BACKEND rather than its routing. */
const SPACE_KEYS = ['provider', 'model', 'dim', 'chunk', 'maxLength', 'dimensions', 'annIndex'];

/**
 * Providers that exist without being declared. Their options come from the
 * top-level server settings (cache dirs, ollama host) unless the config
 * overrides them by re-declaring the same id.
 */
export const PRESET_BASE_URLS = {
    ollama: 'http://127.0.0.1:11434/v1',
    vllm: 'http://127.0.0.1:8000/v1',
    openai: 'https://api.openai.com/v1',
};

export function builtinProviders({ onnxCacheDir = null, clipCacheDir = null, ollamaHost = null, vllmBaseUrl = null } = {}) {
    return {
        onnx: { type: 'onnx', cacheDir: onnxCacheDir },
        clip: { type: 'clip', cacheDir: clipCacheDir || onnxCacheDir },
        blip: { type: 'blip', cacheDir: clipCacheDir || onnxCacheDir },
        // Named presets over the single OpenAI-compatible implementation, both
        // pointing at the port their daemon listens on by default — so "use my
        // local Ollama" is a space referencing `ollama`, with nothing to declare.
        // Deliberately loopback-only: a preset for a hosted API would be an
        // unconfigured provider the admin status view pings on every poll.
        // OpenAI itself is reached by declaring a provider of type `openai`,
        // which is where the API key gets typed anyway.
        ollama: { type: 'openai', baseUrl: ollamaHost || PRESET_BASE_URLS.ollama },
        vllm: { type: 'openai', baseUrl: vllmBaseUrl || PRESET_BASE_URLS.vllm },
    };
}

/**
 * Turn a JSON-expressible matcher into something `router.test()` accepts. JSON
 * has no RegExp literal, so three spellings are supported:
 *   "data/schema/note"  → exact string match
 *   "image/*"                → prefix match  (compiles to /^image\//)
 *   "/^text\\/(plain|md)$/i" → explicit regex, optional trailing flags
 * A RegExp passed in from JS is returned untouched.
 */
export function toMatcher(value) {
    if (value instanceof RegExp) { return value; }
    if (typeof value !== 'string' || value.length === 0) { return value; }

    const explicit = value.match(/^\/(.+)\/([gimsuy]*)$/);
    if (explicit) { return new RegExp(explicit[1], explicit[2]); }

    if (value.endsWith('/*')) {
        return new RegExp(`^${escapeRegex(value.slice(0, -2))}/`);
    }
    return value;
}

function escapeRegex(s) { return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

function normalizeProviders(declared, defaults) {
    const out = { ...defaults };
    if (declared == null) { return out; }
    if (typeof declared !== 'object' || Array.isArray(declared)) {
        throw new Error('inferd config: `providers` must be an object keyed by provider id');
    }
    for (const [id, rawDeclared] of Object.entries(declared)) {
        if (!rawDeclared || typeof rawDeclared !== 'object') {
            throw new Error(`inferd config: provider '${id}' must be an object`);
        }
        const raw = aliasLegacyProvider(rawDeclared);
        // Re-declaring a built-in id merges over its defaults, so overriding just
        // `baseUrl` on `ollama` doesn't drop the rest.
        const base = out[id] || {};
        const type = raw.type || base.type;
        if (!type) { throw new Error(`inferd config: provider '${id}' has no \`type\``); }
        if (!PROVIDER_TYPES.has(type)) {
            throw new Error(`inferd config: provider '${id}' has unknown type '${type}' (known: ${[...PROVIDER_TYPES].join(', ')})`);
        }
        out[id] = { ...base, ...raw, type };
    }
    return out;
}

function normalizeSpace(space, raw, providerIds) {
    if (!raw || typeof raw !== 'object') {
        throw new Error(`inferd config: space '${space}' must be an object`);
    }
    const { provider, model, dim } = raw;
    if (!provider) { throw new Error(`inferd config: space '${space}' has no \`provider\``); }
    if (!providerIds.has(provider)) {
        throw new Error(`inferd config: space '${space}' references undeclared provider '${provider}' (declared: ${[...providerIds].join(', ')})`);
    }
    if (!model) { throw new Error(`inferd config: space '${space}' has no \`model\``); }
    if (!Number.isInteger(dim) || dim <= 0) {
        throw new Error(`inferd config: space '${space}' needs a positive integer \`dim\` (it sizes the Lance table and cannot be guessed)`);
    }
    const out = {};
    for (const key of SPACE_KEYS) {
        if (raw[key] !== undefined) { out[key] = raw[key]; }
    }
    return out;
}

function normalizeRule(raw, i) {
    if (!raw || typeof raw !== 'object') {
        throw new Error(`inferd config: rules[${i}] must be an object`);
    }
    if (!raw.space) { throw new Error(`inferd config: rules[${i}] has no \`space\``); }
    const match = {};
    for (const [key, value] of Object.entries(raw.match || {})) {
        match[key] = key === 'modality' ? value : toMatcher(value);
    }
    return { space: raw.space, match };
}

/**
 * Normalize + validate raw config into `{ providers, spaces, rules }`. Defaults
 * reproduce the built-in CPU-local behaviour exactly.
 *
 * @param {object} [options]
 * @param {object} [options.providers]  declared providers, keyed by id
 * @param {object} [options.spaces]     per-space backends, keyed by space name
 * @param {Array}  [options.rules]      routing rules (defaults to DEFAULT_RULES)
 * @param {string} [options.onnxCacheDir]
 * @param {string} [options.clipCacheDir]
 * @param {string} [options.ollamaHost]
 */
// Per-modality summary generation ("describe" capability — captions for images
// first, audio/text later). BLIP is the local optional default: disabled until
// a workspace explicitly enables image summaries.
// Shape: { summarize: { image: { enabled, provider?, model? }, audio: {...} } }
const SUMMARIZE_MODALITIES = new Set(['image', 'audio', 'text']);
// How a caption model is DRIVEN — a property of the model family, not of the
// provider. Defined next to the worker that implements each one.
const CAPTION_TASKS = new Set(CAPTION_TASK_NAMES);
const DEFAULT_SUMMARIZE = {
    // SmolVLM-500M-Instruct is the runtime default captioner: an instruct VLM,
    // so the description is steerable (DEFAULT_CAPTION_PROMPT) instead of the
    // one terse, repetition-prone sentence vit-gpt2 produced. Still disabled
    // until a workspace opts in — captioning is expensive, not free.
    // `task` and `prompt` are deliberately NOT set here. This object merges
    // key-wise into whatever a workspace saves, so a sticky `task: 'instruct'`
    // would follow a user who switches to a plain captioner and build the wrong
    // pipeline. Both already have correct fallbacks downstream:
    // resolveCaptionTask() reads 'instruct' off the model id, and the worker
    // falls back to DEFAULT_CAPTION_PROMPT.
    image: { enabled: false, provider: 'blip', model: 'HuggingFaceTB/SmolVLM-500M-Instruct' },
};

function normalizeSummarize(raw, providerIds) {
    if (raw === undefined || raw === null) { return {}; }
    if (typeof raw !== 'object' || Array.isArray(raw)) {
        throw new Error('inferd config: `summarize` must be an object keyed by modality');
    }
    const out = {};
    for (const [modality, spec] of Object.entries(raw)) {
        if (!SUMMARIZE_MODALITIES.has(modality)) {
            throw new Error(`inferd config: summarize has unknown modality '${modality}' (known: ${[...SUMMARIZE_MODALITIES].join(', ')})`);
        }
        if (!spec || typeof spec !== 'object') { continue; }
        const entry = { enabled: spec.enabled === true };
        if (spec.provider !== undefined) {
            if (typeof spec.provider !== 'string' || !providerIds.has(spec.provider)) {
                throw new Error(`inferd config: summarize.${modality} references unknown provider '${spec.provider}'`);
            }
            entry.provider = spec.provider;
        }
        if (spec.model !== undefined) {
            if (typeof spec.model !== 'string' || !spec.model) {
                throw new Error(`inferd config: summarize.${modality}.model must be a non-empty string`);
            }
            entry.model = spec.model;
        }
        // Which FAMILY the model belongs to, because the runner differs: a plain
        // captioner takes an image and nothing else, an instruct VLM takes an
        // image plus a prompt, and Florence-2 takes an image plus a task token.
        // Left unset it is inferred from the model id (see resolveCaptionTask).
        if (spec.task !== undefined) {
            if (!CAPTION_TASKS.has(spec.task)) {
                throw new Error(`inferd config: summarize.${modality}.task must be one of ${[...CAPTION_TASKS].join(', ')}`);
            }
            entry.task = spec.task;
        }
        // Only instruct VLMs can be steered ("describe the architecture…"); a
        // plain captioner ignores it, which is why it is not an error to set.
        if (spec.prompt !== undefined) {
            if (typeof spec.prompt !== 'string') {
                throw new Error(`inferd config: summarize.${modality}.prompt must be a string`);
            }
            entry.prompt = spec.prompt;
        }
        if (entry.enabled && (!entry.provider || !entry.model)) {
            throw new Error(`inferd config: summarize.${modality} is enabled but missing provider/model`);
        }
        out[modality] = entry;
    }
    return out;
}

export function normalizeConfig(options = {}) {
    const providers = normalizeProviders(options.providers, builtinProviders(options));
    const providerIds = new Set(Object.keys(providers));

    const rawRules = Array.isArray(options.rules) && options.rules.length ? options.rules : DEFAULT_RULES;
    // Whole-space replacement, not a per-key merge: declaring a space declares
    // it fully. Partial overrides are merged across layers before normalization.
    const rawSpaces = { ...DEFAULT_SPACES, ...(options.spaces || {}) };

    const spaces = {};
    for (const [space, raw] of Object.entries(rawSpaces)) {
        spaces[space] = normalizeSpace(space, raw, providerIds);
    }

    const rules = rawRules.map(normalizeRule);
    if (rules.length === 0) { throw new Error('inferd config: `rules` is empty — nothing would ever embed'); }
    for (const rule of rules) {
        if (!spaces[rule.space]) {
            throw new Error(`inferd config: rule routes to space '${rule.space}', which has no configured backend (declared spaces: ${Object.keys(spaces).join(', ') || 'none'})`);
        }
    }

    const rawSummarize = options.summarize;
    if (rawSummarize != null && (typeof rawSummarize !== 'object' || Array.isArray(rawSummarize))) {
        normalizeSummarize(rawSummarize, providerIds);
    }
    const summarize = {};
    for (const modality of new Set([...Object.keys(DEFAULT_SUMMARIZE), ...Object.keys(rawSummarize || {})])) {
        summarize[modality] = { ...(DEFAULT_SUMMARIZE[modality] || {}), ...(rawSummarize?.[modality] || {}) };
    }
    return { providers, spaces, rules, summarize: normalizeSummarize(summarize, providerIds) };
}

/**
 * Merge configuration layers, lowest precedence first —
 * `mergeConfigLayers(serverDefault, userConfig)`.
 *
 * Providers merge by id and spaces merge KEY-WISE, so a user who only wants a
 * different text model sends `{ spaces: { text: { model, dim } } }` and inherits
 * the provider underneath. (normalizeConfig replaces whole spaces; that is the
 * right call for a single hand-written file, and this is the right call for
 * layering. The merged result still has to validate, so a half-specified space
 * fails loudly rather than embedding with a mismatched dim.)
 *
 * Routing rules do NOT merge — a layer that declares them replaces them
 * wholesale. Interleaving ordered match rules from two sources produces routing
 * nobody wrote or can predict.
 */
export function mergeConfigLayers(...layers) {
    const out = { providers: {}, spaces: {} };
    let rules = null;
    for (const layer of layers) {
        if (!layer || typeof layer !== 'object') { continue; }
        for (const [id, spec] of Object.entries(layer.providers || {})) {
            if (!spec || typeof spec !== 'object') { continue; }
            out.providers[id] = { ...(out.providers[id] || {}), ...spec };
        }
        // Summarize merges key-wise per modality, like spaces: a workspace that
        // only flips `image.enabled` inherits the provider/model underneath.
        for (const [modality, spec] of Object.entries(layer.summarize || {})) {
            if (!spec || typeof spec !== 'object') { continue; }
            out.summarize = out.summarize || {};
            out.summarize[modality] = { ...(out.summarize[modality] || {}), ...spec };
        }
        for (const [space, cfg] of Object.entries(layer.spaces || {})) {
            if (!cfg || typeof cfg !== 'object') { continue; }
            out.spaces[space] = { ...(out.spaces[space] || {}), ...cfg };
        }
        if (Array.isArray(layer.rules) && layer.rules.length) { rules = layer.rules; }
    }
    if (rules) { out.rules = rules; }
    return out;
}

/**
 * Strip secrets from a config for anything that leaves the server. API keys are
 * write-only over the API: a GET reports whether one is set, never its value.
 */
export function redactConfig(config) {
    const providers = {};
    for (const [id, spec] of Object.entries(config?.providers || {})) {
        const { apiKey, headers, ...rest } = spec || {};
        providers[id] = {
            ...rest,
            ...(apiKey ? { apiKeySet: true } : {}),
            ...(headers ? { headerNames: Object.keys(headers) } : {}),
        };
    }
    return { ...config, providers };
}

export default { normalizeConfig, mergeConfigLayers, redactConfig, toMatcher, builtinProviders, PROVIDER_TYPES, PRESET_BASE_URLS, LEGACY_PROVIDER_TYPES };
