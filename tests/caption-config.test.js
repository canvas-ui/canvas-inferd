import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

import { normalizeConfig } from '../src/config.js';
import { resolveCaptionTask } from '../src/providers/caption-task.js';

/**
 * A caption model's FAMILY decides how it is driven — image alone, image plus
 * prompt, or image plus a task token — so picking a better model is never just
 * a change of model id. These cover the config surface that carries that
 * choice, and the inference used when a config does not state one.
 */

describe('summarize config', () => {
    test('carries the model family and prompt through normalisation', () => {
        const config = normalizeConfig({
            summarize: {
                image: {
                    enabled: true,
                    provider: 'blip',
                    model: 'HuggingFaceTB/SmolVLM-500M-Instruct',
                    task: 'instruct',
                    prompt: 'Name the architectural style.',
                },
            },
        });
        assert.deepEqual(config.summarize.image, {
            enabled: true,
            provider: 'blip',
            model: 'HuggingFaceTB/SmolVLM-500M-Instruct',
            task: 'instruct',
            prompt: 'Name the architectural style.',
        });
    });

    test('rejects a family the worker has no runner for', () => {
        assert.throws(
            () => normalizeConfig({ summarize: { image: { enabled: true, provider: 'blip', model: 'x/y', task: 'magic' } } }),
            /task must be one of/,
        );
    });

    test('the default captioner is SmolVLM-500M-Instruct, still disabled', () => {
        // Changed deliberately: vit-gpt2 produced one terse, repetition-prone
        // sentence. The blast radius is small because the default is DISABLED —
        // anyone actually captioning saved a config, and the UI always writes
        // provider/model/task explicitly. Only a hand-written config that sets
        // nothing but `enabled: true` inherits the new model.
        const config = normalizeConfig({});
        assert.equal(config.summarize.image.model, 'HuggingFaceTB/SmolVLM-500M-Instruct');
        assert.equal(config.summarize.image.enabled, false);
        // No sticky task/prompt: they must stay inferable so overriding the
        // model alone doesn't drag an instruct pipeline onto a plain captioner.
        assert.equal(config.summarize.image.task, undefined);
        assert.equal(resolveCaptionTask(config.summarize.image.model), 'instruct');
    });
});

describe('caption family inference', () => {
    test('an explicit task always wins', () => {
        assert.equal(resolveCaptionTask('anything/at-all', 'florence2'), 'florence2');
    });

    test('instruct VLMs and Florence-2 are recognised by id', () => {
        assert.equal(resolveCaptionTask('HuggingFaceTB/SmolVLM-500M-Instruct'), 'instruct');
        assert.equal(resolveCaptionTask('onnx-community/Florence-2-base-ft'), 'florence2');
        assert.equal(resolveCaptionTask('Qwen/Qwen2-VL-2B'), 'instruct');
    });

    test('anything else is assumed to be a plain captioner', () => {
        assert.equal(resolveCaptionTask('Xenova/vit-gpt2-image-captioning'), 'caption');
        assert.equal(resolveCaptionTask(''), 'caption');
    });
});
