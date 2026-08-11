'use strict';

import os from 'node:os';
import { pipeline, RawImage, env } from '@huggingface/transformers';
import { DEFAULT_CAPTION_PROMPT, PIPELINE_FOR_TASK, resolveCaptionTask } from './caption-task.js';

/**
 * The local caption worker: one forked child, one model, images in, sentences
 * out. Three model FAMILIES are supported, because nothing better than a plain
 * captioner can be driven the same way:
 *
 *   caption   `image-to-text` — image in, one short sentence out. The old
 *             default (vit-gpt2) is this, and it is weak: greedy decoding on a
 *             repetitive subject produced captions like "a wall of wallpapers
 *             of a wall of wallpapers of a wall of…".
 *   instruct  `image-text-to-text` — an instruct VLM that takes a PROMPT, so
 *             "describe the architecture, materials and setting" is expressible
 *             rather than hoped for. This is the family worth using.
 *   florence2 Florence-2, driven by a task token (`<MORE_DETAILED_CAPTION>`)
 *             instead of free text.
 *
 * The family is configured (`summarize.image.task`) or inferred from the model
 * id. Note that the BLIP ONNX repos this file is named after are all 401/gated
 * on the Hub now — hence the family plumbing rather than a swap of model id.
 */

const MODEL = process.env.CANVAS_BLIP_MODEL || 'Xenova/vit-gpt2-image-captioning';
const DTYPE = process.env.CANVAS_BLIP_DTYPE || 'q8';
const THREADS = Math.max(1, Number(process.env.CANVAS_INFERD_THREADS) || Math.min(4, os.cpus().length || 4));
const MAX_NEW_TOKENS = Math.max(16, Number(process.env.CANVAS_BLIP_MAX_TOKENS) || 96);

/** Task token Florence-2 needs; the detailed variant is the useful one here. */
const FLORENCE_TASK = process.env.CANVAS_BLIP_FLORENCE_TASK || '<MORE_DETAILED_CAPTION>';

const TASK = resolveCaptionTask(MODEL, process.env.CANVAS_BLIP_TASK || null);

let runnerPromise;
function runner() {
    if (!runnerPromise) {
        if (process.env.CANVAS_BLIP_CACHE) { env.cacheDir = process.env.CANVAS_BLIP_CACHE; }
        env.allowRemoteModels = true;
        runnerPromise = pipeline(PIPELINE_FOR_TASK[TASK], MODEL, {
            dtype: DTYPE,
            session_options: { intraOpNumThreads: THREADS, interOpNumThreads: 1 },
        });
    }
    return runnerPromise;
}

/**
 * Decoding options shared by every family.
 *
 * `no_repeat_ngram_size` is the fix for the degenerate loop: greedy decoding
 * with nothing forbidding repetition will happily emit the same 3-gram until it
 * runs out of budget, which is literally what "a wall of wallpapers of a wall
 * of wallpapers" is. Blocking repeated 3-grams ends that, and the repetition
 * penalty discourages the milder version of the same failure.
 */
function generationOptions() {
    return {
        do_sample: false,
        max_new_tokens: MAX_NEW_TOKENS,
        no_repeat_ngram_size: 3,
        repetition_penalty: 1.15,
    };
}

/** transformers.js returns a few shapes depending on family; normalise them. */
function extractText(output) {
    const first = Array.isArray(output) ? output[0] : output;
    const generated = first?.generated_text;
    if (typeof generated === 'string') { return generated.trim(); }
    // Chat-style output: the assistant turn is last.
    if (Array.isArray(generated)) {
        const last = generated[generated.length - 1];
        const content = last?.content;
        if (typeof content === 'string') { return content.trim(); }
        if (Array.isArray(content)) {
            return content.map((part) => (typeof part === 'string' ? part : part?.text || '')).join(' ').trim();
        }
    }
    return '';
}

// Florence-2 echoes its own task token back; a caption that starts with
// "<MORE_DETAILED_CAPTION>" would be indexed verbatim otherwise.
function stripTaskToken(text) {
    return text.replace(/^<[A-Z_]+>\s*/, '').trim();
}

async function describe(buffer, prompt) {
    const bytes = Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer);
    const image = await RawImage.fromBlob(new Blob([bytes]));
    const run = await runner();

    let output;
    if (TASK === 'caption') {
        output = await run(image, generationOptions());
    } else if (TASK === 'florence2') {
        output = await run(image, FLORENCE_TASK, generationOptions());
    } else {
        const instruction = prompt || process.env.CANVAS_BLIP_PROMPT || DEFAULT_CAPTION_PROMPT;
        const messages = [{
            role: 'user',
            content: [{ type: 'image' }, { type: 'text', text: instruction }],
        }];
        output = await run(image, messages, generationOptions());
    }

    const text = stripTaskToken(extractText(output));
    if (!text) { throw new Error(`caption model '${MODEL}' returned no caption`); }
    return text;
}

let chain = Promise.resolve();
process.on('message', (message) => {
    chain = chain.then(async () => {
        try {
            process.send({ id: message.id, text: await describe(message.buffer, message.prompt) });
        } catch (error) {
            process.send({ id: message.id, error: error.message });
        }
    });
});

process.send({ ready: true });
