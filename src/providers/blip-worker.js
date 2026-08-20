'use strict';

import os from 'node:os';
import {
    pipeline,
    RawImage,
    env,
    AutoProcessor,
    AutoModelForImageTextToText,
    Florence2ForConditionalGeneration,
} from '@huggingface/transformers';
import { DEFAULT_CAPTION_PROMPT, resolveCaptionTask } from './caption-task.js';

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

const MODEL = process.env.CANVAS_BLIP_MODEL || 'HuggingFaceTB/SmolVLM-500M-Instruct';
const DTYPE = process.env.CANVAS_BLIP_DTYPE || 'q8';
const THREADS = Math.max(1, Number(process.env.CANVAS_INFERD_THREADS) || Math.min(4, os.cpus().length || 4));
const MAX_NEW_TOKENS = Math.max(16, Number(process.env.CANVAS_BLIP_MAX_TOKENS) || 96);

/** Task token Florence-2 needs; the detailed variant is the useful one here. */
const FLORENCE_TASK = process.env.CANVAS_BLIP_FLORENCE_TASK || '<MORE_DETAILED_CAPTION>';

const TASK = resolveCaptionTask(MODEL, process.env.CANVAS_BLIP_TASK || null);

/**
 * Only the plain-captioner family has a `pipeline()` wrapper — there is no
 * `image-text-to-text` pipeline in transformers.js (v4.2 lists the supported
 * set in its own error, and that is not among them). The VLM families are
 * therefore driven through processor + model classes directly.
 */
let runnerPromise;
function runner() {
    if (!runnerPromise) {
        if (process.env.CANVAS_BLIP_CACHE) { env.cacheDir = process.env.CANVAS_BLIP_CACHE; }
        env.allowRemoteModels = true;
        const modelOptions = {
            dtype: DTYPE,
            session_options: { intraOpNumThreads: THREADS, interOpNumThreads: 1 },
        };
        if (TASK === 'caption') {
            runnerPromise = pipeline('image-to-text', MODEL, modelOptions)
                .then((generate) => ({ kind: 'caption', generate }));
        } else {
            const ModelClass = TASK === 'florence2' ? Florence2ForConditionalGeneration : AutoModelForImageTextToText;
            runnerPromise = Promise.all([
                AutoProcessor.from_pretrained(MODEL),
                ModelClass.from_pretrained(MODEL, modelOptions),
            ]).then(([processor, model]) => ({ kind: TASK, processor, model }));
        }
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

/** Instruct VLM: chat template + image, so the prompt actually steers it. */
async function describeInstruct({ processor, model }, image, prompt) {
    const instruction = prompt || process.env.CANVAS_BLIP_PROMPT || DEFAULT_CAPTION_PROMPT;
    const messages = [{
        role: 'user',
        content: [{ type: 'image' }, { type: 'text', text: instruction }],
    }];
    const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
    const inputs = await processor(text, [image], { do_image_splitting: false });
    const generated = await model.generate({ ...inputs, ...generationOptions() });
    // generate() returns the PROMPT plus the answer; keep only what was added,
    // or every caption starts with the instruction we just sent.
    const promptLength = inputs.input_ids.dims.at(-1);
    const answer = generated.slice(null, [promptLength, null]);
    return processor.batch_decode(answer, { skip_special_tokens: true })[0] || '';
}

/** Florence-2: a task token rather than free text, and its own post-processing. */
async function describeFlorence({ processor, model }, image) {
    const prompts = processor.construct_prompts(FLORENCE_TASK);
    const textInputs = processor.tokenizer(prompts);
    const visionInputs = await processor.image_processor(image);
    const generated = await model.generate({ ...textInputs, ...visionInputs, ...generationOptions() });
    const decoded = processor.batch_decode(generated, { skip_special_tokens: false })[0] || '';
    const parsed = processor.post_process_generation(decoded, FLORENCE_TASK, image.size);
    const value = parsed?.[FLORENCE_TASK];
    return typeof value === 'string' ? value : decoded;
}

async function describe(buffer, prompt) {
    const bytes = Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer);
    const image = await RawImage.fromBlob(new Blob([bytes]));
    const run = await runner();

    let raw;
    if (run.kind === 'caption') {
        raw = extractText(await run.generate(image, generationOptions()));
    } else if (run.kind === 'florence2') {
        raw = await describeFlorence(run, image);
    } else {
        raw = await describeInstruct(run, image, prompt);
    }

    const text = stripTaskToken(String(raw || '').trim());
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
