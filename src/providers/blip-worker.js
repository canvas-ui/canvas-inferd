'use strict';

import os from 'node:os';
import { pipeline, RawImage, env } from '@huggingface/transformers';

const MODEL = process.env.CANVAS_BLIP_MODEL || 'Xenova/blip-image-captioning-base';
const DTYPE = process.env.CANVAS_BLIP_DTYPE || 'q8';
const THREADS = Math.max(1, Number(process.env.CANVAS_INFERD_THREADS) || Math.min(4, os.cpus().length || 4));

let captionerPromise;
function captioner() {
    if (!captionerPromise) {
        if (process.env.CANVAS_BLIP_CACHE) { env.cacheDir = process.env.CANVAS_BLIP_CACHE; }
        captionerPromise = pipeline('image-to-text', MODEL, {
            dtype: DTYPE,
            session_options: { intraOpNumThreads: THREADS, interOpNumThreads: 1 },
        });
    }
    return captionerPromise;
}

async function describe(buffer) {
    const image = await RawImage.fromBlob(new Blob([buffer]));
    const output = await (await captioner())(image, {
        do_sample: false,
        num_beams: 3,
        max_new_tokens: 40,
    });
    const text = output?.[0]?.generated_text?.trim();
    if (!text) { throw new Error(`BLIP '${MODEL}' returned no caption`); }
    return text;
}

let chain = Promise.resolve();
process.on('message', (message) => {
    chain = chain.then(async () => {
        try {
            process.send({ id: message.id, text: await describe(message.buffer) });
        } catch (error) {
            process.send({ id: message.id, error: error.message });
        }
    });
});

process.send({ ready: true });
