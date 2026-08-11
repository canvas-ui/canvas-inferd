'use strict';

/**
 * Which FAMILY a caption model belongs to, and how each is driven.
 *
 * This is deliberately separate from blip-worker.js: the worker is a program
 * (it wires IPC handlers on import), so anything that needs to reason about
 * caption families — config validation, tests, a future model picker on the
 * server — must be able to import the rules without forking a child.
 */

/**
 *   caption   image in, one sentence out (`image-to-text`)
 *   instruct  image + prompt, so the description can be steered
 *   florence2 image + a task token such as `<MORE_DETAILED_CAPTION>`
 */
export const CAPTION_TASKS = ['caption', 'instruct', 'florence2'];

/** transformers.js pipeline that runs each family. */
export const PIPELINE_FOR_TASK = {
    caption: 'image-to-text',
    instruct: 'image-text-to-text',
    florence2: 'image-text-to-text',
};

export const DEFAULT_CAPTION_PROMPT = 'Describe this photograph in two or three sentences. '
    + 'If it shows a building or interior, name the architectural style, the main materials and the setting. '
    + 'Describe only what is visible.';

/**
 * Family from the model id, for configs that do not state one. Inference is a
 * convenience for hand-written configs — the UI always sends `task` explicitly,
 * because a wrong guess builds the wrong pipeline and fails at load time.
 */
export function resolveCaptionTask(model, configured = null) {
    if (configured) { return configured; }
    const id = String(model || '').toLowerCase();
    if (id.includes('florence')) { return 'florence2'; }
    // Instruct-tuned VLMs advertise it in the name (SmolVLM-*-Instruct, Qwen*-VL,
    // LLaVA, moondream, Idefics). Anything else is assumed a plain captioner.
    if (/instruct|-vl|llava|moondream|idefics/.test(id)) { return 'instruct'; }
    return 'caption';
}
