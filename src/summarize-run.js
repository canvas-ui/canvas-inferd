'use strict';

import debugInstance from 'debug';

const debug = debugInstance('canvas:inferd:summarize');

/**
 * ImageSummaryRun — captioning a workspace's images, start to finish.
 *
 * This is an inference WORKFLOW: pick the images, feed each to a vision model,
 * write the caption back. It used to run inside canvas-server, which meant the
 * server owned a retry policy for a model worker it could not see, a
 * consecutive-failure limit for a model it did not load, and a `workerDead`
 * error contract from a process it did not manage. All of that is knowledge
 * about the models, so it belongs on this side of the socket; the server keeps
 * only what is genuinely its own — which documents are images, the bytes, and
 * where a caption is stored.
 *
 * The run is deliberately in-memory and per-workspace: it is a foreground job a
 * user starts and watches, not durable state. A daemon restart abandons it, and
 * the user starts it again — which is also the retry path for a crashed model
 * worker, since inferd will not respawn one on its own.
 */

// A misconfiguration fails identically for every image — a model that cannot
// load, a family the runner has no path for. Marching through 1400 images to
// report the same sentence 1400 times is noise, so a run that only ever fails
// gives up early and says why.
const CONSECUTIVE_FAILURE_LIMIT = 5;
const MAX_REPORTED_ERRORS = 20;

export class ImageSummaryRun {
    #wsId;
    #adapter;
    #describe;
    #cancelled = false;
    #promise = null;
    #status = { running: false, total: 0, described: 0, skipped: 0, failed: 0 };

    /**
     * @param {string} wsId
     * @param {object} adapter        the workspace seam (imageDocumentIds, resolveInput, setSummary)
     * @param {Function} describe     (bytes, { contentType }) => Promise<string>
     */
    constructor(wsId, adapter, describe) {
        this.#wsId = wsId;
        this.#adapter = adapter;
        this.#describe = describe;
    }

    get running() { return this.#status.running; }

    get status() {
        // Copy: callers render this into a response while the run mutates it.
        return { ...this.#status, errors: this.#status.errors ? [...this.#status.errors] : undefined };
    }

    async start({ force = false } = {}) {
        if (this.#status.running) {
            return { started: false, error: 'Image summary generation is already running', status: this.status };
        }

        const ids = await this.#adapter.imageDocumentIds();
        this.#status = {
            running: true, total: ids.length, described: 0, skipped: 0, failed: 0,
            startedAt: new Date().toISOString(),
        };
        this.#cancelled = false;
        this.#promise = this.#run(ids, { force })
            .catch((error) => {
                this.#status.aborted = true;
                this.#status.abortedReason = error.message || String(error);
            })
            .finally(() => {
                this.#promise = null;
                this.#status = { ...this.#status, running: false, finishedAt: new Date().toISOString() };
            });
        return { started: true, status: this.status };
    }

    stop() {
        if (!this.#status.running) {
            return { stopped: false, error: 'No image summary run is in progress', status: this.status };
        }
        this.#cancelled = true;
        return { stopped: true, status: this.status };
    }

    async #run(ids, { force }) {
        let consecutiveFailures = 0;

        for (const id of ids) {
            if (this.#cancelled) { this.#status.cancelled = true; return; }
            try {
                const input = await this.#adapter.resolveInput(id);
                if (!input || input.skip || input.modality !== 'image' || !input.bytes) {
                    this.#status.skipped++;
                    continue;
                }
                if (!force && input.summary) { this.#status.skipped++; continue; }

                const text = await this.#describe(input.bytes, { contentType: input.contentType || null });
                await this.#adapter.setSummary(id, text);
                this.#status.described++;
                consecutiveFailures = 0;
            } catch (error) {
                this.#status.failed++;
                consecutiveFailures++;
                const errors = this.#status.errors || (this.#status.errors = []);
                if (errors.length < MAX_REPORTED_ERRORS) {
                    errors.push({ id, error: error.message || String(error) });
                }
                // The model worker died (OOM, native crash) and will not be
                // respawned. Carrying on would mark every remaining image failed
                // against a provider that cannot answer — stop and say why,
                // leaving the un-attempted images for a later run.
                if (error?.workerDead) {
                    this.#abort(error.message || 'model worker died');
                    return;
                }
                if (consecutiveFailures >= CONSECUTIVE_FAILURE_LIMIT) {
                    this.#abort(`${consecutiveFailures} images in a row failed — ${error.message || String(error)}`);
                    return;
                }
            }
        }
    }

    #abort(reason) {
        this.#status.aborted = true;
        this.#status.abortedReason = reason;
        debug(`workspace ${this.#wsId}: image summaries stopped — ${reason}`);
    }
}

export default ImageSummaryRun;
