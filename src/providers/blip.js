'use strict';

import { fork } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import debugInstance from 'debug';
import { describeWorkerExit, logWorkerExit, workerDeadError } from './worker-exit.js';

const debug = debugInstance('canvas:inferd:blip');

const WORKER_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), 'blip-worker.js');
const DEFAULT_MODEL = 'Xenova/vit-gpt2-image-captioning';

export default class BlipProvider {
    id = 'blip';
    #cacheDir;
    #model;
    #dtype;
    #workers = new Map();
    // model -> why it died. A CRASHED worker is not respawned: reloading
    // GB-scale weights per image is what turned one OOM kill into a caption run
    // that thrashed through thousands of images, re-forking and being killed
    // again each time. One crash stops the model. A TIMEOUT is different — that
    // worker is merely wedged, so it still respawns.
    #dead = new Map();
    #seq = 1;

    constructor({ cacheDir = null, model = null, dtype = null, id = null } = {}) {
        if (id) { this.id = id; }
        this.#cacheDir = cacheDir;
        this.#model = model;
        this.#dtype = dtype;
    }

    #resolveModel(model) {
        return model || this.#model || process.env.CANVAS_BLIP_MODEL || DEFAULT_MODEL;
    }

    modelCached(model) {
        if (!this.#cacheDir) { return null; }
        try { return fs.existsSync(path.join(this.#cacheDir, ...this.#resolveModel(model).split('/'))); }
        catch (_) { return null; }
    }

    #worker(model) {
        const existing = this.#workers.get(model);
        if (existing) { return existing; }

        const tombstone = this.#dead.get(model);
        if (tombstone) { throw workerDeadError(`${tombstone}. Not respawning — fix the cause (memory) and start a new run.`); }

        const env = { ...process.env, CANVAS_BLIP_MODEL: model };
        if (this.#cacheDir) { env.CANVAS_BLIP_CACHE = this.#cacheDir; }
        if (this.#dtype) { env.CANVAS_BLIP_DTYPE = this.#dtype; }
        const child = fork(WORKER_PATH, [], { env, serialization: 'advanced' });
        // `expected` marks OUR OWN teardown (stop, timeout kill) so those exits
        // stay quiet and anything else is reported as the failure it is.
        const entry = { child, pending: new Map(), expected: false };
        debug(`blip worker spawned (${model}, pid ${child.pid}, dtype ${this.#dtype || process.env.CANVAS_BLIP_DTYPE || 'q8'})`);
        const fail = (reason, dead = false) => {
            for (const request of entry.pending.values()) {
                request.reject(dead ? workerDeadError(reason) : new Error(reason));
            }
            entry.pending.clear();
            if (this.#workers.get(model) === entry) { this.#workers.delete(model); }
        };
        child.on('message', (message) => {
            if (message?.ready) { debug(`blip worker ready (${model}, pid ${child.pid})`); return; }
            const request = entry.pending.get(message?.id);
            if (!request) { return; }
            entry.pending.delete(message.id);
            if (message.error) { request.reject(new Error(message.error)); }
            else { request.resolve(message.text); }
        });
        child.on('exit', (code, signal) => {
            logWorkerExit(debug, 'BLIP', {
                model, pid: child.pid, code, signal,
                pending: entry.pending.size, expected: entry.expected,
            });
            // The cause travels with the rejection, not just to the log: a
            // caption run records one of these per image (Workspace's summary
            // status), so the UI shows "killed by the OOM killer" rather than a
            // bare "worker exited" the operator then has to go hunting for.
            const reason = describeWorkerExit('BLIP', model, code, signal);
            if (!entry.expected) { this.#dead.set(model, reason); }
            fail(reason, !entry.expected);
        });
        child.on('error', (error) => {
            console.error(`inferd: BLIP worker (${model}) failed to start or lost its channel: ${error.message}`);
            fail(error.message);
        });
        this.#workers.set(model, entry);
        return entry;
    }

    async describeImage(buffer, { model } = {}) {
        if (!Buffer.isBuffer(buffer) && !(buffer instanceof Uint8Array)) {
            throw new Error('BlipProvider.describeImage requires image bytes');
        }
        const resolved = this.#resolveModel(model);
        const worker = this.#worker(resolved);
        const id = this.#seq++;
        const cold = this.modelCached(resolved) === false;
        const timeoutMs = Math.max(1000, Number(process.env.CANVAS_BLIP_TIMEOUT_MS) || (cold ? 600000 : 120000));
        return await new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                worker.pending.delete(id);
                worker.expected = true;
                console.error(`inferd: BLIP worker (${resolved}) wedged for ${timeoutMs}ms — killing it; the next caption respawns a clean one`);
                try { worker.child.kill(); } catch (_) { /* ignore */ }
                if (this.#workers.get(resolved) === worker) { this.#workers.delete(resolved); }
                reject(new Error(`BLIP worker timeout after ${timeoutMs}ms (${resolved})`));
            }, timeoutMs);
            timer.unref?.();
            worker.pending.set(id, {
                resolve: (text) => { clearTimeout(timer); resolve(text); },
                reject: (error) => { clearTimeout(timer); reject(error); },
            });
            worker.child.send({ id, buffer });
        });
    }

    /**
     * Re-arm a model the breaker stopped. Called when a new run is started
     * deliberately — the operator has had their say (more RAM, a smaller
     * model), so the next attempt is allowed to spawn again.
     */
    resetWorkers(model = null) {
        if (model) { return this.#dead.delete(model); }
        const had = this.#dead.size > 0;
        this.#dead.clear();
        return had;
    }

    status() {
        return {
            id: this.id,
            type: 'blip',
            cacheDir: this.#cacheDir,
            defaultModel: this.#resolveModel(null),
            dtype: this.#dtype || process.env.CANVAS_BLIP_DTYPE || 'q8',
            // pid included so an operator can point `ps`/`top` at the process
            // actually holding the weights.
            workers: [...this.#workers.entries()].map(([model, entry]) => ({ model, pid: entry.child.pid, pending: entry.pending.size })),
            // Models the breaker has stopped — visible in the status endpoint so
            // "nothing happens when I press generate" has an answer on screen.
            dead: [...this.#dead.entries()].map(([model, reason]) => ({ model, reason })),
        };
    }

    async stop() {
        for (const entry of this.#workers.values()) {
            entry.expected = true;
            try { entry.child.kill(); } catch (_) { /* ignore */ }
            for (const request of entry.pending.values()) { request.reject(new Error('BLIP provider stopped')); }
        }
        this.#workers.clear();
        this.#dead.clear();
    }
}
