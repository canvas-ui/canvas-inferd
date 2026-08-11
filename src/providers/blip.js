'use strict';

import { fork } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const WORKER_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), 'blip-worker.js');
const DEFAULT_MODEL = 'Xenova/blip-image-captioning-base';

export default class BlipProvider {
    id = 'blip';
    #cacheDir;
    #model;
    #dtype;
    #workers = new Map();
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

        const env = { ...process.env, CANVAS_BLIP_MODEL: model };
        if (this.#cacheDir) { env.CANVAS_BLIP_CACHE = this.#cacheDir; }
        if (this.#dtype) { env.CANVAS_BLIP_DTYPE = this.#dtype; }
        const child = fork(WORKER_PATH, [], { env, serialization: 'advanced' });
        const entry = { child, pending: new Map() };
        const fail = (reason) => {
            for (const request of entry.pending.values()) { request.reject(new Error(reason)); }
            entry.pending.clear();
            if (this.#workers.get(model) === entry) { this.#workers.delete(model); }
        };
        child.on('message', (message) => {
            if (message?.ready) { return; }
            const request = entry.pending.get(message?.id);
            if (!request) { return; }
            entry.pending.delete(message.id);
            if (message.error) { request.reject(new Error(message.error)); }
            else { request.resolve(message.text); }
        });
        child.on('exit', () => fail(`BLIP worker exited (${model})`));
        child.on('error', (error) => fail(error.message));
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

    status() {
        return {
            id: this.id,
            type: 'blip',
            cacheDir: this.#cacheDir,
            defaultModel: this.#resolveModel(null),
            dtype: this.#dtype || process.env.CANVAS_BLIP_DTYPE || 'q8',
            workers: [...this.#workers.entries()].map(([model, entry]) => ({ model, pending: entry.pending.size })),
        };
    }

    async stop() {
        for (const entry of this.#workers.values()) {
            try { entry.child.kill(); } catch (_) { /* ignore */ }
            for (const request of entry.pending.values()) { request.reject(new Error('BLIP provider stopped')); }
        }
        this.#workers.clear();
    }
}
