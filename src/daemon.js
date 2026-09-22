'use strict';

import net from 'node:net';
import fs from 'node:fs/promises';
import path from 'node:path';
import debugInstance from 'debug';
import Inferd from './index.js';
import { testProvider } from './test-provider.js';
import { redactConfig } from './config.js';
import { checkConfigEndpoints, checkEndpoint, endpointFor } from './endpoint-guard.js';
import { RpcPeer } from './rpc/peer.js';
import { ImageSummaryRun } from './summarize-run.js';

const debug = debugInstance('canvas:inferd:daemon');

/**
 * InferdDaemon — canvas-inferd as its own process.
 *
 * Inference is the only part of Canvas that drags native code (onnxruntime,
 * transformers.js) into a dependency tree, and two ONNX versions cannot share
 * one process — inferd already had to isolate CLIP in a child for exactly that
 * reason. Running the whole service out-of-process moves that entire problem
 * out of canvas-server: the server keeps no model runtime, no ONNX pin, and a
 * segfaulting model worker can no longer take the API down with it.
 *
 * The socket carries a symmetric peer (see rpc/peer.js), not a request/response
 * API, because the work is genuinely two-way: the server enqueues a document,
 * the daemon asks back for its bytes, then hands vectors back for storage. The
 * server owns the data; the daemon owns the models. Neither can be a pure
 * client of the other.
 */
export class InferdDaemon {
    #socketPath;
    #server = null;
    #inferd;
    #peers = new Set();
    // wsId -> the peer that registered it, so a workspace callback goes back to
    // the process that actually holds that workspace.
    #workspacePeers = new Map();
    // wsId -> ImageSummaryRun. Captioning is a foreground job the user starts
    // and watches; its state lives here because the models do.
    #summaryRuns = new Map();
    #logger;

    constructor({ socketPath, logger = console, ...inferdOptions } = {}) {
        if (!socketPath) { throw new Error('InferdDaemon requires a socketPath'); }
        this.#socketPath = socketPath;
        this.#logger = logger;
        this.#inferd = new Inferd({
            ...inferdOptions,
            // Per-user config lives in canvas-server's user store, so resolving
            // it is a call back over the socket.
            resolveUserConfig: (userId) => this.#resolveUserConfig(userId),
        });
    }

    get inferd() { return this.#inferd; }
    get socketPath() { return this.#socketPath; }

    async start() {
        await fs.mkdir(path.dirname(this.#socketPath), { recursive: true });
        // A socket file left by a killed process refuses bind with EADDRINUSE
        // even though nobody is listening; probe before removing so a genuinely
        // live daemon is never yanked out from under its clients.
        await this.#clearStaleSocket();

        this.#server = net.createServer((socket) => this.#onConnection(socket));
        await new Promise((resolve, reject) => {
            this.#server.once('error', reject);
            this.#server.listen(this.#socketPath, () => {
                this.#server.removeListener('error', reject);
                resolve();
            });
        });
        // The socket is the server's private control channel: owner-only.
        await fs.chmod(this.#socketPath, 0o600).catch(() => {});
        this.#logger.info?.({ socketPath: this.#socketPath }, 'inferd daemon listening');
        return this;
    }

    async #clearStaleSocket() {
        try { await fs.stat(this.#socketPath); } catch { return; } // nothing there
        const live = await new Promise((resolve) => {
            const probe = net.connect(this.#socketPath);
            const done = (answer) => { probe.destroy(); resolve(answer); };
            probe.once('connect', () => done(true));
            probe.once('error', () => done(false));
            probe.setTimeout(1000, () => done(false));
        });
        if (live) { throw new Error(`inferd daemon already listening on ${this.#socketPath}`); }
        await fs.unlink(this.#socketPath).catch(() => {});
    }

    #onConnection(socket) {
        socket.setNoDelay(true);
        const peer = new RpcPeer(socket, { label: 'inferd-daemon' });
        this.#peers.add(peer);
        peer.handleAll(this.#methods(peer));
        peer.on('close', () => this.#onPeerGone(peer));
        peer.on('error', (error) => debug(`peer error: ${error.message}`));
        debug('client connected');
    }

    // A dropped connection means the workspaces it registered no longer have a
    // reachable adapter — leaving them registered would queue work that can
    // never resolve its input.
    #onPeerGone(peer) {
        this.#peers.delete(peer);
        for (const [wsId, owner] of [...this.#workspacePeers]) {
            if (owner !== peer) { continue; }
            this.#workspacePeers.delete(wsId);
            try { this.#inferd.unregisterWorkspace(wsId); }
            catch (error) { debug(`unregister ${wsId} after disconnect failed: ${error.message}`); }
        }
        debug('client disconnected');
    }

    #resolveUserConfig(userId) {
        // User config is not workspace-scoped, so it goes to any live client.
        // In practice one canvas-server owns a daemon; if several ever share
        // one, per-user config would need routing of its own.
        const [peer] = this.#peers;
        if (!peer) { return null; }
        return peer.call('server.resolveUserConfig', [userId]);
    }

    /** The adapter inferd sees for a workspace — every call is an RPC home. */
    #adapterFor(peer, wsId) {
        return {
            resolveInput: (docId) => peer.call('ws.resolveInput', [wsId, docId]),
            storeVectors: (docId, schema, updatedAt, chunks, opts) =>
                peer.call('ws.storeVectors', [wsId, docId, schema, updatedAt, chunks, opts]),
            getUnembedded: (space, schemas) => peer.call('ws.getUnembedded', [wsId, space, schemas]),
            documentIdsUnderScope: (scope) => peer.call('ws.documentIdsUnderScope', [wsId, scope]),
            clearSpace: (space) => peer.call('ws.clearSpace', [wsId, space]),
            onQueueDrained: () => peer.call('ws.onQueueDrained', [wsId]),
            // Used by the captioning run: which documents are images, and where
            // a caption goes. Both are facts only the server holds.
            imageDocumentIds: () => peer.call('ws.imageDocumentIds', [wsId]),
            setSummary: (docId, text) => peer.call('ws.setSummary', [wsId, docId, text]),
        };
    }

    #summaryRunFor(wsId, peer) {
        let run = this.#summaryRuns.get(wsId);
        if (!run) {
            run = new ImageSummaryRun(
                wsId,
                this.#adapterFor(peer, wsId),
                (bytes, opts) => this.#inferd.describeImage(wsId, bytes, opts),
            );
            this.#summaryRuns.set(wsId, run);
        }
        return run;
    }

    /**
     * The daemon's method table. Everything canvas-server needs from inferd,
     * and nothing else — the surface is the contract, so a method missing here
     * is a deliberate "the server does not get to do that".
     */
    #methods(peer) {
        const inferd = this.#inferd;
        return {
            'inferd.ping': () => ({ ok: true, pid: process.pid }),

            // ── Workspace registration ──────────────────────────────────────
            'inferd.registerWorkspace': (wsId, opts = {}) => {
                this.#workspacePeers.set(wsId, peer);
                inferd.registerWorkspace(wsId, this.#adapterFor(peer, wsId), opts);
                return true;
            },
            'inferd.unregisterWorkspace': (wsId) => {
                this.#workspacePeers.delete(wsId);
                this.#summaryRuns.get(wsId)?.stop();
                this.#summaryRuns.delete(wsId);
                inferd.unregisterWorkspace(wsId);
                return true;
            },

            // ── Work ────────────────────────────────────────────────────────
            'inferd.enqueueMany': (wsId, docIds) => inferd.enqueueMany(wsId, docIds) ?? true,
            'inferd.reconcile': (wsId, opts) => inferd.reconcile(wsId, opts),
            'inferd.drained': (wsId) => inferd.drained(wsId),
            'inferd.pause': (wsId) => inferd.pause(wsId) ?? true,
            'inferd.resume': (wsId) => inferd.resume(wsId) ?? true,

            // ── Query-side embedding ────────────────────────────────────────
            'inferd.embedQueryForWorkspace': (wsId, text, space) =>
                inferd.embedQueryForWorkspace(wsId, text, space),
            'inferd.embedImageQuery': (wsId, bytes, contentType, space) =>
                inferd.embedImageQuery(wsId, bytes, contentType, space),

            // ── Vision ──────────────────────────────────────────────────────
            'inferd.describeImage': (wsId, bytes, opts) => inferd.describeImage(wsId, bytes, opts),
            'inferd.resetDescribeWorkers': (wsId) => inferd.resetDescribeWorkers(wsId),

            // Captioning a whole workspace: the loop, the failure policy and the
            // run state all live here (see summarize-run.js). The server starts
            // it and polls; it does not drive it.
            'inferd.startImageSummaries': async (wsId, opts = {}) => {
                const ctx = await inferd.contextForWorkspace(wsId);
                if (!ctx.config.summarize?.image?.enabled) {
                    throw new Error('image summaries are disabled — enable summarize.image first');
                }
                // A previous run may have been stopped by a crashed model
                // worker, which inferd refuses to respawn on its own. Starting a
                // run by hand is the deliberate retry that re-arms it.
                try { await inferd.resetDescribeWorkers(wsId); } catch { /* best effort */ }
                return this.#summaryRunFor(wsId, peer).start(opts);
            },
            'inferd.stopImageSummaries': (wsId) => this.#summaryRunFor(wsId, peer).stop(),
            'inferd.imageSummaryStatus': (wsId) => this.#summaryRunFor(wsId, peer).status,

            // ── Config / status ─────────────────────────────────────────────
            // contextForWorkspace returns a live object (providers, router
            // instances). Only its config and routing rules are readable data,
            // so that is all that crosses — the rest cannot be serialised and
            // has no meaning in another process.
            'inferd.contextFor': async (userId) => {
                const ctx = await inferd.contextFor(userId);
                return { config: ctx.config, ...(ctx.invalid ? { invalid: ctx.invalid } : {}) };
            },
            'inferd.workspacesOf': (userId) => inferd.workspacesOf(userId),
            'inferd.testProvider': (provider, opts) => testProvider(inferd, provider, opts),
            'inferd.contextForWorkspace': async (wsId) => {
                const ctx = await inferd.contextForWorkspace(wsId);
                return { config: ctx.config, router: { rules: ctx.router?.rules || [] } };
            },
            'inferd.spaceConfigsForWorkspace': (wsId, opts) => inferd.spaceConfigsForWorkspace(wsId, opts),
            'inferd.invalidateWorkspace': (wsId, config) => inferd.invalidateWorkspace(wsId, config) ?? true,
            'inferd.invalidateUser': (userId) => inferd.invalidateUser(userId) ?? true,
            'inferd.validate': (config, opts) => inferd.validate(config, opts),
            'inferd.serverConfig': () => inferd.serverConfig,
            'inferd.setServerConfig': (config) => inferd.setServerConfig(config) ?? true,
            'inferd.status': () => inferd.status(),
            'inferd.workspaceStatus': (wsId) => inferd.workspaceStatus(wsId),

            // Pure helpers the API routes need. They live here so canvas-server
            // needs no inferd import at all — the whole point of the split.
            'inferd.redactConfig': (config) => redactConfig(config),
            'inferd.checkConfigEndpoints': (config, opts) => checkConfigEndpoints(config, opts),
            'inferd.checkEndpoint': (url, opts) => checkEndpoint(url, opts),
            'inferd.endpointFor': (spec) => endpointFor(spec),
        };
    }

    async stop() {
        for (const peer of this.#peers) { peer.destroy(); }
        this.#peers.clear();
        this.#workspacePeers.clear();
        try { await this.#inferd.stop(); } catch (error) { debug(`inferd stop failed: ${error.message}`); }
        if (this.#server) {
            await new Promise((resolve) => this.#server.close(resolve));
            this.#server = null;
        }
        await fs.unlink(this.#socketPath).catch(() => {});
    }
}

export default InferdDaemon;
