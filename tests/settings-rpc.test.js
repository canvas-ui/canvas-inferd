import test from 'node:test';
import assert from 'node:assert/strict';
import net from 'node:net';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { InferdDaemon } from '../src/daemon.js';
import { RpcPeer } from '../src/rpc/peer.js';

// Exercise the actual wire boundary; no model downloads or inference required.
test('settings RPC supports user inheritance, provider tests and endpoint policies', async t => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'inferd-settings-'));
    const daemon = new InferdDaemon({ socketPath: path.join(dir, 'daemon.sock'), logger: { info() {}, warn() {}, error() {} } });
    t.after(async () => { await daemon.stop(); await rm(dir, { recursive: true, force: true }); });
    await daemon.start();
    const socket = net.connect(daemon.socketPath);
    await once(socket, 'connect');
    const peer = new RpcPeer(socket);
    t.after(() => peer.destroy());
    peer.handleAll({ 'server.resolveUserConfig': async () => ({ spaces: { text: { model: 'user-model' } } }) });
    const ctx = await peer.call('inferd.contextFor', ['alice']);
    assert.equal(ctx.config.spaces.text.model, 'user-model');
    assert.equal(ctx.providers, undefined);
    assert.deepEqual(await peer.call('inferd.workspacesOf', ['alice']), []);
    const problems = await peer.call('inferd.checkConfigEndpoints', [{ providers: { remote: { type: 'openai', baseUrl: 'http://127.0.0.1:8000/v1' } } }, { allowHosts: ['approved.local'] }]);
    assert.ok(problems.length > 0, 'allowlist must survive the RPC boundary');
    const modalities = [];
    daemon.inferd.providerFor = () => ({
        modelCached: async () => true,
        embedQuery: async () => { modalities.push('text'); return { vector: [1, 0], dim: 2 }; },
        embedImage: async ([bytes], _model, opts) => {
            assert.ok(Buffer.isBuffer(bytes));
            assert.equal(bytes.subarray(1, 4).toString(), 'PNG');
            assert.deepEqual(opts.contentTypes, ['image/png']);
            modalities.push('image');
            return { vectors: [[0, 1, 0]], dim: 3 };
        },
    });
    assert.deepEqual(await peer.call('inferd.testProvider', [{ type: 'clip' }, { model: 'fake', modality: 'image', probe: true }]), { cached: true, modality: 'image' });
    assert.deepEqual(modalities, [], 'cache probe must not load a model');
    for (const [modality, dim] of [['text', 2], ['image', 3]]) {
        const result = await peer.call('inferd.testProvider', [{ type: 'clip' }, { model: 'fake', modality }]);
        assert.equal(result.ok, true);
        assert.equal(result.dim, dim);
    }
    assert.deepEqual(modalities, ['text', 'image']);
    await assert.rejects(peer.call('inferd.testProvider', [{}, { modality: 'audio' }]), /modality/);
});
