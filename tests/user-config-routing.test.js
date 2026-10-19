import test from 'node:test';
import assert from 'node:assert/strict';
import net from 'node:net';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { InferdDaemon } from '../src/daemon.js';
import { RpcPeer } from '../src/rpc/peer.js';

async function connect(daemon, t, userConfig) {
    const socket = net.connect(daemon.socketPath);
    await once(socket, 'connect');
    const peer = new RpcPeer(socket);
    t.after(() => peer.destroy());
    peer.handleAll({ 'server.resolveUserConfig': async () => userConfig });
    return peer;
}

test('per-user config comes only from a peer that declared it, whatever the connect order', async t => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'inferd-routing-'));
    const daemon = new InferdDaemon({ socketPath: path.join(dir, 'daemon.sock'), logger: { info() {}, warn() {}, error() {} } });
    t.after(async () => { await daemon.stop(); await rm(dir, { recursive: true, force: true }); });
    await daemon.start();

    // A workspace child connects first and has no user store.
    const child = await connect(daemon, t, null);
    const hub = await connect(daemon, t, { spaces: { text: { model: 'hub-model' } } });
    await hub.call('inferd.provideUserConfig', []);

    const ctx = await child.call('inferd.contextFor', ['alice']);
    assert.equal(ctx.config.spaces.text.model, 'hub-model');

    hub.destroy();
    await new Promise(r => setTimeout(r, 50));
    const without = await child.call('inferd.contextFor', ['bob']);
    assert.notEqual(without.config?.spaces?.text?.model, 'hub-model');
});
