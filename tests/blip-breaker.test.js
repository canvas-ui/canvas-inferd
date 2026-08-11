import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

import BlipProvider from '../src/providers/blip.js';

/**
 * The crash breaker, exercised against a REALLY killed worker.
 *
 * The bug this exists for: an OOM-killed caption worker was respawned on the
 * next image, reloading GB-scale weights straight back into the condition that
 * killed it — so one kill became a run that thrashed through thousands of
 * images. A crash must stop the model; only a deliberate new run re-arms it.
 *
 * SIGKILL from outside is exactly the shape the kernel's OOM killer produces.
 */

const IMAGE = Buffer.from([0xff, 0xd8, 0xff, 0xe0]);

// Give the fork a moment to exist, then take it out from under the provider.
async function killWorker(provider) {
    for (let i = 0; i < 50; i++) {
        const [worker] = provider.status().workers;
        if (worker?.pid) {
            process.kill(worker.pid, 'SIGKILL');
            return worker.pid;
        }
        await new Promise((resolve) => setTimeout(resolve, 20));
    }
    throw new Error('worker never spawned');
}

describe('BLIP crash breaker', () => {
    test('a killed worker fails its request and is not respawned', async () => {
        const provider = new BlipProvider({ model: 'stub/never-loads' });
        try {
            const inFlight = provider.describeImage(IMAGE);
            await killWorker(provider);

            const error = await inFlight.then(() => null, (e) => e);
            assert.ok(error, 'the in-flight caption should reject');
            // The caller learns it must stop, not retry.
            assert.equal(error.workerDead, true);
            assert.equal(error.code, 'INFERD_WORKER_DEAD');

            // The next call must NOT fork a replacement.
            const second = await provider.describeImage(IMAGE).then(() => null, (e) => e);
            assert.ok(second, 'the next caption should reject too');
            assert.equal(second.workerDead, true);
            assert.match(second.message, /Not respawning/);
            assert.equal(provider.status().workers.length, 0);
            assert.equal(provider.status().dead.length, 1);
        } finally {
            await provider.stop();
        }
    });

    test('an explicit reset re-arms it, and stop() clears the tombstone', async () => {
        const provider = new BlipProvider({ model: 'stub/never-loads' });
        try {
            const inFlight = provider.describeImage(IMAGE);
            await killWorker(provider);
            await inFlight.catch(() => {});
            assert.equal(provider.status().dead.length, 1);

            assert.equal(provider.resetWorkers(), true, 'reset reports it re-armed something');
            assert.equal(provider.status().dead.length, 0);
            assert.equal(provider.resetWorkers(), false, 'a second reset has nothing to do');
        } finally {
            await provider.stop();
        }
    });
});
