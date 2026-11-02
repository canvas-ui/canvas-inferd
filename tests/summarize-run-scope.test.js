import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ImageSummaryRun } from '../src/summarize-run.js';

test('a scoped caption run only visits the images the workspace returns for that path', async () => {
    const asked = [];
    const described = [];
    const adapter = {
        imageDocumentIds: async (scope) => { asked.push(scope); return scope ? [2] : [1, 2, 3]; },
        resolveInput: async (id) => ({ modality: 'image', bytes: Buffer.from([id]), contentType: 'image/png' }),
        setSummary: async (id, text) => { described.push([id, text]); },
    };
    const run = new ImageSummaryRun('ws', adapter, async () => 'a photo');

    const result = await run.start({ scope: 'ctx://photos' });
    assert.equal(result.started, true);
    assert.equal(result.status.total, 1);
    assert.equal(result.status.scope, 'ctx://photos');
    while (run.running) { await new Promise(r => setImmediate(r)); }
    assert.deepEqual(asked, ['ctx://photos']);
    assert.deepEqual(described, [[2, 'a photo']]);

    await run.start();
    while (run.running) { await new Promise(r => setImmediate(r)); }
    assert.deepEqual(asked, ['ctx://photos', null]);
    assert.equal(run.status.scope, undefined);
});
