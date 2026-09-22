// A small real PNG exercises image decoding as well as embedding.
const TEST_PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAwElEQVR4nNXOQREAAAyDMKRX+kTw4BYFYc9RByzqgEUdsKgDFnXAog5Y1AGLOmBRByzqgEUdsKgDFnXAog5Y1AGLOmBRByzqgEUdsKgDFnXAog5Y1AGLOmBRByzqgEUdsKgDFnXAog5Y1AGLOmBRByzqgEUdsKgDFnXAog5Y1AGLOmBRByzqgEUdsKgDFnXAog5Y1AGLOmBRByzqgEUdsKgDFnXAog5Y1AGLOmBRByzqgEUdsKgDFnXAog5Y1AHrAB8+AWlViFU0AAAAAElFTkSuQmCC', 'base64');

export async function testProvider(inferd, provider, { model, modality = 'text', probe = false } = {}) {
    if (!['text', 'image'].includes(modality)) { throw new Error('modality must be text or image'); }
    const instance = inferd.providerFor(provider);
    if (probe) {
        const cached = typeof instance.modelCached === 'function' ? await instance.modelCached(model) : null;
        return { cached, modality };
    }
    const started = Date.now();
    let vector; let dim;
    if (modality === 'image') {
        if (typeof instance.embedImage !== 'function') { throw new Error('provider type does not support image embedding'); }
        const result = await instance.embedImage([TEST_PNG], { model }, { contentTypes: ['image/png'] });
        vector = result.vectors?.[0];
        dim = result.dim;
    } else {
        ({ vector, dim } = await instance.embedQuery('canvas embedding connectivity check', { model }));
    }
    return { ok: Array.isArray(vector), dim: dim || vector?.length || 0, latencyMs: Date.now() - started, modality };
}
