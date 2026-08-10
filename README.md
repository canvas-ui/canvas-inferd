# canvas-inferd

The Canvas inference service. Extracted from `canvas-server/src/services/embedd`
(renamed inferd 2026-08) with full history.

**Capabilities**
- `embed` — text + image embeddings: local ONNX (fastembed) and CLIP/SigLIP
  (forked worker), remote Ollama and OpenAI-compatible endpoints. Per-user and
  per-workspace config layering, provider pooling, SSRF endpoint guard,
  per-workspace queues with a durable reconcile ledger (synapsd bitmaps).
- `describe` — planned: vision-model captions (qwen-vl) written back to
  `metadata.summary` (rails already live in synapsd/canvas-server).
- `extract` — planned: hidden-state / anchor projection experiments.

synapsd stores and namespaces vectors; inferd produces them. Neither ranks —
that's the database's job. See canvas-server's `TODO.multimodal.md` for the
roadmap this service anchors.

## Consumption

canvas-server consumes this repo as a git dependency
(`"canvas-inferd": "github:canvas-ui/canvas-inferd#main"`) and refreshes it to
latest main on every deployment update.

```
npm install
npm test
```
