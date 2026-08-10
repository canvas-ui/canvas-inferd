# canvas-inferd

The Canvas inference service, extracted from Canvas Server with full history.

Inferd produces model output. SynapsD stores vectors and performs candidate
filtering and ranking. Canvas Server owns the workspace adapter between them.
Raw media and model runtimes never enter SynapsD.

## Current capabilities

- Text embeddings through local ONNX/fastembed, Ollama, and OpenAI-compatible
  providers.
- Joint text/image embeddings through a forked CLIP/SigLIP worker or an
  OpenAI-compatible endpoint.
- Four-layer configuration: built-in defaults, server, user, then workspace.
- Provider pooling, endpoint validation, a server-wide inference concurrency
  limit, and isolated per-workspace queues.
- Durable reconciliation through SynapsD's model-keyed `seen` and `vectors`
  bitmap ledgers.
- Ephemeral text and image query embeddings. Query inputs are never indexed.

The built-in spaces are:

| Space | Model | Dimensions |
|---|---|---:|
| `text` | `bge-small-en-v1.5` | 384 |
| `image` | `Xenova/clip-vit-base-patch32` | 512 |

Providers implement:

```js
embedText(texts, rule)
embedQuery(text, rule)
embedImage(images, rule, metadata)
status()
stop()
```

`embedImage` may reject unsupported providers.

## Workspace adapter

Canvas Server registers each workspace with:

```js
inferd.registerWorkspace(workspaceId, {
  resolveInput,
  storeVectors,
  getUnembedded,
  clearSpace,
  documentIdsUnderScope,
  onQueueDrained,
})
```

`resolveInput(docId)` returns text or image input plus its schema, timestamp,
comment, and generated summary. Inferd routes and batches it, then calls
`storeVectors`. `reconcile()` asks `getUnembedded` for the durable backlog, so a
restart or failed job requires no separate queue database.

The public query-side methods are:

- `embedQueryForWorkspace(workspaceId, text, space)`
- `embedImageQuery(workspaceId, bytes, contentType, space)`

SynapsD receives their vectors through Canvas Server's typed match descriptor
and ranks only documents which survived structural filtering.

## Planned capabilities

- `describe`: captions, summaries, and transcripts. Generated text is written
  to `metadata.summary`; SynapsD already indexes that field for FTS and as the
  reserved text chunk `-2`.
- `extract`: model hidden states and projected semantic anchors.
- Priority scheduling before live streams become a caller: interactive work,
  background derivation, then bulk reconciliation.
- Streaming input orchestration for camera, audio, logs, and messages. Inferd
  owns inference and time-dependent cue decay; QuerySession owns bitmap state.

Semantic anchors use their own model-keyed space and `anchor:` query namespace.
`geo:` remains physical GPS. Anchor storage and pure bitmap algebra belong in
SynapsD; codebooks, projection, clocks, and model calls belong here. The first
anchor prototype should use the existing contrastively trained joint image
space. Gemma per-layer representations remain an experiment until they beat
that baseline.

## Consumption

canvas-server consumes this repo as a git dependency
(`"canvas-inferd": "github:canvas-ui/canvas-inferd#main"`) and refreshes it to
latest main on every deployment update.

```
npm install
npm test
```
