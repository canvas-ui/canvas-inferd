# canvas-inferd TODO

## Direction

**Remote API endpoints are the intended default.** The local runtimes (onnx,
clip, blip) are experiments for specific use-case scenarios — offline, private,
or edge-bound work — not the path everything should walk. Design new work so
the remote provider is the ordinary case and a resident local model is the
opt-in.

## Open

### Configurable resident-worker cap
Right now every distinct model gets its own forked child, each holding its
weights resident, and nothing bounds how many coexist. On a small box (the
8 GB VPS that surfaced this) BLIP + CLIP + the server + LanceDB is enough to
put the kernel OOM killer in play — it takes the fattest process, which is
always a model worker.

Wanted: a configurable cap on resident workers — `1` being a sensible default
for small deployments — evicting least-recently-used workers past the limit.
Config layer as usual (built-in ← server inferd.json ← user ← workspace), so a
big machine can raise it and an edge box can pin it to one.

Interaction to keep in mind: eviction must not look like a crash. The crash
breaker (see below) keys off UNEXPECTED exits, so an evicting stop has to mark
`entry.expected = true` the way `stop()` and the timeout kill already do.

### Follow-ups from the OOM incident (2026-08-11)
- Breaker is per-model and re-arms on an explicit new run
  (`resetDescribeWorkers`) or provider stop. Consider surfacing "stopped, why"
  in the workspace status endpoint beyond the caption run's own status.
- Embedding (queue-driven) has the same exposure as captioning did: a crashed
  CLIP worker now trips the breaker rather than re-forking per document, but
  the queue's own retry/backoff behaviour against a dead provider has not been
  reviewed.
- `worker-exit.js` centralises exit reporting for the forked providers. ONNX
  runs a worker THREAD (dies with the process, cannot be OOM-killed alone) and
  is deliberately not on the breaker.
