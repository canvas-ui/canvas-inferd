'use strict';

/**
 * Why a forked model worker died, said out loud.
 *
 * These children hold a whole model resident (BLIP/CLIP are GB-scale), which
 * makes them the fattest process on the box and so the OOM killer's first
 * pick. That death is SILENT by construction: the kernel SIGKILLs the child,
 * the parent sees a bare 'exit', and the caller only ever learns that its own
 * request rejected. Diagnosing one of these took a journalctl trawl for
 * `oom-kill` precisely because no part of the process ever said a word.
 *
 * Hence the split: an expected exit stays on `debug`, an abnormal one goes to
 * console.error whether or not DEBUG is set. The exit SIGNAL is the one fact
 * that names the cause, so it must survive default log settings — a
 * debug-gated line is the same as no line at all on a server nobody has
 * DEBUG=canvas:inferd:* exported on.
 */

// Node reports an OS kill as `signal` with a null code. The wording stops
// short of certainty — SIGKILL is also what `kill -9` and a container memory
// limit look like — but it points at the log that settles it.
const SIGNAL_HINTS = {
    SIGKILL: 'killed by the OS, typically the OOM killer — confirm with `journalctl -u canvas-server | grep -i oom-kill`',
    SIGSEGV: 'segfault inside the native runtime (onnxruntime)',
    SIGABRT: 'native abort, often an allocation failure inside onnxruntime',
    SIGTERM: 'terminated externally',
    SIGHUP: 'controlling terminal went away',
};

/**
 * An error a caller should STOP on rather than retry: the model worker died and
 * the provider is refusing to respawn it. Consumers driving a per-item loop
 * (canvas-server's caption run) branch on `workerDead` to abort the whole run
 * instead of re-entering a fork → load → get-killed cycle per item.
 */
export function workerDeadError(message) {
    const error = new Error(message);
    error.workerDead = true;
    error.code = 'INFERD_WORKER_DEAD';
    return error;
}

/** One-line cause, reused as the rejection message so callers surface it too. */
export function describeWorkerExit(label, model, code, signal) {
    const cause = signal
        ? (SIGNAL_HINTS[signal] || `killed by ${signal}`)
        : `exit code ${code}`;
    return `${label} worker (${model}) died — ${cause}`;
}

/**
 * Log a worker exit at the right volume. `expected` covers our own teardown
 * (provider stop, timeout kill), which is noise rather than news.
 */
export function logWorkerExit(debug, label, { model, pid, code, signal, pending = 0, expected = false }) {
    const who = `${label} worker (${model}, pid ${pid ?? '?'})`;
    if (expected) {
        debug(`${who} exited on request (code ${code}, signal ${signal || 'none'})`);
        return;
    }
    if (!signal && code === 0) {
        debug(`${who} exited cleanly`);
        return;
    }
    const dropped = pending ? `; ${pending} in-flight request(s) failed` : '';
    // The parent's own RSS dates the event against the machine's memory
    // ceiling — the child's is already gone by the time we get here.
    const rss = Math.round(process.memoryUsage.rss() / (1024 * 1024));
    console.error(`inferd: ${describeWorkerExit(label, model, code, signal)}${dropped}. Parent RSS ${rss} MB.`);
}
