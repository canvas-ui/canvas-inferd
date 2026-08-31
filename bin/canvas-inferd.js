#!/usr/bin/env node
'use strict';

/**
 * canvas-inferd — run the inference service as its own process.
 *
 *   canvas-inferd --socket /run/canvas/inferd.sock [--config path.json]
 *
 * Everything model-related lives on this side of the socket: the ONNX runtime,
 * transformers.js, the model cache. canvas-server holds none of it and talks to
 * this process over the socket (see src/daemon.js).
 */

import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { InferdDaemon } from '../src/daemon.js';

function parseArgs(argv) {
    const args = {};
    for (let i = 2; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith('--')) { continue; }
        const eq = arg.indexOf('=');
        if (eq > -1) { args[arg.slice(2, eq)] = arg.slice(eq + 1); }
        else if (argv[i + 1] && !argv[i + 1].startsWith('--')) { args[arg.slice(2)] = argv[++i]; }
        else { args[arg.slice(2)] = true; }
    }
    return args;
}

const args = parseArgs(process.argv);

if (args.help) {
    process.stdout.write(`canvas-inferd — inference daemon

  --socket <path>   unix socket to listen on
                    (default: $CANVAS_INFERD_SOCKET, else <tmp>/canvas-inferd.sock)
  --config <path>   JSON config file (providers, spaces, rules, summarize)
  --cache-dir <p>   model cache directory (default: $CANVAS_INFERD_CACHE_DIR)
  --help            this text
`);
    process.exit(0);
}

const socketPath = args.socket
    || process.env.CANVAS_INFERD_SOCKET
    || path.join(os.tmpdir(), 'canvas-inferd.sock');

let fileConfig = {};
const configPath = args.config || process.env.CANVAS_INFERD_CONFIG;
if (configPath) {
    try {
        fileConfig = JSON.parse(await fs.readFile(configPath, 'utf8'));
    } catch (error) {
        process.stderr.write(`canvas-inferd: cannot read config ${configPath}: ${error.message}\n`);
        process.exit(1);
    }
}

const daemon = new InferdDaemon({
    socketPath,
    onnxCacheDir: args['cache-dir'] || process.env.CANVAS_INFERD_CACHE_DIR || fileConfig.cacheDir,
    ollamaHost: fileConfig.ollamaHost,
    vllmBaseUrl: fileConfig.vllmBaseUrl,
    concurrency: fileConfig.concurrency,
    providers: fileConfig.providers,
    spaces: fileConfig.spaces,
    rules: fileConfig.rules,
    summarize: fileConfig.summarize,
    logger: {
        info: (obj, msg) => process.stdout.write(`${msg || ''} ${JSON.stringify(obj)}\n`),
        warn: (obj, msg) => process.stderr.write(`${msg || ''} ${JSON.stringify(obj)}\n`),
    },
});

try {
    await daemon.start();
} catch (error) {
    process.stderr.write(`canvas-inferd: ${error.message}\n`);
    process.exit(1);
}

// Model workers hold native handles and a socket file sits on disk; both need
// an orderly wind-down or the next start trips over them.
let stopping = false;
const shutdown = async (signal) => {
    if (stopping) { return; }
    stopping = true;
    process.stderr.write(`canvas-inferd: ${signal}, shutting down\n`);
    try { await daemon.stop(); } catch { /* going away regardless */ }
    process.exit(0);
};
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
