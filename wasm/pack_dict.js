#!/usr/bin/env node
// Packs MeCab binary dictionary files into a single blob for WASM loading.
// Usage: node pack_dict.js /path/to/dict/dir output.bin
//
// Expected files in dict dir:
//   sys.dic, matrix.def (or matrix.bin), char.def (or char.bin), unk.def (or unk.dic)
//
// The output blob has a 36-byte header followed by the raw file data:
//
//   Offset  Size  Field
//   ──────  ────  ─────────────────────────────────────────
//        0     4  magic = 0x4D434142  ("MCAB" in LE)
//        4     4  offset of sys.dic data  (from blob start)
//        8     4  length of sys.dic data
//       12     4  offset of matrix.bin data
//       16     4  length of matrix.bin data
//       20     4  offset of char.bin data
//       24     4  length of char.bin data
//       28     4  offset of unk.dic data
//       32     4  length of unk.dic data
//       36     …  data sections (in order listed above)

'use strict';

const fs = require('fs');
const path = require('path');

const MAGIC = 0x4D434142; // "MCAB" as LE u32
const HEADER_SIZE = 36;   // 4 (magic) + 4 * 2 * 4 (4 sections × offset+length × 4 bytes)

/**
 * Packs a MeCab dictionary directory into a single WASM-loadable blob.
 *
 * @param {string} dictDir   Path to directory containing MeCab binary files.
 * @param {string} outputPath  Destination path for the packed blob.
 */
function packDictionary(dictDir, outputPath) {
    // Primary file names and their fallback alternatives.
    const files = ['sys.dic', 'matrix.bin', 'char.bin', 'unk.dic'];
    const fallbacks = {
        'matrix.bin': 'matrix.def',
        'char.bin':   'char.def',
        'unk.dic':    'unk.def',
    };

    const buffers = files.map((f) => {
        const primary  = path.join(dictDir, f);
        const fallback = path.join(dictDir, fallbacks[f] || f);

        if (fs.existsSync(primary)) {
            console.log(`  ${f}: ${fs.statSync(primary).size} bytes`);
            return fs.readFileSync(primary);
        }
        if (fs.existsSync(fallback)) {
            console.log(`  ${f} (from ${path.basename(fallback)}): ${fs.statSync(fallback).size} bytes`);
            return fs.readFileSync(fallback);
        }
        console.warn(`  WARNING: ${f} not found (included as empty section)`);
        return Buffer.alloc(0);
    });

    // Build the 36-byte header.
    const header = Buffer.alloc(HEADER_SIZE);
    header.writeUInt32LE(MAGIC >>> 0, 0);

    let offset = HEADER_SIZE;
    for (let i = 0; i < 4; i++) {
        header.writeUInt32LE(offset >>> 0,           4 + i * 8);
        header.writeUInt32LE(buffers[i].length >>> 0, 8 + i * 8);
        offset += buffers[i].length;
    }

    const output = Buffer.concat([header, ...buffers]);
    fs.writeFileSync(outputPath, output);
    console.log(`Packed ${output.length} bytes -> ${outputPath}`);
}

// ── CLI entry point ──────────────────────────────────────────────────────────

if (require.main === module) {
    if (process.argv.length < 4) {
        console.error('Usage: node pack_dict.js <dict_dir> <output.bin>');
        process.exit(1);
    }

    const dictDir    = process.argv[2];
    const outputPath = process.argv[3];

    if (!fs.existsSync(dictDir) || !fs.statSync(dictDir).isDirectory()) {
        console.error(`Error: '${dictDir}' is not a valid directory`);
        process.exit(1);
    }

    packDictionary(dictDir, outputPath);
}

module.exports = { packDictionary };
