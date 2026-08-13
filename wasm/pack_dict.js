#!/usr/bin/env node
// Packs MeCab binary dictionary files into a single blob for WASM loading.
// Usage: node pack_dict.js /path/to/dict/dir output.bin
//
// Required files in dict dir (all four, all BINARY — see below):
//   sys.dic, matrix.bin, char.bin, unk.dic
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
//
// ── Why there is no .def fallback ────────────────────────────────────────────
// char.def / matrix.def / unk.def are MeCab *source* files: human-readable text
// (EUC-JP in stock IPADIC). The mecrab readers map their binary counterparts and
// reject anything whose size does not match the exact layout formula, so a blob
// carrying .def text can never load — and it fails inside `loadDictionary()` in
// the browser, far from the mistake. Earlier revisions of this script silently
// substituted the .def text, and silently wrote empty sections for files that
// were missing outright, producing exactly such blobs. Every section is now
// required and validated before anything is written.

'use strict';

const fs = require('fs');
const path = require('path');

const MAGIC = 0x4D434142; // "MCAB" as LE u32
const HEADER_SIZE = 36;   // 4 (magic) + 4 * 2 * 4 (4 sections × offset+length × 4 bytes)

/** Magic constant of a MeCab sys.dic / unk.dic header (mecrab dict/sys_dic.rs). */
const DICTIONARY_MAGIC_ID = 0xef718f77;
/** DIC_VERSION accepted by the reader. */
const DIC_VERSION = 102;
/** Header size of sys.dic / unk.dic. */
const DIC_HEADER_SIZE = 72;
/** Number of CharInfo entries in char.bin (CharDef::TABLE_SIZE). */
const CHARINFO_TABLE_SIZE = 0xFFFF;

/** Source (text) file that is NOT a substitute for each required binary file. */
const TEXT_SOURCE_OF = {
    'matrix.bin': 'matrix.def',
    'char.bin': 'char.def',
    'unk.dic': 'unk.def',
};

/** Error type for every "this blob would not load" condition. */
class PackError extends Error {}

/**
 * Validate a sys.dic / unk.dic section.
 *
 * @param {Buffer} buf   File contents.
 * @param {string} name  File name (for messages).
 * @param {number} expectedType  0 = sys.dic, 2 = unk.dic.
 */
function validateDic(buf, name, expectedType) {
    if (buf.length < DIC_HEADER_SIZE) {
        throw new PackError(
            `${name} is ${buf.length} bytes — too small for the ${DIC_HEADER_SIZE}-byte MeCab dictionary header.`);
    }
    const magic = buf.readUInt32LE(0);
    const claimedSize = (magic ^ DICTIONARY_MAGIC_ID) >>> 0;
    if (claimedSize !== buf.length) {
        throw new PackError(
            `${name} is not a MeCab binary dictionary: its magic encodes a file size of ${claimedSize}, ` +
            `but the file is ${buf.length} bytes. Compile the dictionary first ` +
            `(kizame dict compile, or mecab-dict-index).`);
    }
    const version = buf.readUInt32LE(4);
    if (version !== DIC_VERSION) {
        throw new PackError(`${name} has dictionary version ${version}; mecrab requires ${DIC_VERSION}.`);
    }
    const dictType = buf.readUInt32LE(8);
    if (dictType !== expectedType) {
        throw new PackError(
            `${name} has dict_type=${dictType}, expected ${expectedType} ` +
            `(0 = system dictionary, 2 = unknown-word dictionary).`);
    }
}

/**
 * Validate char.bin: 4 byte csize + csize*32 name bytes + 65535 * 4 CharInfo bytes,
 * exactly — `CharDef::parse_bytes` rejects every other size.
 *
 * @param {Buffer} buf  File contents.
 */
function validateCharBin(buf) {
    if (buf.length < 4) {
        throw new PackError(`char.bin is ${buf.length} bytes — too small to hold its category count.`);
    }
    const csize = buf.readUInt32LE(0);
    const expected = 4 + csize * 32 + CHARINFO_TABLE_SIZE * 4;
    if (buf.length !== expected) {
        throw new PackError(
            `char.bin is ${buf.length} bytes, but a ${csize}-category table requires exactly ${expected}. ` +
            `That mismatch is what packing the char.def text file looks like — compile it to char.bin first.`);
    }
}

/**
 * Validate matrix.bin: u16 lsize + u16 rsize + lsize*rsize i16 costs, exactly.
 *
 * @param {Buffer} buf  File contents.
 */
function validateMatrixBin(buf) {
    if (buf.length < 4) {
        throw new PackError(`matrix.bin is ${buf.length} bytes — too small to hold its size header.`);
    }
    const lsize = buf.readUInt16LE(0);
    const rsize = buf.readUInt16LE(2);
    const expected = 4 + lsize * rsize * 2;
    if (buf.length !== expected) {
        throw new PackError(
            `matrix.bin is ${buf.length} bytes, but its ${lsize}×${rsize} header requires exactly ${expected}. ` +
            `That mismatch is what packing the matrix.def text file looks like — compile it to matrix.bin first.`);
    }
}

/** Required sections, in blob order, with their validators. */
const SECTIONS = [
    { name: 'sys.dic', validate: (buf) => validateDic(buf, 'sys.dic', 0) },
    { name: 'matrix.bin', validate: validateMatrixBin },
    { name: 'char.bin', validate: validateCharBin },
    { name: 'unk.dic', validate: (buf) => validateDic(buf, 'unk.dic', 2) },
];

/**
 * Read one required section, refusing to substitute its .def source text.
 *
 * @param {string} dictDir  Directory holding the compiled dictionary.
 * @param {string} name     Binary file name.
 * @returns {Buffer} File contents.
 */
function readSection(dictDir, name) {
    const file = path.join(dictDir, name);
    if (!fs.existsSync(file)) {
        const source = TEXT_SOURCE_OF[name];
        if (source && fs.existsSync(path.join(dictDir, source))) {
            throw new PackError(
                `${name} is missing; only its source file ${source} is present. ` +
                `${source} is MeCab source text (EUC-JP in stock IPADIC), not a binary section — ` +
                `packing it produces a blob that fails to load in the browser. Compile the dictionary ` +
                `first ("kizame dict compile --input ${dictDir} --output <dir>", or ` +
                `"mecab-dict-index -d ${dictDir} -o <dir>"), then pack the output directory.`);
        }
        throw new PackError(
            `${name} is missing from ${dictDir}. All four sections ` +
            `(sys.dic, matrix.bin, char.bin, unk.dic) are required — an empty section ` +
            `produces a blob that fails to load.`);
    }

    const buf = fs.readFileSync(file);
    if (buf.length === 0) {
        throw new PackError(`${name} is empty — a zero-length section produces a blob that fails to load.`);
    }
    return buf;
}

/**
 * Packs a MeCab dictionary directory into a single WASM-loadable blob.
 *
 * @param {string} dictDir   Path to directory containing MeCab binary files.
 * @param {string} outputPath  Destination path for the packed blob.
 * @throws {PackError} If any of the four binary sections is missing, empty, or
 *   not in its binary format — i.e. whenever the blob would fail to load.
 */
function packDictionary(dictDir, outputPath) {
    const buffers = SECTIONS.map(({ name, validate }) => {
        const buf = readSection(dictDir, name);
        validate(buf);
        console.log(`  ${name}: ${buf.length} bytes`);
        return buf;
    });

    // Build the 36-byte header.
    const header = Buffer.alloc(HEADER_SIZE);
    header.writeUInt32LE(MAGIC >>> 0, 0);

    let offset = HEADER_SIZE;
    for (let i = 0; i < SECTIONS.length; i++) {
        header.writeUInt32LE(offset >>> 0, 4 + i * 8);
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

    try {
        packDictionary(dictDir, outputPath);
    } catch (err) {
        if (err instanceof PackError) {
            console.error(`Error: ${err.message}`);
            process.exit(1);
        }
        throw err;
    }
}

module.exports = { packDictionary, PackError };
