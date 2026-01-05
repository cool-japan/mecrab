#!/usr/bin/env bash
# MeCrab-specific cargo wrapper
# Unsets global RUSTFLAGS to avoid OpenBLAS linkage from other projects

env -u RUSTFLAGS cargo "$@"
