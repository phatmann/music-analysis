#!/usr/bin/env bash
set -e
BASEDIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

if [[ ! -f $BASEDIR/fma_metadata/tracks.csv ]]; then
    unxz $BASEDIR/fma_metadata/tracks.csv.xz
fi

uvicorn main:app --reload --host 0.0.0.0 --port 3001
