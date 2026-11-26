#!/bin/sh
# GPU 機に渡す学習用バンドルを作る。中身は testdata/discord/model/ の 3 ファイル。
set -eu
cd "$(dirname "$0")/.."
out=${1:-discord_corpus_pc.tgz}
tar czf "$out" --exclude output.txt -C testdata/discord model
echo "wrote $out"
ls -lh "$out"
