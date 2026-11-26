#!/bin/sh
# scripts/*.py の argparse フラグ重複をファイル単位で検出する（--dir の二重定義など）
set -eu
cd "$(dirname "$0")/.."
fail=0
for f in scripts/*.py; do
	dup=$(grep -o 'add_argument("--[a-z0-9-]*"' "$f" | sort | uniq -d || true)
	if [ -n "$dup" ]; then
		echo "$f:"
		echo "$dup"
		fail=1
	fi
done
[ "$fail" -eq 0 ] && echo "ok: no duplicate flags"
exit "$fail"
