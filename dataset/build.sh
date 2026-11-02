#!/usr/bin/env bash
# Build the published dataset into $STAGE (run by `hf publish dataset`): train.jsonl (5,000 questions, seed 0) and
# test.jsonl (1,000, seed 1, less any question whose text is also in train, renumbered).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${STAGE:?set STAGE to the folder to write to}"
uv run python -m dataset.generate --n 5000 --seed 0 --out "$STAGE/train.jsonl"
uv run python -m dataset.generate --n 1000 --seed 1 --out "$STAGE/all_test.jsonl"
uv run python -c "
import json, sys
train = {json.loads(l)['text'] for l in open(sys.argv[1])}
test = [r for l in open(sys.argv[2]) if (r := json.loads(l))['text'] not in train]
open(sys.argv[3], 'w').write(''.join(json.dumps({**r, 'id': i}, ensure_ascii=False) + '\n' for i, r in enumerate(test)))
" "$STAGE/train.jsonl" "$STAGE/all_test.jsonl" "$STAGE/test.jsonl"
rm "$STAGE/all_test.jsonl"
