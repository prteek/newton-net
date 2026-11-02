#!/usr/bin/env bash
# Publish to Hugging Face (locally: `hf auth login` as prteek; in CI: HF_TOKEN):
#   dataset  prteek/newton-net-questions: train (5,000 questions, seed 0) and test (1,000, seed 1, less any text also in
#            train) from dataset/generate.py, with dataset/README.md as the card
#   model    prteek/newton-net: the model code so far (model/*.py), with model/README.md as the card; no weights yet
#   space    prteek/newton-net: space/ (app, card, requirements, diagram) plus the model code the app imports
# Usage: ./publish.sh [message] [dataset|model|space ...]   (default: dataset model space)
# Set HF_CLI=hf to use an installed hf CLI instead of `uvx hf`.
set -euo pipefail
cd "$(dirname "$0")"
msg="${1:-Update}"
shift || true
if [ $# -gt 0 ]; then targets=("$@"); else targets=(dataset model space); fi
read -r -a hf <<< "${HF_CLI:-uvx --from huggingface_hub hf}"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

for t in "${targets[@]}"; do
  case "$t" in
    dataset)
      mkdir "$stage/dataset"
      uv run python -m dataset.generate --n 5000 --seed 0 --out "$stage/dataset/train.jsonl"
      uv run python -m dataset.generate --n 1000 --seed 1 --out "$stage/dataset/all_test.jsonl"
      uv run python -c "
import json, sys
train = {json.loads(l)['text'] for l in open(sys.argv[1])}
test = [r for l in open(sys.argv[2]) if (r := json.loads(l))['text'] not in train]
open(sys.argv[3], 'w').write(''.join(json.dumps({**r, 'id': i}, ensure_ascii=False) + '\n' for i, r in enumerate(test)))
" "$stage/dataset/train.jsonl" "$stage/dataset/all_test.jsonl" "$stage/dataset/test.jsonl"
      rm "$stage/dataset/all_test.jsonl"
      cp dataset/README.md "$stage/dataset/README.md"
      "${hf[@]}" upload prteek/newton-net-questions "$stage/dataset" . --repo-type dataset --commit-message "$msg" ;;
    model)
      mkdir -p "$stage/model/model"
      cp model/README.md "$stage/model/"
      cp model/__init__.py model/extract.py model/reading.py model/planner.py "$stage/model/model/"
      "${hf[@]}" upload prteek/newton-net "$stage/model" . --repo-type model --commit-message "$msg" ;;
    space)
      mkdir -p "$stage/space/model"
      cp space/app.py space/README.md space/requirements.txt space/architecture.svg "$stage/space/"
      cp model/__init__.py model/extract.py model/reading.py model/planner.py "$stage/space/model/"
      "${hf[@]}" upload prteek/newton-net "$stage/space" . --repo-type space --commit-message "$msg" ;;
    *)
      echo "unknown target: $t" >&2; exit 1 ;;
  esac
done
