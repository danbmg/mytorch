#!/bin/sh
## Defense demo (T1.14): the midterm checklist, step by step. Press Enter between steps.
## Needs data/ (python3 scripts/dedup_split.py) and my_torch_network_basic.nn.

cd "$(dirname "$0")/.."
REPO=$(pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

step() {
    printf '\n\033[1m== %s\033[0m\n' "$1"
    shift
    printf '$ %s\n' "$*"
    sh -c "$*"
    printf '\033[2m(exit %s, Enter to continue)\033[0m' "$?"
    read _
}

step "Build" "make re"
step "Help" "./my_torch_analyzer --help"
step "Usage error: message on stderr, exit 84" "./my_torch_analyzer --predict my_torch_network_basic.nn"
printf '8/8/8/8/8/8/k7/K7 b - - 0 1\n8/8/8/8/8/8/k7/K8 b - - 0 1\n' > "$TMP/bad.txt"
step "Invalid FEN: the file and the line are named, nothing on stdout" \
    "./my_torch_analyzer --predict my_torch_network_basic.nn $TMP/bad.txt"
step "Generate 3 blank networks from a config" "cd $TMP && $REPO/my_torch_generator $REPO/configs/basic.conf 3 && ls -l *.nn"
step "What a network file contains (binary weights, readable header)" \
    "python3 scripts/inspect_nn.py my_torch_network_basic.nn"
head -n 5000 data/train.txt > "$TMP/train_small.txt"
step "Train (progress on stderr), saved elsewhere: the original is untouched" \
    "./my_torch_analyzer --train --epochs 3 --save $TMP/trained.nn $TMP/basic_1.nn $TMP/train_small.txt"
step "Train without --save: the network is replaced, the old one kept as .bak" \
    "./my_torch_analyzer --train --epochs 1 $TMP/basic_2.nn $TMP/train_small.txt && ls -l $TMP/basic_2.nn*"
step "Predict: one line per board, in order" "head -n 5 data/test.txt && ./my_torch_analyzer --predict my_torch_network_basic.nn data/test.txt | head -n 5"
step "Accuracy on the whole test set (71 557 boards never seen)" \
    "./my_torch_analyzer --predict my_torch_network_basic.nn data/test.txt > $TMP/pred.txt && sed -E 's/^([^ ]+ ){6}//' data/test.txt | paste -d'|' $TMP/pred.txt - | awk -F'|' '{n++; ok += (\$1 == \$2)} END {printf \"%d / %d = %.2f%%\\n\", ok, n, 100 * ok / n}'"
step "Unit tests" "make tests_run"
printf '\nGraphs: docs/benchmarks.md. Training reports: docs/report/. Decisions: docs/decisions.md.\n'
