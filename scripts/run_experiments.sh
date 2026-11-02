#!/bin/sh
## Every experiment of docs/benchmarks.md (T2.3-T2.5), about 1h30 on 8 cores. Graphs and CSV go to docs/img/.
## usage: sh scripts/run_experiments.sh [NAME ...]   (default: all; needs data/chess.npz from scripts/dedup_split.py)

set -e
cd "$(dirname "$0")/.."
D=data/chess.npz
OUT=docs/img
SMALL="--train-samples 100000 --val-samples 30000"  # few enough boards for overfitting to show
FULL="--val-samples 30000"                          # the whole training set
NO_ES="training.early_stopping=null"
[ -f "$D" ] || { echo "$D missing: run python3 scripts/dedup_split.py first" >&2; exit 84; }
mkdir -p "$OUT"

ONLY=" $* "
run() { [ "$ONLY" = "  " ] || case "$ONLY" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
compare() { name=$1; shift; python3 scripts/compare.py --data $D --base configs/basic.conf --out "$OUT/$name.png" "$@"; }

run regularization && compare regularization $SMALL --epochs 40 \
    --variant "none: layers.hidden.dropout=0 training.l2=0 $NO_ES" \
    --variant "l2: layers.hidden.dropout=0 $NO_ES" \
    --variant "dropout: training.l2=0 $NO_ES" \
    --variant "l2 + dropout: $NO_ES" \
    --variant "l2 + dropout + early stopping: " \
    --title "Regularisation: train (dashed) vs validation (solid), 100k training boards"

run schedules && compare schedules $FULL --epochs 30 \
    --variant "constant: lr_schedule={\"name\":\"constant\"} $NO_ES" \
    --variant "step: lr_schedule={\"name\":\"step\",\"step_size\":10,\"gamma\":0.5} $NO_ES" \
    --variant "exponential: lr_schedule={\"name\":\"exponential\",\"gamma\":0.95} $NO_ES" \
    --variant "cosine: lr_schedule={\"name\":\"cosine\",\"t_max\":30} $NO_ES" \
    --variant "plateau: lr_schedule={\"name\":\"plateau\",\"factor\":0.5,\"patience\":2} $NO_ES" \
    --title "Learning rate schedules (whole training set)"

run activations && compare activations $SMALL --epochs 30 \
    --variant "relu: $NO_ES" \
    --variant "leaky_relu: layers.hidden.activation=leaky_relu $NO_ES" \
    --variant "elu: layers.hidden.activation=elu $NO_ES" \
    --variant "tanh: layers.hidden.activation=tanh $NO_ES" \
    --variant "sigmoid: layers.hidden.activation=sigmoid $NO_ES" \
    --title "Hidden activation functions"

run batch && compare batch_size $SMALL --epochs 30 \
    --variant "batch 32: training.batch_size=32 $NO_ES" \
    --variant "batch 128: training.batch_size=128 $NO_ES" \
    --variant "batch 256: training.batch_size=256 $NO_ES" \
    --variant "batch 1024: training.batch_size=1024 $NO_ES" \
    --variant "batch 4096: training.batch_size=4096 $NO_ES" \
    --title "Batch size: stability of the convergence vs time"

run optimizers && compare optimizers $SMALL --epochs 30 \
    --variant "sgd (lr 0.1): optimizer={\"name\":\"sgd\",\"lr\":0.1} $NO_ES" \
    --variant "momentum (lr 0.01): optimizer={\"name\":\"momentum\",\"lr\":0.01} $NO_ES" \
    --variant "rmsprop (lr 0.001): optimizer={\"name\":\"rmsprop\",\"lr\":0.001} $NO_ES" \
    --variant "adam (lr 0.001): $NO_ES" \
    --title "Optimizers, each with its usual learning rate"

for search in architecture random genetic; do
    run $search && python3 scripts/hpo.py configs/search/$search.json --data $D --out $OUT/hpo_$search
done

# the best architectures of the searches, confirmed on the whole training set
run candidates && python3 scripts/compare.py --data $D $FULL --out $OUT/candidates.png \
    configs/basic.conf configs/wide.conf configs/deep.conf configs/deeper.conf \
    --title "Candidates on the whole training set (572k boards)"

run data && python3 scripts/learning_curve.py --data $D --out $OUT/learning_curve.png
run perf && python3 scripts/bench_perf.py --data $D --out $OUT/perf
exit 0
