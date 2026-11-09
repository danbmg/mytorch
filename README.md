# my_torch

Réseau de neurones **écrit de zéro** (Python + numpy, aucune bibliothèque de ML) et analyseur d'échiquier : pour chaque position FEN, il prédit `Nothing`, `Check White`, `Check Black`, `Checkmate White` ou `Checkmate Black` (couleur = camp qui donne l'échec).

## Résultats (test : 71 557 positions jamais vues)

`--predict` répond avec la moyenne du réseau sur les 4 versions symétriques de chaque plateau (couleurs échangées, colonnes a↔h inversées) ; la colonne « plateau seul » donne le réseau sans cette moyenne. Le F1 macro est celui de `--predict`.

| Réseau | Architecture | Accuracy (`--predict`) | plateau seul | F1 macro | Entraînement |
|---|---|---|---|---|---|
| **`my_torch_network_final.nn`** | 773 > 1024 > 512 > 5 | **95,34 %** | 95,01 % | **0,942** | 71 epochs, 3,2 h, train ×4 (les 4 versions de chaque plateau) |
| réseau précédent (`git show 70d6402:my_torch_network_final.nn`) | 773 > 1024 > 512 > 5 | 94,21 % | 93,82 % | 0,929 | 59 epochs, 3,6 h, train ×2 (couleurs échangées) |
| `docs/networks/baseline_basic.nn` (référence) | 773 > 256 > 128 > 5 | 89,61 % | 87,95 % | 0,880 | 30 epochs, 3,4 min |
| `bonus/my_torch_network_stalemate.nn` (6 classes, test de 82 057 positions) | 773 > 512 > 256 > 6 | 94,42 % (pats : F1 0,928) | 93,93 % | 0,932 | 50 epochs, 84 min, train ×4 |

Comment on est arrivé au réseau final : `docs/benchmarks.md`. Pourquoi chaque choix : `docs/decisions.md`. Rapports d'entraînement : `docs/report/`.

## Construire, tester, utiliser

Python ≥ 3.10 et numpy (matplotlib pour les scripts). Sur Ubuntu : `sudo apt install python3-numpy python3-matplotlib python3-pytest libopenblas0-pthread`. Sans `libopenblas0-pthread`, le numpy d'Ubuntu calcule avec le BLAS de référence : mêmes réponses, mais le `--predict` du test complet passe de 8 s à 75 s (`docs/benchmarks.md` §7). C'est aussi le cas du dump (image `epitechcontent/epitest-docker` : Ubuntu 26.04, Python 3.14, numpy 2.3) : 27 s pour les 71 557 plateaux, 0,5 s pour 1 000.

```bash
make && make tests_run                                                  # my_torch_analyzer, my_torch_generator, 50 tests
./my_torch_analyzer --predict my_torch_network_final.nn boards.txt       # une prédiction par ligne, rien d'autre sur stdout
./my_torch_analyzer --train --save new.nn my_torch_network_final.nn train.txt   # sans --save : écrase et garde un .bak
./my_torch_generator configs/basic.conf 3                               # basic_1.nn … basic_3.nn, réseaux vierges
python3 scripts/inspect_nn.py my_torch_network_final.nn                  # couches, hyperparamètres, poids
sh scripts/demo.sh                                                      # démo de soutenance pas à pas
```

Erreur (argument, fichier, FEN invalide avec son numéro de ligne, réseau illisible) : message sur stderr et code 84.

## Décrire un réseau (`configs/*.conf`, JSON)

```json
{"input_size": 773,
 "layers": [{"size": 1024, "activation": "relu", "dropout": 0.3}, {"size": 512, "activation": "relu", "dropout": 0.3},
            {"size": 5, "activation": "softmax"}],
 "optimizer": {"name": "adam", "lr": 0.001}, "lr_schedule": {"name": "plateau", "factor": 0.5, "patience": 2},
 "training": {"epochs": 80, "batch_size": 256, "l2": 0.0001, "grad_clip": 5.0, "early_stopping": {"patience": 6}},
 "labels": ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"], "seed": 42}
```

| Clé | Choix |
|---|---|
| `layers[i]` | `size`, `activation` (`relu`, `leaky_relu`, `elu`, `sigmoid`, `tanh`, `linear`, `softmax`), `dropout`, `init` (`auto`, `he`, `xavier`, `lecun`, `uniform`) |
| `loss`, `dtype` | `cross_entropy` (sortie softmax) ou `mse` ; `float32` ou `float64` |
| `optimizer` | `sgd`, `momentum`, `rmsprop`, `adam` ; `lr` = learning rate initial |
| `lr_schedule` | `constant`, `step`, `exponential`, `cosine`, `plateau` |
| `training` | `epochs`, `batch_size`, `l2`, `grad_clip`, `early_stopping`, `shuffle`, `eval_train_samples` |

Seuls `input_size` et `layers` sont obligatoires : le reste a une valeur par défaut, et une clé invalide est refusée en la nommant (`layers[0].activation: unknown 'relux'`).

**Fichier `.nn`** : archive npz avec un en-tête JSON lisible (config complète, activation de chaque couche, hyperparamètres, labels, historique) et les poids en **binaire float32**. Il porte une version de format (un fichier plus récent que le programme est refusé). L'écriture est atomique (fichier temporaire puis renommage) et `.bak` garde la version remplacée.

## Tout reproduire

```bash
python3 scripts/dedup_split.py                  # dédup + split 80/10/10 stratifié -> data/{train,val,test}.txt, data/chess.npz
python3 bonus/validate_movegen.py --symmetries  # les 4 versions de chaque position ont le label qu'on leur donne
python3 scripts/train.py configs/large.conf --data data/chess.npz --out my_torch_network_final.nn --report docs/report/final --flip-colors --flip-files
python3 scripts/evaluate.py my_torch_network_final.nn --data data/chess.npz   # accuracy de test en 5 et en 3 classes
sh scripts/run_experiments.sh                   # toutes les expériences de docs/benchmarks.md (~1h30 sur 8 cœurs)
python3 bonus/gen_stalemate.py                  # bonus : voir bonus/README.md
```

Seeds fixées partout (split, initialisation, mélange) : sur la même machine, avec le même numpy et le même nombre de threads BLAS (`OPENBLAS_NUM_THREADS`), la même commande redonne le même réseau, au bit près. Ailleurs, la bibliothèque de calcul matriciel (BLAS) peut additionner dans un autre ordre : les poids diffèrent dans les dernières décimales, et l'entraînement suit un chemin proche sans être identique. Le réseau livré a été entraîné dans un conteneur `python:3.12-slim` + numpy 1.26.4, OpenBLAS 6 threads (`docs/benchmarks.md` §10).

## Organisation

```
my_torch/    module de réseaux de neurones, générique (aucune notion d'échecs)
analyzer/    échecs : FEN, encodage 773, symétries, labels, ligne de commande
generator/   my_torch_generator
scripts/     dédup/split, entraînement + rapport, comparaisons, recherche d'hyperparamètres, benchmarks
configs/     configs de réseaux, espaces de recherche (search/)
tests/       gradients numériques, XOR, sauvegarde, config, symétries, analyzer, générateur de coups
docs/        benchmarks.md, decisions.md, img/ (graphes + CSV), report/ (rapports d'entraînement)
bonus/       stalemate : générateur de coups, génération de pats, réseau à 6 classes
```
