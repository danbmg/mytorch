# Module `my_torch` — référence

Bibliothèque de réseaux de neurones écrite de zéro avec numpy. Elle **ne connaît rien aux échecs** : elle crée, entraîne, sauvegarde et recharge des réseaux entièrement connectés décrits par une configuration JSON. Tout ce qui touche au FEN vit dans `analyzer/`.

| Fichier | Rôle |
|---|---|
| `my_torch/activations.py` | relu, leaky_relu, elu, sigmoid, tanh, linear, softmax : sortie et gradient, sur tout le batch |
| `my_torch/losses.py` | cross_entropy, binary_cross_entropy, mse |
| `my_torch/init.py` | initialisations he, xavier, lecun, uniform, et `auto` |
| `my_torch/optimizers.py` | sgd, momentum (option Nesterov), rmsprop, adam |
| `my_torch/schedules.py` | baisse du learning rate : constant, step, exponential, cosine, plateau |
| `my_torch/config.py` | validation de la config + valeurs par défaut |
| `my_torch/layers.py` | couche `Dense` avec dropout inversé |
| `my_torch/network.py` | `Network` : forward, backprop, `fit`, `predict`, MC dropout |
| `my_torch/serialization.py` | fichiers `.nn` (écriture atomique, versionnés) |
| `my_torch/metrics.py` | accuracy, matrice de confusion, précision / rappel / F1 par classe |
| `my_torch/parallel.py` | plusieurs entraînements en parallèle (processus + mémoire partagée) |
| `my_torch/hpo.py` | recherche d'hyperparamètres : grille, aléatoire, algorithme génétique |

## 1. Utilisation en Python

```python
from my_torch import Network, load_network

net = Network(config)                       # réseau vierge, poids tirés avec config["seed"]
history = net.fit(X, Y, X_val, Y_val, verbose=True)   # Y : indices de classe (N,) ou one-hot
net.predict(X)                              # (N,) indices de classe
net.predict_proba(X)                        # (N, n_out) probabilités
mean, std = net.predict_mc(X, n_samples=30) # MC dropout : moyenne + incertitude
net.evaluate(X, Y)                          # {"loss": ..., "accuracy": ...}
net.save("model.nn", backup=True)           # backup : garde l'ancien fichier en model.nn.bak
net = load_network("model.nn")              # ou Network.load(...)
```

- `fit(..., epochs=5, batch_size=128, ...)` remplace pour cet appel les réglages de `config["training"]` ; `verbose=True` affiche une ligne par epoch sur stderr (et un pourcentage dans un terminal).
- `X` peut être en `uint8` : le module convertit chaque mini-batch en float32. Le jeu d'entraînement en `uint8` occupe 4× moins de mémoire (440 Mo au lieu de 1,8 Go).
- Un nouvel appel à `fit` sur un réseau chargé reprend l'entraînement : `meta["epochs_trained"]` et l'historique s'accumulent.
- Erreurs : `ConfigError` (config invalide, avec le chemin de la clé fautive), `NetworkFormatError` (fichier `.nn` inutilisable), `TrainingError` (loss qui diverge), `ValueError` (données de la mauvaise forme). Ce sont toutes des `ValueError` ou des `RuntimeError` : l'analyzer peut les afficher sur stderr et sortir avec 84.

## 2. Configuration (`configs/*.conf`, JSON)

Exemple : `configs/basic.conf`. Seuls `input_size` et `layers` sont obligatoires ; tout le reste a une valeur par défaut. La config **normalisée** (avec tous les défauts) est celle qu'on retrouve dans le `.nn`.

| Clé | Défaut | Valeurs |
|---|---|---|
| `input_size` | — | entier > 0 (773 pour l'échiquier) |
| `layers[i].size` | — | neurones de la couche |
| `layers[i].activation` | — | `relu`, `leaky_relu`, `elu`, `sigmoid`, `tanh`, `linear`, `softmax` |
| `layers[i].alpha` | 0.01 / 1.0 | pente de `leaky_relu` / `elu` |
| `layers[i].dropout` | 0 | dans [0, 1), interdit sur la couche de sortie |
| `layers[i].init` | `init` global | `auto`, `he`, `xavier`, `lecun`, `uniform` |
| `init` | `auto` | `auto` : he pour la famille ReLU, xavier pour sigmoid / tanh / softmax, lecun pour linear |
| `loss` | `cross_entropy` | `cross_entropy` (exige une sortie softmax), `binary_cross_entropy` (exige sigmoid), `mse` |
| `dtype` | `float32` | `float32`, `float64` |
| `optimizer` | `{"name": "adam"}` | `sgd`, `momentum` (`momentum`, `nesterov`), `rmsprop` (`rho`, `eps`), `adam` (`beta1`, `beta2`, `eps`) ; `lr` = **learning rate initial** |
| `lr_schedule` | `{"name": "constant"}` | `step` (`step_size`, `gamma`), `exponential` (`gamma`), `cosine` (`t_max`, `lr_min`), `plateau` (`factor`, `patience`, `min_lr`, `min_delta`) |
| `training.epochs` | 20 | nombre maximum d'epochs |
| `training.batch_size` | 256 | taille des mini-batchs |
| `training.l2` | 0 | régularisation L2 (sur les poids, pas sur les biais) |
| `training.grad_clip` | null | plafond de la norme globale des gradients |
| `training.early_stopping` | null | `true` ou `{"patience": 5, "min_delta": 0, "restore_best": true}` |
| `training.shuffle` | true | mélange les données à chaque epoch (seedé) |
| `training.eval_train_samples` | 20000 | nombre d'exemples d'entraînement sur lesquels la loss et l'accuracy sont re-mesurées à chaque epoch, dropout éteint (0 = moyenne pendant l'epoch) |
| `labels` | null | noms des sorties (le module les garde sans les interpréter) |
| `seed` | tirée au hasard puis gardée | rend l'initialisation et le mélange reproductibles |
| `name`, `description` | — | texte libre |

Une config invalide est refusée avec la clé en cause : `layers[0].activation: unknown 'relux' (choices: relu, leaky_relu, ...)`.

## 3. Fichier `.nn`

Archive `.npz` numpy (un zip) :

| Entrée | Contenu |
|---|---|
| `header.npy` | JSON UTF-8 : `format`, `format_version`, date, **config normalisée** (couches, neurones, activation de chaque couche, init, optimiseur, learning rate et sa baisse, régularisation, seed), `labels`, `meta` (epochs entraînés, meilleure val_loss, temps), tableau des couches, historique d'entraînement |
| `W0.npy`, `b0.npy`, … | poids et biais de chaque couche, en **binaire** float32 |

- **Pourquoi pas tout en texte** : les poids en binaire sont exacts (pas d'arrondi décimal), environ 3× plus petits et rapides à charger. L'en-tête reste lisible : `python3 scripts/inspect_nn.py FILE.nn` (`--json`, `--history`), ou `unzip -p FILE.nn header.npy`.
- **Version de format** : un fichier d'une version plus récente que le programme est refusé avec un message clair. `serialization._upgrade()` est l'endroit où convertir les anciennes versions quand le format changera.
- **Écriture atomique** : écriture dans un fichier temporaire du même dossier, puis `os.replace`. Une sauvegarde interrompue laisse l'ancien fichier intact. `save(path, backup=True)` garde en plus l'ancien en `path.bak`. C'est la parade au piège « `--train` sans `--save` écrase le réseau d'origine ».
- `np.savez` reçoit un **fichier ouvert** et pas un nom : avec un nom, numpy ajouterait `.npz` (`x.nn.npz`).

## 4. Générateur

```
make
./my_torch_generator configs/basic.conf 3      # -> basic_1.nn basic_2.nn basic_3.nn
```

Les fichiers sont créés dans le dossier courant, nommés d'après la config. Si la config a une `seed`, le réseau i reçoit `seed + i - 1` : relancer la commande redonne les mêmes poids. Tous les fichiers de config sont vérifiés **avant** d'écrire quoi que ce soit. En cas d'erreur : message sur stderr et code 84.

## 5. Scripts d'analyse

Ils lisent un **dataset encodé** `.npz` : `X_train`, `Y_train`, `X_val`, `Y_val` (et optionnellement `X_test`, `Y_test`, `labels`). `X` est de forme (N, input_size), en uint8 de préférence ; `Y` contient des indices de classe. C'est le contrat avec la partie analyzer, qui produit ce fichier à partir de `data/{train,val,test}.txt`.

| Commande | Sert à |
|---|---|
| `python3 scripts/compare.py --data D.npz --base configs/basic.conf --variant "none: training.l2=0 layers.hidden.dropout=0" --variant "all: " --out docs/img/reg.png` | courbes train (tirets) / validation (trait plein) de 2 à 8 configs sur un même graphe, plus le tableau des écarts train/val (`.md`, `.csv`) |
| `python3 scripts/hpo.py configs/search/architecture.json --data D.npz --plot` | balayage nombre de couches × neurones |
| `python3 scripts/hpo.py configs/search/random.json --data D.npz --plot` | recherche aléatoire (24 configs) |
| `python3 scripts/hpo.py configs/search/genetic.json --data D.npz --plot` | algorithme génétique (12 individus × 5 générations) |
| `python3 scripts/bench_perf.py --data D.npz` | mesures de vitesse (section 7) |

`hpo.py` écrit `trials.csv`, `results.json`, `best.conf` (prêt pour l'entraînement ou le générateur), `summary.md` et `hpo.png`. Dans `--variant` comme dans un espace de recherche, une clé est un chemin dans la config : `training.l2`, `optimizer.lr`, `layers.hidden.dropout` (`hidden` = toutes les couches sauf la sortie, `*` = toutes). Les gènes `hidden.layers`, `hidden.size`, `hidden.ratio`, `hidden.activation` et `hidden.dropout` reconstruisent les couches cachées.

## 6. Choix techniques

| Choix | Pourquoi | Alternative écartée |
|---|---|---|
| Softmax + cross-entropy fusionnés | gradient sur les logits = `(p - y) / N` ; la loss est calculée par log-softmax, donc jamais de `log(0)` | gradient de la loss puis jacobienne du softmax : plus de calcul, instable quand p → 0 |
| Init `auto` selon l'activation | garde la variance du signal à peu près constante d'une couche à l'autre (vérifié dans `tests/test_network.py::TestInit`) | une seule init pour tout : ReLU + xavier sous-alimente les couches profondes |
| Dropout inversé | on multiplie par 1/(1-p) pendant l'entraînement, donc rien à corriger à l'inférence | dropout « classique » : il faut multiplier par (1-p) à l'inférence |
| MC dropout (`predict_mc`) | plusieurs passes avec le dropout allumé : la dispersion mesure l'incertitude du réseau | une seule passe : aucune idée de la confiance |
| Early stopping qui restaure le meilleur epoch | on garde les poids de la meilleure val_loss, pas ceux du dernier epoch (déjà en overfitting) | s'arrêter sans restaurer |
| L2 sur les poids seulement | les biais ne font pas d'overfitting : les pénaliser ne fait que décaler les sorties | L2 sur tous les paramètres |
| Écrêtage de la norme globale des gradients | une seule échelle pour tous les gradients : leur direction est conservée | écrêtage valeur par valeur : déforme la direction |
| float32 | 2,3× plus rapide que float64 (section 7) ; la précision suffit (les tests de gradient tournent en float64) | float64 partout |
| 1 thread BLAS par processus, parallélisme par processus | sur des matrices de 256 lignes, 12 threads BLAS sont **plus lents** qu'un seul ; on parallélise plutôt les entraînements indépendants | laisser numpy prendre tous les cœurs pour un seul entraînement |
| Processus `spawn` + mémoire partagée | les données (440 Mo) ne sont pas copiées dans chaque processus ; `spawn` fonctionne pareil sur toutes les versions de Python | des threads Python (le GIL bloque le code Python) ; `fork` (déconseillé quand des threads BLAS tournent, et plus le défaut à partir de Python 3.14) |
| Remise à 0 des nombres dénormalisés toutes les 100 étapes | avec L2, les poids des entrées toujours à 0 (un pion en 1ʳᵉ ou 8ᵉ rangée) décroissent vers 0 sans l'atteindre ; une fois sous 1,2e-38 (« dénormalisés »), chaque calcul est 10 à 100× plus lent | rien : un entraînement L2 de 40 epochs passait de 18 s à 44 s |

## 7. Mesures de vitesse

Laptop Intel Core 5 120U (2 cœurs performants + 8 cœurs économes, 12 threads, 15 W), réseau `configs/basic.conf` (773 → 256 → 128 → 5, 231 685 paramètres), 100 000 exemples d'entraînement. Commande : `python3 scripts/bench_perf.py --data D.npz`.

| Mesure | Réglage | Résultat |
|---|---|---|
| threads BLAS (batch 256) | 1 / 2 / 4 / 12 | 1,43 / 1,02 / 1,68 / 2,26 s par epoch |
| type des nombres | float64 → float32 | 3,26 → 1,39 s par epoch (2,3×) |
| vectorisation | 1 exemple à la fois → mini-batch de 256 | 808 → 72 348 exemples/s (90×) |
| dénormalisés (L2, fin d'entraînement) | sans → avec remise à 0 | 0,73 → 0,48 s par epoch |
| entraînements parallèles (12 entraînements de 3 epochs) | 1 → 2 → 4 → 8 → 12 processus | gain 1,0 → 1,8 → 2,3 → 3,2 → 3,2× |

Le gain plafonne vers 3× : la machine n'a que 2 cœurs performants et une enveloppe de 15 W, donc la fréquence baisse quand tous les cœurs tournent. Sur une machine de bureau, le même code passe à l'échelle plus loin. Pour un **seul** entraînement, `MY_TORCH_THREADS=2` est le meilleur réglage sur cette machine (1,02 s par epoch).

## 8. Limites

- Couches denses uniquement : pas de convolution, alors que l'échiquier est une grille 8×8.
- L'état de l'optimiseur (moments d'Adam) n'est pas sauvegardé : reprendre un entraînement repart avec des moments à zéro. Les sauver doublerait la taille du fichier.
- `predict_mc` coûte `n_samples` passes : à réserver à l'analyse, pas à `--predict` sur un gros fichier.
