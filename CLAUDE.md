# CLAUDE.md — my_torch (Epitech, binôme)

Réseau de neurones **from scratch** + analyseur d'échiquier FEN → `Nothing` / `Check <Color>` / `Checkmate <Color>`.
Ce fichier = contexte projet + plan de travail. **Cocher les tâches au fil de l'eau.**

---

## 0. À compléter / hypothèses

| Point | Valeur | Statut |
|---|---|---|
| Membre A (module NN) | Dan — `danbmg` | ok (rôles inversables) |
| Membre B (analyzer + data + benchs) | `ricardofrcn` | ok |
| Date midterm | `<JJ/MM>` | à remplir |
| Date rendu final | `<JJ/MM>` | à remplir |
| Langage | Python 3 + **numpy** uniquement pour le calcul | hypothèse, à confirmer |
| Format de sortie | avec couleur (`Check White`, `Checkmate Black`), comme le dataset | hypothèse, à confirmer avec l'AER |
| `python-chess` | autorisé **uniquement** dans `scripts/` (génération/vérif de données), jamais dans `my_torch/` ni `analyzer/` | à valider avec l'AER |

---

## 1. Contraintes du sujet (MUST)

- Binaire `my_torch_analyzer`, construit par `Makefile` (`all`, `re`, `clean`, `fclean`).
- Erreurs → **stderr** + `exit 84`. Succès → `exit 0`.
- Apprentissage **supervisé**. **Interdit** : pytorch, tensorflow, keras, jax, sklearn… (tout lib de NN).
- Module NN générique : **créer**, **sauvegarder/charger**, **entraîner** un réseau.
- Pousser un réseau pré-entraîné dont le nom **commence par `my_torch_network`** (évalué par l'autograder).
- Garder **tous** les scripts + datasets d'entraînement (datasets non pushés, mais à montrer en défense).
- Lutter contre l'**overfitting**. Justifier **chaque** hyperparamètre.
- `README` obligatoire ; benchmarks **avec preuves visuelles** (graphes).
- Bonus (stalemate, chess AI…) → dossier `bonus/`.

### Usage attendu
```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```
- `--predict` : 1 prédiction par ligne, **même ordre** que le fichier. Ligne = FEN (6 champs) + label attendu optionnel. **stdout = prédictions uniquement.**
- `--train` : ligne = FEN + label. Sauve dans `SAVEFILE` si `--save`, sinon écrase `LOADFILE`.

---

## 2. Dataset (`data/`, gitignoré)

| Classe | Fichiers | Lignes | White | Black |
|---|---|---|---|---|
| Nothing | `nothing/{10,20,many}_pieces.txt` | 300 000 | — | — |
| Check | `check/…` | 249 300 | 135 587 | 113 713 |
| Checkmate | `checkmate/…` | 173 719 | 94 295 | 79 424 |
| **Total** | 9 fichiers | **723 019** | | |

- Format : `<FEN 6 champs> <label>` ; `Nothing` sans couleur.
- **Couleur = camp qui donne l'échec** (`… b … Check White` = les Blancs mettent le roi noir en échec).
- 0 FEN invalide · 7 460 doublons (mêmes 4 premiers champs FEN), 0 conflit de label → **dédupliquer avant le split** (sinon fuite train→test).
- Pas de stalemate dans le dataset → le bonus demande de générer des données.
- Déséquilibre léger (Checkmate sous-représenté, surtout `many_pieces`) → split **stratifié** + suivre le F1 par classe.

---

## 3. Choix techniques de départ (à justifier par les benchs)

- **Entrée (773)** : 12 plans one-hot (6 pièces × 2 couleurs) × 64 cases = 768 + trait (1) + roques (4). En passant ignoré (209 lignes seulement).
- **Sortie (5)** : `Nothing, Check White, Check Black, Checkmate White, Checkmate Black` — softmax + cross-entropy. Stalemate = 6ᵉ classe (bonus).
- **Baseline** : `773 → 256 ReLU → 128 ReLU → 5 softmax`, init He, Adam lr=1e-3, batch 256, L2=1e-4, dropout 0.2, early stopping (patience 5 sur val loss).
- **Split** : 80 / 10 / 10 stratifié, seed fixe, après déduplication.
- **Format `.nn`** : `np.savez` → config JSON + poids + état normalisation.
  ⚠️ `np.savez("x.nn")` crée `x.nn.npz` → **toujours passer un file handle** : `with open(path, "wb") as f: np.savez(f, ...)`.
- **Configs** (`configs/*.conf`) : JSON (couches, activations, init, optimiseur, hyperparamètres).

---

## 4. Arborescence cible

```
my_torch/            # module NN — AUCUNE notion d'échecs ici
  layers.py activations.py losses.py init.py optimizers.py
  network.py serialization.py metrics.py
analyzer/            # spécifique échecs
  main.py cli.py fen.py labels.py
generator/main.py    # my_torch_generator
configs/             # *.conf (JSON)
scripts/             # dedup_split.py train.py hpo.py benchmark.py plot.py
tests/               # pytest
docs/                # design.md benchmarks.md + docs/img/*.png
bonus/
my_torch_network_*.nn   # réseau(x) pré-entraîné(s), à la racine
Makefile README.md CLAUDE.md
```

Makefile : `make` génère `my_torch_analyzer` et `my_torch_generator` = wrapper Python exécutable (shebang + `sys.path` vers le repo + `from analyzer.main import main`). Approche testée.

---

## 5. Contrat d'interface (à figer en Phase 0 → A et B avancent en parallèle)

```python
# my_torch/network.py   (A)
class Network:
    def __init__(self, config: dict, seed: int | None = None): ...
    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray  # (N, n_out) probas
    def fit(self, X, Y, X_val=None, Y_val=None, **hp) -> dict                # historique loss/acc
    def predict(self, X: np.ndarray) -> np.ndarray                           # (N,) indices classes
    def save(self, path: str) -> None
    @staticmethod
    def load(path: str) -> "Network"

# analyzer/fen.py   (B)
def fen_to_vector(fen: str) -> np.ndarray          # (773,) float32, ValueError si FEN invalide
def parse_line(line: str) -> tuple[str, str | None] # (fen, label|None)
# analyzer/labels.py   (B)
LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
```
Tant que `Network` n'est pas prêt, B travaille avec un **stub** (prédictions aléatoires, même API).

---

## 6. Plan & répartition

Estimations en jours-homme (total 18.75). Chaque tâche = 1 item du GitHub Project, même ID (voir §8).

### Phase 0 — Setup (les deux, J1)
- [ ] **T0.1** (B, 0.5) Repo, `.gitignore` (`data/`, `__pycache__`, venv), Makefile squelette, `pytest`.
- [ ] **T0.2** (A+B, 0.5) Valider ensemble le contrat §5 + créer les stubs.
- [ ] **T0.3** (B, 0.5) `scripts/dedup_split.py` : dédup + split stratifié 80/10/10 seedé → `data/{train,val,test}.txt`.

### Phase 1 — Objectif MIDTERM : pipeline complet de bout en bout
**A — module NN**
- [ ] **T1.1** (1) Couche Dense + activations (ReLU, LeakyReLU, sigmoid, tanh, softmax) forward/backward vectorisés.
- [ ] **T1.2** (0.5) Losses : softmax+cross-entropy (combinés, stable), MSE.
- [ ] **T1.3** (0.5) Init : He, Xavier, LeCun, uniforme.
- [ ] **T1.4** (1) Optimiseurs SGD / Momentum / Adam + boucle mini-batch avec shuffle.
- [ ] **T1.5** (1) Création depuis config + `save`/`load` `.nn` (round-trip identique).
- [ ] **T1.6** (0.5) Tests : gradient check numérique, convergence XOR, round-trip save/load.
- [ ] **T1.7** (0.5) `my_torch_generator config nb [config nb …]` → `<nom>_<i>.nn`.

**B — analyzer + data**
- [ ] **T1.8** (1) `fen.py` + `labels.py` : parsing, validation, encodage 773, tests unitaires.
- [ ] **T1.9** (0.5) `cli.py` : `--help`, `--predict | --train [--save]`, toutes les erreurs → stderr + 84.
- [ ] **T1.10** (0.5) Modes predict (ordre conservé, encodage par batch) et train branchés sur `Network`.
- [ ] **T1.11** (0.25) Makefile final (`all re clean fclean` + `tests_run`).
- [ ] **T1.12** (1) `scripts/train.py` + `metrics` (accuracy, matrice de confusion) + courbes loss/acc (matplotlib).

**Les deux**
- [ ] **T1.13** (0.5) Premier réseau entraîné `my_torch_network_basic.nn` + baseline chiffrée sur le test.
- [ ] **T1.14** (0.5) Prépa midterm : README v0, script de démo, 1 graphe, justification de la baseline.

### Phase 2 — Objectif FINAL : optimisation, benchmarks, docs
**A**
- [ ] **T2.1** (1) Régularisation : L2, dropout (+ **MC dropout** à l'inférence), early stopping.
- [ ] **T2.2** (0.5) LR schedule (step / cosine), gradient clipping.
- [ ] **T2.3** (1) Perf : float32, vectorisation, multiprocessing pour la HPO (going further).
- [ ] **T2.4** (1.5) HPO : random search + métaheuristique (algo génétique) sur le set de validation.

**B**
- [ ] **T2.5** (1.5) Suite de benchs : plusieurs courbes sur un même graphe, precision/recall/F1 par classe, résultats par sous-ensemble (10/20/many pièces), temps d'entraînement.
- [ ] **T2.6** (1) Docs : README complet, `docs/design.md` (justification des choix), `docs/benchmarks.md` (graphes).
- [ ] **T2.7** (0.5) Robustesse CLI : FEN malformée, fichier vide/absent, args invalides, labels manquants.

**Les deux**
- [ ] **T2.8** (0.5) Choix du modèle final, push `my_torch_network_*.nn`, **ré-entraînement depuis les scripts** pour prouver la reproductibilité.
- [ ] **T2.9** (opt.) Bonus stalemate dans `bonus/` (génération de données nécessaire).
- [ ] **T2.10** (0.5) Prépa défense finale (datasets prêts à montrer, démo, Q&A sur les hyperparamètres).

Charge : A ≈ 5 j (P1) + 4 j (P2) · B ≈ 4.25 j (P0+P1) + 3 j (P2) · commun ≈ 2.5 j.

---

## 7. Checklists de défense

**Midterm** (barème non connu → viser un pipeline complet) :
- [ ] `make re` OK, `--help` OK, erreurs → 84
- [ ] `./my_torch_generator configs/basic.conf 3` génère 3 `.nn`
- [ ] `--train` sur `train.txt` puis `--predict` sur `test.txt` → accuracy annoncée
- [ ] Courbe loss/accuracy train vs val
- [ ] Architecture + hyperparamètres expliqués ; datasets et scripts montrables

**Final** :
- [ ] Tout le midterm + anti-overfitting démontré (courbes avec/sans régularisation)
- [ ] HPO présentée avec résultats comparés
- [ ] README + docs + benchmarks avec graphes
- [ ] `my_torch_network_*.nn` pushé et chargeable par `make && ./my_torch_analyzer --predict`
- [ ] Reproductibilité prouvée (même config + seed → même résultat)

---

## 8. GitHub Project `mytorch`

Org **EpitechPGE3-2026**, rempli via Claude in Chrome (`PROMPT_CHROME.md`) — 27 draft issues, une par tâche du §6, titre `[Tx.y] …`.

- **Status** : `todo` → `In progress` → `Done` (rien d'autre)
- **Assignees** : `danbmg` (A) · `ricardofrcn` (B) · les deux pour les tâches communes
- **Champs** : `Estimate` (number, j-h) · `Phase` (`P0-Setup` / `P1-Midterm` / `P2-Final`) · `Priority` du template (P0 = setup + midterm, P1 = final, P2 = bonus)
- **Vues** : celles du template (Backlog, Priority board, Team items, Roadmap, My items) + `Midterm` (filtre P0+P1)
- **Corps d'item** : Estimate · Description · Definition of Done (checklist) · Notes
- Pas de milestones (indisponibles sur les draft issues) → le champ `Phase` les remplace.

**Workflow**
- Prendre une tâche → `In progress` (max 2 par personne en parallèle).
- Branche `feat/T1.4-adam`, PR relue par l'autre membre **avant** de passer en `Done` (pas de colonne Review).
- `Done` = DoD de l'item cochée + `make re && make tests_run` vert + tâche cochée au §6.
- `main` toujours fonctionnel.

---

## 9. Commandes

```bash
make re                                   # build
make tests_run                            # pytest tests/
python3 scripts/dedup_split.py data/raw data/
python3 scripts/train.py configs/basic.conf --out my_torch_network_basic.nn
./my_torch_analyzer --train --save out.nn my_torch_network_basic.nn data/train.txt
./my_torch_analyzer --predict out.nn data/test.txt
```

---

## 10. Règles pour Claude

- **numpy uniquement** pour le calcul NN. Jamais d'import torch/tensorflow/keras/jax/sklearn. matplotlib seulement dans `scripts/`.
- `my_torch/` ne connaît rien aux échecs ; toute logique FEN va dans `analyzer/`.
- `--predict` : rien d'autre que les prédictions sur stdout ; logs → stderr.
- Toute erreur utilisateur → message sur stderr + `exit 84`, jamais de traceback.
- Seeds fixées partout (split, init, shuffle) pour la reproductibilité.
- **Après chaque modif : `make re && make tests_run`** + vérifier un `--predict` sur un petit fichier.
- Ne jamais committer `data/`. Ne jamais écraser un `my_torch_network_*.nn` sans le demander.
- Cocher la tâche correspondante dans ce fichier une fois sa DoD remplie, et rappeler de passer l'item en `Done` sur le board.
- Toujours citer l'ID de tâche (`T1.4`) dans les branches, commits et PR.