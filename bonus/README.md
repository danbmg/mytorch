# Bonus : le pat (stalemate)

Le dataset du sujet ne contient **aucun** pat. Il faut donc d'abord fabriquer des données justes, puis apprendre une 6ᵉ classe `Stalemate`.

## 1. Un générateur de coups légaux comme oracle (`movegen.py`)

Pat = pas en échec et aucun coup légal. `movegen.py` énumère les coups (pièces, roque, prise en passant, promotions) et garde ceux qui ne laissent pas son roi attaqué, ce qui couvre clouages et échecs à la découverte. 170 lignes, aucune bibliothèque d'échecs.

Il est prouvé juste : il retrouve **les 723 019 labels du dataset** (`python3 bonus/validate_movegen.py`, 5 s sur 8 cœurs) et le nombre exact de coups des positions de référence « perft » (`tests/bonus/test_movegen.py`). Il vérifie aussi les labels des 4 versions symétriques de chaque position (couleurs échangées, colonnes a↔h, `analyzer/augment.py`) dont le réseau principal s'entraîne : `--symmetries`, 723 019 positions sur 723 019.

Pourquoi un réseau si le générateur sait déjà répondre ? C'est l'objet du projet : le générateur sert d'oracle pour fabriquer les données, et le réseau apprend le motif sans calculer les coups.

## 2. Les données (`gen_stalemate.py`)

On tire des positions aléatoires peu chargées, qu'on étiquette avec le générateur de coups. On garde N pats, N positions **« presque pat »** (`Nothing` avec au plus 2 coups légaux, sinon le réseau apprendrait « peu de pièces = pat »), N/2 `Nothing` quelconques, et N/2 `Check` et N/2 `Checkmate` avec le même matériel.

Avec N = 30 000 : 105 000 positions gardées sur **12,8 millions tirées** (1 position aléatoire sur 430 est un pat), en 141 s sur 4 cœurs. Elles sont dédupliquées, coupées en 80/10/10 et ajoutées aux parts du dataset d'origine (`bonus/data/chess6.npz`). La même seed redonne les mêmes 105 000 positions, au caractère près, et leurs 4 versions symétriques ont toutes le bon label (vérifié par le générateur de coups : un pat reste un pat dans le miroir).

## 3. Le réseau à 6 classes (`bonus/stalemate.conf`, rapport : `bonus/report/report.md`)

512 > 256, 6 sorties, entraîné comme le réseau principal sur les 4 versions de chaque plateau (2 625 784), 50 epochs (84 min). Sur le test (82 057 positions, dont 3 000 pats jamais vus), avec la réponse de `--predict` (moyenne sur les 4 versions) :

| | Précision | Rappel | F1 |
|---|---|---|---|
| **Stalemate** | 0,938 | 0,919 | **0,928** |
| toutes classes | | | accuracy **94,4 %**, F1 macro 0,932 |
| *avant (2 versions de moins au train, réseau seul)* | *0,888* | *0,852* | *0,869 ; accuracy 90,5 %* |

- 92 % des pats sont reconnus. Les ratés partent presque tous en `Nothing` (211 sur 244) : le réseau voit que le roi n'est pas attaqué, pas qu'il n'a plus de case de fuite.
- Les « presque pat » font leur travail : seules 11 des 29 757 positions `Nothing` du dataset d'origine sont prises pour des pats.
- Sur les positions du dataset : 96,4 / 93,8 / 93,7 % sur 10 / 20 / beaucoup de pièces, un peu sous le réseau livré (1024 > 512, 96,9 / 94,7 / 94,8 %), qui a 2,5× plus de paramètres.
- L'entraînement s'est arrêté au plafond de 50 epochs, sa val_loss baissait encore : il reste de la marge.
- Limites : ce sont des pats de fin de partie, avec le matériel de `gen_stalemate.py`. Un pat de milieu de partie, avec beaucoup de pièces bloquées (le pat en 10 coups de Sam Loyd), ou un roi seul bloqué par un roi et un pion (le matériel tiré accompagne toujours un pion d'une pièce) ne leur ressemblent pas : le réseau les rate.

```bash
python3 bonus/gen_stalemate.py
python3 scripts/train.py bonus/stalemate.conf --data bonus/data/chess6.npz --out bonus/my_torch_network_stalemate.nn --report bonus/report --flip-colors --flip-files
./my_torch_analyzer --predict bonus/my_torch_network_stalemate.nn boards.txt   # l'analyzer lit les 6 labels dans le .nn
```
