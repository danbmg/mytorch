# Bonus : le pat (stalemate)

Le dataset du sujet ne contient **aucun** pat. Il faut donc d'abord fabriquer des données justes, puis apprendre une 6ᵉ classe `Stalemate`.

## 1. Un générateur de coups légaux comme oracle (`movegen.py`)

Pat = pas en échec et aucun coup légal. `movegen.py` énumère les coups (pièces, roque, prise en passant, promotions) et garde ceux qui ne laissent pas son roi attaqué, ce qui couvre clouages et échecs à la découverte. 170 lignes, aucune bibliothèque d'échecs.

Il est prouvé juste : il retrouve **les 723 019 labels du dataset** (`python3 bonus/validate_movegen.py`, 24 s) et le nombre exact de coups des positions de référence « perft » (`tests/test_movegen.py`).

Pourquoi un réseau si le générateur sait déjà répondre ? C'est l'objet du projet : le générateur sert d'oracle pour fabriquer les données, et le réseau apprend le motif sans calculer les coups.

## 2. Les données (`gen_stalemate.py`)

On tire des positions aléatoires peu chargées, qu'on étiquette avec le générateur de coups. On garde N pats, N positions **« presque pat »** (`Nothing` avec au plus 2 coups légaux, sinon le réseau apprendrait « peu de pièces = pat »), N/2 `Nothing` quelconques, et N/2 `Check` et N/2 `Checkmate` avec le même matériel.

Avec N = 30 000 : 105 000 positions gardées sur **12,8 millions tirées** (1 position aléatoire sur 430 est un pat), en 141 s sur 4 cœurs. Elles sont dédupliquées, coupées en 80/10/10 et ajoutées aux parts du dataset d'origine (`bonus/data/chess6.npz`).

## 3. Le réseau à 6 classes (`bonus/stalemate.conf`, rapport : `bonus/report/report.md`)

Même architecture que le premier réseau final (512 > 256), 6 sorties, 50 epochs (19 min). Sur le test (82 057 positions, dont 3 000 pats jamais vus) :

| | Précision | Rappel | F1 |
|---|---|---|---|
| **Stalemate** | 0,888 | 0,852 | 0,869 |
| toutes classes | | | accuracy **90,5 %**, F1 macro 0,887 |

- 85 % des pats sont reconnus. Les ratés partent presque tous en `Nothing` (378 sur 445) : le réseau voit que le roi n'est pas attaqué, pas qu'il n'a plus de case de fuite.
- Les « presque pat » font leur travail : seules 18 des 29 757 positions `Nothing` du dataset d'origine sont prises pour des pats.
- Ajouter la classe ne coûte rien aux 5 autres : 93,3 / 89,4 / 90,7 % sur 10 / 20 / beaucoup de pièces, contre 92,3 / 88,4 / 90,5 % pour le premier réseau final.
- Limite : ce sont des pats de fin de partie à peu de pièces. Un pat de milieu de partie, avec beaucoup de pièces bloquées, ne leur ressemble pas.

```bash
python3 bonus/gen_stalemate.py
python3 scripts/train.py bonus/stalemate.conf --data bonus/data/chess6.npz --out bonus/my_torch_network_stalemate.nn --report bonus/report
./my_torch_analyzer --predict bonus/my_torch_network_stalemate.nn boards.txt   # l'analyzer lit les 6 labels dans le .nn
```
