# Choix et décisions

Chiffres : `docs/benchmarks.md` (§ entre parenthèses).

## 1. Choix techniques

| Choix | Pourquoi | Alternative écartée |
|---|---|---|
| **Python + numpy** | aucune bibliothèque de ML autorisée ; numpy fournit le produit matriciel (BLAS), pas de réseau ni de gradient ; disponible sur le dump | module en C++ compilé + front en script (bonus du critère « module ») : il faudrait réécrire un produit matriciel aussi rapide que BLAS, pour un gain nul. On renonce à ce point en connaissance de cause |
| **`my_torch/` séparé d'`analyzer/`** | le module ne voit que des matrices et des indices de classe (il entraîne aussi XOR dans les tests) ; tout ce qui touche au FEN est dans `analyzer/` | un seul programme mêlant échecs et réseau |
| **Encodage one-hot 773** (12 plans × 64 cases + trait + 4 roques) | chaque pièce sur chaque case a son propre poids ; uint8 → 440 Mo pour tout le train | 64 valeurs (−6…6) : 12× plus compact, mais impose un ordre artificiel entre les pièces. Contraintes en balance : taille de l'encodage ↔ difficulté d'apprentissage |
| **Softmax + entropie croisée** | 5 classes exclusives ; gradient fusionné `p − y`, stable | 5 sigmoïdes indépendantes, MSE |
| **ReLU, init He** (`auto`) | 70,7 % contre 63,5 % pour la sigmoïde, qui sature (§6) ; He garde la variance d'une couche à l'autre [He 2015] | sigmoïde, tanh |
| **512 > 256** | 88,9 % contre 88,0 % (256 > 128) et 86,9 % (512 seul) sur tout le train ; une 3ᵉ couche coûte 7 % de temps pour 0 gain (§3). Contraintes : profondeur ↔ temps ↔ quantité de données | la couche unique qui gagnait sur 150k positions |
| **Adam**, lr 1e-3, β 0,9 / 0,999 | valeurs de l'article [Kingma 2015] ; converge 5× plus vite que SGD (§6) ; la recherche aléatoire ne fait pas mieux (§4) | SGD, Momentum, RMSprop |
| **Plateau** (×0,5 après 2 epochs sans progrès) | 89,3 % contre 88,0 à 88,4 % pour les autres schedules (§2) | exponentielle (`basic`), step, cosinus |
| **Batch 256** | 2,2× moins cher que 32 pour 1 point de moins ; 4096 ne converge pas (§6). Contraintes : stabilité ↔ temps | 32, 4096 |
| **L2 1e-4 + dropout 0,3 + early stopping** (patience 5, meilleur epoch restauré) | meilleure val_loss, et 3× moins de calcul grâce à l'early stopping (§1) [Srivastava 2014, Prechelt 1998] | rien : la val_loss est multipliée par 4 |
| **Écrêtage des gradients à 5** | sécurité contre une explosion ponctuelle [Pascanu 2013] ; ne coûte rien | — |
| **float32** | 2,4× plus rapide que float64 (§7) ; les tests de gradient tournent en float64 | float64 |
| **Dédup + split 80/10/10 stratifié, seed 42** | 7 460 doublons : sans dédup, une position tombe dans le train ET le test ; la stratification garde les proportions de chaque classe et sous-ensemble | split brut |
| **Fichier `.nn`** : npz, en-tête JSON + poids binaires | poids exacts, 3× plus petits que du texte ; l'en-tête se lit avec `inspect_nn.py` ; un fichier tout en texte coûterait un point à la grille | tout en JSON |
| **Un processus par entraînement**, mémoire partagée | le GIL bloque les threads Python ; 2,6× sur la recherche d'hyperparamètres (§7) | threads, ou laisser BLAS prendre tous les cœurs (plus lent) |

## 2. Données : origine, licence, limites

- **Dataset fourni par Epitech, sans licence** : usage limité au projet, dépôt privé, aucune redistribution, et le réseau n'est pas présenté comme « entraîné sur des données libres ». Les positions viennent probablement de parties en ligne (non vérifiable). Elles ne contiennent aucune donnée personnelle : un FEN décrit une position, sans joueur ni date.
- **Si on sort de ces limites** : publier le dataset ou le réseau engage le droit d'auteur et le droit des bases de données (directive 96/9/CE). Scraper un site comme chess.com viole ses conditions d'utilisation, et des parties avec les pseudos des joueurs seraient des données personnelles (RGPD).
- **Bonus** : nos pats sont **générés** et étiquetés par notre générateur de coups, prouvé exact sur les 723 019 positions du dataset. Aucune source externe.
- **Ce que le réseau n'a jamais vu** : aucun pat, aucune position illégale, seulement 209 positions avec en passant. Les mats sont rares dans `many_pieces` (28 632 lignes, contre 100 000 dans `20_pieces`).

## 3. Impact business

- **Taille ↔ temps ↔ matériel** : le réseau final gagne 2,2 points pour 2,3× plus de calcul. Pas de GPU sur le dump, donc tout est optimisé pour le CPU (99×, 2,4×, 2,6×, §7). La prédiction ne coûte presque rien (54 000 positions/s) : le coût est à l'entraînement. Pour réentraîner souvent, `basic` suffit (3,4 min).
- **De zéro plutôt qu'un framework** : imposé par le sujet. En entreprise, on prendrait le framework, puisque le temps d'ingénieur coûte plus que le temps machine. Ici, on y gagne zéro dépendance lourde et chaque ligne est explicable.
- **Coût des données** : chaque doublement rapporte 5 à 7 points, sans plateau (§5). Aucun réglage n'a rapporté autant : investir dans les données avant d'investir dans le réglage. Générer des données sûres a un prix : 12,8 millions de tirages pour 30 000 pats (141 s sur 4 cœurs).

## 4. Pièges et parades

| Piège | Parade |
|---|---|
| Accuracy qui masque le déséquilibre : 41,6 % de `Nothing`, donc un réseau constant fait déjà 41,6 % | F1 par classe et macro, matrice de confusion, score par sous-ensemble dans chaque rapport ; split stratifié |
| Réseau sur-appris (100 % train / 70 % validation à 100k positions) | test jamais utilisé pour choisir ; L2, dropout, early stopping ; `--train` garde 10 % du fichier pour valider |
| Fuite train → test (7 460 doublons) | déduplication sur la position avant le split |
| Sauvegarde illisible après un changement de format | `format_version` dans chaque fichier ; version plus récente refusée avec un message clair ; `_upgrade()` prévu pour convertir |
| `--train` sans `--save` écrase le réseau | `.bak` + écriture atomique ; dossier de destination vérifié **avant** l'entraînement ; un Ctrl-C n'écrit rien |
| Label sans couleur (l'exemple du sujet montre `Check`) | labels stockés dans le `.nn`, toujours avec la couleur |
| FEN invalide traitée en silence | chaque ligne est validée ; erreur avec `fichier:ligne`, code 84, rien sur stdout |

## 5. L'utilisateur

`--help` détaillé ; des erreurs qui disent quoi corriger, jamais de traceback. stdout propre en `--predict` : le résumé ne s'affiche sur stderr que dans un terminal. Une ligne par epoch plus un pourcentage, et `curves.png` redessiné à chaque epoch. `inspect_nn.py` montre ce que contient un réseau. 6,5 s par epoch, et `--epochs N` pour un essai rapide.

## 6. « Si vous recommenciez ? »

1. **Choisir les hyperparamètres par successive halving** [Hyperband, Li 2018] : beaucoup de configs à petit budget, puis la meilleure moitié à budget double, jusqu'aux données complètes. Notre recherche sur 150k positions a désigné le pire candidat (§3). Pour le même calcul, les dernières décisions se prennent dans les conditions réelles. C'est une méthode de plus dans `my_torch/hpo.py`, pas une réécriture.
2. **Rééquilibrer les mats des positions chargées** (rappel 77 %, la classe la plus rare) en pondérant la perte par classe. On échange un peu de précision sur `Check` contre du rappel là où on rate le plus.
3. **Le pipeline de données d'abord** : les scripts d'expérience ont attendu le dataset encodé. Il aurait fallu le faire le premier jour, avec un réseau factice.

Envisagé et écarté : donner en entrée les cases attaquées (on ferait le travail du réseau à sa place) ; une couche convolutive (plus naturelle pour des motifs d'attaque, mais une implémentation de plus, et lente sur CPU sans bibliothèque).

**Sources** : He et al., *Delving Deep into Rectifiers*, 2015 · Kingma & Ba, *Adam*, 2015 · Srivastava et al., *Dropout*, 2014 · Prechelt, *Early Stopping — But When?*, 1998 · Pascanu et al., *On the difficulty of training RNNs*, 2013 · Glorot & Bengio, 2010 (Xavier) · Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, 2012 · Li et al., *Hyperband*, 2018.
