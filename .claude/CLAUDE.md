# Repo-level notes

This project tracks session context in `.claude/context.md`.

- Global preferences (identity, code style, Epitech constraints, behavior) live in `~/.claude/CLAUDE.md` — don't duplicate them here.
- Project-specific architecture, interfaces and constraints live in `CLAUDE.md` at the repo root.
- Context updates are **manual only** — triggered by the `/update-context` command, never automatic.

## Git — Claude ne commite pas, ne pousse pas

**Règle absolue.** Claude ne lance jamais `git commit`, `git push`, `git merge`, `git rebase`, ni `gh pr create`. Même quand le travail est fini, testé et vert. Même quand la convention de commit est claire. Même quand il vient d'en discuter avec danbmg.

Ce que Claude fait à la place, quand une livraison est prête :
1. il laisse les fichiers modifiés dans l'arbre de travail, **sans les indexer** (`git add` compris) ;
2. il annonce les messages de commit qu'il propose, dans l'ordre, au format ci-dessous ;
3. il s'arrête là. C'est danbmg qui indexe et qui commite, depuis son IDE.

La raison : c'est son dépôt, son binôme et sa note. L'historique git est à lui, il doit pouvoir relire chaque diff avant qu'il entre dedans. Un commit posé par Claude, c'est un diff qui est passé sans relecture.

La seule exception est un ordre explicite et sans ambiguïté sur le moment même — « commit ça », « pousse la branche ». Un « fais la branche et vas-y » au début d'une tâche n'est **pas** cette autorisation : créer une branche n'autorise pas à la remplir.

## Messages de commit

Le format à proposer à danbmg — c'est lui qui les passe :

- `[+]` pour un ajout, `[~]` pour une reprise de l'existant.
- Description courte, en anglais, en minuscules, sans point final.
- **Un commit par nature de changement**, dans cet ordre : le code, puis les tests, puis la doc. Jamais un seul commit fourre-tout.

Exemple, la livraison du CRUD widgets (`3b28be6`, `5e528c6`, `ed25f35`) :

```
[+] widget instances CRUD backend
[+] unit tests for widgets CRUD
[+] updated doc
```

Une branche par tâche (`feat/widgets-crud`), partie de `origin/main` et non du `main` local, qui traîne souvent derrière. Créer la branche est la seule opération git que Claude fait de lui-même.

## Documentation — uniquement sur demande

**Claude ne met à jour aucun fichier de notes de lui-même.** Ni `CLAUDE.md` (racine), ni `.claude/context.md`, ni `learning.md`. Même après une livraison finie et testée, même quand il y aurait manifestement de quoi écrire.

À la place, Claude termine la tâche, dit en une ligne ce qui mériterait d'être noté et où, et s'arrête. C'est danbmg qui déclenche, par `/update-context` ou en le demandant explicitement.

La raison : ces trois fichiers sont ses notes, pas celles de Claude. Les remplir sans qu'il ait lu ce qui s'y ajoute les transforme en fichiers qu'il ne connaît plus — l'inverse de ce à quoi ils servent, surtout `learning.md`, qui doit tenir en soutenance.

Seule exception : `docs/api.md`. C'est le contrat que theodsh lit pour coder le front, pas une note personnelle ; il reste mis à jour avec la route qu'il décrit, comme l'exige la règle 4 du `CLAUDE.md` racine.

## Notes de soutenance (`learning.md`, gitignoré)

Règle générale, valable sur tout projet d'école — également dans `~/.claude/CLAUDE.md`, recopiée ici pour qu'elle survive à la machine. **Le déclenchement, lui, est propre à ce projet : sur demande uniquement.**

Objectif unique : **pouvoir défendre le code à l'oral sans relire les fichiers**. Ce n'est pas de la doc projet — elle, elle va dans `README.md` / `docs/`.

**Ce fichier ne s'écrit que sur demande** (voir la section ci-dessus). Quand danbmg la formule, deux situations appellent une entrée :
1. **une brique de code livrée** → ce qu'elle fait, pourquoi ce choix *plutôt qu'un autre* (nomme l'alternative écartée), et ce qui casserait sans elle ;
2. **un terme technique sur lequel danbmg a posé une question** (« c'est quoi un CRUD ? », « pourquoi Argon2 ? »).

Entre-temps, Claude signale en une ligne ce qui mériterait une entrée, sans l'écrire.

Ce qui fait qu'une entrée sert à quelque chose :
- **français simple** — pas de jargon pour expliquer le jargon ;
- **rattachée au code réel** : chemins de fichiers, noms de routes, extraits courts. Un tableau qui met le concept abstrait en face de ce qu'il devient dans *ce* projet vaut mieux qu'une définition recopiable depuis Wikipédia ;
- **dire où le concept s'arrête** : ce qui dépasse du cadre, et ce qu'il ne couvre pas. C'est souvent là que se joue la question du jury ;
- **un exemple concret tiré du sujet**, pas un exemple abstrait ;
- **les décisions d'architecture avec leur raison** : le compromis assumé, le piège évité, ce qu'on a refusé de faire.

Tenir aussi une section « questions probables en soutenance » en fin de fichier, alimentée au fil de l'eau.

Entrées numérotées en continu, regroupées par phase/jalon. Insérer une entrée au milieu = décaler les suivantes et resynchroniser le compte dans `.claude/context.md`.
