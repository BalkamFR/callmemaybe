*Ce projet a été créé dans le cadre du cursus 42 par papilaz.*

# Call Me Maybe

## Description

Ce projet implémente un outil de function calling qui traduit des prompts en
langage naturel en appels de fonction structurés (`name` + `parameters`), en
utilisant un petit modèle de langage local (`Qwen/Qwen3-0.6B` par défaut).
Plutôt que de demander au modèle de produire du JSON librement et d'espérer
qu'il respecte la syntaxe, la génération est pilotée par du **constrained
decoding** (décodage contraint) : à chaque étape de génération, les logits
produits par le modèle sont masqués pour que seuls les tokens compatibles
avec un nom de fonction valide, un type de paramètre valide, ou la structure
JSON attendue puissent être sélectionnés. Cela garantit une sortie JSON
syntaxiquement valide et conforme au schéma, même si le modèle sous-jacent
est très petit (500M de paramètres) et serait peu fiable si on le sollicitait
naïvement.

## Instructions

### Prérequis

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) pour la gestion des dépendances

### Installation

```bash
make install
# ou : uv sync
```

### Exécution

```bash
make run
# ou : uv run python -m src
```

Par défaut, le programme lit `data/input/functions_definition.json` et
`data/input/function_calling_tests.json`, et écrit les résultats dans
`data/output/function_calling_results.json`.

Tous les chemins, ainsi que le modèle utilisé, peuvent être surchargés :

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json \
    --model Qwen/Qwen3-0.6B
```

### Autres cibles du Makefile

```bash
make debug        # lance le programme sous pdb
make test         # lance la suite de tests pytest
make lint         # flake8 + mypy (flags requis)
make lint-strict  # flake8 + mypy --strict
make clean        # supprime les caches et la sortie générée
```

## Exemple d'utilisation

Entrée (`data/input/function_calling_tests.json`) :

```json
[
  { "prompt": "Is -4 an even number?" },
  { "prompt": "Format template: Say \"hello\" to {name}" }
]
```

Sortie (`data/output/function_calling_results.json`) :

```json
[
  {
    "prompt": "Is -4 an even number?",
    "name": "fn_is_even",
    "parameters": { "n": -4 }
  },
  {
    "prompt": "Format template: Say \"hello\" to {name}",
    "name": "fn_format_template",
    "parameters": { "template": "Say \"hello\" to {name}" }
  }
]
```

## Explication de l'algorithme

Le pipeline (voir `src/generator.py`) se déroule en trois étapes, toutes
pilotées par le constrained decoding plutôt que par du simple prompting :

1. **Sélection du nom de fonction** (`select_tool_name`) : le prompt et la
   liste des fonctions disponibles sont encodés une fois. Ensuite, token par
   token, les logits sont masqués (`src/masking.py::mask_logits`) pour que
   seuls les tokens qui prolongent un préfixe valide d'un nom de fonction
   autorisé (ou du repli `fn_none`) puissent être choisis
   (`get_allowed_next_tokens`). La boucle s'arrête dès qu'aucun candidat
   n'est plus extensible, ce qui la borne naturellement à la longueur du
   nom de fonction le plus long.

2. **Génération des paramètres** (`generate_parameters`) : pour chaque
   paramètre de la fonction choisie :
   - Les **nombres** sont générés avec les logits restreints aux chiffres,
     au point décimal, et au signe moins (sous ses deux formes tokenisées,
     avec et sans espace initial, car le tokenizer fusionne un espace
     précédent avec le signe pour les nombres négatifs).
   - Les **chaînes** sont générées librement (le modèle recopie
     généralement la partie pertinente du prompt), puis tronquées au
     premier guillemet de fermeture non échappé. Tout ce que le modèle a
     généré au-delà de ce point (JSON en trop, continuation hallucinée)
     est écarté, et le reste de la structure JSON est ajouté par le
     programme lui-même, pas par le modèle — c'est ce qui évite un JSON
     malformé.
   - Les **booléens** sont générés avec les logits restreints aux seuls
     tokens de `true` et `false`, garantissant que la valeur est toujours
     l'un des deux.

3. **Assemblage et nettoyage** (`generate_tool_call` + `clean_json_output`) :
   le nom de fonction et les paramètres sont assemblés en une chaîne JSON,
   les virgules en trop sont supprimées, le premier objet `{...}` équilibré
   est extrait (en écartant tout ce qui est affiché après), et les valeurs
   de type chemin de fichier sont débarrassées des espaces parasites.

Le nombre maximal d'étapes de génération pour les nombres et les chaînes est
calculé à partir de bornes réelles plutôt que de constantes arbitraires : la
borne des nombres est la longueur de la plus grande représentation possible
d'un float JSON, et la borne des chaînes dépend de la longueur du prompt
utilisateur (puisque les valeurs de type chaîne en sont recopiées), pour que
la boucle ne soit jamais interrompue prématurément sur une valeur légitime.

## Choix de conception

- **Le constrained decoding vit entièrement dans `src/generator.py` et
  `src/masking.py`**, séparé du prompting (`src/prompting.py`) et des
  entrées/sorties (`src/parsing.py`, `src/utils.py`), pour que chaque
  aspect puisse être testé et raisonné indépendamment.
- **`Config` est un `pydantic.BaseModel`** (`src/parsing.py`) plutôt qu'un
  simple dictionnaire, pour que la configuration d'exécution (chemins de
  fichiers, nom du modèle) soit validée et auto-documentée.
- **Le code d'affichage/couleurs vit dans son propre module**
  (`src/display.py`), séparé de la vraie logique de function calling dans
  `src/generator.py` et `src/utils.py`.
- **Les chaînes sont générées librement puis tronquées après coup**, plutôt
  que de tenter de contraindre chaque caractère, car la valeur cible
  (recopiée depuis le prompt utilisateur) n'est pas connue à l'avance et
  peut contenir des caractères arbitraires, y compris des guillemets à
  échapper.

## Difficultés rencontrées

- **Les nombres négatifs disparaissaient silencieusement.** Le modèle ne
  choisissait presque jamais le token `-` seul avant un chiffre, car le
  tokenizer fusionne un espace précédent avec le signe moins en un seul
  token (`" -"`) qui n'était pas dans l'ensemble autorisé. Corrigé en
  ajoutant ce token fusionné à l'ensemble autorisé et en ne codant plus en
  dur un espace avant le nombre.
- **Tronquer les chaînes sans corrompre le JSON.** Au départ, le code
  gardait tout ce que le modèle avait généré jusqu'à et y compris son
  propre guillemet de fermeture, puis ajoutait un séparateur par-dessus —
  produisant un JSON malformé du type `"query": "...", "..."` quand le
  modèle tentait de continuer le JSON lui-même. Le correctif reconstruit la
  valeur uniquement à partir du texte précédant le premier guillemet non
  échappé, en écartant tout ce qui a été généré après.
- **Un prompt contenant un guillemet interne**
  (`Say "hello" to {name}`) n'était pas recopié correctement par le très
  petit modèle quand la génération était insuffisamment contrainte. Rendre
  le prompt few-shot explicite sur l'échappement des guillemets internes
  (avec un exemple concret) a suffi à corriger ce cas, une fois le bug de
  corruption JSON ci-dessus également corrigé.

## Analyse de performance

- **Précision** : 100% sur les prompts partagés avec
  `data/correction/function_calling_corrections.json` (8/8), largement
  au-dessus de l'objectif de 90% fixé par le sujet.
- **Validité** : chaque sortie générée est un JSON valide et parseable —
  aucune erreur de syntaxe — car la structure JSON elle-même (accolades,
  séparateurs, guillemets) est écrite par le programme, pas échantillonnée
  depuis le modèle.
- **Vitesse** : l'ensemble des 11 prompts de test, chargement du modèle
  inclus, s'exécute en environ 10 à 15 secondes sur CPU, très en dessous
  du budget de 5 minutes.
- **Fiabilité** : la génération est déterministe (décodage glouton via
  `argmax`, pas d'échantillonnage), donc les résultats sont stables d'une
  exécution à l'autre.

## Stratégie de test

Les tests unitaires (`tests/`, lancés avec `make test`) couvrent chaque
fonction pure ne nécessitant pas de charger le modèle : le nettoyage des
paramètres et l'extraction JSON (`test_utils.py`), les primitives de
masquage des logits (`test_masking.py`), la construction du prompt
(`test_prompting.py`), et la classe `Config`, y compris l'analyse des
arguments CLI et la gestion d'erreurs sur des fichiers d'entrée malformés
(`test_parsing.py`).

La validation de bout en bout se fait en exécutant le pipeline complet sur
les fichiers fournis dans `data/input/` et en comparant la sortie à
`data/correction/function_calling_corrections.json`.

## Ressources

- Sujet du projet : *« call me maybe — Introduction to function calling in
  LLMs »* (fourni séparément, non inclus dans ce dépôt)
- [Documentation Hugging Face `transformers`](https://huggingface.co/docs/transformers)
- [Documentation pydantic](https://docs.pydantic.dev/)

### Usage de l'IA

- **Comprendre la tokenisation** tout au début du projet : comment le
  tokenizer de `llm_sdk` encode/décode le texte, et comment un espace
  précédent peut fusionner avec le caractère suivant en un seul token —
  une compréhension qui a ensuite permis d'expliquer pourquoi les nombres
  négatifs disparaissaient.
- **Écriture d'une partie de la suite de tests unitaires** (`tests/`) pour
  les fonctions pures ne dépendant pas du modèle.
- **Rédaction de cette documentation.**

Chaque changement a été relu, testé contre les données fournies, et
compris avant d'être conservé.
