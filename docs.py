"""Documentation technique du projet Call Me Maybe.

Ce module résume l'API du SDK, les contraintes imposées par l'école 42,
ainsi que l'architecture à concevoir pour le projet.
"""

# =====================================================================
# 1. MÉTHODES PUBLIQUES DU SDK (llm_sdk.Small_LLM_Model)
# =====================================================================
# Interdiction formelle d'accéder aux attributs privés (_model, _tokenizer,
# _device, etc.) sous peine d'invalidation du projet.

# Instanciation :
model = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
# -> Charge le modèle et le tokenizer en mémoire locale.
# -> À instancier une seule fois au démarrage du programme.

# Encodage texte -> identifiants :
tensor_ids = model.encode(text: str) -> torch.Tensor
# -> Convertit du texte brut en IDs de tokens.
# -> Retourne un tenseur 2D de forme (1, N).
# -> Pour extraire la liste d'entiers utilisable : tensor_ids[0].tolist()

# Calcul des logits :
logits = model.get_logits_from_input_ids(input_ids: list[int]) -> list[float]
# -> Prend en entrée l'historique complet des IDs de tokens accumulés.
# -> Retourne une liste de 151 936 scores bruts réels (un par token possible).

# Décodage identifiants -> texte :
text = model.decode(ids: torch.Tensor | list[int]) -> str
# -> Traduit une liste d'IDs ou un ID unique en chaîne de caractères lisible.

# Récupération du vocabulaire :
path = model.get_path_to_vocab_file() -> str
# -> Retourne le chemin d'accès absolu vers le fichier vocab.json sur le disque.
# -> Permet d'associer chaque sous-chaîne de caractères à son token ID.


# =====================================================================
# 2. RÈGLES ET BIBLIOTHÈQUES AUTORISÉES (Section IV du sujet)
# =====================================================================
# - Python >= 3.10, respect strict de PEP 257 (docstrings) et flake8.
# - Typage complet (typing) validé par mypy sans aucune erreur.
# - Packages autorisés :
#     * numpy : calculs matriciels, argmax, masquage de tenseurs.
#     * pydantic : obligatoire pour TOUTES les classes de données.
#     * json : manipulation des fichiers I/O et sérialisation.
#     * argparse : gestion des arguments de la ligne de commande.
# - Packages strictement interdits dans src/ :
#     * torch, transformers, huggingface_hub, dspy, outlines, guidance.
#     * Tout doit passer par les méthodes publiques de llm_sdk.


# =====================================================================
# 3. STRUCTURE DES DONNÉES D'ENTRÉE ET SORTIE
# =====================================================================
# Entrées (data/input/) :
# 1. functions_definition.json :
#    Liste d'objets définissant les outils disponibles :
#    [
#      {
#        "name": "fn_add_numbers",
#        "description": "Add two numbers together...",
#        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
#        "returns": {"type": "number"}
#      },
#      ...
#    ]
#
# 2. function_calling_tests.json :
#    Liste de prompts utilisateur à traiter :
#    [
#      {"prompt": "What is the sum of 2 and 3?"},
#      ...
#    ]

# Sortie (data/output/function_calling_results.json) :
# Format imposé à 100 % (aucun texte superflu, JSON strictement valide) :
# [
#   {
#     "prompt": "What is the sum of 2 and 3?",
#     "name": "fn_add_numbers",
#     "parameters": {"a": 2.0, "b": 3.0}
#   },
#   ...
# ]

