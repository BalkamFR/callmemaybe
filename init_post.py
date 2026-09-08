#!/usr/bin/env python3
"""Script de reconfiguration automatique lors d'un changement de poste à 42."""

import getpass
import os
from pathlib import Path
import subprocess
import sys


def setup_workspace() -> None:
    user = getpass.getuser()
    project_dir = Path(__file__).resolve().parent
    venv_link = project_dir / ".venv"

    goinfre_dir = Path("/goinfre") / user
    local_venv = goinfre_dir / "callmemaybe_venv"
    uv_cache = goinfre_dir / "uv-cache"
    hf_cache = goinfre_dir / "hf-cache"

    print(f"🔧 Configuration pour l'utilisateur : {user}")

    # 1. Création des répertoires locaux sur le goinfre
    for folder in (local_venv, uv_cache, hf_cache):
        folder.mkdir(parents=True, exist_ok=True)
    print(f"📁 Dossiers goinfre créés/vérifiés dans : {goinfre_dir}")

    # 2. Nettoyage de l'ancien lien symbolique ou dossier .venv
    if venv_link.is_symlink() or venv_link.exists():
        if venv_link.is_symlink():
            venv_link.unlink()
        else:
            # Sécurité au cas où .venv serait un vrai dossier
            subprocess.run(["rm", "-rf", str(venv_link)], check=True)
        print("🧹 Ancien .venv supprimé.")

    # 3. Création du lien symbolique vers le venv local
    venv_link.symlink_to(local_venv)
    print(f"🔗 Lien symbolique créé : {venv_link} -> {local_venv}")

    # 4. Préparation de l'environnement d'exécution pour uv
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(uv_cache)
    env["HF_HOME"] = str(hf_cache)

    # 5. Synchronisation des paquets avec uv
    print("\n📦 Installation des dépendances via uv sync...")
    try:
        subprocess.run(["uv", "sync"], env=env, check=True, cwd=project_dir)
        print("\n✅ Environnement prêt et synchronisé sur ce poste !")
    except subprocess.CalledProcessError as err:
        print(f"\n❌ Erreur lors de uv sync (code {err.returncode})", file=sys.stderr)
        sys.exit(err.returncode)


if __name__ == "__main__":
    setup_workspace()