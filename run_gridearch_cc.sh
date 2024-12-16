#!/bin/bash
#SBATCH --job-name=mti865_run_gridsearch
#SBATCH --account=def-chdesa
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/output_search_%j.txt  # Log de sortie
#SBATCH --error=logs/error_search_%j.txt   # Log d'erreur
#SBATCH --time=08:00:00             # Temps maximum (adapter)
#SBATCH --mem=16G                    # Mémoire (adapter)
# Charger les modules nécessaires
module load python/3.10
module load scipy-stack 

# Créer un environnement virtuel temporaire (facultatif si vous en avez besoin)
source .venv/bin/activate
# pip install jupyter

# Run training script
python3 segmentation_challenge_script.py

# Désactiver l'environnement virtuel (facultatif)
deactivate
