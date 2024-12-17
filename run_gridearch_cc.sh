#!/bin/bash
#SBATCH --job-name=mti865_run_gridsearch
#SBATCH --account=def-chdesa
##SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/output_search_%j.txt  # Log de sortie
#SBATCH --error=logs/error_search_%j.txt   # Log d'erreur
#SBATCH --time=08:00:00             # Temps maximum (adapter)
#SBATCH --mem=16G                    # Mémoire (adapter)
# Charger les modules nécessaires
module load python/3.10 cuda cudnn
module load scipy-stack 


# Créer un environnement virtuel temporaire (facultatif si vous en avez besoin)
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

# Copier les données dans le répertoire temporaire pour accélérer l'accès
mkdir $SLURM_TMPDIR/mti865_data
tar xf data.tar -C $SLURM_TMPDIR/mti865_data

# Run training script
python3 segmentation_challenge_script.py --data_path $SLURM_TMPDIR/mti865_data

# Désactiver l'environnement virtuel (facultatif)
deactivate
