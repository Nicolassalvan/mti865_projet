#!/bin/bash
#SBATCH --job-name=run_notebooks
#SBATCH --output=logs/output_%j.txt  # Log de sortie
#SBATCH --error=logs/error_%j.txt   # Log d'erreur
#SBATCH --account=def-chdesa
#SBATCH --time=1:00:00             # Temps maximum (adapter)
#SBATCH --mem=16G                    # Mémoire (adapter)
#SBATCH --cpus-per-task=4           # Nombre de CPU (adapter)
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=nicolas.salvan.1@ens.etsmtl.ca # Mettez votre mail ici 

# Charger les modules nécessaires
module load python/3.10
module load scipy-stack 

# Créer un environnement virtuel temporaire (facultatif si vous en avez besoin)
source my_env/bin/activate


# Définir les dossiers
NOTEBOOK_DIR="notebooks"
OUTPUT_DIR="output"

# Créer le dossier output s'il n'existe pas
mkdir -p $OUTPUT_DIR

# Parcourir tous les fichiers notebooks et les exécuter avec jupyter nbconvert
for notebook in $NOTEBOOK_DIR/*.ipynb; do
    notebook_name=$(basename "$notebook")
    output_notebook="$OUTPUT_DIR/$notebook_name"

    echo "[ ==== Exécution de $notebook... ==== ]"
    jupyter nbconvert --to notebook --execute --output "$output_notebook" "$notebook" --ExecutePreprocessor.timeout=-1
    echo "[ ==== Exécution de $notebook terminée ==== ]"
    echo "[ ==== Notebook enregistré sous $output_notebook ==== ]"
    echo ""

    
done

# Désactiver l'environnement virtuel (facultatif)
deactivate
