# mti865_projet
Projet de MTI865 - Computer vision 

# Idées de modification de l'entraînement 

- Self-training : rajouter des exemples de GT où le modèle est confiant 

- Consistence de transformation : Les transformations des données non étiquettées doivent être aussi appliquées à leur prédiction : ajout d'un terme de Loss 

https://neptune.ai/blog/pytorch-loss-functions 


# Problèmes à checker 

L'augmentation augmente pas le nombre d'exemples : toujours 24/25 

Optimisation : option num_workers dans le DataLoader 

Unlabeled or Unlabelled ? 