from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from progressBar import printProgressBar
from utils import *

"""
fonctions utils pour le modele student : 
- to_var : permet de prendre un tensor et le convertir en variable
- mask_to_rgb : convertit un masque de segmentation en une image rgb
- inferenceTeacher permet de générer les pseudo labels et enregistrer les distributions de proba dans Data/train/Img-UnlabeledProbabilities/{numEpoch},
permet aussi enregistrer les images prédites dans Data/train/Img-UnlabeledPredictions/{numEpoch}
permet aussi enregistrer les comparaisons image d'entrée/prédiction dans Results/Images/TeacherUnlabeledPredictions
- plot_net_predictions_without_ground_truth : affichage sur tensorboard des images quand on ne dispose pas du GT
- dynamic_weight_kl_div : loss utilisée pour les pseudo labels pendant l'entrainement. Applique des poids plus forts si le modele se trompe sur les classes à prédire et les classes et s'il prédit des classes improbables. On utilise une KL div pondérée
- distillation_loss : KL div classique utilisée pendant les tests
- get_teacher_proba : permet de récupérer les distributions de probabilités du teacher pour les unlabeled. elles sont enregistrées dans Data/train/Img-UnlabeledProbabilities/{num_epoch_teacher}



"""

labels = {0: "Background", 1: "Foreground"}
LABEL_TO_COLOR = {
    0: [0, 0, 0], # Background
    1: [67, 67, 67], # Right ventricle (RV)
    2: [154, 154, 154], # Myocardium (MYO)
    3: [255, 255, 255] # Left ventricle (LV)
}

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def inferenceTeacher(net, img_batch, epoch, device):

    net.eval() #se mettre en mode evaluation 
    total = len(img_batch)
    print("len image batch : ", total)



    for i, data in enumerate(img_batch):
        images, img_names = data
        images = to_var(images).to(device)

        net_predictions = net(images)
        probs = torch.softmax(net_predictions, dim=1)
        y_pred = torch.argmax(probs, dim=1)

        # Sauvegarde image prédite vs image de base dans Results/Images/TeacherUnlabeledPredictions
        path = os.path.join(f"./Results/Images/TeacherUnlabeledPredictions/{epoch}") #creation chemin de sauvegarde
        if not os.path.exists(path):
            os.makedirs(path)
        fig = plot_net_predictions_without_ground_truth(images, y_pred, img_names, len(images))
        fig.savefig(os.path.join(path, str(i) + ".png"))

        #Sauvegarde image prédite dans Data/train/Img-UnlabeledPredictions pour s'en servir comme data
        save_folder = os.path.join(f"./Data/train/Img-UnlabeledPredictions/{epoch}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        #on doit garder les distributions de probabilités prédites
        save_folder_probs = os.path.join(f"./Data/train/Img-UnlabeledProbabilities/{epoch}")
        if not os.path.exists(save_folder_probs):
            os.makedirs(save_folder_probs)

        
        for j in range(len(images)) :
            mask_pred = y_pred[j].cpu().detach().numpy() #get predicted image
            img_name = os.path.basename(img_names[j])  # extrait le nom du fichier
            img_name = os.path.splitext(img_name)[0]  # retire l'extension
            save_path = os.path.join(save_folder, f"{img_name}.png")
            plt.imsave(save_path, mask_pred, cmap='gray')

            prob_map = probs[j].cpu().detach().numpy()  # distribution de proba pour l'image
            save_path_prob = os.path.join(save_folder_probs, f"{img_name}.npy")
            np.save(save_path_prob, prob_map)
               
       
        plt.close(fig)

    printProgressBar(total, total, done="[Inference] Teacher Inference Done !")


def plot_net_predictions_without_ground_truth(imgs, masks_pred, img_names, batch_size):

    fig, ax = plt.subplots(2, batch_size, figsize=(20, 15))

    for i in range(batch_size):

        img = np.transpose(imgs[i].cpu().detach().numpy(), (1, 2, 0))
        mask_pred = masks_pred[i].cpu().detach().numpy()
        

        img_name = os.path.basename(img_names[i])  # path image
        img_name = os.path.splitext(img_name)[0] #only image name
        

        ax[0, i].imshow(img) #image d'entrée 
        ax[0, i].set_title("Input Image " + img_name)
        
        ax[1, i].imshow(mask_to_rgb(mask_pred)) #image sur laquelle on applique les prédictions
        ax[1, i].set_title("Predicted Mask")

    return fig


def dynamic_weight_kl_div(y_pred_student, y_pred_teacher, confidence=0.7, alpha=0.1, temperature = 5.0):
    """
    Calcule une KL divergence pondérée avec accent sur les erreurs significatives
    
    Args:
    - soft_y_pred_student : tensor de proba du student [num_classes, H, W]
    - soft_y_pred_teacher : tensor de proba du teacher [num_classes, H, W]
    - confidence : seuil de confiance pour considérer une classe comme dominante
    - alpha : poids pour les classes non dominantes
    
    Returns:
    - loss : Perte pondérée KL
    """
    soft_y_pred_teacher = torch.softmax(y_pred_teacher / temperature, dim=1)
    soft_y_pred_student = torch.softmax(y_pred_student / temperature, dim=1)
    
    # classes dominantes : fortement prédites par le teacher
    max_probs, _ = soft_y_pred_teacher.max(dim=0)  # probas des classes prédites
    dominant_mask = (soft_y_pred_teacher >= confidence)  # masque pour récupérer classes dominantes prédites avec confiance

    # de base, on met un poids faible pour la loss mais on met un poids élevé pour les classes predites avec confiance
    weights = torch.ones_like(soft_y_pred_teacher) * alpha  # par défaut : poids réduit
    weights[dominant_mask] = 1.0  # mais poids élevé pour les classes dominantes

    # on applique pénalité pour les grosses erreurs du student
    penalty = torch.abs(soft_y_pred_student - soft_y_pred_teacher)  # différence entre student et teacher
    penalty_weight = penalty * (1 - soft_y_pred_teacher)  # plus la classe est improbable pour le teacher, plus l'erreur est pénalisée
    weights += penalty_weight

    # KL divergence pondérée : resultat
    kl_div = torch.nn.functional.kl_div(soft_y_pred_student.log(), soft_y_pred_teacher, reduction='none')  # [num_classes, H, W]
    weighted_kl = kl_div * weights

    # moyenne sur toutes les classes et tous les pixels
    return weighted_kl.mean()


#KL div classique
def distillation_loss(y_pred_student, y_pred_teacher, temperature = 5.0, alpha=0.1) :


    max_probs, _ = teacher_probs.max(dim=0)

    soft_y_pred_teacher = torch.softmax(y_pred_teacher / temperature, dim=1)
    soft_y_pred_student = torch.softmax(y_pred_student / temperature, dim=1)

    # on applique ponderation en fonction des classes predites par le teacher(ponderation moindre quand on calcule la loss pour classe non predite par teacher)

    kl_div = torch.nn.functional.kl_div(soft_y_pred_student.log(), soft_y_pred_teacher, reduction='none')  # [num_classes, H, W]
    #nn.KLDivLoss(reduction='batchmean')(torch.log(soft_y_pred_student), soft_y_pred_teacher)


    # Moyenne sur toutes les classes et tous les pixels
    return kl_div
    
    
    # soft_y_pred_teacher = torch.softmax(y_pred_teacher / temperature, dim=1)
    # soft_y_pred_student = torch.softmax(y_pred_student / temperature, dim=1)
    
    # loss = nn.KLDivLoss(reduction='batchmean')(torch.log(soft_y_pred_student), soft_y_pred_teacher)
    
    # return loss

def get_teacher_proba (num_epoch_teacher, img_names, device) :
    #on recupere les distributions de proba du teacher pour les unlabeled
    teacher_probs_path = f"./Data/train/Img-UnlabeledProbabilities/{num_epoch_teacher}" 
    batch_teacher_probs = []
    #confidence_threshold = 0.25
    for img_name in img_names: #parcourir toutes les images du batch
        img_base_name = os.path.basename(img_name)
        img_base_name = os.path.splitext(img_base_name)[0]
        prob_file_path = os.path.join(teacher_probs_path, f"{img_base_name}.npy") #on trouve le chemin exact de la distribution de proba de l'image

        teacher_probs = torch.tensor(np.load(prob_file_path)).to(device)

        # on doit charger la distribution de probas et la convertir en tenseur pytorch
        
        batch_teacher_probs.append(teacher_probs)
    segmentation_classes_teacher = torch.stack(batch_teacher_probs)


    
    return segmentation_classes_teacher


