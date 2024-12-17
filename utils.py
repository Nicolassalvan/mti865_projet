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
import scipy.io as sio
import pdb
import time
from os.path import isfile, join
import statistics
from PIL import Image
from medpy.metric.binary import dc, hd, asd, assd
import scipy.spatial

# from scipy.spatial.distance import directed_hausdorff


labels = {0: "Background", 1: "Foreground"}
LABEL_TO_COLOR = {
    0: [0, 0, 0], # Background
    1: [67, 67, 67], # Right ventricle (RV)
    2: [154, 154, 154], # Myocardium (MYO)
    3: [255, 255, 255] # Left ventricle (LV)
}


def compute_dsc(pred, gt):
    dsc_all = []
    # pdb.set_trace()
    for i_b in range(pred.shape[0]):
        pred_id = pred[i_b, 1, :]
        gt_id = gt[i_b, 0, :]
        dsc_all.append(dc(pred_id.cpu().data.numpy(), gt_id.cpu().data.numpy()))

    dsc = np.asarray(dsc_all)

    return dsc.mean()


def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
        imageNames = [
            f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))
        ]
        imageNames.sort()

    return imageNames


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    # pdb.set_trace()
    return (x == 1).float()


def plot_net_predictions(imgs, true_masks, masks_pred, batch_size):

    fig, ax = plt.subplots(3, batch_size, figsize=(20, 15))

    for i in range(batch_size):

        img = np.transpose(imgs[i].cpu().detach().numpy(), (1, 2, 0))
        mask_pred = masks_pred[i].cpu().detach().numpy()
        mask_true = np.transpose(true_masks[i].cpu().detach().numpy(), (1, 2, 0))

        ax[0, i].imshow(img, cmap="gray")
        # ax[1, i].imshow(mask_to_rgb(mask_pred), cmap="gray")
        ax[1, i].imshow(mask_pred, cmap="hot")
        ax[1, i].set_title("Predicted")
        ax[2, i].imshow(mask_true, cmap="hot")
        ax[2, i].set_title("Ground truth")

    return fig

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



def mask_to_rgb(mask):
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for i in np.unique(mask):
        rgb[mask == i] = LABEL_TO_COLOR[i]
    return rgb


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.33333334  # for ACDC this value
    return (batch / denom).round().long().squeeze()


def inference(net, img_batch, modelName, epoch):
    """
    Function to perform inference on a batch of images and save the results
    
    """
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax().cuda()
    CE_loss = nn.CrossEntropyLoss().cuda()

    losses = []
    for i, data in enumerate(img_batch):

        printProgressBar(
            i, total, prefix="[Inference] Getting segmentations...", length=30
        )
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
        print(net_predictions.shape)
        segmentation_classes = getTargetSegmentation(labels)
        CE_loss_value = CE_loss(net_predictions, segmentation_classes)
        losses.append(CE_loss_value.cpu().data.numpy())
        pred_y = softMax(net_predictions)
        masks = torch.argmax(pred_y, dim=1)

        path = os.path.join("./Results/Images/", modelName, str(epoch))

        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(
            torch.cat(
                [
                    images.data,
                    labels.data,
                    masks.view(labels.shape[0], 1, 256, 256).data / 3.0,
                ]
            ),
            os.path.join(path, str(i) + ".png"),
            padding=0,
        )

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    losses = np.asarray(losses)

    return losses.mean()

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

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()

class EarlyTermination:
    def __init__(self, max_epochs_without_improvement=1, min_delta=0):
        self.max_epochs_without_improvement = max_epochs_without_improvement
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float('inf') # Set to infinity so that the first validation loss is always lower

    def should_terminate(self, validation_loss):
        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.max_epochs_without_improvement:
                return True
        return False

#loss qui compare les prédictions du student avec celles du softmax mais weighted
def distillation_loss(y_pred_student, y_pred_teacher, temperature = 5.0, alpha=0.1) :


    max_probs, _ = teacher_probs.max(dim=0)

    soft_y_pred_teacher = torch.softmax(y_pred_teacher / temperature, dim=1)
    soft_y_pred_student = torch.softmax(y_pred_student / temperature, dim=1)

    # on applique ponderation en fonction des classes predites par le teacher(ponderation moindre quand on calcule la loss pour classe non predite par teacher)
    weights = teacher_probs.clone()
    weights[max_probs.unsqueeze(0) != teacher_probs] *= alpha
    kl_div = torch.nn.functional.kl_div(soft_y_pred_student.log(), soft_y_pred_teacher, reduction='none')  # [num_classes, H, W]
    weighted_kl = kl_div * weights

    # Moyenne sur toutes les classes et tous les pixels
    return weighted_kl.mean()
    
    
    # soft_y_pred_teacher = torch.softmax(y_pred_teacher / temperature, dim=1)
    # soft_y_pred_student = torch.softmax(y_pred_student / temperature, dim=1)
    
    # loss = nn.KLDivLoss(reduction='batchmean')(torch.log(soft_y_pred_student), soft_y_pred_teacher)
    
    # return loss


def dynamic_weight_kl_div(y_pred_student, y_pred_teacher, confidence_threshold=0.7, alpha=0.1, temperature = 5.0):
    """
    Calcule une KL divergence pondérée avec accent sur les erreurs significatives.
    
    Args:
    - soft_y_pred_student : Tensor de probabilités du student [num_classes, H, W]
    - soft_y_pred_teacher : Tensor de probabilités du teacher [num_classes, H, W]
    - confidence : Seuil de confiance pour considérer une classe comme dominante
    - alpha : Poids pour les classes non dominantes
    
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
        #a voir : retirer les probas incertaines (inf a 0.3 par exemple
        # max_probs, _ = teacher_probs.max(dim=0)  # on recuperre la proba des classes predites
        # low_confidence_mask = max_probs < confidence_threshold
        # teacher_probs[low_confidence_mask] = 0.00001
        
        # on doit charger la distribution de probas et la convertir en tenseur pytorch
        
        batch_teacher_probs.append(teacher_probs)
    segmentation_classes_teacher = torch.stack(batch_teacher_probs)


    
    return segmentation_classes_teacher
