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
from medpy.metric.binary import dc, hd, asd, assd, jc
import scipy.spatial
from torch import tensor
from torchmetrics.functional.classification import binary_jaccard_index

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


# Inspiré de https://github.com/IvLabs/stagewise-knowledge-distillation/issues/12
# et https://github.com/IvLabs/stagewise-knowledge-distillation/blob/master/semantic_segmentation/utils/metrics.py

def mean_iou(model, dataloader):
    # IOU over the entire dataset for each class
    ious_class=list()
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    nb_im=0

    num_classes= 4
    for sem_class in range(num_classes):
        class_intersection=0
        class_union=0
        for idx, data in enumerate(dataloader):
            ## GET IMAGES, LABELS and IMG NAMES
            images, labels, img_names = data
            ### From numpy to torch variables
            labels = to_var(labels).to(device)
            images = to_var(images).to(device)
            for i in range(images.shape[0]):
                #print("images.len()=",images.len())
                prediction = model(images)
                prediction = F.softmax(prediction, dim=1)
                prediction = torch.argmax(prediction, axis=1).squeeze(1)

                # probs = torch.softmax(prediction, dim=1)
                # y_pred = torch.argmax(probs, dim=1)

                prediction = prediction.view(-1)
                labels = labels.view(-1)
                pred_inds = (prediction == sem_class)
                target_inds = (labels == sem_class)
                if target_inds.long().sum().item() == 0:
                    # intersection_now = float('nan')
                    union_now1 = float('nan')
                else: 
                    intersection_now = (pred_inds[target_inds]).long().sum().item()
                    union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
                # Compute the total intersection and total union for each class over the entire dataset
                class_intersection+= intersection_now
                class_union+=union_now
                nb_im+=1
        print("nb_im=",nb_im)
        iou_now = float(class_intersection) / float(class_union)
        ious_class.append(iou_now)

    return (sum(ious_class) / len(ious_class)) #return mean of iou for each class

def mIOU(label, pred, num_classes=4):
    pred = F.softmax(pred, dim=1)              
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)

def mean_dsc(model, dataloader):
    # IOU over the entire dataset for each class
    dice_class=list()
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    nb_im=0
    smooth=1e-6

    num_classes= 4
    for sem_class in range(num_classes):
        class_intersection=0
        class_denom=0
        for idx, data in enumerate(dataloader):
            ## GET IMAGES, LABELS and IMG NAMES
            images, labels, img_names = data
            for i in enumerate(images):
                #print("images.len()=",images.len())
                ### From numpy to torch variables
                labels = to_var(labels).to(device)
                images = to_var(images).to(device)

                prediction = model(images)
                prediction = F.softmax(prediction, dim=1)
                prediction = torch.argmax(prediction, axis=1).squeeze(1)

                # probs = torch.softmax(prediction, dim=1)
                # y_pred = torch.argmax(probs, dim=1)

                prediction = prediction.view(-1)
                labels = labels.view(-1)
                pred_inds = (prediction == sem_class)
                target_inds = (labels == sem_class)
                # print("target_inds size=",target_inds.size())
                # print("pred_inds size=",pred_inds.size())

                if target_inds.long().sum().item() == 0:
                    # intersection = float('nan')
                    denom1 = float('nan')
                else: 
                    intersection = (pred_inds[target_inds]).long().sum().item()
                    denom = pred_inds.long().sum().item() + target_inds.long().sum().item()
                # Compute the total intersection and total union for each class over the entire dataset
                class_intersection+= intersection
                class_denom+=denom
                nb_im+=1
        print("nb_im=",nb_im)
        dice = (float(2 * class_intersection) + smooth) / (float(class_denom) + smooth)
        dice_class.append(dice)

    return (sum(dice_class) / len(dice_class)) #return mean of iou for each class


def dice_coeff(mask1, mask2, smooth=1e-6, num_classes=4):
    dice = 0
    for sem_class in range(num_classes):
        # mask1 = mask1.view(-1)
        # mask2 = mask2.view(-1)
        print ("mask1.size=",mask1.size())
        print ("mask2.size=",mask2.size())
        pred_inds = (mask2 == sem_class)
        target_inds = (mask1 == sem_class)
        intersection = (pred_inds[target_inds]).long().sum().item()
        denom = pred_inds.long().sum().item() + target_inds.long().sum().item()
        dice += (float(2 * intersection) + smooth) / (float(denom) + smooth)
    return dice / num_classes

def iou(mask1, mask2, num_classes=4, smooth=1e-6):
    avg_iou = 0
    for sem_class in range(num_classes):
        pred_inds = (mask2 == sem_class)
        # print("pred_inds=",pred_inds)
        # print("pred_inds size=",pred_inds.size())
        target_inds = (mask1 == sem_class)
        # print("target_inds size=",target_inds.size())
        
        # print("target_inds=",target_inds)
        if target_inds.long().sum().item() == 0:
            # intersection = float('nan')
            denom1 = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
        avg_iou += (float(intersection_now + smooth) / float(union_now + smooth))
    return (avg_iou / num_classes)

def iou2(mask1, mask2, num_classes=4, smooth=1e-6):
    # avg_iou = list()
    avg_iou = 0
    # mask1_new=mask1.type(torch.int)
    # mask2_new=mask2.type(torch.int)
    #print("mask1=",mask1)
    # print("mask2=",mask2)
    print("mask2.size=",mask2.size())
    print("mask1.size=",mask1.size())
    print("mask1.shape[0]=",mask1.shape[0])
    # print("mask1.shape[1]=",mask1.shape[1])
    # print("mask1.shape[2]=",mask1.shape[2])
    # print("mask1.shape[3]=",mask1.shape[3])
    # for i_batch in range (mask1.shape[0]):
    #     for sem_class in range (mask1.shape[1]):
    #         intersection = (mask1[i_batch,sem_class,:,:] & mask2[i_batch,sem_class,:,:]).float().sum((2,3))
    #         union = (mask1[i_batch,sem_class,:,:] | mask2[i_batch,sem_class,:,:]).float().sum((2,3))
    #         union_prev=(mask1 | mask2).float()
    #         print("union=",union)
    #         #print("union_prev=",union_prev)
    #         #print("intersection=",intersection)
    #         avg_iou [sem_class]+= (float(intersection + smooth) / float(union + smooth))
    for i_batch in range (mask1.shape[0]):
        # intersection = (mask1[i_batch,:,:] & mask2[i_batch,:,:]).float().sum((1,2))
        # union = (mask1[i_batch,:,:] | mask2[i_batch,:,:]).float().sum((1,2))
        print ("i_batch=",i_batch)
        intersection = (mask1[i_batch] and mask2[i_batch]).float().sum((1,2))
        union = (mask1[i_batch] or mask2[i_batch]).float().sum((1,2))
        union_prev=(mask1 | mask2).float()
        print("union=",union)
        #print("union_prev=",union_prev)
        #print("intersection=",intersection)
        # avg_iou [sem_class]+= (float(intersection + smooth) / float(union + smooth))
        avg_iou += (float(intersection + smooth) / float(union + smooth))
    # for sem_class in range (mask1.shape[1]):
    #     avg_iou [sem_class]=avg_iou [sem_class]/mask1.shape[0]
    return avg_iou

def dice_score_class(model,dataloader):
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dice_class=[0,0,0,0]
    num_classes=4
    smooth=1e-6
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, img_names = data
        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        dice_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        for i_batch in range (dice_target.shape[0]):
            #Show first image and mask
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                intersect3 = torch.sum(torch.mul(probs[i_batch][classe], dice_target[i_batch][classe]), dim=(0,1))+ smooth
                
                den3 = torch.sum(probs[i_batch][classe].pow(2) + dice_target[i_batch][classe].pow(2), dim=(0,1))+ smooth
                
                dice_class3=(2.0 * (intersect3 / den3))

                dice_class[classe]+= dice_class3.item()
            nb_it+=1
            
    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class


def dice_score_class2(model,dataloader): # DICE score with dc from medpy
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dice_class=[0,0,0,0]
    num_classes=4
    smooth=1e-6
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, img_names = data

        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        dice_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        for i_batch in range (dice_target.shape[0]):
            #Show first image and mask
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                dice_class[classe]+= dc(probs[i_batch][classe].data.numpy(), dice_target[i_batch][classe].data.numpy())
            nb_it+=1

    for classe in range (num_classes):
        dice_class[classe]=dice_class[classe]/nb_it
    return dice_class


def jaccard_score_class(model,dataloader): #Jaccard (or IOU) score
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    iou_class=[0,0,0,0]
    num_classes=4
    smooth=1e-6
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, img_names = data
        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        iou_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()

        for i_batch in range (iou_target.shape[0]):
            #Show first image and mask
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Jaccard (or IOU) coefficient
                intersect3 = torch.sum(torch.mul(probs[i_batch][classe], iou_target[i_batch][classe]), dim=(0,1))+ smooth
                
                den3 = torch.sum(probs[i_batch][classe].pow(2) + iou_target[i_batch][classe].pow(2), dim=(0,1))+ smooth
                den5=den3-intersect3

                iou_c=(intersect3 / den5)
                iou_class[classe]+= iou_c.item()
            nb_it+=1
            
    for classe in range (num_classes):
        iou_class[classe]=iou_class[classe]/nb_it
    return iou_class

def jaccard_score_class2(model,dataloader): #Jaccard (or IOU) score with binary jaccard coeff from torchmetrics
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    jaccard=[0,0,0,0]
    num_classes=4
    smooth=1e-6
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, img_names = data

        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        jaccard_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        for i_batch in range (jaccard_target.shape[0]):
            #Show first image and mask
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                jaccard[classe]+= binary_jaccard_index(probs[i_batch][classe], jaccard_target[i_batch][classe],0.15).item()
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard


def jaccard_score_class3(model,dataloader):  #Jaccard (or IOU) score with jc from medpy
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    jaccard=[0,0,0,0]
    num_classes=4
    smooth=1e-6
    nb_it=0
    for idx, data in enumerate(dataloader):
        images, labels, img_names = data

        labels = to_var(labels).to(device)
        images = to_var(images).to(device)

        net_predictions = model(images)
        probs = torch.softmax(net_predictions, dim=1)
        y_pred = torch.argmax(probs, dim=1)

        segmentation_classes = getTargetSegmentation(labels)
        jaccard_target = F.one_hot(segmentation_classes, num_classes = num_classes).permute(0,3,1,2).contiguous()
        for i_batch in range (jaccard_target.shape[0]):
            #Show first image and mask
            for classe in range (num_classes):
                # Calculate the intersection and denominator parts for the Dice coefficient
                jaccard[classe]+= jc(probs[i_batch][classe].data.numpy(), jaccard_target[i_batch][classe].data.numpy())
            nb_it+=1

    for classe in range (num_classes):
        jaccard[classe]=jaccard[classe]/nb_it
    return jaccard


def miou(model, dataloader):
    # Set device depending on the availability of GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():  # Apple M-series of chips
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    ious = list()
    new_iou=list()
    for idx, data in enumerate(dataloader):
        images, labels, img_names = data
        labels = to_var(labels).to(device)
        images = to_var(images).to(device)
        prediction = model(images)
        prediction = F.softmax(prediction, dim=1)
        print ("prediction.size 1=",prediction.size())
        prediction = torch.argmax(prediction, axis=1)#.squeeze(1)
        print ("prediction.size 2=",prediction.size())
        # prediction = prediction.view(-1)
        # labels = labels.view(-1)
        print ("labels.size 1=",labels.size())
        segmentation_classes = getTargetSegmentation(labels)
        labels = F.one_hot(segmentation_classes, num_classes = 4).permute(0,3,1,2).contiguous()
        labels = torch.argmax(labels, axis=1)
        print ("labels.size 2=",labels.size())
        new_iou=iou2(labels, prediction, num_classes=4)
        for i in ious:
            ious[i]+=new_iou[i]
        #ious.append(iou(labels, prediction, num_classes=4))
    for i in ious:
        ious[i]=ious[i]/idx
    return ious
    #return (sum(ious) / len(ious))



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
        ax[1, i].imshow(mask_to_rgb(mask_pred))
        ax[1, i].set_title("Predicted")
        ax[2, i].imshow(mask_true, cmap="gray")
        ax[2, i].set_title("Ground truth")

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


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()
