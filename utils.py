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
import matplotlib.pyplot as plt

# from scipy.spatial.distance import directed_hausdorff




labels = {0: "Background", 1: "Foreground"}
LABEL_TO_COLOR = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}

def computeDSC(pred, gt):
    """
    Compute the Dice Similarity Coefficient between the predicted segmentation and the ground truth segmentation.

    Args:
        pred (torch.Tensor): The predicted segmentation.
        gt (torch.Tensor): The ground truth segmentation.
    
    Returns:
        float: The Dice Similarity Coefficient between the predicted segmentation and the ground truth
    """
    dscAll = []
    pdb.set_trace()
    for i_b in range(pred.shape[0]):
        pred_id = pred[i_b, 1, :]
        gt_id = gt[i_b, 0, :]
        dscAll.append(dc(pred_id.cpu().data.numpy(), gt_id.cpu().data.numpy()))

    DSC = np.asarray(dscAll)

    return DSC.mean()


def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
        imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

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


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.33333334  # for ACDC this value
    return (batch / denom).round().long().squeeze()


from scipy import ndimage


def inference(net, img_batch, modelName, epoch):
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax().cuda()
    CE_loss = nn.CrossEntropyLoss().cuda()

    losses = []
    for i, data in enumerate(img_batch):

        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
        segmentation_classes = getTargetSegmentation(labels)
        CE_loss_value = CE_loss(net_predictions, segmentation_classes)
        losses.append(CE_loss_value.cpu().data.numpy())
        pred_y = softMax(net_predictions)
        masks = torch.argmax(pred_y, dim=1)

        path = os.path.join('./Results/Images/', modelName, str(epoch))

        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(
            torch.cat([images.data, labels.data, masks.view(labels.shape[0], 1, 256, 256).data / 3.0]),
            os.path.join(path, str(i) + '.png'), padding=0)

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    losses = np.asarray(losses)

    return losses.mean()


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()


def plot_net_predictions(imgs, true_masks, masks_pred, batch_size):

    fig, ax = plt.subplots(3, batch_size, figsize=(20, 15))

    for i in range(batch_size):

        img = np.transpose(imgs[i].cpu().detach().numpy(), (1, 2, 0))
        mask_pred = masks_pred[i].cpu().detach().numpy()
        mask_true = np.transpose(true_masks[i].cpu().detach().numpy(), (1, 2, 0))

        ax[0, i].imshow(img)
        ax[1, i].imshow(mask_to_rgb(mask_pred))
        ax[1, i].set_title("Predicted")
        ax[2, i].imshow(mask_true)
        ax[2, i].set_title("Ground truth")

    return fig


def mask_to_rgb(mask):
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for i in np.unique(mask):
        rgb[mask == i] = LABEL_TO_COLOR[i]
    return rgb