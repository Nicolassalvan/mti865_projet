from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from progressBar import printProgressBar

import medicalDataLoader
import argparse
import utils

from UNet_Base import *
import random
import torch
import pdb
import matplotlib.pyplot as plt
import numpy as np
import os

NUM_CLASSES = 4

# Define global hyperparameters
TOTAL_EPOCHS = 120  # Number of epochs to train for
BATCH_SIZE_VAL = 4

# Set device depending on the availability of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():  # Apple M-series of chips
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Define image and mask transformations
transform = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.137], std=[0.1733])])

mask_transform = v2.Compose(
    [
        v2.ToTensor(),
    ]
)


def define_dataloaders(root_dir: str, batch_size: int, batch_size_val: int):
    print(f"Dataset: {root_dir} ")

    train_set_full = medicalDataLoader.MedicalImageDataset(
        "train",
        root_dir,
        transform=transform,
        mask_transform=mask_transform,
        augment=True,
        equalize=False,
    )

    train_loader_full = DataLoader(
        train_set_full,
        batch_size=batch_size,
        worker_init_fn=np.random.seed(0),
        num_workers=0,
        shuffle=True,
    )

    val_set = medicalDataLoader.MedicalImageDataset(
        "val",
        root_dir,
        transform=transform,
        mask_transform=mask_transform,
        equalize=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size_val,
        worker_init_fn=np.random.seed(0),
        num_workers=0,
        shuffle=False,
    )

    return train_loader_full, val_loader


def run_training(
    writer: SummaryWriter,
    batch_size: int = 8,
    batch_size_val: int = 4,
    lr: float = 0.01,
    weight_decay: float = 1e-5,
    data_dir: str = "./data/",
):
    print("-" * 40)
    print("~~~~~~~~  Starting the training... ~~~~~~")
    print("-" * 40)

    print(f"Using device: {device}")

    print("~~~~~~~~~~~ Creating the UNet model ~~~~~~~~~~")
    modelName = f"UNet_{batch_size}_{lr}_{weight_decay}"
    print(" Model Name: {}".format(modelName))

    ## CREATION OF YOUR MODEL
    net = UNet(NUM_CLASSES).to(device)

    print(
        "Total params: {0:,}".format(
            sum(p.numel() for p in net.parameters() if p.requires_grad)
        )
    )

    # DEFINE YOUR OUTPUT COMPONENTS (e.g., SOFTMAX, LOSS FUNCTION, ETC)
    CE_loss = torch.nn.CrossEntropyLoss().to(device)

    ce_loss_weight = 0.7
    dice_loss_weight = 0.3

    ## DEFINE YOUR OPTIMIZER
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    ### To save statistics ####
    train_losses = []
    train_dc_losses = []
    val_losses = []
    # val_dc_losses = []

    best_loss_val = 1000

    directory = f"Results/Statistics/{modelName}"

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    if not os.path.exists(directory):
        os.makedirs(directory)

    ## START THE TRAINING
    train_loader_full, val_loader = define_dataloaders(
        data_dir, batch_size, batch_size_val
    )

    early_terminator = utils.EarlyTermination(max_epochs_without_improvement=7, min_delta=0.01)
    ## FOR EACH EPOCH
    for epoch in range(TOTAL_EPOCHS):
        net.train()

        num_batches = len(train_loader_full)
        # print("Number of batches: ", num_batches)

        running_train_loss = 0
        running_dice_loss = 0

        # Training loop
        for idx, data in enumerate(train_loader_full):
            ### Set to zero all the gradients
            net.zero_grad()
            optimizer.zero_grad()

            ## GET IMAGES, LABELS and IMG NAMES
            images, labels, _ = data

            ### From numpy to torch variables
            labels = utils.to_var(labels).to(device)
            images = utils.to_var(images).to(device)

            # Forward pass
            net_predictions = net(
                images
            )  # Predictions have shape [batch_size, num_classes, height, width]

            # Get the segmentation classes
            segmentation_classes = utils.getTargetSegmentation(labels)
            # Modify segmentation classes to be one-hot encoded (shape [batch_size, num_classes, height, width])
            dice_target = (
                F.one_hot(segmentation_classes, num_classes=NUM_CLASSES)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            # Compute the loss
            loss_ce = ce_loss_weight * CE_loss(net_predictions, segmentation_classes)
            loss_dice = dice_loss_weight * DiceLoss()(net_predictions, dice_target)
            loss = loss_ce + loss_dice

            running_train_loss += loss.item()
            # dice_loss = dice_coefficient(net_predictions, labels)
            # dice_loss = utils.compute_dsc(net_predictions, labels)
            # running_dice_loss += dice_loss

            # Backprop
            loss.backward()
            optimizer.step()

            # Add the loss to the tensorboard every 5 batches
            if idx % 10 == 0:
                writer.add_scalar(
                    "Loss/train",
                    running_train_loss / (idx + 1),
                    epoch * len(train_loader_full) + idx,
                )

            if idx % 100 == 0:
                # Also add visualizations of the images
                probs = torch.softmax(net_predictions, dim=1)
                y_pred = torch.argmax(probs, dim=1)
                writer.add_figure(
                    "predictions vs. actuals",
                    utils.plot_net_predictions(images, labels, y_pred, batch_size),
                    global_step=epoch * len(train_loader_full) + idx,
                )

            # THIS IS JUST TO VISUALIZE THE TRAINING
            printProgressBar(
                idx + 1,
                num_batches,
                prefix="[Training] Epoch: {} ".format(epoch),
                length=15,
                suffix=" Loss: {:.4f}, ".format(running_train_loss / (idx + 1)),
            )

        train_loss = running_train_loss / num_batches
        train_losses.append(train_loss)

        train_dc_loss = running_dice_loss / num_batches
        train_dc_losses.append(train_dc_loss)

        net.eval()
        val_running_loss = 0
        val_running_dc = 0

        # Validation loop
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                images, labels, img_names = data

                labels = utils.to_var(labels).to(device)
                images = utils.to_var(images).to(device)

                net_predictions = net(images)

                segmentation_classes = utils.getTargetSegmentation(labels)
                dice_target = (
                    F.one_hot(segmentation_classes, num_classes=NUM_CLASSES)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

                loss_ce = ce_loss_weight * CE_loss(
                    net_predictions, segmentation_classes
                )
                loss_dice = dice_loss_weight * DiceLoss()(net_predictions, dice_target)
                loss = loss_ce + loss_dice

                val_running_loss += loss.item()
                # dice_loss = dice_coefficient(net_predictions, labels)
                # dice_loss = utils.compute_dsc(net_predictions, labels)
                # val_running_dc += dice_loss

                if idx % 10 == 0:
                    writer.add_scalar(
                        "Loss/val",
                        val_running_loss / (idx + 1),
                        epoch * len(val_loader) + idx,
                    )

                printProgressBar(
                    idx + 1,
                    len(val_loader),
                    prefix="[Validation] Epoch: {} ".format(epoch),
                    length=15,
                    suffix=" Loss: {:.4f}, ".format(val_running_loss / (idx + 1)),
                )

        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)

        # Check if model performed best and save it if true
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            if not os.path.exists("./models/" + modelName):
                os.makedirs("./models/" + modelName)
            torch.save(
                net.state_dict(), "./models/" + modelName + "/" + str(epoch) + "_Epoch"
            )
            np.save(os.path.join(directory, "Losses.npy"), train_losses)

        printProgressBar(
            num_batches,
            num_batches,
            done="[Epoch: {}, TrainLoss: {:.4f}, TrainDice: {:.4f}, ValLoss: {:.4f}".format(
                epoch, train_loss, train_dc_loss, val_loss
            ),
        )

        # Early stopping
        if early_terminator.should_terminate(val_loss):
            print("Stopping early as validation loss has not decreased within threshold")
            break

    writer.flush()  # Flush the writer to ensure that all the data is written to disk

    # Return best overall loss as comparison metric for grid search
    return best_loss_val


def perform_gridsearch(data_dir: str):
    """
    Perform a grid search over specified hyperparameters to find the best combination.

    This function iterates over all possible combinations of batch sizes, learning rates,
    and weight decays, runs training for each combination, and tracks the combination
    that results in the lowest loss.

    Args:
        writer (SummaryWriter): An instance of SummaryWriter to log training information.

    Returns:
        None
    """
    # Define hyperparameters for grid search
    batch_sizes = [4, 8, 16]
    lrs = [0.01, 0.001]
    weight_decays = [1e-5, 1e-6]

    run_history = set()
    best_run = {
        "batch_size": None,
        "lr": None,
        "weight_decay": None,
        "loss": 9000,
    }

    possible_combination_len = len(batch_sizes) * len(lrs) * len(weight_decays)
    print(f"Running grid search for {possible_combination_len} combinations")

    # Create list with all possible permutations of hyperparameters
    permutations = [
        (batch_size, lr, weight_decay)
        for batch_size in batch_sizes
        for lr in lrs
        for weight_decay in weight_decays
    ]

    try:
        for perm in permutations:
            writer = SummaryWriter()
            batch_size, lr, weight_decay = perm
            print(
                f"Running training for batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}"
            )
            loss_val = run_training(
                writer, batch_size, BATCH_SIZE_VAL, lr, weight_decay, data_dir
            )
            run_history.add((batch_size, lr, weight_decay, loss_val))
            if loss_val < best_run["loss"]:
                print(
                    f"New best run found: batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}, loss={loss_val}"
                )
                best_run["batch_size"] = batch_size
                best_run["lr"] = lr
                best_run["weight_decay"] = weight_decay
                best_run["loss"] = loss_val
            writer.close()
    except KeyboardInterrupt:
        print("Grid search interrupted. Closing writer and printing best run.")
        writer.close()
        print(f"Best run: {best_run}")
        print(f"Run history: {run_history}")
        return

    print(f"Grid search completed. Best run: {best_run}")
    print(f"Run history: {run_history}")

def perform_random_search():
    param_ranges = {
        "batch_size": [4, 8, 16],
        "lr": [0.01, 0.001],
        "weight_decay": [1e-5, 1e-6]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Challenge Script")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/",
        help="Path to the data directory",
    )
    args = parser.parse_args()

    # Ensure the data directory exists
    if not os.path.exists(args.data_dir):
        print("Data directory does not exist. Please check the path.")
        exit(1)

    # Pass the data directory to the grid search function
    perform_gridsearch(args.data_dir)
