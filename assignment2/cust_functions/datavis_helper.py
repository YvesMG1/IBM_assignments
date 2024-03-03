import matplotlib.pyplot as plt
import numpy as np
from cust_functions.processing_helper import Smoothed_H5Dataset

def plot_conf_matrix(cm):
    """Function to plot confusion matrix and normalized confusion matrix."""

    # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalise confusion matrix
    axes[1].matshow(cm / cm.sum(axis=1), cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(x=j, y=i, s=round(cm[i, j] / cm.sum(axis=1)[i], 2), va='center', ha='center', size='xx-large')
    axes[1].set_title("Normalised Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


def plot_metrics(train_results, val_results):
    """Function to plot training and validation loss, recall, precision and f1 score."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Metrics")

    epochs = range(1, len(train_results["train_loss"]) + 1)

    # Plot training and validation loss on the left column
    axes[0][0].plot(epochs, train_results["train_loss"], 'b-', label='Train Loss')
    axes[0][0].plot(epochs, val_results["val_loss"], 'r-', label='Validation Loss')
    axes[0][0].set_xlabel('Epochs')
    axes[0][0].set_ylabel('Loss')
    axes[0][0].legend()
    
    # Plot training and validation recall on the right column
    axes[0][1].plot(epochs, train_results["train_recall"], 'b-', label='Train Recall')
    axes[0][1].plot(epochs, val_results["val_recall"], 'r-', label='Validation Recall')
    axes[0][1].set_xlabel('Epochs')
    axes[0][1].set_ylabel('Recall')
    axes[0][1].legend()

    # Plot training and validation precision on the left column
    axes[1][0].plot(epochs, train_results["train_precision"], 'b-', label='Train Precision')
    axes[1][0].plot(epochs, val_results["val_precision"], 'r-', label='Validation Precision')
    axes[1][0].set_xlabel('Epochs')
    axes[1][0].set_ylabel('Precision')
    axes[1][0].legend()

    # Plot training and validation f1 score on the right column
    axes[1][1].plot(epochs, train_results["train_f1_score"], 'b-', label='Train F1 Score')
    axes[1][1].plot(epochs, val_results["val_f1_score"], 'r-', label='Validation F1 Score')
    axes[1][1].set_xlabel('Epochs')
    axes[1][1].set_ylabel('F1 Score')
    axes[1][1].legend()

    plt.tight_layout()
    plt.show()


def plot_with_overlays(rgb_image, ground_truth, predictions = None, model_names = None):
    """Function to plot multiple predictions vs ground truth overlayed on rgb image."""

    # Normalize RGB image if necessary
    if rgb_image.max() > 1:
        rgb_image = rgb_image / 255.0
    
    # Determine the number of prediction images to plot
    if predictions is not None:
        num_predictions = len(predictions)
    
    # Create subplots: 1 for the original image, 1 for the ground truth, and the rest for predictions
    if predictions is None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    else:
        fig, axes = plt.subplots(1, 2 + num_predictions, figsize=(20, 10))
    
    # Original RGB Image
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original RGB Image", fontsize=10)
    axes[0].axis('off')

    # Ground Truth Overlay
    gt_mask = np.zeros_like(rgb_image)
    gt_mask[..., 1] = ground_truth  # Green channel for ground truth
    gt_overlayed_image = np.where(gt_mask.any(-1, keepdims=True), gt_mask, rgb_image)
    axes[1].imshow(gt_overlayed_image)
    axes[1].set_title("Ground Truth", fontsize=10)
    axes[1].axis('off')

    # Predictions Overlay
    if predictions is not None:
        for i, prediction in enumerate(predictions, start=2):  # Start at index 2 because 0 and 1 are original and GT
            pred_mask = np.zeros_like(rgb_image)
            pred_mask[..., 0] = prediction  # Red channel for prediction
            pred_overlayed_image = np.where(pred_mask.any(-1, keepdims=True), pred_mask, rgb_image)
            axes[i].imshow(pred_overlayed_image)
            axes[i].set_title(f"Pred for {model_names[i-2]}", fontsize=10)
            axes[i].axis('off')

    plt.show()


def plot_patch_distribution_per_image(h5_files, blt_files, patch_size=(128, 128), thresholds=[]):
    """Function to plot distribution of patches above BLT percentage thresholds."""
    
    # Calculate the minimum distance between the thresholds
    if len(thresholds) > 1:
        min_distance = min(np.diff(sorted(thresholds)))
    else:
        min_distance = 1.0  # default width when there's only one threshold

    # Define the bar width as a fraction of the minimum distance between thresholds
    bar_width = min_distance * 0.8

    patches_above_threshold = np.zeros((len(thresholds), len(h5_files) + 1))
    total_patches = np.zeros(len(h5_files) + 1)

    # Load data and count patches
    for i, threshold in enumerate(thresholds):
        for j, h5_file in enumerate(h5_files):
            dataset = Smoothed_H5Dataset([h5_file], [blt_files[j]], patch_size, threshold, return_rgb_nir_separately=True)
            patches_above_threshold[i, j] = len(dataset.patch_starts)
            total_patches[j] = len(dataset.patch_starts) + dataset.patches_not_relevant

    fig, axes = plt.subplots(1, len(h5_files) + 1, figsize=(15, 5))
    fig.suptitle("Distribution above BLT percentage threshold")

    # Plot for each image
    for i, ax in enumerate(axes[:-1]):
        ax.bar(thresholds, patches_above_threshold[:, i] / total_patches[i] * 100, width=bar_width, color='blue', edgecolor='black', capsize=7)
        ax.set_title(f"Image {i+1}")
        ax.set_xlabel('Threshold for BLT area (%)')
        ax.set_ylabel('Percentage of patches')

    # Plot for total
    axes[-1].bar(thresholds, patches_above_threshold.sum(axis=1) / total_patches.sum() * 100, width=bar_width, color='blue', edgecolor='black', capsize=7)
    axes[-1].set_title("Total")
    axes[-1].set_xlabel('Threshold for BLT area (%)')
    axes[-1].set_ylabel('Percentage of patches')

    plt.tight_layout()
    plt.show()


def plot_patch_distribution(h5_files, blt_files, patch_size=(128, 128), thresholds=[], H5Dataset=None):
    """Function to plot distrubtion of relevant patches over BLT percentage thresholds."""

    relevant = []
    irrelevant = []

    for i in range(len(thresholds)):
        dataset = H5Dataset(h5_files, blt_files, patch_size, thresholds[i], return_rgb_nir_separately=True)
        relevant.append(len(dataset.patch_starts))
        irrelevant.append(dataset.patches_not_relevant)

    barWidth = 0.3
    re = np.arange(len(relevant))
    ire = [x + barWidth for x in re]

    # Create blue bars
    plt.bar(re, relevant, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='relevant patches')
    # Create cyan bars
    plt.bar(ire, irrelevant, width = barWidth, color = 'red', edgecolor = 'black', capsize=7, label='irrelevant patches')
    
    plt.xticks([r + barWidth for r in range(len(relevant))], thresholds)
    plt.ylabel('number of patches')
    plt.xlabel('threshold for BLT area')
    plt.legend(loc='center right')
    
    plt.show()