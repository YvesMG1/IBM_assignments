import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import numpy as np
from cust_functions.processing_helper import H5Dataset


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
    axes[0][1].plot(epochs, val_results["val_recall"], 'r-', label='Validation Recall')
    axes[0][1].set_xlabel('Epochs')
    axes[0][1].set_ylabel('Recall')
    axes[0][1].legend()

    # Plot training and validation precision on the left column
    axes[1][0].plot(epochs, val_results["val_precision"], 'r-', label='Validation Precision')
    axes[1][0].set_xlabel('Epochs')
    axes[1][0].set_ylabel('Precision')
    axes[1][0].legend()

    # Plot training and validation f1 score on the right column
    axes[1][1].plot(epochs, val_results["val_f1_score"], 'r-', label='Validation F1 Score')
    axes[1][1].set_xlabel('Epochs')
    axes[1][1].set_ylabel('F1 Score')
    axes[1][1].legend()

    plt.tight_layout()
    plt.show()


def plot_loss(train_results, val_results):
    """Function to plot training and validation loss."""
    fig, axes = plt.subplots(1, 1, figsize=(15, 10))
    epochs = range(1, len(train_results["train_loss"]) + 1)

    # Plot training and validation loss
    axes.plot(epochs, np.log(train_results["train_loss"]), 'b-', label='Train Loss')
    axes.plot(epochs, np.log(val_results["val_loss"]), 'r-', label='Validation Loss')
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Loss (log scale)')
    axes.legend()
    plt.show()


def plot_conf_matrix_multiclass(cm, class_labels=None):
    """Function to plot confusion matrix and normalized confusion matrix for multiclass."""

    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    axes[0].matshow(cm, cmap=plt.cm.Blues, alpha=0.9)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='small')
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_xticks(np.arange(cm.shape[1]), labels=class_labels)
    axes[0].set_yticks(np.arange(cm.shape[0]), labels=class_labels)

    # Normalise confusion matrix
    axes[1].matshow(cmn, cmap=plt.cm.Blues, alpha=0.9)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(x=j, y=i, s=round(cmn[i, j], 2), va='center', ha='center', size='large')
    axes[1].set_title("Normalised Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_xticks(np.arange(cm.shape[1]), labels=class_labels)
    axes[1].set_yticks(np.arange(cm.shape[0]), labels=class_labels)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    plt.tight_layout()
    plt.show()


def plot_dw_with_overlays(rgb_image, ground_truth, NUM_CLASSES, predictions = None, model_names = None):
    """Function to plot multiple predictions vs ground truth overlayed on rgb image."""

    # Normalize RGB image if necessary
    if rgb_image.max() > 1:
        rgb_image = rgb_image / 255.0
        print('normalizing rgb image')
    
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

    cmap = ListedColormap(["black", "darkgreen", "springgreen", "lightseagreen", 
                               "darkorange", "olive", "red", "tan", "cyan", "blue", "white"])
    # Define a normalization from values -> colors
    norm = colors.BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], 11)
    # Add a legend for labels
    legend_labels = {"black": "00 nodata",
                         "darkgreen": "01 trees",
                         "springgreen": "02 grass", 
                         "lightseagreen": "03 flooded veg.", 
                         "darkorange": "04 crops",
                         "olive": "05 shrub & scrub",
                         "red": "06 built-up", 
                         "tan": "07 bare",
                         "cyan": "08 snow & ice",
                         "blue": "09 water",
                         "white": "10 clouds"}

    patches = [Patch(color=color, label=label)
            for color, label in legend_labels.items()]
    # Ground Truth Overlay
    axes[1].imshow(ground_truth,cmap=cmap,norm=norm, interpolation='none')
    axes[1].set_title("Ground Truth", fontsize=10)
    axes[1].axis('off')
    axes[1].legend(handles=patches)

    # Predictions Overlay
    if predictions is not None:
        for i, prediction in enumerate(predictions, start=2):  # Start at index 2 because 0 and 1 are original and GT
            axes[i].imshow(prediction,cmap=cmap,norm=norm, interpolation='none')
            axes[i].set_title(f"Pred for {model_names[i-2]}", fontsize=10)
            axes[i].axis('off')

    plt.show()