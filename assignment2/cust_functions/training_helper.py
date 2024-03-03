import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def train_model(model, optimizer, criterion, train_loader, device, NUM_CLASSES, model_type = 'CNN'):
    """Function to train CNN and DNN models."""

    model.train()
    running_loss = 0.0
    running_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    
    for images, labels in train_loader:

        # Reshape images and labels for DNN and CNN
        if model_type == 'CNN':
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).squeeze(1).long()
        
        elif model_type == 'DNN':
            b, c, h, w = images.shape
            images = images.view(b*h*w, c)
            labels = labels.view(b*h*w)
            images, labels = images.to(device), labels.to(device)
        
        # Train model
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update running loss
        #running_loss += loss
        running_loss += loss.item()
        
        # Get predictions and labels
        if model_type == 'CNN':
            predicted = torch.argmax(outputs, dim=1).flatten()
            labels = labels.flatten()
        elif model_type == 'DNN':
            predicted = torch.argmax(outputs.data, 1)

        # Update running confusion matrix
        cm =  confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
        running_confusion_matrix += cm
        
    return running_loss, running_confusion_matrix



def validate_model(model, criterion, val_loader, device, NUM_CLASSES,  model_type = 'CNN'):
    """Function to validate CNN and DNN models."""

    model.eval()
    running_val_loss = 0.0
    running_val_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    with torch.no_grad():
        for images, labels in val_loader:

            # Reshape images and labels for DNN and CNN
            if model_type == 'CNN':
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze(1).long()
            elif model_type == 'DNN':
                b, c, h, w = images.shape
                images = images.view(b*h*w, c)
                labels = labels.view(b*h*w)
                images, labels = images.to(device), labels.to(device)

            # Get predictions and labels
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            if model_type == 'CNN':
                predicted = torch.argmax(outputs, dim=1).flatten()
                labels = labels.flatten()
            elif model_type == 'DNN':
                predicted = torch.argmax(outputs.data, 1)

            # Update running confusion matrix
            cm =  confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
            running_val_confusion_matrix += cm

    return running_val_loss, running_val_confusion_matrix


def load_model(model, model_path, device, NUM_CHANNELS=4, NUM_CLASSES=2):
    """Function to load a model from a file."""
    model = model(NUM_CHANNELS, NUM_CLASSES)
    model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    return model


def predict_on_patch(model, dataset, idx, device):
    """Function to predict on a single patch."""
    patch = dataset[idx][0].unsqueeze(0)
    patch = patch.to(device)
    pred = model(patch)
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    return pred


def merge_predictions(model, device, image_height, image_width, custom_dataset, patch_size):
    """Function to load predicted patches and generate overall image."""
    # create array to store all prediction patches
    BLT_pred = np.full((image_height, image_width), np.nan, dtype=np.float32)
    print(BLT_pred.shape)

    # fetch all predictions and append to big final image
    for m in range(len(custom_dataset.patch_starts)):

        # assess pixel indices
        p = custom_dataset.patch_starts[m]
        i, j = p[1], p[2]

        # prediction
        sample_patch = custom_dataset[m][0].unsqueeze(0)  # fetching patch
        sample_patch = sample_patch.to(device)
        sample_pred = model(sample_patch)  # running predictions: vector assigning scores to each class
        sample_pred = torch.argmax(sample_pred, dim=1).squeeze(0).cpu().numpy()  # max score for main class in np array

        # Calculate the end positions, taking care of not going out of the image boundaries
        end_i = min(i + patch_size[0], image_height)
        end_j = min(j + patch_size[1], image_width)

        # Determine the size of the prediction to use
        use_pred_h = end_i - i
        use_pred_w = end_j - j

        # Cut down the patch if it's larger than the expected size
        trimmed_pred = sample_pred[:use_pred_h, :use_pred_w]

        # merge to overall image
        BLT_pred[i:end_i, j:end_j] = trimmed_pred

    return BLT_pred


def testing_model(model, val_loader, device, NUM_CLASSES,  model_type = 'CNN'):
    """Function to validate CNN and DNN models."""
    model.eval()
    running_val_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    predicted_list=[]

    with torch.no_grad():
        for images, labels in val_loader:

            # Reshape images and labels for DNN and CNN
            if model_type == 'CNN':
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze(1).long()
            elif model_type == 'DNN':
                b, c, h, w = images.shape
                images = images.view(b*h*w, c)
                labels = labels.view(b*h*w)
                images, labels = images.to(device), labels.to(device)

            # Get predictions and labels
            outputs = model(images)
            
            if model_type == 'CNN':
                predicted = torch.argmax(outputs, dim=1).flatten()
                labels = labels.flatten()
            elif model_type == 'DNN':
                predicted = torch.argmax(outputs.data, 1)
                predicted_list.append(torch.argmax(outputs.data, 1).cpu().numpy())


            # Update running confusion matrix
            cm =  confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
            running_val_confusion_matrix += cm

    return  running_val_confusion_matrix, predicted_list 
