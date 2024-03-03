import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

# Function to train CNN and DNN models
def train_model(model, optimizer, criterion, train_loader, device, NUM_CLASSES):
    """Function to train CNN"""

    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Reshape images and labels for CNN
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).squeeze(1).long()
        
        # Train model
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    return running_loss


def validate_model(model, criterion, val_loader, device, NUM_CLASSES, class_labels):
    """Function to validate CNN"""

    model.eval()
    running_val_loss = 0.0
    running_val_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    with torch.no_grad():
        for images, labels in val_loader:
            # Reshape images and labels for CNN
            images, labels = images.to(device), labels.to(device).squeeze(1).long()

            # Get predictions and labels
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            # Get predictions and labels
            predicted = torch.argmax(outputs, dim=1).flatten()
            labels = labels.flatten()

            # Update running confusion matrix
            cm =  confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), 
                                   labels=class_labels)
            running_val_confusion_matrix += cm

    return running_val_loss, running_val_confusion_matrix


def validate_model_simple(model, criterion, val_loader, device):
    """Function to validate CNN without calculating the confusion matrix"""

    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).squeeze(1).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

    return running_val_loss


def calculate_metrics(conf_matrix):
    precision = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=0) + 1e-5)
    recall = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=1) + 1e-5)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-5)

    return precision, recall, f1_score


def calculate_reduced_confusion_matrix(conf_matrix, exclude_classes):
    # Convert to numpy array if it's not already
    conf_matrix = np.array(conf_matrix)

    # Exclude rows and columns corresponding to the classes to ignore
    reduced_conf_matrix = np.delete(conf_matrix, exclude_classes, axis=0)  # Remove rows
    reduced_conf_matrix = np.delete(reduced_conf_matrix, exclude_classes, axis=1)  # Remove columns

    return reduced_conf_matrix


def train_score_fusion(fusion_model, cnn_model, dataloader, criterion, optimizer, device, time_steps=3):
    fusion_model.train()
    cnn_model.eval()  # Make sure the existing model is in eval mode
    running_loss = 0.0
    if time_steps == 3:
        for image_2, image_1, image_0, labels in dataloader:
            # Move images and labels to device
            image_2 = image_2.float().to(device)
            image_1 = image_1.float().to(device)
            image_0 = image_0.float().to(device)
            labels = labels.to(device).squeeze(1).long()

            sequence = []
            with torch.no_grad():
                # Generate predictions from the existing model
                predictions_2 = cnn_model.module(image_2, return_intermediate=True)
                predictions_1 = cnn_model.module(image_1, return_intermediate=True)
                predictions_0 = cnn_model.module(image_0, return_intermediate=True)

                # Append predictions to sequence
                sequence.append(predictions_2)
                sequence.append(predictions_1)
                sequence.append(predictions_0)

            # Concatenate predictions as sequencial input for fusion model
            predictions = torch.stack(sequence, dim=1)

            # Train the fusion network
            optimizer.zero_grad()
            output = fusion_model(predictions)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    if time_steps == 5:
        for image_4, image_3, image_2, image_1, image_0, labels in dataloader:
            # Move images and labels to device
            image_4 = image_4.to(device).float()
            image_3 = image_3.to(device).float()
            image_2 = image_2.to(device).float()
            image_1 = image_1.to(device).float()
            image_0 = image_0.to(device).float()
            labels = labels.to(device).squeeze(1).long()

            sequence = []
            with torch.no_grad():
                # Generate predictions from the existing model
                predictions_4 = cnn_model.module(image_4, return_intermediate=True)
                predictions_3 = cnn_model.module(image_3, return_intermediate=True)
                predictions_2 = cnn_model.module(image_2, return_intermediate=True)
                predictions_1 = cnn_model.module(image_1, return_intermediate=True)
                predictions_0 = cnn_model.module(image_0, return_intermediate=True)

                # Append predictions to sequence
                sequence.append(predictions_4)
                sequence.append(predictions_3)
                sequence.append(predictions_2)
                sequence.append(predictions_1)
                sequence.append(predictions_0)
                
            # Concatenate predictions as sequencial input for fusion model
            predictions = torch.stack(sequence, dim=1)

            # Train the fusion network
            optimizer.zero_grad()
            output = fusion_model(predictions)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    return running_loss


def validate_score_fusion_simple(fusion_model, cnn_model, dataloader, criterion, device, time_steps=3):
    fusion_model.eval()
    cnn_model.eval()
    running_loss = 0.0

    if time_steps == 3:
        with torch.no_grad():
            for image_2, image_1, image_0, labels in dataloader:

                # Move images and labels to device
                image_2 = image_2.float().to(device)
                image_1 = image_1.float().to(device)
                image_0 = image_0.float().to(device)
                labels = labels.to(device).squeeze(1).long()

                sequence = []
                # Generate predictions from the existing model
                predictions_2 = cnn_model.module(image_2, return_intermediate=True)
                predictions_1 = cnn_model.module(image_1, return_intermediate=True)
                predictions_0 = cnn_model.module(image_0, return_intermediate=True)

                # Append predictions to sequence
                sequence.append(predictions_2)
                sequence.append(predictions_1)
                sequence.append(predictions_0)

                # Concatenate predictions as sequencial input for fusion model
                predictions = torch.stack(sequence, dim=1)

                # Train the fusion network
                output = fusion_model(predictions)
                loss = criterion(output, labels)
                running_loss += loss.item()

    if time_steps == 5:
        with torch.no_grad():
            for image_4, image_3, image_2, image_1, image_0, labels in dataloader:
                # Move images and labels to device
                image_4 = image_4.to(device).float()
                image_3 = image_3.to(device).float()
                image_2 = image_2.to(device).float()
                image_1 = image_1.to(device).float()
                image_0 = image_0.to(device).float()
                labels = labels.to(device).squeeze(1).long()

                sequence = []
                # Generate predictions from the existing model
                predictions_4 = cnn_model.module(image_4, return_intermediate=True)
                predictions_3 = cnn_model.module(image_3, return_intermediate=True)
                predictions_2 = cnn_model.module(image_2, return_intermediate=True)
                predictions_1 = cnn_model.module(image_1, return_intermediate=True)
                predictions_0 = cnn_model.module(image_0, return_intermediate=True)

                # Append predictions to sequence
                sequence.append(predictions_4)
                sequence.append(predictions_3)
                sequence.append(predictions_2)
                sequence.append(predictions_1)
                sequence.append(predictions_0)

                # Concatenate predictions as sequencial input for fusion model
                predictions = torch.stack(sequence, dim=1)

                # Train the fusion network
                output = fusion_model(predictions)
                loss = criterion(output, labels)
                running_loss += loss.item()
    
    return running_loss


def validate_score_fusion(fusion_model, cnn_model, dataloader, criterion, device, timesteps, NUM_CLASSES, class_labels):

    fusion_model.eval()
    cnn_model.eval()
    running_loss = 0.0
    running_confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    if timesteps == 3:
        with torch.no_grad():
            for image_2, image_1, image_0, labels in dataloader:
                # Move images and labels to device
                image_2 = image_2.float().to(device)
                image_1 = image_1.float().to(device)
                image_0 = image_0.float().to(device)
                labels = labels.to(device).squeeze(1).long()

                sequence = []
                # Generate predictions from the existing model
                predictions_2 = cnn_model.module(image_2, return_intermediate=True)
                predictions_1 = cnn_model.module(image_1, return_intermediate=True)
                predictions_0 = cnn_model.module(image_0, return_intermediate=True)

                # Append predictions to sequence
                sequence.append(predictions_2)
                sequence.append(predictions_1)
                sequence.append(predictions_0)

                # Concatenate predictions as sequencial input for fusion model
                predictions = torch.stack(sequence, dim=1)
                
                # Train the fusion network
                output = fusion_model(predictions)
                loss = criterion(output, labels)
                running_loss += loss.item()

                # Get predictions and labels
                predicted = torch.argmax(output, dim=1).flatten()
                labels = labels.flatten()

                # Update running confusion matrix
                cm =  confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), 
                                    labels=class_labels)
                running_confusion_matrix += cm
    
    if timesteps == 5:
        with torch.no_grad():
            for image_4, image_3, image_2, image_1, image_0, labels in dataloader:
                # Move images and labels to device
                image_4 = image_4.to(device).float()
                image_3 = image_3.to(device).float()
                image_2 = image_2.to(device).float()
                image_1 = image_1.to(device).float()
                image_0 = image_0.to(device).float()
                labels = labels.to(device).squeeze(1).long()

                sequence = []
                # Generate predictions from the existing model
                predictions_4 = cnn_model.module(image_4, return_intermediate=True)
                predictions_3 = cnn_model.module(image_3, return_intermediate=True)
                predictions_2 = cnn_model.module(image_2, return_intermediate=True)
                predictions_1 = cnn_model.module(image_1, return_intermediate=True)
                predictions_0 = cnn_model.module(image_0, return_intermediate=True)

                # Append predictions to sequence
                sequence.append(predictions_4)
                sequence.append(predictions_3)
                sequence.append(predictions_2)
                sequence.append(predictions_1)
                sequence.append(predictions_0)

                # Concatenate predictions as sequencial input for fusion model
                predictions = torch.stack(sequence, dim=1)

                # Train the fusion network
                output = fusion_model(predictions)
                loss = criterion(output, labels)
                running_loss += loss.item()

                # Get predictions and labels
                predicted = torch.argmax(output, dim=1).flatten()
                labels = labels.flatten()

                # Update running confusion matrix
                cm =  confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), 
                                    labels=class_labels)
                running_confusion_matrix += cm

    return running_loss, running_confusion_matrix


def loadfusionmodel(model, model_path, device, NUM_CLASSES):
    """Function to load a model from a file."""
    model = model(input_channels=32, output_size=NUM_CLASSES, hidden_size=64)
    model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    return model


def predict_on_patch_score(fusion_model, cnn_model, dataset, idx, device, timesteps): 
    """Function to predict on a single patch."""
    fusion_model.eval()
    cnn_model.eval()
    pred = None
    with torch.no_grad():
        if timesteps == 3:
            patch2, patch1, patch0 = dataset[idx][0].unsqueeze(0), dataset[idx][1].unsqueeze(0), dataset[idx][2].unsqueeze(0)
            patch2, patch1, patch0 = patch2.to(device).float(), patch1.to(device).float(), patch0.to(device).float()

            sequence = []
            # Generate predictions from the existing model
            predictions_2 = cnn_model.module(patch2, return_intermediate=True)
            predictions_1 = cnn_model.module(patch1, return_intermediate=True)
            predictions_0 = cnn_model.module(patch0, return_intermediate=True)

            # Append predictions to sequence
            sequence.append(predictions_2)
            sequence.append(predictions_1)
            sequence.append(predictions_0)
            
            # Concatenate predictions as sequencial input for fusion model
            predictions = torch.stack(sequence, dim=1)

            # Predict on the fusion model
            output = fusion_model(predictions)
            output = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            return pred
        
        if timesteps == 5:
            patch4, patch3, patch2, patch1, patch0 = dataset[idx][0].unsqueeze(0), dataset[idx][1].unsqueeze(0), dataset[idx][2].unsqueeze(0), dataset[idx][3].unsqueeze(0), dataset[idx][4].unsqueeze(0)
            patch4, patch3, patch2, patch1, patch0 = patch4.to(device).float(), patch3.to(device).float(), patch2.to(device).float(), patch1.to(device).float(), patch0.to(device).float()

            sequence = []
            with torch.no_grad():
                # Generate predictions from the existing model
                predictions_4 = cnn_model.module(patch4, return_intermediate=True)
                predictions_3 = cnn_model.module(patch3, return_intermediate=True)
                predictions_2 = cnn_model.module(patch2, return_intermediate=True)
                predictions_1 = cnn_model.module(patch1, return_intermediate=True)
                predictions_0 = cnn_model.module(patch0, return_intermediate=True)

                # Append predictions to sequence
                sequence.append(predictions_4)
                sequence.append(predictions_3)
                sequence.append(predictions_2)
                sequence.append(predictions_1)
                sequence.append(predictions_0)
            
                # Concatenate predictions as sequencial input for fusion model
                predictions = torch.stack(sequence, dim=1)

                # Predict on the fusion model
                output = fusion_model(predictions)
                output = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                return pred
            

