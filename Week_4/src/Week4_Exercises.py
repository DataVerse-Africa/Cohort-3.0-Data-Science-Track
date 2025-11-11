"""
Week 4 Exercises: Deep Learning & Computer Vision
DataVerse Africa Internship Cohort 3.0

Practice exercises to reinforce learning concepts from the main notebook.
Complete these exercises to deepen your understanding of computer vision and neural networks.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# EXERCISE 1: Image Preprocessing Techniques
# ============================================================================

def exercise_1_image_preprocessing():
    """
    Exercise 1: Implement and compare different image preprocessing techniques
    
    Tasks:
    1. Load an image from the grapevine dataset
    2. Apply different preprocessing techniques
    3. Compare the effects visually
    4. Discuss which techniques might be most useful for our classification task
    """
    
    print("EXERCISE 1: Image Preprocessing Techniques")
    print("=" * 50)
    
    # TODO: Load a sample image from the dataset
    # dataset_path = Path('../Grapevine_Leaves_Image_Dataset')
    # sample_image_path = dataset_path / 'Ak' / 'Ak (1).png'
    
    # TODO: Implement the following preprocessing techniques:
    # 1. Resize to different dimensions (128x128, 256x256, 512x512)
    # 2. Convert to grayscale
    # 3. Apply Gaussian blur with different kernel sizes
    # 4. Apply histogram equalization
    # 5. Apply edge detection (Canny)
    # 6. Apply morphological operations (erosion, dilation)
    
    # TODO: Create a visualization showing original vs processed images
    
    # TODO: Write a function that applies multiple preprocessing steps in sequence
    
    pass  # Remove this when you implement the exercise


def exercise_1_solution():
    """
    Solution for Exercise 1
    """
    dataset_path = Path('../Grapevine_Leaves_Image_Dataset')
    sample_image_path = dataset_path / 'Ak' / 'Ak (1).png'
    
    # Load image
    img = cv2.imread(str(sample_image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply different preprocessing techniques
    img_resized_128 = cv2.resize(img_rgb, (128, 128))
    img_resized_256 = cv2.resize(img_rgb, (256, 256))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_rgb, (15, 15), 0)
    img_eq = cv2.equalizeHist(img_gray)
    edges = cv2.Canny(img_gray, 100, 200)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img_gray, kernel, iterations=1)
    img_dilation = cv2.dilate(img_gray, kernel, iterations=1)
    
    # Visualize results
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_resized_128)
    axes[0, 1].set_title('Resized 128x128')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_resized_256)
    axes[0, 2].set_title('Resized 256x256')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_gray, cmap='gray')
    axes[1, 0].set_title('Grayscale')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_blur)
    axes[1, 1].set_title('Gaussian Blur')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img_eq, cmap='gray')
    axes[1, 2].set_title('Histogram Equalization')
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(edges, cmap='gray')
    axes[2, 0].set_title('Edge Detection')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(img_erosion, cmap='gray')
    axes[2, 1].set_title('Erosion')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(img_dilation, cmap='gray')
    axes[2, 2].set_title('Dilation')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXERCISE 2: CNN Architecture Experimentation
# ============================================================================

def exercise_2_cnn_architectures():
    """
    Exercise 2: Design and compare different CNN architectures
    
    Tasks:
    1. Create 3 different CNN architectures with varying complexity
    2. Train each model on a subset of the data
    3. Compare their performance and training time
    4. Analyze the trade-offs between model complexity and performance
    """
    
    print("EXERCISE 2: CNN Architecture Experimentation")
    print("=" * 50)
    
    # TODO: Define three different CNN architectures:
    # 1. Simple CNN (2-3 conv layers)
    # 2. Medium CNN (4-5 conv layers)
    # 3. Deep CNN (6+ conv layers)
    
    # TODO: For each architecture, consider:
    # - Number of filters in each layer
    # - Filter sizes (3x3, 5x5, etc.)
    # - Pooling strategies
    # - Dropout rates
    # - Dense layer sizes
    
    # TODO: Train each model and compare:
    # - Training time
    # - Final accuracy
    # - Overfitting behavior
    # - Number of parameters
    
    pass  # Remove this when you implement the exercise


def create_simple_cnn(input_shape, num_classes):
    """Simple CNN with 2 conv layers"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_medium_cnn(input_shape, num_classes):
    """Medium CNN with 4 conv layers"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def create_deep_cnn(input_shape, num_classes):
    """Deep CNN with 6 conv layers"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# ============================================================================
# EXERCISE 3: Data Augmentation Impact Analysis
# ============================================================================

def exercise_3_data_augmentation():
    """
    Exercise 3: Analyze the impact of different data augmentation strategies
    
    Tasks:
    1. Train models with different augmentation strategies
    2. Compare performance with and without augmentation
    3. Analyze which augmentation techniques are most effective
    4. Visualize the augmented images to understand the transformations
    """
    
    print("EXERCISE 3: Data Augmentation Impact Analysis")
    print("=" * 50)
    
    # TODO: Create different ImageDataGenerator configurations:
    # 1. No augmentation
    # 2. Light augmentation (small rotations, shifts)
    # 3. Heavy augmentation (large rotations, flips, zooms)
    # 4. Custom augmentation (your choice of parameters)
    
    # TODO: Train the same model architecture with each augmentation strategy
    
    # TODO: Compare the results and analyze:
    # - Which strategy prevents overfitting best?
    # - Which strategy achieves highest validation accuracy?
    # - How do the augmented images look?
    
    pass  # Remove this when you implement the exercise


# ============================================================================
# EXERCISE 4: Transfer Learning Comparison
# ============================================================================

def exercise_4_transfer_learning():
    """
    Exercise 4: Compare different pre-trained models for transfer learning
    
    Tasks:
    1. Try different pre-trained models (VGG16, ResNet50, MobileNetV2, EfficientNet)
    2. Compare their performance on our dataset
    3. Analyze the trade-offs between model size and accuracy
    4. Experiment with fine-tuning vs feature extraction
    """
    
    print("EXERCISE 4: Transfer Learning Comparison")
    print("=" * 50)
    
    # TODO: Load different pre-trained models:
    # - VGG16
    # - ResNet50
    # - MobileNetV2
    # - EfficientNetB0 (if available)
    
    # TODO: For each model, try:
    # 1. Feature extraction (freeze base model)
    # 2. Fine-tuning (unfreeze some layers)
    
    # TODO: Compare:
    # - Model size (number of parameters)
    # - Training time
    # - Final accuracy
    # - Inference speed
    
    pass  # Remove this when you implement the exercise


# ============================================================================
# EXERCISE 5: Model Interpretability
# ============================================================================

def exercise_5_model_interpretability():
    """
    Exercise 5: Understand what your model is learning
    
    Tasks:
    1. Visualize convolutional layer activations
    2. Create class activation maps (CAM)
    3. Analyze which features the model focuses on
    4. Identify potential biases or issues in the model
    """
    
    print("EXERCISE 5: Model Interpretability")
    print("=" * 50)
    
    # TODO: Implement visualization functions for:
    # 1. Filter visualizations
    # 2. Feature maps from different layers
    # 3. Grad-CAM for understanding model decisions
    # 4. Confusion matrix analysis
    
    # TODO: Apply these techniques to understand:
    # - What patterns the model recognizes
    # - Which parts of the image are most important
    # - Where the model makes mistakes and why
    
    pass  # Remove this when you implement the exercise


def visualize_feature_maps(model, image, layer_names):
    """
    Visualize feature maps from specified layers
    """
    # Create a model that outputs feature maps
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations
    activations = activation_model.predict(np.expand_dims(image, axis=0))
    
    # Plot feature maps
    for layer_name, activation in zip(layer_names, activations):
        n_features = activation.shape[-1]
        size = activation.shape[1]
        
        # Display first 16 feature maps
        n_cols = 4
        n_rows = min(4, n_features // n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
        
        for i in range(n_rows * n_cols):
            if i < n_features:
                ax = axes[i // n_cols, i % n_cols] if n_rows > 1 else axes[i % n_cols]
                ax.imshow(activation[0, :, :, i], cmap='viridis')
                ax.set_title(f'Feature {i}')
                ax.axis('off')
        
        plt.suptitle(f'Feature Maps from {layer_name}')
        plt.tight_layout()
        plt.show()


# ============================================================================
# EXERCISE 6: Custom Loss Functions and Metrics
# ============================================================================

def exercise_6_custom_metrics():
    """
    Exercise 6: Implement custom loss functions and evaluation metrics
    
    Tasks:
    1. Implement focal loss for handling class imbalance
    2. Create custom metrics for multi-class classification
    3. Compare different loss functions on the same model
    4. Analyze when custom losses might be beneficial
    """
    
    print("EXERCISE 6: Custom Loss Functions and Metrics")
    print("=" * 50)
    
    # TODO: Implement:
    # 1. Focal loss function
    # 2. Top-k accuracy metric
    # 3. Per-class precision and recall
    # 4. Balanced accuracy for imbalanced datasets
    
    # TODO: Train models with different loss functions and compare results
    
    pass  # Remove this when you implement the exercise


def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss implementation for handling class imbalance
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
        
        # Calculate focal loss
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = -alpha_t * tf.pow((1 - p_t), gamma) * tf.log(p_t)
        
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    
    return focal_loss_fixed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Week 4 Exercises: Deep Learning & Computer Vision")
    print("=" * 60)
    print()
    print("Available exercises:")
    print("1. Image Preprocessing Techniques")
    print("2. CNN Architecture Experimentation")
    print("3. Data Augmentation Impact Analysis")
    print("4. Transfer Learning Comparison")
    print("5. Model Interpretability")
    print("6. Custom Loss Functions and Metrics")
    print()
    print("To run an exercise, call the corresponding function:")
    print("e.g., exercise_1_image_preprocessing()")
    print()
    print("Solutions are available for Exercise 1.")
    print("For other exercises, implement the TODO sections.")
    print()
    print("Remember to:")
    print("- Document your findings")
    print("- Compare different approaches")
    print("- Analyze the trade-offs")
    print("- Prepare insights for your presentation")