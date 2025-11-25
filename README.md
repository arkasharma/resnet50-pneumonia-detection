# Pneumonia Detection from Chest X-ray Images

## Project Overview
This project aims to develop and fine-tune a Convolutional Neural Network (CNN) model to classify chest X-ray images as either `NORMAL` or `PNEUMONIA`. The goal is to create an accurate and robust model for assisting in the diagnosis of pneumonia, a critical medical condition.

## Dataset
The dataset used is the "Chest X-Ray Images (Pneumonia)" dataset, which is a collection of X-ray images categorized into 'NORMAL' and 'PNEUMONIA' classes. 

**Initial Dataset Split (before preprocessing):**
- **Training Set**: 5232 images (1349 NORMAL, 3883 PNEUMONIA)
- **Validation Set**: 0 images (Originally had a small validation set that was merged into training)
- **Test Set**: 624 images (234 NORMAL, 390 PNEUMONIA)

**Key Dataset Characteristics:**
- **Class Imbalance**: The dataset is heavily imbalanced, with a significantly higher number of PNEUMONIA cases (approximately 2.88:1 ratio) compared to NORMAL cases in the training set.
- **Image Dimensions**: Images vary in dimensions (e.g., widths from 384 to 2916 pixels, heights from 127 to 2663 pixels).
- **Brightness/Contrast**: Pneumonia images show greater variability in average brightness and tend to have lower contrast, suggesting a more hazy or cloudy appearance.

## Data Preprocessing and Augmentation
To address the dataset's characteristics and prepare it for model training:
1.  **Validation Set Merging**: The original small validation set was merged into the training set to allow for proper cross-validation or a more robust split later.
2.  **Image Resizing**: All images were resized to (224, 224) to match the input requirements of the ResNet50 model.
3.  **Color Mode Conversion**: Grayscale X-ray images were converted to 3-channel RGB to align with ResNet50's expected input.
4.  **Normalization**: Pixel values were rescaled using `preprocess_input` (ImageNet normalization) to match the distribution of images ResNet50 was originally trained on.
5.  **Data Augmentation**: `ImageDataGenerator` was used for augmentation during training to improve generalization.
6.  **Class Weights**: Class weights were computed using `sklearn.utils.class_weight.compute_class_weight` to counteract the class imbalance during training.

**Final Data Splits (after preprocessing and train/validation split from augmented training data):**
-   **Training Set**: 4187 images (1080 NORMAL, 3107 PNEUMONIA)
-   **Validation Set**: 1045 images (269 NORMAL, 776 PNEUMONIA)
-   **Test Set**: 624 images (234 NORMAL, 390 PNEUMONIA)

## Model Architecture: ResNet50 Transfer Learning

### Baseline Model

1.  **Base Model**: ResNet50, pre-trained on ImageNet, was used as the feature extractor. The top (classification) layer was excluded (`include_top=False`).
2.  **Frozen Layers**: Initially, all layers of the `ResNet50` base model were frozen (`layer.trainable = False`) to leverage pre-learned features and reduce computational cost.
3.  **Custom Classification Head**: A new classification head was added on top of the frozen `ResNet50` base.
4.  **Compilation**: The model was compiled with Adam (`learning_rate=1e-4`), `binary_crossentropy` loss, and metrics `accuracy`, `Precision`, `Recall`.
5.  **Callbacks**: `ModelCheckpoint` and `EarlyStopping` were used during training.

### Baseline Evaluation Results (on Test Set)
-   **Accuracy**: 0.8974
-   **Precision**: 0.8956
-   **Recall**: 0.9462
-   **F1 Score**: 0.9202
-   **Confusion Matrix:**
    ```
    [[191  43]
     [ 21 369]]
    ```

The baseline model showed strong performance, especially in recall, indicating its effectiveness at identifying pneumonia cases. This is crucial in a medical context where false negatives are highly undesirable.

## Fine-Tuning Strategy

A multi-dimensional hyperparameter search was performed to optimize the model further. Instead of an exhaustive grid search, a curated subset of configurations was explored to balance performance and computational efficiency. Key hyperparameters tuned included:

1.  **Unfreeze Depth**: How many top layers of the `ResNet50` backbone to unfreeze and retrain.
2.  **Learning Rate**: Different learning rates for the Adam optimizer.
3.  **Optimizer**: Comparison between Adam and SGD (with momentum).
4.  **Dropout Rate**: Different dropout rates in the classification head.
5.  **L2 Regularization**: Introduction of L2 regularization in the dense layers.

## Final Model Configuration
Based on the fine-tuning experiments, the optimal configuration for the final model was:
-   **Unfreeze Layers**: Last 10 layers of ResNet50.
-   **Learning Rate**: `1e-4`.
-   **Optimizer**: Adam.
-   **Dropout**: `0.3`.
-   **L2 Regularization**: `0.01` in the dense layers.

## Final Model Evaluation Results (on Test Set)
-   **Accuracy**: 0.9103
-   **Precision**: 0.9113
-   **Recall**: 0.9487
-   **F1 Score**: 0.9296
-   **Confusion Matrix:**
    ```
    [[198  36]
     [ 20 370]]
    ```

## How to Run the Project
1.  **Clone the Repository**:
    ```bash
    git clone resnet50-pneumonia-detection
    cd resnet50-pneumonia-detection
    ```
2.  **Setup Environment**:
    Ensure you have Python installed. It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Mount Google Drive (if using Colab)**:
    The notebook assumes data and model checkpoints are stored in Google Drive. Ensure your `BASE_DIR` points to the correct location of your `chest_xray` dataset on Google Drive.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive/Math_156_Project/chest_xray'
    ```
4.  **Run the Notebook**: Open the `.ipynb` file in Google Colab or Jupyter Notebook and run all cells sequentially or run the `.py` file through the terminal.

## Dependencies
-   `numpy`
-   `matplotlib`
-   `pillow`
-   `seaborn`
-   `scikit-learn`
-   `tensorflow`
-   `tensorflow-addons`
-   `tqdm`
-   `pandas`
-   `opencv-python`