
# Detection and Classification of Fruits Using Transfer Learning and Fine-Tuning

## Introduction

This project focuses on classifying fruits using the Fruits-360 dataset by leveraging transfer learning and fine-tuning techniques. Three pretrained convolutional neural network (ConvNet) architectures—VGG16, ResNet50, and InceptionV3—were evaluated. The VGG16 model emerged as the best-performing architecture, with additional analysis using class activation map (CAM) techniques for explainability.

## Dataset

The Fruits-360 dataset was used for this project and can be accessed [here](https://www.kaggle.com/datasets/moltean/fruits/data). The dataset was preprocessed as follows:

- **Normalization**: Pixel values scaled to [0, 1].
- **Resizing**: Images resized to 100x100 pixels.
- **Label Encoding**: Converted categorical labels into numerical values.
- **Splitting**: Divided into training, validation, and test sets.

Principal Component Analysis (PCA) was applied to reduce dimensions while preserving 95% of variance, improving computational efficiency.

## Models and Training Methodology

### Pretrained Models

1. **VGG16**: Achieved the highest accuracy and F1-score after fine-tuning.
2. **ResNet50**: Demonstrated strong feature extraction but underperformed compared to VGG16.
3. **InceptionV3**: Delivered competitive performance but did not outperform VGG16.

### Fine-Tuning Steps

- Freeze all convolutional layers and replace the classifier with new fully connected layers.
- Unfreeze top convolutional layers and train with a lower learning rate.

**Implementation Details**:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Batch Size**: 64
- **Epochs**: 50 for feature extraction; 20 for fine-tuning

## Performance Evaluation

| Model      | Accuracy (Fine-Tuning) | F1-Score (Fine-Tuning) |
|------------|-------------------------|-------------------------|
| **VGG16**  | Highest                | Highest                |
| ResNet50   | Moderate               | Moderate               |
| InceptionV3| Lower                  | Lower                  |

Training and validation loss/accuracy plots showed the effectiveness of fine-tuning. Overfitting was mitigated using dropout and early stopping.

## Explainability with CAM Techniques

To enhance interpretability, the fine-tuned VGG16 model was analyzed using the following CAM techniques:

- **GradCAM**: Highlighted significant regions contributing to predictions.
- **GradCAM++**: Provided refined visual explanations.
- **ScoreCAM**: Used class-specific saliency maps.

### Visualization Examples

- **Best Class**: Correctly classified fruit with clear activation regions.
- **Worst Class**: Misclassified fruit with ambiguous activations.

## Project Structure

- **Dataset Folders**:
  - `fruits-360_dataset_100x100`: Preprocessed images (100x100 pixels).
  - `Test`, `Training`, `Validation`: Dataset splits.
- **`main.ipynb`**: Primary notebook for running training and testing workflows.

## Conclusion

This project demonstrated the effectiveness of transfer learning and fine-tuning in fruit classification. The VGG16 model excelled in both accuracy and explainability. PCA contributed to computational efficiency, while CAM techniques made predictions interpretable.

## References

1. [Fruits-360 Dataset](https://www.kaggle.com/datasets/moltean/fruits/data)
2. [Keras Pretrained Models](https://keras.io/api/applications/)
3. [Transfer Learning Implementation](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/first_edition/5.3-using-a-pretrained-convnet.ipynb)
4. [CAM Visualization](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/first_edition/5.4-visualizing-what-convnets-learn.ipynb)

---
