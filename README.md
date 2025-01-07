
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
![image](https://github.com/user-attachments/assets/f4a1e6dd-bcb3-4a76-b8e8-f85ee1a7245c)
![image](https://github.com/user-attachments/assets/3afb790c-1352-4904-887d-1e3f101230e9)
![image](https://github.com/user-attachments/assets/137ba70d-1b9b-4735-b84d-d74eddfb7fe5)


## Models and Training Methodology
### Trained Models
![image](https://github.com/user-attachments/assets/08114025-1e28-4995-b20f-3c742e17042a)

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

### No Fine-Tuning Models
![image](https://github.com/user-attachments/assets/24577371-fb8b-4304-bffd-7623de3de0d1)
![image](https://github.com/user-attachments/assets/f22eef84-09d3-472c-9083-90c67af0d711)

### Fine-Tuning Models
![image](https://github.com/user-attachments/assets/af05206d-56f2-458d-bac4-f60b72cea906)
![image](https://github.com/user-attachments/assets/3d999565-ecbd-4a91-8eee-024ecff563bc)

Training and validation loss/accuracy plots showed the effectiveness of fine-tuning. Overfitting was mitigated using dropout and early stopping.

## Explainability with CAM Techniques

To enhance interpretability, the fine-tuned VGG16 model was analyzed using the following CAM techniques:

- **GradCAM**: Highlighted significant regions contributing to predictions.
- **GradCAM++**: Provided refined visual explanations.
- **ScoreCAM**: Used class-specific saliency maps.

### Visualization Examples

- **Best Class**: Correctly classified fruit with clear activation regions.(**VGG16 Fine-Tuning Model**)
![image](https://github.com/user-attachments/assets/ff014b45-9fc9-49cc-b8a8-6977dd38ca39)
![image](https://github.com/user-attachments/assets/6ef15a97-a497-4669-b453-0abf3c96dbc1)

- **Worst Class**: Misclassified fruit with ambiguous activations.(**ResNet50 Fine-Tuning Model**)
![image](https://github.com/user-attachments/assets/9eceb87c-82d4-4a82-8861-a01454deb62e)
![image](https://github.com/user-attachments/assets/db8035f6-ee45-4474-81e4-518aa525526d)

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
