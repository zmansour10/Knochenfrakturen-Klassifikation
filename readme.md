# Knochenfrakturen-Klassifikation (Bone Fracture Classification)

This project was created for Milestone 1 in the Deep Learning course. The goal is to build and compare different custom CNN architectures for classifying bone fractures in X-ray images.

The dataset contains 10 fracture classes. Before training, the images were checked, cleaned, split into train/validation/test sets, and preprocessed into a consistent format. After that, several CNN models were trained with Keras/TensorFlow and compared using test accuracy, macro F1-score, and confusion matrices.

trices.

## Project Structure

```text
project/
├── dataset/                    # original dataset
├── dataset_processed/          # preprocessed RGB images
├── dataset_processed_gray/     # grayscale-standardized images

├── scripts/                    # preprocessing, training, and evaluation scripts
└── results/                    # saved models, metrics, plots, and CSV files
````

## What is included?

* data checking and cleaning
* grayscale-standardized dataset
* training of multiple custom CNN architectures
* comparison using accuracy, macro F1-score, and confusion matrix

## Main Result

The first CNN models with Global Average Pooling were stable, but their accuracy was low. The best compact model was `CNN-06`, which used grayscale images and stronger dropout.

The best experiment was `CNN-12_StridedConv_Flatten_BN_LeakyReLU`. This model achieved the highest test accuracy, but it also showed clear overfitting.

