import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# WEEK 5 - DEEP LEARNING APPLICATION IN DATA SCIENCE
# Fashion-MNIST Image Classification using CNN
# ============================================================

print("=" * 75)
print("WEEK 5 - DEEP LEARNING APPLICATION IN DATA SCIENCE")
print("FASHION-MNIST IMAGE CLASSIFICATION USING CNN")
print("=" * 75)


# ============================================================
# 1. CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("figures", exist_ok=True)
os.makedirs("report", exist_ok=True)


# ============================================================
# 2. BASIC SETTINGS
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS = 10
BATCH_SIZE = 64

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


print("\nTensorFlow Version:", tf.__version__)
print("Number of Epochs:", EPOCHS)
print("Batch Size:", BATCH_SIZE)


# ============================================================
# 3. LOAD FASHION-MNIST DATASET
# ============================================================

print("\n" + "=" * 75)
print("LOADING FASHION-MNIST DATASET")
print("=" * 75)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("\nDataset loaded successfully.")

print("\nTraining data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)

print("Testing data shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)

print("\nNumber of training images:", len(X_train))
print("Number of testing images:", len(X_test))

print("Image dimensions:", X_train.shape[1], "x", X_train.shape[2])

print("Number of classes:", len(class_names))


# ============================================================
# 4. DISPLAY SAMPLE IMAGES
# ============================================================

print("\nCreating sample image visualization...")

plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.suptitle("Sample Images from Fashion-MNIST Dataset")
plt.tight_layout()

plt.savefig(
    "figures/sample_fashion_mnist_images.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: figures/sample_fashion_mnist_images.png")


# ============================================================
# 5. CLASS DISTRIBUTION
# ============================================================

print("\nCreating class distribution graph...")

class_counts = np.bincount(y_train, minlength=10)

class_distribution = pd.DataFrame({
    "Class_ID": range(10),
    "Class_Name": class_names,
    "Number_of_Images": class_counts
})

print("\nClass Distribution:")
print(class_distribution.to_string(index=False))

class_distribution.to_csv(
    "report/class_distribution.csv",
    index=False
)

plt.figure(figsize=(12, 6))

plt.bar(class_names, class_counts)

plt.title("Class Distribution in Fashion-MNIST Training Dataset")
plt.xlabel("Fashion Category")
plt.ylabel("Number of Images")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    "figures/class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: figures/class_distribution.png")


# ============================================================
# 6. PREPROCESSING
# ============================================================

print("\n" + "=" * 75)
print("DATA PREPROCESSING")
print("=" * 75)

print("\nOriginal pixel value range:")
print("Minimum:", X_train.min())
print("Maximum:", X_train.max())


# Convert pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# Add channel dimension
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


print("\nAfter normalization:")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("Minimum pixel value:", X_train.min())
print("Maximum pixel value:", X_train.max())


# ============================================================
# 7. SAVE PREPROCESSING INFORMATION
# ============================================================

preprocessing_info = pd.DataFrame({
    "Parameter": [
        "Original Image Size",
        "Image Channels",
        "Original Pixel Range",
        "Normalized Pixel Range",
        "Training Samples",
        "Testing Samples",
        "Number of Classes"
    ],
    "Value": [
        "28 x 28",
        "1 - Grayscale",
        "0 - 255",
        "0 - 1",
        len(X_train),
        len(X_test),
        len(class_names)
    ]
})

preprocessing_info.to_csv(
    "report/preprocessing_information.csv",
    index=False
)


# ============================================================
# 8. BUILD CNN MODEL
# ============================================================

print("\n" + "=" * 75)
print("BUILDING CONVOLUTIONAL NEURAL NETWORK")
print("=" * 75)

model = models.Sequential([
    
    # Input Layer
    layers.Input(shape=(28, 28, 1)),

    # First Convolution Block
    layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Second Convolution Block
    layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Convert feature maps into a vector
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout to reduce overfitting
    layers.Dropout(0.5),

    # Output Layer
    layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 10. DISPLAY MODEL SUMMARY
# ============================================================

print("\nMODEL ARCHITECTURE:\n")

model.summary()


# Save model summary
with open("report/model_summary.txt", "w") as file:

    model.summary(
        print_fn=lambda line: file.write(line + "\n")
    )


# ============================================================
# 11. CREATE ARCHITECTURE DIAGRAM
# ============================================================

print("\nCreating neural network architecture diagram...")

fig, ax = plt.subplots(figsize=(15, 7))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

ax.axis("off")

architecture = [
    (1, 5, "Input\n28 × 28 × 1"),
    (2.5, 5, "Conv2D\n32 Filters\n3 × 3 + ReLU"),
    (4, 5, "MaxPooling\n2 × 2"),
    (5.5, 5, "Conv2D\n64 Filters\n3 × 3 + ReLU"),
    (7, 5, "MaxPooling\n2 × 2"),
    (8.5, 6.5, "Flatten"),
    (8.5, 4, "Dense\n128 + ReLU"),
    (8.5, 1.8, "Dropout\n0.5"),
    (6.5, 1.8, "Output\n10 Classes\nSoftmax")
]

for x, y, text in architecture:

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="black"
        )
    )


# Arrows
arrows = [
    ((1.5, 5), (2.0, 5)),
    ((3.0, 5), (3.5, 5)),
    ((4.5, 5), (5.0, 5)),
    ((6.0, 5), (6.5, 5)),
    ((7.5, 5), (8.0, 6.2)),
    ((8.5, 5.8), (8.5, 4.7)),
    ((8.5, 3.5), (8.5, 2.5)),
    ((8.0, 1.8), (7.0, 1.8))
]

for start, end in arrows:

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->")
    )


ax.set_title(
    "CNN Architecture for Fashion-MNIST Classification",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "figures/cnn_architecture.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: figures/cnn_architecture.png")


# ============================================================
# 12. DEFINE CALLBACKS
# ============================================================

print("\nSetting up training callbacks...")

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_learning_rate = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.000001
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n" + "=" * 75)
print("STARTING MODEL TRAINING")
print("=" * 75)

print("\nTraining may take several minutes depending on your computer.")

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[
        early_stopping,
        reduce_learning_rate
    ],
    verbose=1
)

end_time = time.time()

training_time = end_time - start_time

print("\nTraining completed.")

print(
    "Training time: "
    f"{training_time / 60:.2f} minutes"
)


# ============================================================
# 14. DETERMINE ACTUAL EPOCHS
# ============================================================

actual_epochs = len(history.history["loss"])

print("\nNumber of epochs actually completed:", actual_epochs)


# ============================================================
# 15. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history.history)

history_df.insert(
    0,
    "Epoch",
    range(1, actual_epochs + 1)
)

history_df.to_csv(
    "report/training_history.csv",
    index=False
)

print("Saved: report/training_history.csv")


# ============================================================
# 16. TRAINING AND VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "figures/training_validation_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: figures/training_validation_accuracy.png")


# ============================================================
# 17. TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "figures/training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: figures/training_validation_loss.png")


# ============================================================
# 18. EVALUATE MODEL ON TEST DATA
# ============================================================

print("\n" + "=" * 75)
print("MODEL EVALUATION")
print("=" * 75)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print(
    f"Test Accuracy Percentage: {test_accuracy * 100:.2f}%"
)


# ============================================================
# 19. MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_probability = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    y_probability,
    axis=1
)


# ============================================================
# 20. PRECISION, RECALL AND F1 SCORE
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\nOverall Evaluation Metrics:")

print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

classification_report_text = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print(classification_report_text)

with open(
    "report/classification_report.txt",
    "w"
) as file:

    file.write(classification_report_text)


# Save classification report as CSV
classification_report_dict = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

classification_report_df = pd.DataFrame(
    classification_report_dict
).transpose()

classification_report_df.to_csv(
    "report/classification_report.csv"
)

print("Saved: report/classification_report.csv")


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

print("\nCreating confusion matrix...")

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(11, 9))

plt.imshow(cm)

plt.title("Confusion Matrix - Fashion-MNIST CNN")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(10),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(10),
    class_names
)

for i in range(10):

    for j in range(10):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "figures/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: figures/confusion_matrix.png")


# ============================================================
# 23. SAVE CONFUSION MATRIX AS CSV
# ============================================================

confusion_matrix_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

confusion_matrix_df.to_csv(
    "report/confusion_matrix.csv"
)


# ============================================================
# 24. SAMPLE PREDICTIONS
# ============================================================

print("\nCreating sample prediction visualization...")

sample_indices = np.arange(20)

sample_predictions = []

for index in sample_indices:

    actual_class = class_names[y_test[index]]
    predicted_class = class_names[y_pred[index]]

    confidence = (
        np.max(y_probability[index]) * 100
    )

    correct = (
        actual_class == predicted_class
    )

    sample_predictions.append([
        index,
        actual_class,
        predicted_class,
        confidence,
        correct
    ])


sample_predictions_df = pd.DataFrame(
    sample_predictions,
    columns=[
        "Image_Index",
        "Actual_Class",
        "Predicted_Class",
        "Confidence_Percentage",
        "Correct"
    ]
)

sample_predictions_df.to_csv(
    "report/sample_predictions.csv",
    index=False
)


# ============================================================
# 25. PLOT SAMPLE PREDICTIONS
# ============================================================

plt.figure(figsize=(14, 10))

for position, index in enumerate(sample_indices):

    plt.subplot(4, 5, position + 1)

    plt.imshow(
        X_test[index].squeeze(),
        cmap="gray"
    )

    predicted_label = class_names[y_pred[index]]

    actual_label = class_names[y_test[index]]

    confidence = (
        np.max(y_probability[index]) * 100
    )

    plt.title(
        f"Actual: {actual_label}\n"
        f"Pred: {predicted_label}\n"
        f"{confidence:.1f}%"
    )

    plt.axis("off")


plt.suptitle(
    "Sample Predictions from Fashion-MNIST CNN",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "figures/sample_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: figures/sample_predictions.png")


# ============================================================
# 26. SAVE FINAL RESULTS
# ============================================================

final_results = pd.DataFrame({
    "Metric": [
        "Test Loss",
        "Test Accuracy",
        "Test Accuracy Percentage",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1 Score",
        "Training Time Minutes",
        "Epochs Completed",
        "Batch Size"
    ],
    "Value": [
        test_loss,
        test_accuracy,
        test_accuracy * 100,
        precision,
        recall,
        f1,
        training_time / 60,
        actual_epochs,
        BATCH_SIZE
    ]
})

final_results.to_csv(
    "report/final_results.csv",
    index=False
)

print("\nSaved: report/final_results.csv")


# ============================================================
# 27. SAVE TRAINED MODEL
# ============================================================

print("\nSaving trained model...")

model.save(
    "report/fashion_mnist_cnn.keras"
)

print("Saved: report/fashion_mnist_cnn.keras")


# ============================================================
# 28. SAVE PROJECT INFORMATION
# ============================================================

project_information = pd.DataFrame({
    "Parameter": [
        "Project Title",
        "Dataset",
        "Task",
        "Deep Learning Model",
        "Framework",
        "Optimizer",
        "Loss Function",
        "Input Size",
        "Output Classes",
        "Batch Size",
        "Maximum Epochs",
        "Actual Epochs",
        "Dropout",
        "Training Samples",
        "Testing Samples"
    ],
    "Value": [
        "Fashion-MNIST Image Classification",
        "Fashion-MNIST",
        "Multi-class Image Classification",
        "Convolutional Neural Network (CNN)",
        "TensorFlow / Keras",
        "Adam",
        "Sparse Categorical Crossentropy",
        "28 x 28 x 1",
        "10",
        BATCH_SIZE,
        EPOCHS,
        actual_epochs,
        "0.5",
        len(X_train),
        len(X_test)
    ]
})

project_information.to_csv(
    "report/project_information.csv",
    index=False
)


# ============================================================
# 29. FINAL CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nFINAL RESULTS")
print("-" * 50)

print(f"Test Loss       : {test_loss:.4f}")
print(f"Test Accuracy   : {test_accuracy:.4f}")
print(f"Accuracy        : {test_accuracy * 100:.2f}%")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1 Score        : {f1:.4f}")
print(f"Training Time   : {training_time / 60:.2f} minutes")
print(f"Epochs Completed: {actual_epochs}")

print("\nGenerated Figures:")
print("1. sample_fashion_mnist_images.png")
print("2. class_distribution.png")
print("3. cnn_architecture.png")
print("4. training_validation_accuracy.png")
print("5. training_validation_loss.png")
print("6. confusion_matrix.png")
print("7. sample_predictions.png")

print("\nGenerated Report Files:")
print("1. class_distribution.csv")
print("2. preprocessing_information.csv")
print("3. model_summary.txt")
print("4. training_history.csv")
print("5. classification_report.txt")
print("6. classification_report.csv")
print("7. confusion_matrix.csv")
print("8. sample_predictions.csv")
print("9. final_results.csv")
print("10. project_information.csv")
print("11. fashion_mnist_cnn.keras")

