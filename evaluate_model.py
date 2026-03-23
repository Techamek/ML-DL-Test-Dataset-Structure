import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Load the  model
model_path = os.path.join('models', 'touchmodel.h5')

if not os.path.exists(model_path):
    print(f"ERROR: {model_path} not found.")
else:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Load the dataset
    raw_test_ds = tf.keras.utils.image_dataset_from_directory(
        'valData', 
        image_size=(224, 224),
        batch_size=32,
        shuffle=False
    )

    class_names = raw_test_ds.class_names
    print(f"Detected classes: {class_names}")

    # Normalize the data
    test_ds = raw_test_ds.map(lambda x, y: (x / 255.0, y))

    # Generate Predictions
    print("Generating predictions...")
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    # Final Results
    print("\n### Evaluation Results ###")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
