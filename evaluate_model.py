import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = 'test_model.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DIR = 'TestData'

# Load the Trained Model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Prepare Test Data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Run Evaluation
print("Evaluating model performance...")
results = model.evaluate(test_generator)
print(f"\nOverall Test Accuracy: {results[1]*100:.2f}%")

# Generate Confusion Matrix
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Create the matrix
cm = confusion_matrix(y_true, y_pred)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_size=class_labels))
