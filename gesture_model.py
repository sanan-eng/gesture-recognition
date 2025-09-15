# gesture_model_fixed.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import warnings
import gc
from tensorflow.keras import backend as K

warnings.filterwarnings('ignore')

# ------------------ Constants (change paths if you want) ------------------
IMG_SIZE = (224, 224)  # Increased size for EfficientNet
BATCH_SIZE = 32  # Increased batch size
TEST_BATCH_SIZE = 16  # Smaller batch size for evaluation to avoid memory issues
EPOCHS = 30  # Reduced epochs
# Point these to the top-level folders you have (script will auto-detect nested class folder)
TRAIN_PATH = r"E:\CODECLAUSE INTERN\gesture\HGR dataset\train"
TEST_PATH = r"E:\CODECLAUSE INTERN\gesture\HGR dataset\multi_user_test"
VAL_SPLIT = 0.15
SEED = 123


# ------------------ Helpers ------------------
def find_class_root(path):
    """
    If path contains exactly one directory and that directory likely contains classes,
    return the deeper directory. Otherwise return the path unchanged.
    This makes it robust to structures like: train/train/0..19
    """
    try:
        entries = sorted(os.listdir(path))
    except Exception as e:
        raise ValueError(f"Cannot list {path}: {e}")

    # If there is exactly one directory and that directory contains multiple subdirectories,
    # use it as the class root.
    dirs = [d for d in entries if os.path.isdir(os.path.join(path, d))]
    if len(dirs) == 1:
        candidate = os.path.join(path, dirs[0])
        # Check if candidate has multiple subfolders (likely class dirs)
        sub = [s for s in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, s))]
        if len(sub) > 1:
            return candidate
    return path


# ------------------ Data Loading ------------------
def load_and_preprocess_data(test_batch_size=BATCH_SIZE):
    # auto-detect class root
    class_root_train = find_class_root(TRAIN_PATH)
    class_root_test = find_class_root(TEST_PATH)

    print("DEBUG: class_root_train =", class_root_train)
    print("DEBUG: class_root_test  =", class_root_test)

    # get class names
    class_names = sorted([d for d in os.listdir(class_root_train) if os.path.isdir(os.path.join(class_root_train, d))])
    if len(class_names) == 0:
        raise ValueError(f"No class subfolders found in {class_root_train}. Each class should be a subfolder.")
    num_classes = len(class_names)
    print("DEBUG: class_names =", class_names)
    print("DEBUG: num_classes  =", num_classes)

    # Use image_dataset_from_directory with validation_split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        class_root_train,
        labels='inferred',
        label_mode='int',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        subset='training',
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        class_root_train,
        labels='inferred',
        label_mode='int',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        subset='validation',
        seed=SEED
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        class_root_test,
        labels='inferred',
        label_mode='int',
        image_size=IMG_SIZE,
        batch_size=test_batch_size,  # Use smaller batch size for evaluation
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE

    # Preprocessing function for EfficientNet
    def preprocess(image, label):
        # EfficientNet expects inputs in the range [-1, 1]
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    # Debug: print a few labels to ensure they are correct
    sample_labels = []
    for images, labels in train_ds.take(2):  # two batches
        sample_labels.extend(labels.numpy().tolist())
    print("DEBUG: sample train labels (first 2 batches):", sample_labels[:20])

    sample_val_labels = []
    for _, labels in val_ds.take(1):
        sample_val_labels.extend(labels.numpy().tolist())
    print("DEBUG: sample val labels (first batch):", sample_val_labels[:20])

    return train_ds, val_ds, test_ds, class_names, num_classes


# ------------------ Augmentation ------------------
def create_augmentation_model():
    return keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),  # Reduced rotation
        layers.RandomZoom(0.1),  # Reduced zoom
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.05),  # Reduced contrast
    ])


# ------------------ Model ------------------
def create_model(num_classes):
    base_model = applications.EfficientNetB0(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        create_augmentation_model(),
        base_model,
        layers.Dropout(0.3),  # Reduced dropout
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),  # Reduced regularization
        layers.BatchNormalization(),
        layers.Dropout(0.2),  # Reduced dropout
        layers.Dense(num_classes, activation='softmax')
    ])
    return model, base_model


# ------------------ Class weights ------------------
def calculate_class_weights(dataset, num_classes):
    # accumulate labels (flatten)
    labels = []
    for _, label in dataset.unbatch():
        labels.append(int(label.numpy()))
    labels = np.array(labels, dtype=np.int32)
    # if something odd, print distribution
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print("DEBUG: label distribution (train):", dist)

    cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=labels)
    # clip to reasonable range
    cw = np.clip(cw, 0.5, 5.0)
    return dict(enumerate(cw))


# ------------------ Training ------------------
def train():
    train_ds, val_ds, test_ds, class_names, num_classes = load_and_preprocess_data()
    class_weights = calculate_class_weights(train_ds, num_classes)
    print("DEBUG: class_weights =", class_weights)

    model, base_model = create_model(num_classes)

    # Use multiclass-appropriate metrics
    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # Increased learning rate
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger('training_log.csv'),
        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
    ]

    print("🚀 Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Optional fine-tuning
    print("🔧 Fine-tuning last layers of base model...")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # Unfreeze more layers
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=history.epoch[-1] + 10,
        initial_epoch=history.epoch[-1],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Clear memory before evaluation
    K.clear_session()
    gc.collect()

    # Reload best model for evaluation
    if os.path.exists('best_model.keras'):
        print("Loading best model for evaluation...")
        model = keras.models.load_model('best_model.keras')

    # Evaluate with smaller batch size to avoid memory issues
    print("📊 Evaluating on test set...")
    _, _, test_ds_small = load_and_preprocess_data(test_batch_size=TEST_BATCH_SIZE)[:3]
    test_res = model.evaluate(test_ds_small, verbose=1)
    print("Test results:", test_res)

    # Save
    model.save('gesture_model.keras')
    np.save('class_names.npy', class_names)
    metadata = {
        'input_size': IMG_SIZE,
        'classes': class_names,
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'test_results': test_res
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    return history, history_fine


# ------------------ Evaluation ------------------
# ------------------ Evaluation ------------------
def evaluate_model():
    """Evaluate an existing model"""
    if not os.path.exists('best_model.keras'):
        print("No model found for evaluation. Please train a model first.")
        return

    # Clear memory
    K.clear_session()
    gc.collect()

    # Load model
    print("Loading best model for evaluation...")
    model = keras.models.load_model('best_model.keras')

    # Try to load class names, or regenerate them if missing
    try:
        class_names = np.load('class_names.npy', allow_pickle=True)
        print("Loaded class names from file.")
    except FileNotFoundError:
        print("Class names file not found. Generating class names from training directory...")
        # Regenerate class names from training directory
        class_root_train = find_class_root(TRAIN_PATH)
        class_names = sorted([d for d in os.listdir(class_root_train)
                              if os.path.isdir(os.path.join(class_root_train, d))])
        np.save('class_names.npy', class_names)
        print(f"Generated and saved class names: {class_names}")

    # Load test data with smaller batch size
    print("Loading test data...")
    _, _, test_ds = load_and_preprocess_data(test_batch_size=TEST_BATCH_SIZE)[:3]

    # Evaluate
    print("📊 Evaluating on test set...")
    test_res = model.evaluate(test_ds, verbose=1)
    print("Test results:", test_res)

    # Update metadata
    metadata = {
        'input_size': IMG_SIZE,
        'classes':list( class_names),
        'evaluation_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'test_results': test_res
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Evaluation complete!")
    model.save('gesture_model.keras')


# ------------------ Main ------------------
if __name__ == "__main__":
    if os.path.exists('best_model.keras'):
        print("Model exists. Skipping training and proceeding to evaluation.")
        evaluate_model()
    else:
        print("No model found. Starting training...")
        history, history_fine = train()