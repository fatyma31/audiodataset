# ── 1. Libraries ────────────────────────────────────────────
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import MobileNetV2

print("✅ All libraries imported!")
print(f"   TensorFlow : {tf.__version__}")
print(f"   Librosa    : {librosa.__version__}")


# ── 2. Configuration ────────────────────────────────────────
DATASET_PATH = r"C:\Users\MS\Desktop\audiodataset"   # apna path yahan

SAMPLE_RATE  = 22050
DURATION     = 4
HOP_LENGTH   = 512
N_FFT        = 2048
N_MELS       = 128
IMG_SIZE     = 128      # MobileNetV2 ke liye 128x128 RGB

BATCH_SIZE   = 16       # chota batch — kam data ke liye better
EPOCHS       = 50
IGNORE       = {'.ipynb_checkpoints', 'model', 'Untitled.ipynb'}

CLASSES = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
    and d not in IGNORE
])

print(f"\n✅ Dataset    : {DATASET_PATH}")
print(f"✅ Classes    : {len(CLASSES)}")
total = 0
for c in CLASSES:
    n = len([f for f in os.listdir(os.path.join(DATASET_PATH, c))
             if f.lower().endswith(('.wav','.mp3','.ogg','.flac'))])
    total += n
    print(f"   📁 {c:<22s} → {n} files")
print(f"\n   📊 Total: {total} files")


# ── 3. EDA ──────────────────────────────────────────────────
counts = {c: len([f for f in os.listdir(os.path.join(DATASET_PATH, c))
                  if f.lower().endswith(('.wav','.mp3','.ogg','.flac'))])
          for c in CLASSES}

plt.figure(figsize=(14, 5))
bars = plt.bar(counts.keys(), counts.values(),
               color=plt.cm.tab20.colors[:len(counts)])
plt.title('Samples per Class', fontsize=16, fontweight='bold')
plt.ylabel('Number of Files')
plt.xticks(rotation=30, ha='right')
for bar, val in zip(bars, counts.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3, str(val),
             ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()


# ── 4. Feature Extraction ───────────────────────────────────
# FIX: Mel Spectrogram ko 3-channel RGB image banao
# MobileNetV2 ko 3 channels chahiye (ImageNet pretrained)
# Mel ko magma colormap se RGB mein convert karo

def mel_to_rgb(file_path):
    """
    Audio → Mel Spectrogram → RGB image (128x128x3)
    MobileNetV2 ke liye 3 channels zaroori hain
    """
    try:
        audio, _ = librosa.load(
            file_path, sr=SAMPLE_RATE,
            duration=DURATION, mono=True
        )
        target = SAMPLE_RATE * DURATION
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE,
            n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize 0-1
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)

        # Resize to 128x128
        mel_resized = tf.image.resize(
            mel_norm[..., np.newaxis], [IMG_SIZE, IMG_SIZE]
        ).numpy().squeeze()

        # Grayscale → RGB (3 channels) using colormap
        cmap = plt.get_cmap('magma')
        rgb = cmap(mel_resized)[:, :, :3]          # (128, 128, 3) float [0,1]
        rgb = (rgb * 255).astype(np.float32)        # [0, 255] range
        rgb = tf.keras.applications.mobilenet_v2.preprocess_input(rgb)  # [-1,1]

        return rgb

    except Exception as e:
        print(f"   [ERROR] {file_path}: {e}")
        return None


# ── 5. Data Augmentation ────────────────────────────────────
# FIX: Augmentation add kiya — original problem yahi tha
# 480 samples bohot kam hain, augmentation se effectively ~2400+ banenge

def augment_audio(audio, sr):
    """
    Random augmentation apply karo audio pe
    Yeh overfitting rokta hai aur generalization badhata hai
    """
    aug_type = np.random.randint(0, 4)

    if aug_type == 0:
        # Time stretch (slow/fast karo)
        rate = np.random.uniform(0.8, 1.2)
        audio = librosa.effects.time_stretch(audio, rate=rate)

    elif aug_type == 1:
        # Pitch shift (awaz ooper/neeche karo)
        steps = np.random.uniform(-3, 3)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

    elif aug_type == 2:
        # Background noise add karo
        noise = np.random.randn(len(audio)) * 0.005
        audio = audio + noise

    elif aug_type == 3:
        # Volume change
        gain = np.random.uniform(0.7, 1.3)
        audio = audio * gain

    return audio


def mel_to_rgb_with_aug(file_path, augment=False):
    """
    Audio load → optional augmentation → Mel → RGB
    """
    try:
        audio, sr = librosa.load(
            file_path, sr=SAMPLE_RATE,
            duration=DURATION, mono=True
        )
        target = SAMPLE_RATE * DURATION
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))

        # Augment karo sirf training data pe
        if augment:
            audio = augment_audio(audio, sr)
            # Re-pad if needed after time stretch
            if len(audio) < target:
                audio = np.pad(audio, (0, target - len(audio)))
            else:
                audio = audio[:target]

        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE,
            n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
        mel_resized = tf.image.resize(
            mel_norm[..., np.newaxis], [IMG_SIZE, IMG_SIZE]
        ).numpy().squeeze()

        cmap = plt.get_cmap('magma')
        rgb = cmap(mel_resized)[:, :, :3]
        rgb = (rgb * 255).astype(np.float32)
        rgb = tf.keras.applications.mobilenet_v2.preprocess_input(rgb)

        return rgb

    except Exception as e:
        print(f"   [ERROR] {file_path}: {e}")
        return None


# ── 6. Load All Files ───────────────────────────────────────
print("\n⏳ Loading and extracting features...")
print("   (augmented copies bhi ban rahe hain)\n")

X, y = [], []
AUG_PER_SAMPLE = 4    # har original ke 4 augmented copies → 480*5 = 2400 samples

for label in CLASSES:
    folder = os.path.join(DATASET_PATH, label)
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.wav','.mp3','.ogg','.flac'))]
    print(f"⏳ {label:<22s} ({len(files)} files × {AUG_PER_SAMPLE+1} = {len(files)*(AUG_PER_SAMPLE+1)})...")

    for fname in files:
        path = os.path.join(folder, fname)

        # Original
        feat = mel_to_rgb_with_aug(path, augment=False)
        if feat is not None:
            X.append(feat); y.append(label)

        # Augmented copies
        for _ in range(AUG_PER_SAMPLE):
            feat = mel_to_rgb_with_aug(path, augment=True)
            if feat is not None:
                X.append(feat); y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n✅ Feature extraction complete!")
print(f"   Shape  : {X.shape}")
print(f"   Samples: {len(y)}")


# ── 7. Label Encoding + Split ───────────────────────────────
le    = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_enc, len(CLASSES))

idx = np.arange(len(y_enc))
idx_train, idx_test = train_test_split(idx, test_size=0.15,
                                        random_state=42, stratify=y_enc)
idx_train, idx_val  = train_test_split(idx_train, test_size=0.12,
                                        random_state=42)

X_train, y_train = X[idx_train], y_cat[idx_train]
X_val,   y_val   = X[idx_val],   y_cat[idx_val]
X_test,  y_test  = X[idx_test],  y_cat[idx_test]

print(f"\n✅ Data split:")
print(f"   Train : {len(idx_train)}")
print(f"   Val   : {len(idx_val)}")
print(f"   Test  : {len(idx_test)}")


# ── 8. MobileNetV2 Model ────────────────────────────────────
# FIX: Scratch CNN hataya, MobileNetV2 Transfer Learning lagaya
# MobileNetV2 ImageNet pe train hai → features already strong hain
# Kam data pe bhi 80-95% accuracy deta hai

def build_mobilenet_model(num_classes):
    """
    MobileNetV2 Transfer Learning
    Phase 1: Sirf top layers train karo (base frozen)
    Phase 2: Fine-tune karo (last 30 layers unfreeze)
    """
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,       # apna classifier lagaenge
        weights='imagenet'       # pretrained weights
    )
    base.trainable = False       # Phase 1: freeze karo

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='MobileNetV2_AudioClassifier')
    return model, base


model, base_model = build_mobilenet_model(len(CLASSES))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
print("\n✅ MobileNetV2 model ready!")


# ── 9. Training ─────────────────────────────────────────────
os.makedirs(os.path.join(DATASET_PATH, "model"), exist_ok=True)
MODEL_SAVE = os.path.join(DATASET_PATH, "model", "mobilenet_audio_best.keras")

cbs_phase1 = [
    callbacks.ModelCheckpoint(MODEL_SAVE, save_best_only=True,
                               monitor='val_accuracy', verbose=1),
    callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                             monitor='val_accuracy', verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=4,
                                 min_lr=1e-6, verbose=1)
]

print("\n🚀 Phase 1: Top layers training (base frozen)...")
print(f"   Epochs     : {EPOCHS}")
print(f"   Batch size : {BATCH_SIZE}\n")

history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cbs_phase1
)
print("\n✅ Phase 1 complete!")


# ── Phase 2: Fine-tuning ─────────────────────────────────────
# FIX: Fine-tuning step add kiya — accuracy aur badhti hai
print("\n🚀 Phase 2: Fine-tuning (last 30 layers unfreeze)...")

base_model.trainable = True
for layer in base_model.layers[:-30]:   # pehle layers frozen rakhein
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # bohot chota LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

MODEL_SAVE_FT = os.path.join(DATASET_PATH, "model", "mobilenet_finetuned_best.keras")
cbs_phase2 = [
    callbacks.ModelCheckpoint(MODEL_SAVE_FT, save_best_only=True,
                               monitor='val_accuracy', verbose=1),
    callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                             monitor='val_accuracy', verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=3,
                                 min_lr=1e-8, verbose=1)
]

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=BATCH_SIZE,
    callbacks=cbs_phase2,
    initial_epoch=len(history1.history['accuracy'])
)
print("\n✅ Phase 2 fine-tuning complete!")


# ── 10. Results ─────────────────────────────────────────────
# Combine histories
all_acc     = history1.history['accuracy']     + history2.history.get('accuracy', [])
all_val_acc = history1.history['val_accuracy'] + history2.history.get('val_accuracy', [])
all_loss    = history1.history['loss']         + history2.history.get('loss', [])
all_val_loss= history1.history['val_loss']     + history2.history.get('val_loss', [])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MobileNetV2 — Training Results', fontsize=16, fontweight='bold')

axes[0].plot(all_acc,     label='Train Accuracy', color='royalblue', lw=2)
axes[0].plot(all_val_acc, label='Val Accuracy',   color='orange',    lw=2)
axes[0].axvline(x=len(history1.history['accuracy'])-1,
                color='gray', linestyle='--', alpha=0.5, label='Fine-tune start')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(all_loss,     label='Train Loss', color='royalblue', lw=2)
axes[1].plot(all_val_loss, label='Val Loss',   color='orange',    lw=2)
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*45}")
print(f"  🎯 Test Accuracy : {test_acc*100:.2f}%")
print(f"  📉 Test Loss     : {test_loss:.4f}")
print(f"{'='*45}")


# ── 11. Confusion Matrix ────────────────────────────────────
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(13, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5)
plt.title('Confusion Matrix — MobileNetV2', fontsize=16, fontweight='bold')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n📋 Classification Report:")
print('='*60)
print(classification_report(y_true, y_pred, target_names=le.classes_))


# ── 12. Per-Class Accuracy Bar Chart ───────────────────────
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
colors = ['#2ecc71' if a >= 90 else '#f39c12' if a >= 75 else '#e74c3c'
          for a in per_class_acc]

plt.figure(figsize=(14, 6))
bars = plt.bar(le.classes_, per_class_acc, color=colors)
plt.axhline(y=np.mean(per_class_acc), color='navy', linestyle='--', lw=2,
            label=f'Average: {np.mean(per_class_acc):.1f}%')
plt.title('Per Class Accuracy', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim([0, 110])
plt.xticks(rotation=30, ha='right')
plt.legend()
for bar, val in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()
print("\n🟢 Green = 90%+ | 🟡 Orange = 75-90% | 🔴 Red = <75%")

print(f"\n✅ Model saved at: {MODEL_SAVE_FT}")
print("✅ Training complete! Expected accuracy: 80-95%")
