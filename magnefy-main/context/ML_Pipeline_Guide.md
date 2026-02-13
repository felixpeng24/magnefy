# ML Pipeline Guide for PD Classification

This document explains each stage of the machine learning pipeline for classifying Partial Discharge (PD) waveforms, tailored to your dataset (1000+ labeled .mat files from the IEEE LEE dataset, organized into folders by discharge type: internal, corona, surface).

---

## How to Read This Guide

Each stage below explains:
- **What** this stage does (plain English)
- **Why** it matters
- **How** it works (technical detail + Python concepts)
- **What you produce** (the output of this stage)
- **Key decisions** you'll need to make

---

## Stage 0: Project Setup & Data Inventory

### What
Before touching any ML code, you need to know exactly what you have. This means cataloging your dataset: how many files per class, what's inside each .mat file, and whether the data is consistent across files.

### Why
ML models are only as good as the data they train on. If one class has 800 files and another has 50, the model will be biased toward the larger class. If some files have different sampling rates or channel structures, your pipeline will break silently and produce garbage results.

### How
Write a script that walks through each folder (internal/, corona/, surface/), counts files, and loads a sample from each to check:
- Number of channels (expect 4: Ch1-Ch4)
- Number of samples per channel (expect ~4.3 million at 125 MHz)
- Sampling rate (dt value — expect 8 nanoseconds)
- Value ranges per channel

```python
import os
import scipy.io as sio

dataset_root = "/path/to/LEE_dataset"
class_folders = ["internal", "corona", "surface"]  # adjust to actual folder names

for folder in class_folders:
    folder_path = os.path.join(dataset_root, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    print(f"{folder}: {len(files)} files")

    # Spot-check first file
    sample = sio.loadmat(os.path.join(folder_path, files[0]))
    print(f"  Keys: {[k for k in sample.keys() if not k.startswith('__')]}")
    print(f"  Ch1 shape: {sample['Ch1'].shape}")
    print(f"  dt: {sample['dt']}")
```

### What You Produce
- A table showing: class name, number of files, any anomalies
- Confidence that all files have consistent structure

### Key Decisions
- **Class imbalance**: If one class has far fewer samples, you'll need to address this later (oversampling, undersampling, or class weights). Note it now.
- **Which channels to use**: Your notebook uses Ch1 primarily. You may want to analyze all 4 channels or pick the most informative one. Start with one channel to keep things simple, then expand.
- **Noise class**: Your dataset has internal, corona, and surface — but your project scope also mentions classifying "noise." You'll need to decide whether to create synthetic noise samples, find a noise dataset, or train a 3-class model instead of 4.

---

## Stage 1: Preprocessing & Cleaning

### What
Raw waveforms are noisy and vary in scale. Preprocessing makes them consistent and removes irrelevant information before you extract features.

### Why
If one file's voltages range from -0.01 to 0.04 and another's range from -8 to 8 (which you've already seen in your notebook), a model comparing raw amplitudes between files would be meaningless. Preprocessing puts everything on the same playing field.

### How

#### 1a. Normalization
Scale each waveform so values are comparable across files. Two common approaches:

**Z-score normalization** (what your notebook already does):
```python
normalized = (signal - signal.mean()) / signal.std()
```
This centers the signal at 0 with standard deviation of 1. Good for comparing relative spike sizes.

**Min-max normalization**:
```python
normalized = (signal - signal.min()) / (signal.max() - signal.min())
```
This scales everything to [0, 1]. Good when absolute magnitude matters.

**Recommendation**: Start with z-score. It's what your notebook uses and it works well for spike detection.

#### 1b. Noise Floor Determination
Your project scope requires comparing 3+ noise reduction approaches. Here's what they mean:

1. **Standard deviation threshold** (simplest — your notebook does this):
   - Set threshold at N standard deviations from the mean (you use N=3)
   - Anything below is "noise," anything above is a "PD event"
   - Pro: simple, fast. Con: assumes noise is Gaussian (bell-curve shaped)

2. **Histogram knee method**:
   - Plot a histogram of all amplitude values
   - The "knee" (sharp bend) separates the bulk of noise from PD spikes
   - Pro: doesn't assume Gaussian noise. Con: harder to automate

3. **Wavelet denoising**:
   - Uses wavelet transforms to separate signal from noise at different frequency scales
   - Pro: preserves signal shape better. Con: more complex, has tuning parameters
   - Library: `pywt` (PyWavelets)

```python
import pywt

# Wavelet denoising example
coeffs = pywt.wavedec(signal, 'db4', level=5)
# Zero out the detail coefficients below a threshold
threshold = np.median(np.abs(coeffs[-1])) / 0.6745  # universal threshold
denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
denoised_signal = pywt.waverec(denoised_coeffs, 'db4')
```

#### 1c. Bandpass Filtering (optional but useful)
PD signals typically occupy specific frequency ranges. A bandpass filter removes frequencies outside the range of interest:

```python
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Example: keep 100 kHz to 50 MHz (typical PD range for HFCT)
fs = 125e6  # 125 MHz sampling rate
filtered = bandpass_filter(signal, 100e3, 50e6, fs)
```

### What You Produce
- A preprocessing function that takes a raw .mat file and returns a clean, normalized signal
- Documentation of which noise method you chose and why
- Comparison metrics for the 3+ noise approaches (your Phase 3 deliverable)

### Key Decisions
- **Which noise method**: Try all three, measure how many PD events each detects, compare with visual inspection. There's no universal "best" — it depends on your data.
- **Filter frequencies**: Consult your Technical Info doc and PD literature for typical HFCT frequency ranges.

---

## Stage 2: Event Detection (PD Spike Extraction)

### What
Go from a continuous 4.3-million-point waveform to a list of individual PD events (spikes). Each event becomes one row in your eventual training dataset.

### Why
You don't feed the entire 4.3M-point waveform into a model. That would be like trying to find a needle in a haystack by feeding the entire haystack into a machine. Instead, you find the needles first (the PD spikes), then analyze each needle.

### How
Your notebook already does this with `scipy.signal.find_peaks`. The key parameters:

```python
from scipy.signal import find_peaks

# Detect spikes above threshold
threshold = signal.mean() + 3 * signal.std()
peaks, properties = find_peaks(signal, height=threshold, distance=100)

# For each detected peak, extract a window around it
window_size = 500  # samples on each side of the peak
events = []
for peak_idx in peaks:
    start = max(0, peak_idx - window_size)
    end = min(len(signal), peak_idx + window_size)
    event_waveform = signal[start:end]
    events.append(event_waveform)
```

The `window_size` determines how much of the waveform around each spike you capture. Too small and you miss the tail of the PD pulse; too large and you include unrelated data.

### What You Produce
- A list of PD event windows extracted from each file
- Metadata for each event: which file it came from, where in the waveform it occurred, its label (from the folder name)

### Key Decision
- **Window size**: Start with 500-1000 samples on each side. Look at a few events visually to make sure you're capturing the full PD pulse shape. Adjust as needed.
- **Overlapping events**: Sometimes PD events happen close together. You need to decide whether to merge them or treat them as separate events. The `distance` parameter in `find_peaks` controls this.

---

## Stage 3: Feature Extraction

### What
Transform each PD event window (a chunk of ~1000 data points) into a small set of meaningful numbers (10-20 features). This is the most important stage for classical ML approaches.

### Why
A Random Forest or SVM can't look at 1000 raw data points and "understand" them the way a human looks at a waveform shape. But it *can* learn patterns from structured features like "events with high amplitude + short rise time + occurring at phase angle 90 degrees tend to be internal discharge."

### How
Here are the features your project scope requires, with code for each:

```python
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

def extract_features(event_waveform, dt, phase_angle=None):
    """
    Extract features from a single PD event waveform.

    Args:
        event_waveform: numpy array of the event window
        dt: time step (8e-9 for 125 MHz)
        phase_angle: where in the AC cycle this event occurred (0-360 degrees)

    Returns:
        dict of feature names -> values
    """
    features = {}

    # 1. AMPLITUDE: Peak value of the event
    features['amplitude'] = np.max(np.abs(event_waveform))
    features['peak_positive'] = np.max(event_waveform)
    features['peak_negative'] = np.min(event_waveform)

    # 2. ENERGY: Total energy of the pulse
    features['energy'] = np.sum(event_waveform ** 2) * dt

    # 3. PULSE WIDTH: How long the pulse lasts
    # Measured at 50% of peak amplitude (full width at half maximum)
    half_max = features['amplitude'] / 2
    above_half = np.where(np.abs(event_waveform) > half_max)[0]
    if len(above_half) > 1:
        features['pulse_width'] = (above_half[-1] - above_half[0]) * dt
    else:
        features['pulse_width'] = dt  # single sample width

    # 4. RISE TIME: How quickly the pulse reaches its peak
    peak_idx = np.argmax(np.abs(event_waveform))
    ten_pct = 0.1 * features['amplitude']
    ninety_pct = 0.9 * features['amplitude']
    abs_signal = np.abs(event_waveform[:peak_idx + 1])
    rise_start = np.where(abs_signal >= ten_pct)[0]
    rise_end = np.where(abs_signal >= ninety_pct)[0]
    if len(rise_start) > 0 and len(rise_end) > 0:
        features['rise_time'] = (rise_end[0] - rise_start[0]) * dt
    else:
        features['rise_time'] = 0

    # 5. FALL TIME: How quickly the pulse decays after peak
    abs_signal_after = np.abs(event_waveform[peak_idx:])
    fall_start = np.where(abs_signal_after >= ninety_pct)[0]
    fall_end = np.where(abs_signal_after <= ten_pct)[0]
    if len(fall_start) > 0 and len(fall_end) > 0:
        features['fall_time'] = (fall_end[0] - fall_start[0]) * dt
    else:
        features['fall_time'] = 0

    # 6. PHASE ANGLE: Where in the AC cycle this event occurred
    # This is passed in from the event detection stage
    if phase_angle is not None:
        features['phase_angle'] = phase_angle
        features['phase_sin'] = np.sin(np.radians(phase_angle))
        features['phase_cos'] = np.cos(np.radians(phase_angle))

    # 7. OSCILLATION: How much the signal rings after the main pulse
    after_peak = event_waveform[peak_idx:]
    zero_crossings = np.where(np.diff(np.sign(after_peak)))[0]
    features['num_oscillations'] = len(zero_crossings) // 2

    # 8. FFT FEATURES: Dominant frequency components
    N = len(event_waveform)
    yf = np.abs(fft(event_waveform))[:N // 2]
    xf = fftfreq(N, dt)[:N // 2]
    features['dominant_freq'] = xf[np.argmax(yf)]
    features['spectral_centroid'] = np.sum(xf * yf) / np.sum(yf) if np.sum(yf) > 0 else 0

    # 9. STATISTICAL FEATURES
    features['rms'] = np.sqrt(np.mean(event_waveform ** 2))
    features['crest_factor'] = features['amplitude'] / features['rms'] if features['rms'] > 0 else 0
    features['skewness'] = float(np.mean(((event_waveform - np.mean(event_waveform)) / np.std(event_waveform)) ** 3)) if np.std(event_waveform) > 0 else 0
    features['kurtosis'] = float(np.mean(((event_waveform - np.mean(event_waveform)) / np.std(event_waveform)) ** 4)) if np.std(event_waveform) > 0 else 0

    return features
```

#### About Phase Angle
Phase angle is critical for PD classification. It tells you *where in the AC power cycle* the discharge occurred. To calculate it:

```python
def compute_phase_angle(peak_index, total_samples, ac_frequency=60):
    """
    Estimate the AC phase angle at which a PD event occurred.

    The AC cycle repeats at 60 Hz. If you know the sampling rate and
    the position of the event in the recording, you can determine
    where in the 360-degree cycle it falls.
    """
    dt = 8e-9  # 8 nanoseconds
    time_of_event = peak_index * dt
    ac_period = 1.0 / ac_frequency  # ~16.67 ms for 60 Hz
    phase = (time_of_event % ac_period) / ac_period * 360  # degrees
    return phase
```

Different discharge types cluster at different phase angles — this is the basis of Phase-Resolved PD (PRPD) analysis, which is one of the most powerful diagnostic tools for PD classification.

### What You Produce
A pandas DataFrame where:
- Each **row** is one PD event
- Each **column** is a feature (amplitude, phase_angle, rise_time, etc.)
- Plus a **label column** indicating the discharge type

```
| amplitude | phase_angle | rise_time | pulse_width | ... | label    |
|-----------|-------------|-----------|-------------|-----|----------|
| 0.034     | 87.3        | 2.4e-7    | 1.1e-6      | ... | internal |
| 0.019     | 245.1       | 1.8e-7    | 8.3e-7      | ... | corona   |
| 0.041     | 162.8       | 3.1e-7    | 1.5e-6      | ... | surface  |
```

This DataFrame is what you save as a CSV. **This** is what gets fed into the ML models.

### Key Decisions
- **Which features to start with**: Start with all of them, then use feature importance analysis (Stage 5) to see which ones actually matter.
- **Phase angle calculation**: The calculation above assumes a clean 60 Hz AC signal. If the recording doesn't start at a known phase, you may need to use one of the channels as a phase reference (some setups use a dedicated channel for this).

---

## Stage 4: Dataset Assembly & Splitting

### What
Combine all extracted features from all files into one big DataFrame, then split it into training and testing sets.

### Why
You train the model on one portion of data and test on a completely separate portion it has never seen. This tells you how well the model will perform on new, unseen data — not just how well it memorized the training examples.

### How

```python
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# After running feature extraction on all files:
# all_features is a list of dicts, all_labels is a list of strings
df = pd.DataFrame(all_features)
df['label'] = all_labels

# Check class distribution
print(df['label'].value_counts())

# Split: 80% training, 20% testing
# stratify=df['label'] ensures each split has the same proportion of each class
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Training class distribution:\n{y_train.value_counts()}")
```

#### Important: File-Level Splitting
There's a subtle but critical point here: **you should split by file, not by individual event.** If you split by event, events from the same file could end up in both training and testing sets. Since events from the same file are similar (same equipment, same conditions), the model might appear to perform better than it actually would on truly new data. This is called **data leakage**.

```python
# Better approach: split by file, then extract events
from sklearn.model_selection import GroupShuffleSplit

# file_ids tracks which file each event came from
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=file_ids))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

#### Cross-Validation
For more robust evaluation, use k-fold cross-validation instead of a single split:

```python
# 5-fold cross-validation (also grouped by file)
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=file_ids)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Train and evaluate model for this fold
```

This trains and tests 5 times, each time with a different 20% held out. You report the average performance across all 5 folds.

### What You Produce
- Train/test splits ready for model training
- Documented split strategy (grouped by file, stratified by class)
- Class distribution summary for both splits

### Key Decisions
- **Split ratio**: 80/20 is standard. With 1000+ files, you have plenty of data for both.
- **Random seed**: Setting `random_state=42` makes the split reproducible. Anyone running your code gets the same split.

---

## Stage 5: Model Training & Evaluation

### What
Train ML models on the training set, evaluate on the test set, compare models to find the best one.

### Why
Different models have different strengths. Some work better on small datasets, some handle many features better, some are more interpretable. You try several and pick the best.

### How

#### Approach A: Classical ML on Extracted Features
This uses the feature DataFrame from Stage 3. These models are simpler, faster, and more interpretable.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Scale features (important for SVM and KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use same scaling as training

# Define models to evaluate
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}

# Train and evaluate each
results = {}
for name, model in models.items():
    # SVM and KNN need scaled features; tree-based models don't but it doesn't hurt
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))

    # Store results
    results[name] = {
        'model': model,
        'predictions': y_pred,
    }
```

**What the metrics mean:**
- **Accuracy**: What fraction of all predictions were correct. Can be misleading with imbalanced classes.
- **Precision**: Of the events the model said were "internal," what fraction actually were? High precision = few false alarms.
- **Recall**: Of all actual "internal" events, what fraction did the model catch? High recall = few missed events.
- **F1 Score**: Harmonic mean of precision and recall. A single number balancing both.

```python
# Confusion matrix visualization
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(y_true.unique()),
                yticklabels=sorted(y_true.unique()))
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, results['Random Forest']['predictions'],
                      'Random Forest Confusion Matrix')
```

#### Approach B: CNN on Raw Waveforms
Instead of hand-crafting features, feed the raw PD event waveform directly into a Convolutional Neural Network. The CNN learns its own features automatically.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Reshape events for CNN: (num_events, window_size, 1)
# window_size is the length of each event waveform (e.g., 1000)
X_train_cnn = X_train_raw.reshape(-1, window_size, 1)
X_test_cnn = X_test_raw.reshape(-1, window_size, 1)

# Encode labels as integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Build CNN model
model = models.Sequential([
    layers.Conv1D(32, kernel_size=7, activation='relu', input_shape=(window_size, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train_cnn, y_train_encoded,
    epochs=50,
    batch_size=32,
    validation_split=0.15,  # use 15% of training data for validation during training
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)
```

**How CNNs work (brief)**:
- `Conv1D` layers slide small filters across the waveform, detecting local patterns (spikes, oscillations, slopes)
- `MaxPooling1D` layers downsample, keeping only the most prominent features
- Stacking multiple Conv1D layers lets the network detect increasingly complex patterns
- `Dense` layers at the end combine all detected patterns to make the final classification
- `Dropout` randomly disables neurons during training to prevent overfitting
- `EarlyStopping` stops training when the model stops improving, preventing overfitting

**Why 1D?** Your waveforms are 1-dimensional signals (amplitude over time), so you use `Conv1D`. If you were classifying images, you'd use `Conv2D`. You could also convert waveforms to spectrograms (2D time-frequency images) and use `Conv2D` — that's another valid approach.

#### Feature Importance (for classical models)
This tells you which features matter most for classification:

```python
# Random Forest has built-in feature importance
rf = results['Random Forest']['model']
importance = pd.Series(rf.feature_importances_, index=X_train.columns)
importance.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```

If phase_angle and amplitude dominate, that confirms what PD literature says. If an unexpected feature ranks high, that's worth investigating — it could be a genuine insight or a data artifact.

### What You Produce
- Trained models with performance metrics
- Comparison table: model name, accuracy, precision, recall, F1 per class
- Confusion matrices showing where each model gets confused
- Feature importance rankings
- A recommendation for the top 3 models (your Phase 5 deliverable)

### Key Decisions
- **Classical ML vs CNN vs both**: Your project scope says "evaluate 5+ models." A good split would be 3-4 classical models + 1-2 deep learning models.
- **Hyperparameter tuning**: Each model has settings you can adjust. Start with defaults, get baseline results, then tune the top performers. Use `sklearn.model_selection.GridSearchCV` or `RandomizedSearchCV`.
- **Overfitting watch**: If training accuracy is 99% but test accuracy is 70%, your model memorized the training data instead of learning general patterns. Reduce model complexity, add regularization, or get more data.

---

## Stage 6: Model Export & Deployment

### What
Save the best model(s) so they can be loaded later to classify new PD data without retraining.

### Why
Training takes time and compute. Once you have a good model, you save it and reuse it.

### How

```python
import pickle

# Save classical ML model
best_model = results['Random Forest']['model']
with open('pd_classifier_rf.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,         # need the same scaler used during training
        'feature_names': list(X_train.columns),
    }, f)

# Save CNN model (TensorFlow uses its own format)
model.save('pd_classifier_cnn.keras')

# Load and use later
with open('pd_classifier_rf.pkl', 'rb') as f:
    saved = pickle.load(f)

loaded_model = saved['model']
loaded_scaler = saved['scaler']

# Classify a new event
new_features = extract_features(new_event_waveform, dt=8e-9, phase_angle=87.3)
new_df = pd.DataFrame([new_features])[saved['feature_names']]
new_scaled = loaded_scaler.transform(new_df)
prediction = loaded_model.predict(new_scaled)
print(f"Predicted class: {prediction[0]}")
```

### What You Produce
- `.pkl` file for classical models (includes the model + scaler + feature names)
- `.keras` file for CNN models
- A prediction script that loads the model and classifies new data

---

## The Pipeline at a Glance

```
.mat files (organized by class folder)
    |
    v
[Stage 0] Inventory & validation
    |
    v
[Stage 1] Preprocessing (normalize, denoise, filter)
    |
    v
[Stage 2] Event detection (find PD spikes, extract windows)
    |
    v
[Stage 3] Feature extraction (amplitude, phase, rise time, FFT, etc.)
    |
    v
[Stage 4] Dataset assembly & train/test split (by file, not by event)
    |
    v
[Stage 5] Model training & evaluation (RF, SVM, GBM, CNN, etc.)
    |
    v
[Stage 6] Export best model (.pkl or .keras)
    |
    v
New data --> load model --> classify --> "internal discharge" / "corona" / "surface"
```

---

## Common Pitfalls to Avoid

1. **Data leakage**: Never let events from the same file appear in both training and testing sets. Split by file.

2. **Not scaling features**: SVM and KNN perform poorly on unscaled data. Always scale after splitting (fit scaler on training data only, then transform both train and test).

3. **Ignoring class imbalance**: If you have 800 internal, 150 corona, 50 surface files, the model will learn to always predict "internal." Use `class_weight='balanced'` in sklearn models, or oversample minority classes with SMOTE.

4. **Overfitting**: High training accuracy + low test accuracy = overfitting. Use cross-validation, regularization, and simpler models.

5. **Not versioning your data splits**: Set `random_state` everywhere so results are reproducible. Document which files are in train vs test.

6. **Skipping visualization**: Always plot your waveforms, features, and results. Patterns that look wrong visually usually are wrong.

7. **Premature optimization**: Get a simple pipeline working end-to-end first (even with 2-3 features and 1 model). Then iterate: add features, try models, tune hyperparameters. Don't try to build the perfect pipeline on the first attempt.

---

## Recommended Library Versions

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
tensorflow>=2.13        # only if using CNN approach
PyWavelets>=1.4         # only if using wavelet denoising
seaborn>=0.12           # for confusion matrix plots
h5py>=3.8               # for some .mat file formats
```

---

## Mapping This Guide to Your Project Phases

| Project Phase | Guide Stages | Timeline |
|---|---|---|
| Phase 1: Research | Background reading | Weeks 1-2 |
| Phase 2: Data Collection | Stage 0 (Inventory) | Weeks 3-5 |
| Phase 3: Middleware & Noise | Stages 1-2 (Preprocess, Event Detection) | Weeks 6-8 |
| Phase 4: Feature Extraction | Stage 3 (Feature Extraction) | Weeks 9-10 |
| Phase 5: ML Models | Stages 4-5 (Split, Train, Evaluate) | Weeks 11-13 |
| Phase 6: Deployment | Stage 6 (Export & Demo) | Week 14 |

---

## Next Steps (What to Do Right Now)

1. **Run Stage 0** on your full LEE dataset. Know exactly what you have: how many files per class, whether all files have the same structure, any anomalies.

2. **Pick one file from each class** and run it through Stages 1-3 manually in a notebook. Visualize everything. Make sure the features look reasonable before scaling up.

3. **Build the full pipeline** as a Python script (not just a notebook) that processes all files and outputs a single CSV of features + labels.

4. **Train a Random Forest** as your first baseline model. It's fast, requires no tuning, and gives you feature importance for free.

5. **Iterate**: add features, try more models, tune hyperparameters, compare approaches.
