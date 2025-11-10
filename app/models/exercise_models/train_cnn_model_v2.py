import os, glob, math, random, collections
from typing import List
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence

# =========================
# 설정
# =========================
DATASET_ROOT = r"./dataset"
IMG_SIZE     = (160, 160)
SEQ_LEN      = 24
BATCH_SIZE   = 8
EPOCHS       = 25
VAL_SPLIT    = 0.2
SEED         = 42
LR_BASE      = 3e-4
BACKBONE     = "EfficientNetB0"

AUG_FLIP_PROB       = 0.5
AUG_BRIGHT_DELTA    = 0.10
AUG_CONTRAST_RANGE  = (0.9, 1.1)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# 시퀀스 로딩 & 전처리
# =========================
def _sample_indices(n, k):
    if n >= k:
        return np.linspace(0, n - 1, num=k).astype(int)
    return np.concatenate([np.arange(n), np.full((k - n,), n - 1)])

def load_sequence_fixed(seq_path, seq_len, img_size):
    import cv2
    video_exts = (".mp4",".avi",".mov",".mkv")

    # 폴더 → 프레임 입력
    if os.path.isdir(seq_path):
        imgs = sorted(glob.glob(os.path.join(seq_path, "*.*")))
        idxs = _sample_indices(len(imgs), seq_len)
        frames = []
        for i in idxs:
            with Image.open(imgs[i]) as im:
                im = im.convert("RGB").resize(img_size)
                frames.append(np.array(im) / 255.)
        return np.stack(frames, axis=0)

    # 비디오 → 프레임 입력
    if seq_path.lower().endswith(video_exts):
        cap = cv2.VideoCapture(seq_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idxs = _sample_indices(total, seq_len)
        frames = []
        for frame_idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                frames.append(np.zeros((img_size[0], img_size[1], 3), dtype=np.float32))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, img_size) / 255.
            frames.append(frame)
        cap.release()
        return np.stack(frames, axis=0)

    return np.zeros((seq_len, img_size[0], img_size[1], 3), dtype=np.float32)

# =========================
# 증강
# =========================
def augment(frames):
    x = tf.convert_to_tensor(frames)
    if random.random() < AUG_FLIP_PROB:
        x = tf.image.flip_left_right(x)
    return x.numpy()

# =========================
# 데이터 제너레이터
# =========================
class FrameSequenceGenerator(Sequence):
    def __init__(self, seq_paths, labels, preprocess, training):
        self.seq_paths = seq_paths
        self.labels = labels
        self.preprocess = preprocess
        self.training = training

    def __len__(self):
        return math.ceil(len(self.seq_paths) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch = self.seq_paths[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
        X, y = [], []
        for i, path in enumerate(batch):
            arr = load_sequence_fixed(path, SEQ_LEN, IMG_SIZE)
            if self.training:
                arr = augment(arr)
            arr = self.preprocess(arr * 255.0)
            X.append(arr)
            y.append(self.labels[idx*BATCH_SIZE + i])
        return np.stack(X, axis=0), np.array(y)

# =========================
# Temporal Attention
# =========================
class TemporalAttention(layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1)

    def call(self, x):
        h = self.dense(x)
        e = self.score(h)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(a * x, axis=1)

# =========================
# 모델 정의
# =========================
def build_model(num_classes):
    cnn = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=IMG_SIZE+(3,), pooling="avg")
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    cnn.trainable = False

    inputs = layers.Input(shape=(SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.TimeDistributed(cnn)(inputs)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = TemporalAttention(128)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(LR_BASE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model, preprocess, cnn

# =========================
# 학습 실행
# =========================
def main():
    seq_paths = []
    classes = sorted(os.listdir(DATASET_ROOT))
    labels = []

    for cls in classes:
        cls_dir = os.path.join(DATASET_ROOT, cls)
        items = glob.glob(cls_dir + "/**", recursive=True)
        seq_paths.extend(items)
        labels.extend([cls] * len(items))

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    X_train, X_val, y_train, y_val = train_test_split(seq_paths, y, test_size=VAL_SPLIT, stratify=y, random_state=SEED)

    model, preprocess, backbone = build_model(num_classes)

    train_gen = FrameSequenceGenerator(X_train, y_train, preprocess, True)
    val_gen   = FrameSequenceGenerator(X_val, y_val, preprocess, False)

    # ✅ Keras 포맷으로 저장
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        "best_cnn_lstm_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[ckpt], verbose=1)

    # ✅ 최종 저장
    model.save("cnn_lstm_model.keras")

    with open("labels.txt", "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")

    print("✅ Saved: cnn_lstm_model.keras, labels.txt")

if __name__ == "__main__":
    main()
