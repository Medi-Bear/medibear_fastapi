import os, json, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# ------------------------
# 경로 설정
# ------------------------
MODEL_PATH = "cnn_lstm_model_stronger.h5"
LABEL_PATH = "../labels.txt"
DATASET_ROOT = "./dataset"
IMG_SIZE = (160, 160)
SEQ_LEN = 24
BATCH_SIZE = 8

# ------------------------
# 라벨 로드
# ------------------------
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

print(f"[INFO] Loaded labels: {class_names}")
# -*- coding: utf-8 -*-
import os, glob, json, math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# 설정(필요시 경로만 맞추세요)
# ------------------------------
DATASET_ROOT = "./dataset"
MODEL_PATH   = "best_cnn_lstm_model.h5"          # or "cnn_lstm_model_stronger.h5"
LABEL_PATH   = "labels.txt"

IMG_SIZE     = (160, 160)
SEQ_LEN      = 24
BATCH_SIZE   = 8
SEED         = 42
TEST_SPLIT   = 0.15

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------
# 커스텀 레이어(직렬화 지원)
# ------------------------------
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
class TemporalAttention(layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, activation=None)

    def call(self, x):
        h = self.dense(x)
        e = self.score(h)                  # (B, T, 1)
        a = tf.nn.softmax(e, axis=1)       # (B, T, 1)
        return tf.reduce_sum(a * x, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# ------------------------------
# 유틸: 데이터셋 탐색
# ------------------------------
def list_sequence_dirs_and_labels(root_dir):
    """
    dataset/
      classA/
        seq1/ (이미지 프레임)
        seq2.mp4
      classB/
        seqX/ ...
    """
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    seq_paths, labels = [], []

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    for class_name in sorted(os.listdir(root_dir)):
        cpath = os.path.join(root_dir, class_name)
        if not os.path.isdir(cpath) or class_name.startswith("."):
            continue

        # 1) 클래스 폴더 바로 아래의 동영상 파일
        for fname in sorted(os.listdir(cpath)):
            fpath = os.path.join(cpath, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(video_exts):
                seq_paths.append(fpath); labels.append(class_name)

        # 2) 하위 폴더(프레임 폴더)
        for sub in sorted(os.listdir(cpath)):
            spath = os.path.join(cpath, sub)
            if not os.path.isdir(spath) or sub.startswith("."):
                continue
            has_media = False
            for ext in image_exts + video_exts:
                if glob.glob(os.path.join(spath, f"*{ext}")):
                    has_media = True
                    break
            if has_media:
                seq_paths.append(spath); labels.append(class_name)

    print(f"[INFO] Found {len(seq_paths)} sequences across {len(set(labels))} classes.")
    return seq_paths, labels

# ------------------------------
# 유틸: 시퀀스 로딩(고정 프레임 샘플링)
# ------------------------------
def _sample_indices(n, k):
    if n <= 0:
        return np.zeros((k,), dtype=int)
    if n >= k:
        return np.linspace(0, n - 1, num=k).astype(int)
    base = np.arange(n)
    pad  = np.full((k - n,), n - 1)
    return np.concatenate([base, pad])

def _resize_np(img, size):
    return tf.image.resize(img, size).numpy().astype(np.float32)

def _load_frames_from_dir(dir_path):
    paths = sorted([p for p in glob.glob(os.path.join(dir_path, "*.*"))
                    if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))])
    return paths

def load_sequence_fixed(seq_path: str, seq_len: int, img_size=(160,160)) -> np.ndarray:
    import cv2
    if os.path.isdir(seq_path):
        img_paths = _load_frames_from_dir(seq_path)
        idxs = _sample_indices(len(img_paths), seq_len)
        frames = []
        for i in idxs:
            if len(img_paths) == 0:
                frames.append(np.zeros((img_size[0], img_size[1], 3), dtype=np.float32))
            else:
                from PIL import Image
                with Image.open(img_paths[i]) as im:
                    im = im.convert("RGB")
                    frames.append(_resize_np(np.array(im), img_size) / 255.0)
        return np.stack(frames, axis=0)

    video_exts = (".mp4",".avi",".mov",".mkv",".wmv")
    if os.path.isfile(seq_path) and seq_path.lower().endswith(video_exts):
        cap = cv2.VideoCapture(seq_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idxs = _sample_indices(total, seq_len)
        frames, cur = [], -1
        for target in idxs:
            if cur != target:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            ok, frame = cap.read()
            if not ok:
                frames.append(np.zeros((img_size[0], img_size[1], 3), dtype=np.float32))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = _resize_np(frame, img_size) / 255.0
            frames.append(frame)
            cur = target
        cap.release()
        return np.stack(frames, axis=0)

    # fallback
    return np.zeros((seq_len, img_size[0], img_size[1], 3), dtype=np.float32)

# ------------------------------
# 제너레이터(평가 전용: training=False)
# ------------------------------
class FrameSequenceGenerator(Sequence):
    def __init__(self, seq_paths, labels, batch_size, seq_len, img_size=(160,160),
                 shuffle=False, training=False, preprocess=None, **kwargs):
        super().__init__(**kwargs)
        self.seq_paths = list(seq_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.img_size = img_size
        self.shuffle = shuffle
        self.training = training
        self.preprocess = preprocess
        self.indices = np.arange(len(self.seq_paths))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.seq_paths) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        X, y = [], []
        for i in batch_idx:
            arr = load_sequence_fixed(self.seq_paths[i], self.seq_len, self.img_size)
            # 평가 전용이므로 증강 없음
            if self.preprocess is not None:
                arr = self.preprocess(arr * 255.0)
            X.append(arr); y.append(self.labels[i])
        X = np.stack(X, axis=0)
        y = np.array(y)
        return X, y

# ------------------------------
# 메인
# ------------------------------
def main():
    print("[CWD]", os.getcwd())

    # 1) 라벨
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(f"labels file not found: {LABEL_PATH}")
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    num_classes = len(class_names)
    print(f"[INFO] Loaded labels ({num_classes}): {class_names}")

    # 2) 모델 로드 (커스텀 레이어 매핑)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"TemporalAttention": TemporalAttention}
    )
    print(f"[INFO] Model loaded from {MODEL_PATH}")

    # 3) 데이터스플릿: test 만 구성 (학습 때와 동일 split 로직)
    seq_paths, label_names = list_sequence_dirs_and_labels(DATASET_ROOT)
    le = LabelEncoder()
    y_all = le.fit_transform(label_names)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        seq_paths, y_all, test_size=TEST_SPLIT, random_state=SEED, stratify=y_all
    )

    # 4) 제너레이터 (EfficientNet 전처리 가정)
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    test_gen = FrameSequenceGenerator(
        X_test, y_test, BATCH_SIZE, SEQ_LEN, IMG_SIZE,
        shuffle=False, training=False, preprocess=preprocess
    )

    steps = len(test_gen)

    # 5) 평가 + 예측
    print("\n[TEST] Evaluating on test set ...")
    test_loss, test_acc = model.evaluate(test_gen, steps=steps, verbose=1)
    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")

    y_prob = model.predict(test_gen, steps=steps, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    ys = []
    for i in range(steps):
        _, yb = test_gen[i]
        ys.append(yb)
    y_true = np.concatenate(ys, axis=0)[:len(y_pred)]

    # 6) 리포트/혼동행렬 저장
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    with open("test_report_stronger.txt", "w", encoding="utf-8") as f:
        f.write(report)
    np.savetxt("test_confusion_matrix_stronger.txt", cm, fmt="%d")

    print("\n[TEST] Classification Report\n", report)
    print("[TEST] Confusion Matrix\n", cm)

    print("\n✅ Saved files:")
    for p in [
        os.path.abspath("test_report_stronger.txt"),
        os.path.abspath("test_confusion_matrix_stronger.txt"),
    ]:
        print(" -", p)

if __name__ == "__main__":
    main()

# ------------------------
# 모델 로드
# ------------------------
from tensorflow.keras.models import load_model
from train_cnn_lstm_model_stronger import TemporalAttention  # ✅ 실제 클래스 import

model = load_model(
    "best_cnn_lstm_model.h5",
    custom_objects={"TemporalAttention": TemporalAttention}  # ✅ 진짜 매핑
)

print("[INFO] Model loaded.")

# ------------------------
# 데이터 불러오기 (테스트만)
# ------------------------
seq_paths, label_names = list_sequence_dirs_and_labels(DATASET_ROOT)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(label_names)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    seq_paths, y, test_size=0.15, random_state=42, stratify=y
)

test_gen = FrameSequenceGenerator(
    X_test, y_test, BATCH_SIZE, SEQ_LEN, IMG_SIZE,
    shuffle=False, training=False,
    preprocess=tf.keras.applications.efficientnet.preprocess_input
)

# ------------------------
# 예측 및 리포트
# ------------------------
print("[TEST] Evaluating ...")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")

y_prob = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

ys = []
for i in range(len(test_gen)):
    _, yb = test_gen[i]
    ys.append(yb)
y_true = np.concatenate(ys, axis=0)[:len(y_pred)]

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
cm = confusion_matrix(y_true, y_pred)

with open("test_report_stronger.txt", "w", encoding="utf-8") as f:
    f.write(report)
np.savetxt("test_confusion_matrix_stronger.txt", cm, fmt="%d")

print("\n✅ Saved:")
print("- test_report_stronger.txt")
print("- test_confusion_matrix_stronger.txt")
