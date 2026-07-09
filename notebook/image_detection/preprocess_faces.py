import os
import cv2
import sys
import torch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(PROJECT_ROOT)

from app.modules.image.face_helper import FaceHelper


# ==========================================================
# DEVICE SETUP (CUDA)
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n======================================")
print(f"[INFO] Using device: {DEVICE}")
print("======================================\n")


# ==========================================================
# PREPROCESS DATASET
# ==========================================================
def preprocess_dataset(dataset_path, face_helper, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    skipped = 0

    print("\n======================================")
    print(f"Processing -> {dataset_path}")
    print("======================================")

    for class_name in ["real", "fake"]:

        class_dir = os.path.join(dataset_path, class_name)

        if not os.path.exists(class_dir):
            print(f"[WARN] Missing folder: {class_dir}")
            continue

        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        print(f"\n{class_name} : {len(images)} images")

        if len(images) == 0:
            continue

        save_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(save_class_dir, exist_ok=True)

        for idx, img_name in enumerate(tqdm(images)):

            img_path = os.path.join(class_dir, img_name)

            try:
                img = cv2.imread(img_path)

                if img is None:
                    skipped += 1
                    continue

                # ==================================================
                # CUDA WORK SHOULD HAPPEN INSIDE face_helper
                # ==================================================
                face = face_helper.extract_face_image(
                    img_path,
                )

                if face is None or face.size == 0:
                    skipped += 1
                    continue

                save_path = os.path.join(
                    save_class_dir,
                    f"{idx}_{img_name}"
                )

                success = cv2.imwrite(save_path, face)

                if success:
                    saved += 1
                else:
                    skipped += 1

            except Exception as e:
                skipped += 1
                print(f"[ERROR] {img_path}")
                print(e)

    print("\n======================================")
    print("Finished Preprocessing")
    print("Saved   :", saved)
    print("Skipped :", skipped)
    print("======================================\n")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    dataset_root = os.path.join(PROJECT_ROOT, "dataset")

    # ✅ PASS DEVICE INTO FACEHELPER
    face_helper = FaceHelper(device=DEVICE)

    for split in ["train", "val", "test"]:

        dataset_path = os.path.join(dataset_root, split)

        if not os.path.exists(dataset_path):
            print(f"[WARNING] Missing: {dataset_path}")
            continue

        output_dir = os.path.join(
            PROJECT_ROOT,
            "cache_faces",
            split
        )

        preprocess_dataset(
            dataset_path,
            face_helper,
            output_dir
        )

    print("\n🚀 ALL FACE PREPROCESSING COMPLETED SUCCESSFULLY")