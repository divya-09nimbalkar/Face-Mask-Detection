"""
Face Mask Detection
Professional portfolio project with training, evaluation, and live inference.
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

MODEL_FILENAME = "mask_detector.keras"
FACE_DETECTOR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMAGE_SIZE = (224, 224)
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


def resolve_model_path(model_path):
    if not os.path.splitext(model_path)[1]:
        model_path += ".keras"
    return model_path


def build_mask_detector(input_shape=(224, 224, 3), classes=2):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    head = base_model.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(classes, activation="softmax")(head)

    model = Model(inputs=base_model.input, outputs=head)

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def load_dataset(dataset_dir, image_size=IMAGE_SIZE):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. Create a folder named 'dataset' with the subfolders 'with_mask' and 'without_mask'."
        )

    data = []
    labels = []
    classes = []

    for label in sorted(os.listdir(dataset_dir)):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue
        classes.append(label)
        for image_name in os.listdir(label_path):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            image_path = os.path.join(label_path, image_name)
            image = load_img(image_path, target_size=image_size)
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)
            labels.append(label)

    if not data:
        raise ValueError("No image files were found in the dataset directory.")

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    summary = pd.DataFrame({"label": lb.inverse_transform(np.argmax(labels, axis=1))})
    print("Dataset summary:")
    print(summary["label"].value_counts())

    return data, labels, lb


def train(dataset_dir, model_path=MODEL_FILENAME):
    print(f"Loading dataset from: {dataset_dir}")
    data, labels, lb = load_dataset(dataset_dir)

    sample_count = len(labels)
    class_count = len(np.unique(np.argmax(labels, axis=1)))
    if sample_count < class_count * 2:
        raise ValueError(
            "Dataset too small for stratified split. Add at least two images per class."
        )

    test_size = 0.20
    min_test_size = class_count / sample_count
    if test_size < min_test_size:
        test_size = min_test_size
        print(
            f"Warning: dataset is small; using test_size={test_size:.2f} "
            "to preserve at least one sample per class in the test set."
        )

    train_x, test_x, train_y, test_y = train_test_split(
        data,
        labels,
        test_size=test_size,
        stratify=np.argmax(labels, axis=1),
        random_state=42,
    )

    model = build_mask_detector(classes=train_y.shape[1])

    print("Starting training...")
    history = model.fit(
        train_x,
        train_y,
        validation_data=(test_x, test_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    model_path = resolve_model_path(model_path)
    print(f"Saving model to {model_path}")
    model.save(model_path)

    predictions = model.predict(test_x, batch_size=BATCH_SIZE)
    report = classification_report(
        np.argmax(test_y, axis=1), np.argmax(predictions, axis=1), target_names=lb.classes_
    )
    print("\nEvaluation report:\n", report)

    return model, lb


def load_trained_model(model_path=MODEL_FILENAME):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved model not found: {model_path}. Train a model using --train and a dataset folder first."
        )
    return load_model(model_path)


def detect_faces(frame, face_detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces


def predict_mask(frame, model, class_names, face_detector):
    faces = detect_faces(frame, face_detector)
    results = []

    for (x, y, w, h) in faces:
        face = frame[y : y + h, x : x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, IMAGE_SIZE)
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        prob = model.predict(face)[0]
        label_idx = np.argmax(prob)
        label = class_names[label_idx]
        confidence = prob[label_idx]

        results.append(((x, y, w, h), label, confidence))

    return results


def annotate_frame(frame, predictions):
    for ((x, y, w, h), label, confidence) in predictions:
        color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)
        label_text = f"{label}: {confidence * 100:.2f}%"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame


def run_image_inference(image_path, model, class_names, face_detector):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    predictions = predict_mask(image, model, class_names, face_detector)
    output = annotate_frame(image.copy(), predictions)
    window_name = "Face Mask Detection"
    cv2.imshow(window_name, output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_inference(video_path, model, class_names, face_detector):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    while True:
        grabbed, frame = capture.read()
        if not grabbed:
            break
        predictions = predict_mask(frame, model, class_names, face_detector)
        output = annotate_frame(frame, predictions)
        cv2.imshow("Face Mask Detection", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def run_webcam_inference(model, class_names, face_detector, camera_index=0):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise ValueError("Unable to open webcam. Verify that your camera is connected.")

    while True:
        grabbed, frame = capture.read()
        if not grabbed:
            break
        predictions = predict_mask(frame, model, class_names, face_detector)
        output = annotate_frame(frame, predictions)
        cv2.imshow("Face Mask Detection", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Face mask detection training and inference")
    parser.add_argument("--train", nargs="?", const="dataset", help="Train model using a dataset directory")
    parser.add_argument("--model", default=MODEL_FILENAME, help="Path to save or load the mask detector model")
    parser.add_argument("--image", help="Run inference on an image file")
    parser.add_argument("--video", help="Run inference on a video file")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam inference")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    return parser.parse_args()


def main():
    args = parse_args()
    face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

    if args.train:
        dataset_dir = args.train if args.train != "dataset" else "dataset"
        train(dataset_dir, model_path=args.model)
        return

    model = load_trained_model(args.model)
    labels = np.array(["with_mask", "without_mask"])

    if args.image:
        run_image_inference(args.image, model, labels, face_detector)
    elif args.video:
        run_video_inference(args.video, model, labels, face_detector)
    elif args.webcam:
        run_webcam_inference(model, labels, face_detector, camera_index=args.camera_index)
    else:
        print("No action specified. Use --train, --image, --video, or --webcam.")


if __name__ == "__main__":
    main()
