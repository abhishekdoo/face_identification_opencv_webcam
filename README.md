# Face Recognition with Keras + OpenCV (macOS)

## ✅ 1. Install Python 3.10 (Required for TensorFlow)

TensorFlow does not reliably support newer Python versions.

``` bash
brew install python@3.10
python3.10 --version
```

------------------------------------------------------------------------

## ✅ 2. Create Virtual Environment

``` bash
python3.10 -m venv facenet_env
source facenet_env/bin/activate
```

Your prompt should now show:

    (facenet_env)

------------------------------------------------------------------------

## ✅ 3. Install Dependencies

Install compatible versions to avoid protobuf conflicts.

``` bash
pip install tensorflow==2.20.0
pip install protobuf==5.29.0
pip install opencv-python
pip install keras-facenet
pip install mtcnn
pip install scipy
pip install numpy
```

✔ `scipy` is REQUIRED by keras-facenet

------------------------------------------------------------------------

## ✅ 4. Folder Structure

Your project MUST look like:

    face_recognition_webcam/
        webcam_face_recognition.py
        faces/
            adi/
                1.jpg
                2.jpg
            rahul/
                1.jpg
                2.jpg

Each folder = one person\
Multiple images improve accuracy

------------------------------------------------------------------------

## ✅ 5. Full Working Code

Save as:

`webcam_face_recognition.py`

``` python
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import os

FACES_DIR = "faces"
SIMILARITY_THRESHOLD = 0.6
FRAME_SCALE = 0.5

print("Loading models...")
embedder = FaceNet()
detector = MTCNN()

known_embeddings = []
known_names = []

print("Loading known faces...")

for person_name in os.listdir(FACES_DIR):

    person_path = os.path.join(FACES_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    person_embeddings = []

    for file in os.listdir(person_path):

        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        path = os.path.join(person_path, file)
        image = cv2.imread(path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)

        if not faces:
            continue

        x, y, w, h = faces[0]['box']
        x, y = abs(x), abs(y)

        face = image_rgb[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (160, 160))
        except:
            continue

        embedding = embedder.embeddings([face])[0]
        person_embeddings.append(embedding)

    if person_embeddings:
        avg_embedding = np.mean(person_embeddings, axis=0)

        known_embeddings.append(avg_embedding)
        known_names.append(person_name)

print("Loaded:", known_names)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(frame_rgb)

    for face_data in faces:

        x, y, w, h = face_data['box']
        x, y = abs(x), abs(y)

        face = frame_rgb[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (160, 160))
        except:
            continue

        embedding = embedder.embeddings([face])[0]

        name = "Unknown"
        best_score = -1

        for i, known_embedding in enumerate(known_embeddings):

            score = cosine_similarity(embedding, known_embedding)

            if score > best_score:
                best_score = score
                name = known_names[i]

        if best_score < SIMILARITY_THRESHOLD:
            name = "Unknown"

        x1 = int(x / FRAME_SCALE)
        y1 = int(y / FRAME_SCALE)
        x2 = int((x + w) / FRAME_SCALE)
        y2 = int((y + h) / FRAME_SCALE)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{name} ({best_score:.2f})"

        cv2.putText(frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

------------------------------------------------------------------------

## ✅ 6. Run Application

``` bash
python webcam_face_recognition.py
```

Press **Q** → Exit
