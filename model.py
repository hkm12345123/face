import cv2
from keras.models import model_from_json
import numpy as np
from keras_facenet import FaceNet
from sklearn.cluster import KMeans
from db import studentRef


protoPath = "models/deploy.prototxt"
detectionPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, detectionPath)

json_file = open("models/final_80_epoch_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("models/final_80_epoch_model.h5")

embedder = FaceNet()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_liveness(face):
    _face = face.astype("float") / 255.0
    _face = np.expand_dims(_face, axis=0)

    soofingPercent = model.predict(_face)[0]
    print(f"Spoofing percent: {soofingPercent}")
    return soofingPercent < 0.5


def get_face(img):
    """
    Return a 160x160 face image
    """
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    faces = net.forward()

    if len(faces) == 0:
        return None

    i = np.argmax(faces[0, 0, :, 2])
    confidence = faces[0, 0, i, 2]

    if confidence > 0.5:
        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = img[startY:endY, startX:endX]
        face = cv2.resize(face, (160, 160))
        return face


def get_embedding(face):
    """
    Return a 512 vector
    """
    face = face.astype("float32")
    face = np.expand_dims(face, axis=0)

    yhat = embedder.embeddings(face)
    return yhat[0]


def face_embeddings(video):
    """
    Return a list of face embeddings from a video
    """
    vc = cv2.VideoCapture(video)
    embeddings = []

    while True:
        (grabbed, frame) = vc.read()

        if not grabbed:
            break

        face = get_face(frame)
        if face is None:
            continue

        embedding = get_embedding(face)
        embeddings.append(embedding)

    return embeddings


def kmeans_clustering(embeddings, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    return centroids, labels


def face_recognition(face):
    students = studentRef.get()

    if students is None:
        return None

    embedding = get_embedding(face)
    recognized_student_id = None
    recognized_student_name = None
    min_distance = float("inf")

    for id, data in students.items():
        centroids = data.get("centroids", [])
        for centroid in centroids:
            distance = np.linalg.norm(centroid - embedding)
            if distance < min_distance:
                min_distance = distance
                recognized_student_id = id
                recognized_student_name = data.get("name", [])

    if min_distance < 0.6:
        return recognized_student_id, recognized_student_name
    else:
        return None
