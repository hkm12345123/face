from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics
from model import (
    predict_liveness,
    allowed_file,
    get_face,
    face_embeddings,
    kmeans_clustering,
    face_recognition,
)
import numpy as np
import cv2
from pathlib import Path
from tempfile import TemporaryDirectory
import uuid
from db import studentRef
from helper import current_time


app = Flask(__name__)
metrics=PrometheusMetrics(app)
metrics.start_http_server(5099)

endpoints = ("one", "two", "three", "four", "five", "error", '/api/v1/attendance','/api/v1/register' )


@app.route("/one")
def first_route():
    time.sleep(random.random() * 0.2)
    return "ok"


@app.route("/two")
def the_second():
    time.sleep(random.random() * 0.4)
    return "ok"


@app.route("/three")
def test_3rd():
    time.sleep(random.random() * 0.6)
    return "ok"


@app.route("/four")
def fourth_one():
    time.sleep(random.random() * 0.8)
    return "ok"


@app.route("/error")
def oops():
    return ":(", 500

@app.route('/api/v1/attendance',methods=['POST'])
def attendance():
    if "file" not in request.files:
        return make_response(jsonify({"message": "No file part"}), 400)
    file = request.files["file"]

    if file.filename == "":
        return make_response(jsonify({"message": "No selected file"}), 400)

    if file and allowed_file(file.filename):
        img_stream = file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face = get_face(img)
        if face is None:
            return make_response(jsonify({"message": "no face found"}), 400)

        isReal = predict_liveness(face)
        if not isReal:
            return make_response(jsonify({"message": "you're fake"}), 400)

        recognizedStudent = face_recognition(face)
        if recognizedStudent is None:
            return make_response(jsonify({"message": "Unknown face"}), 400)

        studentId, studentName = recognizedStudent

        student = studentRef.child(studentId)
        records = student.child("records").get()
        if records is None:
            records = []
        records.append(current_time())
        student.update({"records": records})

        return make_response(jsonify({"id": studentId, "name": studentName}), 200)
    else:
        return make_response(jsonify({"message": "Invalid file type"}), 400)


@app.route('/api/v1/register',methods=['POST'])
def register():
    if "file" not in request.files:
        return make_response(jsonify({"message": "No video file part"}), 400)

    file = request.files["file"]
    name = request.form.get("name")
    id = request.form.get("id")

    if name is None or name == "":
        return make_response(jsonify({"message": "Name is empty"}), 400)

    if id is None or id == "":
        return make_response(jsonify({"message": "Id is empty"}), 400)

    name = name.title()
    id = id.upper()

    if studentRef.child(id).get() is not None:
        return make_response(jsonify({"message": "Id is already exist"}), 400)

    if file.filename == "":
        return make_response(jsonify({"message": "No selected file"}), 400)

    if file and allowed_file(file.filename):
        with TemporaryDirectory() as td:
            unique_filename = str(uuid.uuid4())
            temp_filename = Path(td) / unique_filename
            file.save(temp_filename)

            embeddings = face_embeddings(str(temp_filename))

            if len(embeddings) < 100:
                return make_response(jsonify({"message": "Not enough frames"}), 400)

            centroids, labels = kmeans_clustering(embeddings)

            studentRef.child(id).set(
                {"name": name, "centroids": centroids.tolist()}
            )

        return make_response(jsonify({"message": "success"}), 200)
    else:
        return make_response(jsonify({"message": "Invalid file type"}), 400)

if __name__ == "__main__":
    app.run("0.0.0.0", 5000, threaded=True)