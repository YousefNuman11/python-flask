from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import pickle
import io
import base64
import os

app = Flask(__name__)

ENCODINGS_FILE = "known_faces/encodings.pkl"

# Load known face encodings
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

# Convert distance to similarity
def face_similarity(distance):
    similarity = max(0, min(1, 1 - (distance / 0.7)))
    return similarity * 100

@app.route('/verify-face', methods=['POST'])
def verify_face():
    try:
        data = request.json
        img_data = data.get('image')
        if not img_data:
            return jsonify({"status": "fail", "message": "No image provided"}), 400

        img_bytes = base64.b64decode(img_data)
        image = face_recognition.load_image_file(io.BytesIO(img_bytes))
        unknown_encodings = face_recognition.face_encodings(image)

        if not unknown_encodings:
            return jsonify({"status": "fail", "message": "No face detected"}), 400

        unknown_encoding = unknown_encodings[0]

        best_match_id = None
        best_similarity = 0
        best_distance = None

        for student_id, encodings in known_faces.items():
            distances = face_recognition.face_distance(encodings, unknown_encoding)
            min_distance = np.min(distances)
            similarity = face_similarity(min_distance)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = student_id
                best_distance = min_distance

        if best_similarity >= 80:
            return jsonify({
                "status": "success",
                "student_id": best_match_id,
                "similarity": round(best_similarity, 2),
                "distance": round(best_distance, 4)
            }), 200
        else:
            return jsonify({
                "status": "fail",
                "message": "Face not recognized with sufficient confidence",
                "highest_similarity": round(best_similarity, 2),
                "closest_distance": round(best_distance, 4) if best_distance else None
            }), 403

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/train-face', methods=['POST'])
def train_face():
    try:
        data = request.json
        student_id = data.get("student_id")
        img_data = data.get("image")

        if not student_id or not img_data:
            return jsonify({"status": "fail", "message": "Missing student_id or image"}), 400

        img_bytes = base64.b64decode(img_data)
        image = face_recognition.load_image_file(io.BytesIO(img_bytes))
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            return jsonify({"status": "fail", "message": "No face detected"}), 400

        # Append or create new entry
        if student_id in known_faces:
            known_faces[student_id].extend(encodings)
        else:
            known_faces[student_id] = encodings

        # Save to file
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(known_faces, f)

        return jsonify({
            "status": "success",
            "message": f"{len(encodings)} face(s) added for student {student_id}"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
