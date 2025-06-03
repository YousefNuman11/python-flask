from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import pickle
import io
import base64

app = Flask(__name__)

# Load known face encodings
with open("known_faces/encodings.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Custom function to convert distance to similarity (0 to 100)
def face_similarity(distance):
    # Assume 0.6 as the standard match threshold from face_recognition
    similarity = max(0, min(1, 1 - (distance / 0.6)))
    return similarity * 100

@app.route('/verify-face', methods=['POST'])
def verify_face():
    try:
        data = request.json
        img_data = data['image']
        img_bytes = base64.b64decode(img_data)
        image = face_recognition.load_image_file(io.BytesIO(img_bytes))

        unknown_encoding = face_recognition.face_encodings(image)
        if not unknown_encoding:
            return jsonify({"status": "fail", "message": "No face detected"}), 400

        unknown_encoding = unknown_encoding[0]

        best_match = None
        best_similarity = 0

        for student_id, known_encoding in known_faces.items():
            distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
            similarity = face_similarity(distance)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student_id

        if best_similarity >= 80:
            return jsonify({
                "status": "success",
                "student_id": best_match,
                "similarity": round(best_similarity, 2)
            }), 200
        else:
            return jsonify({
                "status": "fail",
                "message": "Face not recognized with sufficient confidence",
                "highest_similarity": round(best_similarity, 2)
            }), 403

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
