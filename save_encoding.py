# save_encodings.py
import face_recognition
import os
import pickle

known_faces = {}
folder = "known_faces"

for filename in os.listdir(folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        student_id = os.path.splitext(filename)[0]
        image_path = os.path.join(folder, filename)

        print(f"Processing: {filename}")
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_faces[student_id] = encodings[0]
            print(f"[+] Encoded {student_id}")
        else:
            print(f"[!] No face found in {filename}")

# Save to known_faces/encodings.pkl
with open("known_faces/encodings.pkl", "wb") as f:
    pickle.dump(known_faces, f)


print("✅ All encodings saved to known_faces/encodings.pkl")
