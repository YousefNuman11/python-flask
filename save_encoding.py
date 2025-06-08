# save_encodings.py
import face_recognition
import os
import pickle

known_faces = {}
base_folder = "known_faces"

for student_id in os.listdir(base_folder):
    student_folder = os.path.join(base_folder, student_id)

    if not os.path.isdir(student_folder):
        continue  # skip files, we only want directories

    encodings = []

    for filename in os.listdir(student_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(student_folder, filename)
            print(f"Processing: {student_id}/{filename}")

            image = face_recognition.load_image_file(image_path)
            face_encs = face_recognition.face_encodings(image)

            if face_encs:
                encodings.append(face_encs[0])
                print(f"[+] Encoded {filename}")
            else:
                print(f"[!] No face found in {filename}")

    if encodings:
        known_faces[student_id] = encodings
        print(f"[✅] {student_id}: {len(encodings)} encodings saved")

# Save to known_faces/encodings.pkl
with open("known_faces/encodings.pkl", "wb") as f:
    pickle.dump(known_faces, f)

print("✅ All encodings saved to known_faces/encodings.pkl")
