import cv2
import os
import shutil
import random

# Fungsi untuk menyimpan gambar wajah
def save_face_images(name, num_images=100):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    folder_path = os.path.join('dataset/train', name)
    os.makedirs(folder_path, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (224, 224))
            file_path = os.path.join(folder_path, f"{name}_{count}.jpg")
            cv2.imwrite(file_path, face)
            cv2.putText(frame, f"Saving {count}/{num_images}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_images:
            break

    cap.release()
    cv2.destroyAllWindows()

# Panggil fungsi untuk menyimpan gambar wajah
save_face_images('Kroos')


# Fungsi split dataset
def split_dataset(train_dir, validation_dir, validation_split=0.3):
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    for category in os.listdir(train_dir):
        category_path = os.path.join(train_dir, category)
        if not os.path.isdir(category_path):
            continue

        images = os.listdir(category_path)
        random.shuffle(images)
        split_index = int(len(images) * validation_split)
        validation_images = images[:split_index]

        validation_category_path = os.path.join(validation_dir, category)
        os.makedirs(validation_category_path, exist_ok=True)

        for image in validation_images:
            src_path = os.path.join(category_path, image)
            dest_path = os.path.join(validation_category_path, image)
            shutil.move(src_path, dest_path)


# Panggil fungsi untuk memindahkan gambar ke set validasi
split_dataset('dataset/train', 'dataset/validation')
