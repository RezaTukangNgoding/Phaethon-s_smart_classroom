import cv2
import face_recognition

ESP32_CAM_URL = "http://192.168.49.143:81"

# Inisialisasi daftar wajah yang dikenal
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    face_files = ["known_faces/reza.jpg", "known_faces/nawfal.jpg", "known_faces/rasya.jpg"]
    
    for face_file in face_files:
        try:
            image = face_recognition.load_image_file(face_file)
            encodings = face_recognition.face_encodings(image)

            if not encodings:
                print(f"❌ Tidak ada wajah dalam file: {face_file}")
                continue
            
            known_face_encodings.append(encodings[0])
            known_face_names.append(face_file.split("/")[-1].split(".")[0])
            print(f"✅ Wajah dari {face_file} berhasil dimuat.")
        except Exception as e:
            print(f"❌ Gagal memuat wajah dari {face_file}: {e}")

def main():
    load_known_faces()
    stream_url = f"{ESP32_CAM_URL}/stream"
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("❌ Gagal mengakses streaming video.")
        return

    print("📹 Mengakses kamera ESP32-CAM. Tekan 'q' untuk keluar.")

    previous_status = None  # Status deteksi wajah sebelumnya
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Tidak dapat membaca frame.")
            continue

        # Mengubah frame ke format RGB
        rgb_frame = frame[:, :, ::-1]

        if rgb_frame is None or rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            print("⚠️ Frame tidak valid.")
            continue

        # Konversi ke format BGR ke RGB
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        # Deteksi lokasi wajah
        face_locations = face_recognition.face_locations(rgb_frame)

        # Cek perubahan status deteksi wajah
        if len(face_locations) == 0:
            if previous_status != "no_face":
                print("⚠️ Tidak ada wajah terdeteksi.")
                previous_status = "no_face"
            continue  # Lanjutkan ke iterasi berikutnya jika tidak ada wajah
        
        previous_status = "face_detected"

        try:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        except Exception as e:
            print(f"⚠️ Terjadi kesalahan saat encoding wajah: {e}")
            continue

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("ESP32-CAM Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
