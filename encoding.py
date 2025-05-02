import face_recognition
import os
import pickle

# Folder yang berisi gambar wajah yang akan diencoding
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"

def encode_faces():
    known_face_encodings = []
    known_face_names = []

    # Pastikan folder diketahui ada
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"❌ Folder '{KNOWN_FACES_DIR}' tidak ditemukan.")
        return

    # Iterasi melalui semua file di folder
    for filename in os.listdir(KNOWN_FACES_DIR):
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        
        # Pastikan hanya memproses file gambar
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"⚠️ Lewati file bukan gambar: {filename}")
            continue

        try:
            # Load gambar dan proses encoding
            print(f"🔄 Memproses file: {filename}")
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            # Jika tidak ditemukan wajah, abaikan file
            if not encodings:
                print(f"❌ Tidak ada wajah ditemukan dalam file: {filename}")
                continue

            # Gunakan encoding pertama
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])  # Nama file tanpa ekstensi

            print(f"✅ Berhasil menambahkan wajah: {filename}")
        except Exception as e:
            print(f"❌ Gagal memproses file {filename}: {e}")

    # Simpan encoding ke file
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)

    print(f"📁 Semua encoding berhasil disimpan di '{ENCODINGS_FILE}'")

if __name__ == "__main__":
    encode_faces()
