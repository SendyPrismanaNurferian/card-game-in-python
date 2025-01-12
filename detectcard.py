import cv2
import numpy as np
from keras.models import load_model
from clasifierCNN import LoadCitraTraining, Klasifikasi

# Load model yang sudah dilatih
model_path = "card_classification_model.h5"  # Model sudah dilatih sebelumnya
model = load_model(model_path)

# Load label dari dataset
_, _, label_folders = LoadCitraTraining("CardDataSet")

# Fungsi untuk mendeteksi kartu dan memprediksi labelnya
def detect_card_from_camera():
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Gagal membuka kamera.")
        return

    print("Tekan [ESC] untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        detected_cards = detect_and_warp_cards(frame)

        if detected_cards:
            for card, bounding_box, confidence in detected_cards:
                # Gambar bounding box di sekitar setiap kartu
                if bounding_box is not None:
                    cv2.drawContours(frame, [bounding_box], -1, (0, 255, 0), 2)

                # Tampilkan hasil prediksi di layar
                label = card[0]
                conf = card[1]
                cv2.putText(frame, f"{label} ({conf:.2f}%)", 
                            (bounding_box[0][0][0], bounding_box[0][0][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "Tidak ada kartu terdeteksi", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Tampilkan frame dari kamera
        cv2.imshow("Live Camera Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Tekan ESC untuk keluar
            print("Program selesai.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Fungsi deteksi dan warp kartu
def detect_and_warp_cards(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_cards = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Deteksi kontur dengan area > 1000 px
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:  # Kontur dengan 4 titik (persegi panjang/kartu)
                pts_array = get_sorted_corners(approx)

                # Ukuran tetap untuk gambar hasil warp
                fixed_width, fixed_height = 200, 300
                destination = np.array([[0, 0], [fixed_width - 1, 0], 
                                        [fixed_width - 1, fixed_height - 1], [0, fixed_height - 1]], dtype="float32")

                M = cv2.getPerspectiveTransform(pts_array, destination)
                warped = cv2.warpPerspective(frame, M, (fixed_width, fixed_height))

                # Prediksi kartu dan confidence
                label, confidence = classify_card(warped, label_folders)

                # Tambahkan kartu yang sudah di-warp, bounding box, dan confidence ke daftar
                detected_cards.append(((label, confidence), approx, confidence))

    return detected_cards

# Fungsi untuk menegakkan kartu agar orientasi benar
def get_sorted_corners(contour):
    contour = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # Hitung sudut
    s = contour.sum(axis=1)
    rect[0] = contour[np.argmin(s)]  # Titik kiri atas
    rect[2] = contour[np.argmax(s)]  # Titik kanan bawah

    diff = np.diff(contour, axis=1)
    rect[1] = contour[np.argmin(diff)]  # Titik kanan atas
    rect[3] = contour[np.argmax(diff)]  # Titik kiri bawah

    # Urutkan agar vertikal
    return rect

# Fungsi untuk mengklasifikasikan kartu menggunakan model
def classify_card(processed_card, label_folders):
    # Konversi kartu menjadi input untuk model
    processed_card = cv2.resize(processed_card, (128, 128)) 
    processed_card = processed_card.astype("float32") / 255.0  # Normalisasi
    processed_card = np.expand_dims(processed_card, axis=0)  # Tambahkan dimensi batch

    # Prediksi label kartu
    predictions = model.predict(processed_card)[0]
    max_index = np.argmax(predictions)
    confidence = predictions[max_index] * 100  # Konversi ke persen
    label = label_folders[max_index]

    return label, confidence

if __name__ == "__main__":
    detect_card_from_camera()