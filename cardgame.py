import cv2
import numpy as np
from keras.models import load_model
from clasifierCNN import LoadCitraTraining
import random

# Load model yang sudah dilatih
model_path = "card_classification_model.h5"
model = load_model(model_path)

# Load label dari dataset
_, _, label_folders = LoadCitraTraining("CardDataSet")

# Fungsi untuk menentukan nilai kartu berdasarkan label
card_values = {
    "ace": 14,  
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
    "jack": 11, "queen": 12, "king": 13
}

def get_card_value(card_label):
    for key, value in card_values.items():
        if key in card_label:
            return value
    return 0  # Nilai default jika tidak ditemukan

# Fungsi untuk mendeteksi kartu dan memprediksi labelnya
def detect_card():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Gagal membuka kamera.")
        return None, None

    print("Arahkan kamera ke kartu, tekan [SPACE] untuk mengambil, atau [ESC] untuk keluar.")
    detected_label = None
    detected_value = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        # Tampilkan frame
        cv2.imshow("Deteksi Kartu", frame)

        # Menampilkan label dan nilai kartu di layar
        if detected_label:
            cv2.putText(frame, f"{detected_label} (Nilai: {detected_value})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC untuk keluar
            break
        elif key == 32:  # SPACE untuk menangkap gambar
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                    if len(approx) == 4:  # Hanya deteksi kartu berbentuk persegi panjang
                        pts = approx.reshape(4, 2).astype("float32")

                        # Warping ke ukuran tetap
                        dst = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], dtype="float32")
                        M = cv2.getPerspectiveTransform(pts, dst)
                        warped = cv2.warpPerspective(frame, M, (200, 300))

                        # Preprocessing untuk prediksi
                        warped = cv2.resize(warped, (128, 128))
                        warped = warped.astype("float32") / 255.0
                        warped = np.expand_dims(warped, axis=0)

                        # Prediksi label
                        predictions = model.predict(warped)[0]
                        max_index = np.argmax(predictions)
                        detected_label = label_folders[max_index]
                        detected_value = get_card_value(detected_label)

                        print(f"Kartu terdeteksi: {detected_label} (Nilai: {detected_value})")
                        break

    cap.release()
    cv2.destroyAllWindows()
    return detected_label, detected_value

# Fungsi utama untuk menjalankan game sederhana
def card_game():
    print("=== Game Kartu Sederhana ===")
    print("1. Player vs Bot")
    print("2. Player vs Player")
    mode = int(input("Pilih mode (1/2): "))

    player1_name = input("Masukkan nama Player 1: ")
    if mode == 2:
        player2_name = input("Masukkan nama Player 2: ")
    else:
        player2_name = "Bot"

    # Meminta jumlah ronde dari player
    num_rounds = int(input("Masukkan jumlah ronde yang ingin dimainkan: "))

    print("\nAyo mulai permainan!")

    scores = {player1_name: 0, player2_name: 0}

    for round_num in range(1, num_rounds + 1):  # Bermain sesuai jumlah ronde
        print(f"\n=== Ronde {round_num} ===")

        print(f"{player1_name}, silakan deteksi kartu Anda.")
        card1_label, card1_value = detect_card()
        if not card1_label:
            print("Tidak ada kartu terdeteksi. Coba lagi!")
            continue

        if mode == 2:
            print(f"{player2_name}, silakan deteksi kartu Anda.")
            card2_label, card2_value = detect_card()
            if not card2_label:
                print("Tidak ada kartu terdeteksi. Coba lagi!")
                continue
        else:
            card2_label = random.choice(label_folders)
            card2_value = get_card_value(card2_label)
            print(f"Bot memilih kartu: {card2_label} (Nilai: {card2_value})")

        print(f"\n{player1_name} memainkan kartu: {card1_label} (Nilai: {card1_value})")
        print(f"{player2_name} memainkan kartu: {card2_label} (Nilai: {card2_value})")

        if card1_value > card2_value:
            print(f"{player1_name} memenangkan ronde ini!")
            scores[player1_name] += 1
        elif card1_value < card2_value:
            print(f"{player2_name} memenangkan ronde ini!")
            scores[player2_name] += 1
        else:
            print("Ronde ini seri!")

    print("\n=== Skor Akhir ===")
    for player, score in scores.items():
        print(f"{player}: {score}")

    if scores[player1_name] > scores[player2_name]:
        print(f"Selamat {player1_name}, Anda memenangkan permainan!")
    elif scores[player1_name] < scores[player2_name]:
        print(f"Selamat {player2_name}, Anda memenangkan permainan!")
    else:
        print("Permainan berakhir seri!")

if __name__ == "__main__":
    card_game()
