import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import os

# Buat folder hasil
os.makedirs('hasil_erosi_dilasi', exist_ok=True)

# Upload gambar
uploaded = files.upload()
filename = next(iter(uploaded))
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Konversi ke gambar biner (hitam-putih)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Definisi 9 jenis struktur elemen
strels = {
    'Rect_3x3': cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    'Rect_5x5': cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    'Ellipse_3x3': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    'Ellipse_5x5': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    'Cross_3x3': cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
    'Cross_5x5': cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
    'Line_Horizontal': cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
    'Line_Vertical': cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)),
    'Diagonal_Cross': np.array([[1,0,1],[0,1,0],[1,0,1]], dtype=np.uint8)
}

# Fungsi untuk menampilkan dan menyimpan hasil
def show_save_erode_dilate(img, strels):
    fig, axes = plt.subplots(len(strels), 3, figsize=(12, len(strels)*2))
    summary = []

    for i, (name, strel) in enumerate(strels.items()):
        eroded = cv2.erode(binary, strel)
        dilated = cv2.dilate(binary, strel)

        # Simpan hasil erosi dan dilasi
        eroded_path = f'hasil_erosi_dilasi/{name}_eroded.png'
        dilated_path = f'hasil_erosi_dilasi/{name}_dilated.png'
        cv2.imwrite(eroded_path, cv2.bitwise_not(eroded))  # balik warna ke putih-hitam
        cv2.imwrite(dilated_path, cv2.bitwise_not(dilated))

        # Perbandingan: luas area putih (foreground)
        area_input = np.sum(binary == 255)
        area_eroded = np.sum(eroded == 255)
        area_dilated = np.sum(dilated == 255)

        summary.append({
            'Strel': name,
            'Input Area': area_input,
            'Eroded Area': area_eroded,
            'Dilated Area': area_dilated,
            'Erosion Loss': area_input - area_eroded,
            'Dilation Gain': area_dilated - area_input
        })

        # Tampilkan
        axes[i, 0].imshow(binary, cmap='gray')
        axes[i, 0].set_title(f'Input')

        axes[i, 1].imshow(eroded, cmap='gray')
        axes[i, 1].set_title(f'Erosi - {name}')

        axes[i, 2].imshow(dilated, cmap='gray')
        axes[i, 2].set_title(f'Dilasi - {name}')

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

    return summary

# Jalankan proses dan simpan analisa
analysis = show_save_erode_dilate(binary, strels)

# Tampilkan analisa sebagai tabel
import pandas as pd
df = pd.DataFrame(analysis)
display(df)

# Zip hasil dan kirim ke user
import shutil
shutil.make_archive('hasil_erosi_dilasi_zip', 'zip', 'hasil_erosi_dilasi')
files.download('hasil_erosi_dilasi_zip.zip')
