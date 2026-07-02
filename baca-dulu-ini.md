# Foto Kita Blur ✌️

Aplikasi Python interaktif yang menggunakan kamera web untuk mendeteksi gestur tangan secara real-time. Jika aplikasi mendeteksi gestur "Peace" (dua jari / bentuk V), tampilan kamera akan secara otomatis menjadi blur dan menampilkan teks lucu "foto kita blur..".

Aplikasi ini menggunakan teknologi **OpenCV** untuk pemrosesan gambar dan **MediaPipe** (Hand Landmarker) untuk mendeteksi titik koordinat (landmarks) tangan dengan sangat presisi.

## Fitur Utama

- **Real-time Hand Tracking**: Melacak titik koordinat tangan secara langsung dari kamera.
- **Deteksi Gestur Jari**: Mampu menghitung jumlah jari yang ditunjukkan ke kamera.
- **Deteksi Peace Sign (✌️)**: Secara spesifik dapat mendeteksi gestur tangan "Peace" menggunakan logika posisi ruas jari.
- **Efek Otomatis (Blur)**: Tampilan akan otomatis diblur (`GaussianBlur`) jika mendeteksi dua jari atau gestur Peace.
- **Multi-Camera Support**: Tekan tombol `c` untuk beralih ke kamera lain jika ada banyak kamera yang terhubung.
- **Dynamic Resizing**: Gambar dicrop otomatis ke rasio 16:9 agar tampilan tidak distorsi/gepeng saat diresize.

## Prasyarat (Requirements)

Pastikan kamu memiliki Python terinstal. Untuk menjalankan aplikasi ini, kamu perlu menginstal beberapa library berikut:

```bash
pip install opencv-python mediapipe numpy
```

Selain itu, kamu membutuhkan file model pre-trained MediaPipe:
- `hand_landmarker.task`

Pastikan file `hand_landmarker.task` berada dalam satu folder yang sama dengan `main.py`.

## Cara Penggunaan

1. Buka terminal atau command prompt.
2. Navigasikan ke direktori project ini (`d:\fotokitablus`).
3. Jalankan script utama:

```bash
python main.py
```

4. Arahkan tanganmu ke kamera.
5. Tunjukkan dua jari (✌️) dan lihat efek blur yang muncul!

### Kontrol Keyboard

- Tekan `q` : Keluar dari aplikasi.
- Tekan `c` : Ganti / gilir kamera yang aktif.

## Cara Kerja Sistem Deteksi Jari

- Jempol dideteksi dengan mengukur jarak dari ujung jempol ke pangkal kelingking.
- Empat jari lainnya (telunjuk, tengah, manis, kelingking) dideteksi dengan membandingkan titik kordinat Y ujung jari dengan titik Y ruas jari di bawahnya.
- Efek blur akan ter-trigger apabila total jari yang terbaca berjumlah 2 atau memenuhi kondisi logika `is_peace_sign`.
