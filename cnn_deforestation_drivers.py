import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. PERSIAPAN DATA (Simulasi Citra Tumpukan Optik Sentinel-2 + Radar Sentinel-1)
# Simulasi 1000 gambar, 64x64 piksel, dengan 6 saluran (Channels) untuk gabungan optik & radar
X_data = np.random.rand(1000, 64, 64, 6) 

# Kategori: 0 (Pertanian), 1 (Pertambangan), 2 (Kebakaran/Banjir)
y_data = np.random.randint(0, 3, 1000)

# Memisahkan 80% data latih dan 20% data uji
split = int(0.8 * 1000)
X_train, X_test = X_data[:split], X_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

# 2. MEMBANGUN ARSITEKTUR CNN UNTUK MULTI-SENSOR
model = models.Sequential([
    # Input shape dirancang menerima 6 channel kedalaman gambar
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 6)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax') # 3 Output probabilitas kelas driver
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. TRAINING DAN EVALUASI
print("[INFO] Melatih model CNN menggunakan tumpukan citra Sentinel...")
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)

print("\n[INFO] Menguji AI pada Data Citra Baru...")
loss, akurasi = model.evaluate(X_test, y_test, verbose=0)
print(f"Tingkat Akurasi Model AI: {akurasi*100:.2f}%")