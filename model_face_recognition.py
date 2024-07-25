import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
import tensorflowjs as tfjs

# Direktori dataset
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# Pengaturan Augmentasi Gambar yang lebih agresif
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Menghitung jumlah steps per epoch
train_steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Pastikan steps_per_epoch tidak nol
if train_steps_per_epoch == 0:
    train_steps_per_epoch = 1

if validation_steps == 0:
    validation_steps = 1

# Menggunakan VGG16 sebagai base model
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))

# Membekukan layer base model
for layer in base_model.layers:
    layer.trainable = False

# Menambahkan lapisan di atas base model dengan regularisasi L2
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Kompilasi Model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Callback
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('model/face_recognition_model_vgg16.h5', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
]

# Melatih Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# Membuka beberapa layer dari base model untuk fine-tuning
for layer in base_model.layers[-8:]:  # Membuka lebih banyak layer untuk fine-tuning
    layer.trainable = True

# Kompilasi ulang model dengan learning rate yang lebih rendah
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              metrics=['accuracy'])

# Melatih Model dengan fine-tuning
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# Menyimpan Model dalam Format H5
model.save('model/face_recognition_model_vgg16_optimized.h5')

# Menghitung rata-rata akurasi
average_accuracy = sum(history.history['accuracy'] + history_fine.history['accuracy']) / (len(history.history['accuracy']) + len(history_fine.history['accuracy']))
average_val_accuracy = sum(history.history['val_accuracy'] + history_fine.history['val_accuracy']) / (len(history.history['val_accuracy']) + len(history_fine.history['val_accuracy']))

print(f"Model telah disimpan dalam format HDF5 di folder 'model'")
print(f"Rata-rata akurasi pelatihan: {average_accuracy * 100:.2f}%")
print(f"Rata-rata akurasi validasi: {average_val_accuracy * 100:.2f}%")

# Convert model to TensorFlow.js format
output_dir = 'model/tfjs_model'
tfjs.converters.save_keras_model(model, output_dir)

print(f"Model berhasil dikonversi ke TensorFlow.js dan disimpan di {output_dir}")
