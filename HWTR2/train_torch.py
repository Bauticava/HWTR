import os
import csv
import sys
import random
import threading
import time

import torch
import torch.optim as optim

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage


from model import Network
from configs import ModelConfigs
from torch.utils.data import Dataset
import cv2

# Definir si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔹 Usando dispositivo:", device)

base_path = r"D:/RAE/"
dataset_path = os.path.join(base_path, "dataset6_es")
csv_file = os.path.join(dataset_path, "etiquetas.csv")

# Crear el dataset a partir del CSV
dataset = []
vocab = set()
max_len = 0

# Leer el archivo CSV
with open(csv_file, mode="r", encoding="utf-8-sig") as file:
    reader = csv.reader(file)
    count = 0  # Contador de imágenes procesadas
    for row in reader:
        image_path = os.path.join(base_path, row[0])
        label = row[1]

        # Verificar si el archivo de imagen existe
        if not os.path.exists(image_path):
            print(f"Archivo no encontrado: {image_path}")
            continue

        dataset.append([image_path, label])
        vocab.update(list(label))
        vocab.update("ñ")  # 🔥 Forzar inclusión de la letra "ñ"
        max_len = max(max_len, len(label))

        count += 1
        if count % 100 == 0:
            sys.stdout.write(f"\r📥 Se han leído {count} imágenes del CSV...")
            sys.stdout.flush()  # Forzar actualización inmediata

    print("\n✅ Lectura completada.")  # Mensaje final limpio

configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()

# class CustomDataset(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         image_path, label = self.dataset[idx]
#         return image_path, label  # Devuelve la ruta, no la imagen
#
# dataset = CustomDataset(dataset)  # Ahora es un Dataset válido

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
    use_cache=True,
)

# Split the dataset into training and validation sets
random.shuffle(dataset)
train_dataProvider, test_dataProvider = data_provider.split(split = 0.9)

# Augment training data with random brightness, rotation and erode/dilate
# train_dataProvider.augmentors = [
#     RandomBrightness(),
#     RandomErodeDilate(),
#     RandomSharpen(),
#     RandomRotate(angle=10),
#     ]5}

#network = Network(len(configs.vocab), activation="leaky_relu", dropout=0.3)
network = Network(len(configs.vocab), activation="leaky_relu", dropout=0.3).to(device)
loss = CTCLoss(blank=len(configs.vocab))
optimizer = optim.Adam(network.parameters(), lr=configs.learning_rate)

# uncomment to print network summary, torchsummaryX package is required
#summary(network, torch.zeros((1, configs.height, configs.width, 3)))

# put on cuda device if available
#if torch.cuda.is_available():
   # network = network.cuda()

# create callbacks
earlyStopping = EarlyStopping(monitor="val_CER", patience=20, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)
tb_callback = TensorBoard(configs.model_path + "/logs")
reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=10, verbose=1, mode="min", min_lr=1e-6)
model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.height, configs.width, 3),
    verbose=1,
    metadata={"vocab": configs.vocab}
    )

# create model object that will handle training and testing of the network
model = Model(network, optimizer, loss, metrics=[CERMetric(configs.vocab), WERMetric(configs.vocab)])
model.fit(
    train_dataProvider,
     ,
    epochs=12,
    callbacks=[earlyStopping, modelCheckpoint, tb_callback, reduce_lr, model2onnx]
    )

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
test_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))