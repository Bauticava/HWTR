import torch
import torch.nn.functional as F
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image, ImageQt, ImageFilter
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
import numpy as np
import cv2
import Levenshtein

#from inferenceModel import *

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        # Cargar el modelo una sola vez al iniciar la GUI
        self.model = ImageToWordModel(model_path="Models/08_handwriting_recognition_torch/202502201952/model.onnx")

    def initUI(self):
        self.setWindowTitle("Reconocedor de Palabras")
        self.setFixedSize(900, 300)
        self.setWindowIcon(QtGui.QIcon('logo.png'))

        # Quit Action
        exitAct = QAction('&Quit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(qApp.quit)

        # Menubar
        menubar = self.menuBar()

        # Set up the rest of the interface
        self.init_buttons()
        self.init_text()

        # Drawing canvas
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.brushSize = 25
        self.brushColor = Qt.black
        self.lastPoint = QPoint()

        self.show()

    def init_buttons(self):
        # Clear Button
        btn_clear = QPushButton('Borrar', self)
        btn_clear.resize(180, 50)
        btn_clear.move(700, 25)
        btn_clear.setFont(QFont("Arial", 16))
        btn_clear.clicked.connect(self.clear)
        btn_clear.show()

        # Recognize Button
        btn_recognize = QPushButton('Reconocer', self)
        btn_recognize.resize(180, 50)
        btn_recognize.move(700, 75)
        btn_recognize.setFont(QFont("Arial", 16))
        btn_recognize.clicked.connect(self.recognize)
        btn_recognize.show()

    def init_text(self):
        # Predicted digit display
        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        font = self.text.font()
        font.setFamily('Rockwell')
        font.setPointSize(40)
        self.text.setFont(font)
        self.text.resize(180, 80)
        self.text.move(700, 195)
        self.text.show()

        # Display de Palabra
        label_text = QLabel('Prediccion', self)
        label_text.resize(180, 50)
        label_text.setFont(QFont('Arial', 15))
        label_text.move(700, 135)
        label_text.show()

    def recognize(self):
        print("Iniciando reconocimiento...")

        # Asegurarse de que la imagen esté en formato RGB
        imagen = self.image.convertToFormat(QImage.Format_RGB888)

        # Verificar el tamaño de la imagen
        width = imagen.width()
        height = imagen.height()
        print(f"Ancho: {width}, Alto: {height}")
        imagen.save("Models/testeo/img.png")
        print("Imagen guardada como testeo/img.png")

        # Preparar la imagen para el modelo
        input_img = self.prepare_image("Models/testeo/img.png")

        # Hacer la predicción
        try:
            prediction_text = self.model.predict(input_img)
            # Cargar el diccionario
            ruta_diccionario = "D:/RAE/RAE.txt"  # Ruta del archivo
            diccionario_palabras = cargar_diccionario(ruta_diccionario)

            # Ejemplo de corrección
            palabra_corregida = corregir_palabra(prediction_text, diccionario_palabras)

            print(f"Prediccion: {prediction_text} → Corrección: {palabra_corregida}")

            # Mostrar el resultado en la interfaz
            self.text.setText(f'{palabra_corregida}')
        except Exception as e:
            print(f"Error durante la predicción: {e}")

    def prepare_image(self, image_path):
        """
        Convierte una imagen a formato (64, 256) para el modelo ONNX.
        - Recorta el texto automáticamente.
        - Redimensiona sin distorsión, manteniendo proporciones.
        - Normaliza los valores entre 0 y 1.
        """
        im = Image.open(image_path).convert('L')  # Convertir a escala de grises

        # Convertir imagen a array numpy para detectar píxeles no blancos
        image_array = np.array(im)
        non_empty_columns = np.where(image_array.min(axis=0) < 255)[0]
        non_empty_rows = np.where(image_array.min(axis=1) < 255)[0]

        if non_empty_columns.shape[0] > 0 and non_empty_rows.shape[0] > 0:
            im = im.crop((non_empty_columns[0], non_empty_rows[0],
                          non_empty_columns[-1] + 1, non_empty_rows[-1] + 1))  # Recortar texto

        # Obtener dimensiones después del recorte
        width, height = im.size

        # Redimensionar manteniendo proporciones
        target_width, target_height = 256, 64
        aspect_ratio = width / height

        if aspect_ratio > target_width / target_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        im = im.resize((new_width, new_height), Image.Resampling.LANCZOS).filter(ImageFilter.SHARPEN)

        # Crear imagen final con fondo blanco
        new_image = Image.new('L', (target_width, target_height), 255)
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        new_image.paste(im, (left, top))
        new_image.save("Models/testeo/img2.png")
        print("Imagen guardada como testeo/img2.png")

        # Convertir la imagen a numpy array
        img_array = np.array(new_image)

        # Normalizar la imagen entre 0 y 1
        img_array = img_array / 255.0

        return img_array  # Devolver imagen en formato compatible con ONNX


    def clear(self):
        self.image.fill(Qt.white)
        self.update()
        self.textEdit.clear()

    # Eventos de dibujo
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, image: np.ndarray):
        print("Input shapes:", self.input_shapes)
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image = np.expand_dims(image,axis=-1)  # Ahora será [64, 256, 1] porque solo tiene un canal (escala de grises)
        image = np.repeat(image, 3, axis=-1)  # Repetir el canal para tener 3 (RGB) - [64, 256, 3]
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.metadata["vocab"])[0]

        return text

# Cargar palabras desde un archivo de texto
def cargar_diccionario(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        palabras = {line.strip().lower() for line in f}  # Guardar en un conjunto para búsqueda rápida
    return palabras

# Función para corregir la palabra usando Levenshtein
def corregir_palabra(palabra_predicha, diccionario):
    return min(diccionario, key=lambda palabra: Levenshtein.distance(palabra_predicha, palabra))

