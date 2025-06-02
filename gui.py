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
        self.model = ImageToWordModel(model_path="Models/08_handwriting_recognition_torch/202505262127/model.onnx")

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
        self.brushSize = 5
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
        font.setPointSize(30)
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

        # Verificar el tamaño de la imagen
        width = self.image.width()
        height = self.image.height()
        print(f"Ancho: {width}, Alto: {height}")
        self.image.save("Models/pruebas/img.png")
        print("Imagen guardada como pruebas/img.png")

        # Preparar la imagen para el modelo
        input_img = self.prepare_image("Models/pruebas/img.png")

        # Hacer la predicción
        try:

            image_path = "Models\pruebas\img_variable.png"
            image = cv2.imread(image_path.replace("\\", "/"))

            prediction_text = self.model.predict(image)

            # Mostrar el resultado en la interfaz
            self.text.setText(f'{prediction_text}')
        except Exception as e:
            print(f"Error durante la predicción: {e}")

    def prepare_image(self, image_path):
        """
        Convierte una imagen de una palabra manuscrita en un formato con altura fija (64 px)
        y ancho variable, respetando la proporción original y agregando 2 píxeles de margen
        a cada lado.

        Args:
            image_path (str): Ruta de la imagen original.

        Returns:
            PIL.Image: Imagen procesada lista para convertir a tensor.
        """
        # Abrir y convertir a escala de grises
        im = Image.open(image_path).convert('L')
        image_array = np.array(im)

        # Recortar márgenes en blanco (zonas completamente blancas)
        non_empty_columns = np.where(image_array.min(axis=0) < 255)[0]
        non_empty_rows = np.where(image_array.min(axis=1) < 255)[0]

        if non_empty_columns.shape[0] > 0 and non_empty_rows.shape[0] > 0:
            im = im.crop((
                non_empty_columns[0],
                non_empty_rows[0],
                non_empty_columns[-1] + 1,
                non_empty_rows[-1] + 1
            ))

        # Obtener dimensiones originales del recorte
        orig_width, orig_height = im.size

        # Escalar la altura a 64 píxeles y ajustar ancho proporcionalmente
        target_height = 32
        new_width = int((orig_width / orig_height) * target_height)
        im = im.resize((new_width, target_height), Image.Resampling.LANCZOS).filter(ImageFilter.SHARPEN)

        # Agregar 2 píxeles de margen a izquierda y derecha
        final_width = new_width + 4  # 2 píxeles a cada lado
        new_image = Image.new('L', (final_width, target_height), 255)  # Fondo blanco
        new_image.paste(im, (2, 0))  # Pegamos imagen con margen izquierdo

        # Guardar para visualizar si querés
        new_image.save("Models/pruebas/img_variable.png")
        print(f"Imagen guardada como pruebas/img_variable.png (tamaño: {final_width}x{target_height})")

        return new_image

    def clear(self):
        print("Borrando...")
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.update()

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

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.metadata["vocab"])[0]
        return text