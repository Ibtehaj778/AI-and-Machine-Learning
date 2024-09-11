import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QLineEdit, QFileDialog, QMessageBox)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from transformers import pipeline
import pytesseract
from PIL import Image
from gtts import gTTS
import os

class TextProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Processor")
        self.setGeometry(100, 100, 1200, 800)
        
        self.summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail",tokenizer="google/pegasus-cnn_dailymail")
        self.qa_model = pipeline("question-answering", model="distilbert/distilbert-base-uncased",tokenizer="distilbert/distilbert-base-uncased")
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        
        self.create_layout()
        self.apply_styles()

    def create_layout(self):
        # Left panel for text input and image upload
        left_panel = QVBoxLayout()
        
        # Text input
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter or paste your text here...")
        left_panel.addWidget(QLabel("Input Text"))
        left_panel.addWidget(self.input_text)
        
        # Image upload and OCR
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.ocr_button = QPushButton("Perform OCR")
        self.ocr_button.clicked.connect(self.perform_ocr)
        
        left_panel.addWidget(QLabel("Image Upload"))
        left_panel.addWidget(self.upload_button)
        left_panel.addWidget(self.image_label)
        left_panel.addWidget(self.ocr_button)
        
        # Right panel for summarization and Q&A
        right_panel = QVBoxLayout()
        
        # Summarization
        self.summarize_button = QPushButton("Summarize")
        self.summarize_button.clicked.connect(self.summarize_text)
        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)
        
        right_panel.addWidget(QLabel("Summary"))
        right_panel.addWidget(self.summarize_button)
        right_panel.addWidget(self.summary_output)
        
        # Question Answering
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Enter your question here...")
        self.answer_button = QPushButton("Answer")
        self.answer_button.clicked.connect(self.answer_question)
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        
        right_panel.addWidget(QLabel("Question Answering"))
        right_panel.addWidget(self.question_input)
        right_panel.addWidget(self.answer_button)
        right_panel.addWidget(self.answer_output)
        
        # Add panels to main layout
        self.layout.addLayout(left_panel, 1)
        self.layout.addLayout(right_panel, 1)

    def apply_styles(self):
        # Set the dark theme
        dark_palette = """
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-size: 14px;
            }
            QTextEdit, QLineEdit {
                background-color: #3c3f41;
                border: 1px solid #646464;
                border-radius: 4px;
                padding: 2px;
            }
            QPushButton {
                background-color: #365880;
                border: 1px solid #4e78a2;
                border-radius: 4px;
                padding: 5px 15px;
                color: white;
            }
            QPushButton:hover {
                background-color: #4e78a2;
            }
            QPushButton:pressed {
                background-color: #2d4a6d;
            }
            QLabel {
                color: #bdbdbd;
                font-weight: bold;
            }
        """
        self.setStyleSheet(dark_palette)

        # Set fonts
        title_font = QFont("Arial", 16, QFont.Weight.Bold)
        for label in self.findChildren(QLabel):
            label.setFont(title_font)

        # Adjust specific widget properties
        self.image_label.setStyleSheet("background-color: #3c3f41; border: 1px solid #646464;")
        self.summary_output.setStyleSheet("background-color: #3c3f41; border: 1px solid #646464;")
        self.answer_output.setStyleSheet("background-color: #3c3f41; border: 1px solid #646464;")

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.image_path = file_name

    def perform_ocr(self):
        if hasattr(self, 'image_path'):
            try:
                image = Image.open(self.image_path)
                text = pytesseract.image_to_string(image)
                self.input_text.setText(text)
            except Exception as e:
                QMessageBox.critical(self, "OCR Error", f"An error occurred during OCR: {str(e)}")
        else:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")

    def summarize_text(self):
        text = self.input_text.toPlainText()
        if text:
            summary = self.summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            self.summary_output.setText(summary)
        else:
            QMessageBox.warning(self, "No Text", "Please enter some text to summarize.")

    def answer_question(self):
        context = self.input_text.toPlainText()
        question = self.question_input.text()
        if context and question:
            answer = self.qa_model(question=question, context=context)
            self.answer_output.setText(answer['answer'])
            tts = gTTS(text=text, lang='en')
            output_file = 'output.mp3'
            tts.save(output_file)

            if os.name == 'nt':  # Windows
                os.system(f'start {output_file}')
            elif os.name == 'posix':  # macOS
                os.system(f'afplay {output_file}')
            else:
                print("Unsupported OS for automatic audio playback.")
        else:
            QMessageBox.warning(self, "Missing Input", "Please provide both context and a question.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TextProcessorApp()
    window.show()
    sys.exit(app.exec())




