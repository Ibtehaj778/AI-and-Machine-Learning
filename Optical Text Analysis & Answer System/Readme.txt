Optical Text Analysis & Answer System
Overview
The Optical Text Analysis & Answer System is designed to extract, summarize, and answer questions from text extracted from images. It leverages advanced natural language processing (NLP) models and optical character recognition (OCR) to achieve this. The system uses:

Hugging Face Model for Summarization: google/pegasus-cnn_dailymail
Hugging Face Model for Question Answering: distilbert/distilbert-base-uncased
OCR Tool: pytesseract
Features
Text Extraction: Uses pytesseract to extract text from images.
Text Summarization: Summarizes the extracted text using the google/pegasus-cnn_dailymail model.
Question Answering: Provides answers to questions based on the summarized text using the distilbert/distilbert-base-uncased model.
