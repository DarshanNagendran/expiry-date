import cv2
import pytesseract
import re
from datetime import datetime
import easyocr
import threading
import time
import numpy as np

# Set up Tesseract path (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

# Global variables to hold the current frame and result
current_frame = None
current_result = ""


# Function to preprocess the image for better OCR results
def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve contrast
    gray_image = cv2.equalizeHist(gray_image)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    # Apply dilation to make the text more distinct
    kernel = np.ones((1, 1), np.uint8)
    dilated_image = cv2.dilate(adaptive_thresh, kernel, iterations=1)

    # Apply median blur to reduce noise (optional)
    blurred_image = cv2.medianBlur(dilated_image, 3)

    return blurred_image


# Function to extract text using EasyOCR from an image and filter out dates
def extract_dates_with_easyocr(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Use EasyOCR to perform OCR on the processed image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(processed_image, detail=0)

    # Join the result to form a string
    extracted_text = " ".join(result)

    print(f"Extracted Text (via EasyOCR):\n{extracted_text}")

    return filter_dates_from_text(extracted_text)


# Function to extract text using Tesseract from an image and filter out dates
def extract_dates_with_tesseract(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Use Tesseract to perform OCR on the processed image
    extracted_text = pytesseract.image_to_string(processed_image)

    print(f"Extracted Text (via Tesseract):\n{extracted_text}")

    return filter_dates_from_text(extracted_text)


# Function to filter out dates from the extracted text
def filter_dates_from_text(extracted_text):
    date_pattern = r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})'
    month_pattern = r'(\d{1,2}\s(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s\d{4})'
    numeric_date_pattern = r'(\d{6})'

    dates = (
            re.findall(date_pattern, extracted_text) +
            re.findall(month_pattern, extracted_text) +
            re.findall(numeric_date_pattern, extracted_text)
    )

    print(f"Extracted Dates (raw): {dates}")

    expiry_keywords = ['EXP', 'USE BY', 'EXPIRES', 'exp', 'Exp']
    for keyword in expiry_keywords:
        if keyword in extracted_text.upper():
            keyword_index = extracted_text.upper().index(keyword)
            text_after_keyword = extracted_text[keyword_index:].split()
            for word in text_after_keyword:
                if re.match(date_pattern, word) or re.match(month_pattern, word) or re.match(numeric_date_pattern,
                                                                                             word):
                    return [word]

    return dates


# Function to convert a date string to a datetime object
def convert_to_datetime(date_str):
    date_formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y",
        "%d.%m.%Y", "%m.%d.%Y", "%d-%m-%Y", "%d.%m.%y", "%d-%m-%y",
        "%d %b %Y",
        "%d%m%y"
    ]

    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    return None


# Function to check if the date has passed or is in the future
def check_latest_date(dates):
    current_date = datetime.now()

    date_objects = [convert_to_datetime(date_str) for date_str in dates if convert_to_datetime(date_str) is not None]

    if not date_objects:
        return "No valid dates found."

    date_objects.sort(reverse=True)

    latest_date = date_objects[0]

    if latest_date < current_date:
        return f"The latest date {latest_date.strftime('%d-%m-%Y')} has expired."
    else:
        return f"The latest date {latest_date.strftime('%d-%m-%Y')} is in the future."


# Function to process OCR in a separate thread
def ocr_thread(use_tesseract=True):
    global current_frame, current_result
    while True:
        if current_frame is not None:
            if use_tesseract:
                dates = extract_dates_with_tesseract(current_frame)
            else:
                dates = extract_dates_with_easyocr(current_frame)
            result = check_latest_date(dates)
            current_result = result
        time.sleep(1)


# Main logic for camera input
def main(use_tesseract=True):
    global current_frame, current_result
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    threading.Thread(target=ocr_thread, args=(use_tesseract,), daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        current_frame = frame

        if current_result:
            color = (0, 255, 0) if "expired" not in current_result.lower() else (0, 0, 255)
            cv2.putText(frame, current_result, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Choose OCR engine: True for Tesseract, False for EasyOCR
    main(use_tesseract=False)  # Set to True if you want to use Tesseract
