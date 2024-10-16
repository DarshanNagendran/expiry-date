import cv2
import pytesseract
import re
from datetime import datetime
import easyocr

# Set up Tesseract path (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path


# Function to preprocess the image for better OCR results
def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    gray_image = cv2.medianBlur(gray_image, 3)

    # Sharpen the image (optional, for some images it might improve OCR results)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    sharpened = cv2.filter2D(gray_image, -1, kernel)

    # Apply thresholding to make the text stand out
    _, binary_image = cv2.threshold(sharpened, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image


# Function to extract text using OCR from an image and filter out dates
def extract_dates_from_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Use EasyOCR to perform OCR on the processed image
    reader = easyocr.Reader(['en'])  # You can specify languages here
    result = reader.readtext(image_path, detail=0)  # Extract only the text without bounding boxes

    # Join the result to form a string
    extracted_text = " ".join(result)

    # Debugging: Print the extracted text
    print(f"Extracted Text (via EasyOCR):\n{extracted_text}")

    # Regular expression to match dates like DD-MM-YYYY, DD/MM/YYYY, etc.
    date_pattern = r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})'  # For numeric date formats
    month_pattern = r'(\d{1,2}\s(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s\d{4})'  # For formats like 14 JAN 2026
    numeric_date_pattern = r'(\d{6})'  # For dates without separators like 040914

    # Find all matches of the date pattern in the extracted text
    dates = (
        re.findall(date_pattern, extracted_text) +
        re.findall(month_pattern, extracted_text) +
        re.findall(numeric_date_pattern, extracted_text)  # Added numeric date extraction
    )

    # Debugging: Print extracted dates after filtering
    print(f"Extracted Dates (raw): {dates}")

    # Check for expiry-related keywords and extract the date after them
    expiry_keywords = ['EXP', 'USE BY', 'EXPIRES']
    for keyword in expiry_keywords:
        if keyword in extracted_text.upper():
            keyword_index = extracted_text.upper().index(keyword)
            text_after_keyword = extracted_text[keyword_index:].split()
            for word in text_after_keyword:
                if re.match(date_pattern, word) or re.match(month_pattern, word) or re.match(numeric_date_pattern, word):
                    return [word]  # Return the date after the keyword

    # If no expiry-related keywords found, return all detected dates
    return dates


# Function to convert a date string to a datetime object
def convert_to_datetime(date_str):
    # List of date formats to try, including textual month formats
    date_formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y",
        "%d.%m.%Y", "%m.%d.%Y", "%d-%m-%Y", "%d.%m.%y", "%d-%m-%y",
        "%d %b %Y",  # For dates like "14 JAN 2026"
        "%d%m%y"  # For dates without separators like 040914 (DDMMYY)
    ]

    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    return None


# Function to check if the date has passed or is in the future, considering only the latest date
def check_latest_date(dates):
    current_date = datetime.now()

    # Convert the list of date strings to datetime objects
    date_objects = [convert_to_datetime(date_str) for date_str in dates if convert_to_datetime(date_str) is not None]

    if not date_objects:
        return "No valid dates found."

    # Sort dates in descending order (latest first)
    date_objects.sort(reverse=True)

    # Get the latest date (first one after sorting)
    latest_date = date_objects[0]

    # Check if the latest date has passed or not
    if latest_date < current_date:
        return f"The latest date {latest_date.strftime('%d-%m-%Y')} has expired."
    else:
        return f"The latest date {latest_date.strftime('%d-%m-%Y')} is in the future."


# Main logic
if __name__ == "__main__":
    # Image path for the uploaded image
    image_path = 'images/e2.png'  # Update the path if necessary

    # Extract dates from the uploaded image
    dates = extract_dates_from_image(image_path)

    # Check only the most recent (latest) date
    result = check_latest_date(dates)

    # Print the result
    print("Date Check Result:")
    print(result)
