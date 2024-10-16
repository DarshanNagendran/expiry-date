# Expiry Date Detection System Using OCR

This project uses OpenCV, EasyOCR, and Tesseract to capture live camera feed and detect expiry dates from product labels or packaging. The system extracts and analyzes dates using OCR, processes them, and checks if they have expired or are still valid. 

## Features

- **Real-time Camera Feed**: Captures frames from the camera to detect expiry dates.
- **OCR Integration**: Utilizes EasyOCR and Tesseract for text extraction.
- **Expiry Date Detection**: Identifies and processes dates in various formats (e.g., `DD/MM/YYYY`, `DD-MMM-YYYY`, etc.).
- **Date Validation**: Compares extracted dates with the current date to determine if a product has expired or is still valid.
- **Live Feedback on Frame**: Displays expiry status directly on the camera feed, with color-coded feedback (Green for valid dates, Red for expired).

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- EasyOCR
- PyTesseract
- Tesseract-OCR installed locally (for Windows users, set the correct path in the script)
  
### Python Packages:
Install the required dependencies using pip:

```bash
pip install opencv-python easyocr pytesseract
```

### Tesseract Installation (Windows Users):
Download and install Tesseract-OCR from here and set the `tesseract_cmd` path correctly in the script:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe
```

## How It Works
- **Preprocessing:** The camera feed is converted to grayscale and sharpened to improve text detection.
- **OCR:** The processed frame is analyzed by EasyOCR and Tesseract to extract text.
- **Date Extraction:** Using regex, the system searches for potential date formats in the text.
- **Date Validation:** The system checks the extracted dates and determines if they are expired or valid by comparing them with the current date.
- **Display on Camera Feed:** The results (expiry status) are displayed in real-time on the camera feed.

## How to Run
1. Ensure your camera is connected and functional.
2. Run the script:
```bash
python expiry_date_detection.py
```
3. The camera feed will open in a window. The system will process each frame and display the expiry status directly on the feed. Press q to quit the application.

## Example Usage
- Detects dates in various formats, such as 14/10/2024, 14-OCT-2024, or 140924 (DDMMYY).
- Identifies keywords like "EXP", "EXPIRES", "USE BY", etc., and extracts dates following these terms.

## Troubleshooting
- **Camera Not Opening:** Ensure that the correct camera index is used in cv2.VideoCapture().
- **No Dates Detected:** Check if the image quality is sufficient, and ensure proper lighting.
- **Incorrect Date Format:** Modify the date_formats list in the convert_to_datetime() function to add or adjust date formats as needed.
  
## Future Enhancements
- Support for more date formats.
- Extend to handle multiple languages in OCR.
- Optimize performance for better real-time analysis.

## License
This project is licensed under the MIT License.



