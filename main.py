from flask import Flask, render_template, jsonify, send_file
import mysql.connector
import csv
from expiry_camera import main

app = Flask(__name__)

# Database connection details
DATABASE_CONFIG = {
    "product_expiry": {
        "host": "localhost",
        "user": "root",
        "password": "grishmasuresh",
        "database": "expiry_date"
    },
}

def connect_database(table_name):
    """Connect to the respective database based on table name."""
    config = DATABASE_CONFIG.get(table_name)
    if not config:
        raise ValueError(f"No database configuration found for table: {table_name}")
    return mysql.connector.connect(**config)

@app.route('/')
def index():
    """Render the web page with action buttons."""
    return render_template('index.html')

@app.route('/expiry', methods=['GET'])
def brand_recognition():
    """Run Brand Recognition and return the result."""
    result = main(use_tesseract=False)
    return jsonify({"status": "Success", "result": result})

@app.route('/export/<table>', methods=['GET'])
def export_data(table):
    """Export data from the specified table in its corresponding database."""
    try:
        connection = connect_database(table)
        cursor = connection.cursor()

        # Fetch all data from the specified table
        query = f"SELECT * FROM {table}"
        cursor.execute(query)
        rows = cursor.fetchall()
        headers = [i[0] for i in cursor.description]

        # Write data to CSV file
        file_path = f"{table}_data.csv"
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)

        return send_file(file_path, as_attachment=True, download_name=f"{table}_data.csv")

    except mysql.connector.Error as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
