from flask import Flask, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='front-end')

# Habilitar CORS para el app
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

# Enrutamiento para archivos estaticos (HTML, CSS, JS, imagenes)
@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, port=8000)