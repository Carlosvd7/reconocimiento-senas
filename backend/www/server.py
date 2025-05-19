from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

current_text = ""

@app.route("/update_letter", methods=["POST"])
def update_letter():
    global current_text
    data = request.json
    letter = data.get("letter", "")
    if letter:
        current_text += letter
        print(f"✅ Letra recibida: {letter}")
    return jsonify({"message": "Letra recibida", "current_text": current_text})

@app.route("/update_word", methods=["POST"])
def update_word():
    global current_text
    word = request.json.get("word", "")
    if word:
        current_text += word + " "
        print(f"✅ Palabra recibida: {word}")
    return jsonify({"message": "Palabra recibida", "current_text": current_text})

@app.route("/get_text", methods=["GET"])
def get_text():
    return jsonify({"current_text": current_text})

@app.route("/clear", methods=["POST"])
def clear_text():
    global current_text
    current_text = ""
    print("🧹 Texto borrado desde el frontend")
    return jsonify({"message": "Texto borrado", "current_text": current_text})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
