from flask import Flask, request, jsonify, render_template
import whisper
import deepl
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import http.client
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

load_dotenv()

# Initialize Whisper and DeepL
whisper_model = whisper.load_model("large-v3-turbo")
deepl_api_token = os.getenv("DEEPL_API_KEY")
if not deepl_api_token:
    raise ValueError("DEEPL_API_KEY environment variable is not set")
translator = deepl.Translator(deepl_api_token)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        title = request.form.get('title')
        language = request.form.get('language')
        genre = request.form.get('genre')
        file = request.files.get('file')

        # feltöltött fájl elhelyezése
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        else:
            return jsonify({"error": "No file uploaded"}), 400
        
        print(f"Received file: {file_path}")

        # szöveg kinyerése
        extracted_lyrics = whisper_model.transcribe(file_path)
        original_lyrics = extracted_lyrics["text"]

        print(f"Extracted lyrics.")
        print(extracted_lyrics)

        # szöveg fordítása
        translated_lyrics = translator.translate_text(original_lyrics, target_lang=language)

        print(f"Translated lyrics.")
        print(translated_lyrics)

        musicapi_key = os.getenv("MUSICAPI_API_KEY")
        if not musicapi_key:
            raise ValueError("MUSICAPI_API_KEY environment variable is not set")
        
        # zene generálása

        musicapi_key = os.getenv("MUSICAPI_API_KEY")

        conn = http.client.HTTPSConnection("api.musicapi.ai")

        payload = json.dumps({
            "custom_mode": True,
            "prompt": translated_lyrics,
            "title": title,
            "tags": genre,
            "negative_tags": "piano",
            "gpt_description_prompt": "",
            "make_instrumental": False,
            "mv": "sonic-v3-5"
        })
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer 3d97324dd8c80b0a12b888ec9decfab8'
        }
        conn.request("POST", "/api/v1/sonic/create", payload, headers)
        res = conn.getresponse()
        data = res.read()

        # task id kinyerése az api válaszból
        response_json = json.loads(data)
        if response_json.get("code") == 200 and "data" in response_json:
            task_id = response_json["data"].get("task_id")
            if not task_id:
                return jsonify({"error": "Task ID not found in response"}), 400
        else:
            return jsonify({"error": "Invalid API response"}), 400


        # zene link lekérdezése
        payload = ''
        headers = {
            'Authorization': 'Bearer 3d97324dd8c80b0a12b888ec9decfab8'
        }
        conn.request("GET", f"/api/v1/sonic/task/{task_id}", payload, headers)
        musicapires = conn.getresponse()
        musicapidata = musicapires.read()

        musicapi_json = json.loads(musicapidata)

        if musicapi_json.get("code") == 200 and "data" in musicapi_json:
            audio_url = next((item["audio_url"] for item in musicapi_json["data"] if "audio_url" in item), None)
            if audio_url:
                return jsonify({"audio_url": audio_url})
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500
    

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        if file:
            os.remove(file_path)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)