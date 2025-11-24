from flask import Flask, request, jsonify, render_template, session
import whisper
import deepl
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import http.client
import json
from agent import MusicAgentChat
from datetime import timedelta

app = Flask(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'uploads')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

load_dotenv()

whisper_model = whisper.load_model("large-v3-turbo")
deepl_api_token = os.getenv("DEEPL_API_KEY")
if not deepl_api_token:
    raise ValueError("DEEPL_API_KEY environment variable is not set")
translator = deepl.Translator(deepl_api_token)

agent = MusicAgentChat()

def generate_music_from_file(file_path, title, language, genre):
    """
    Generate music using the MusicAPI with uploaded audio file.
    
    Args:
        file_path: Path to the uploaded audio file
        title: Song title
        language: Target language for lyrics
        genre: Music genre
        
    Returns:
        dict: Contains audio_url on success or error message
    """
    try:
        extracted_lyrics = whisper_model.transcribe(file_path)
        original_lyrics = extracted_lyrics["text"]
        print(f"Extracted lyrics: {original_lyrics}")

        translated_lyrics = translator.translate_text(original_lyrics, target_lang=language)
        print(f"Translated lyrics: {translated_lyrics}")

        musicapi_key = os.getenv("MUSICAPI_API_KEY")
        if not musicapi_key:
            raise ValueError("MUSICAPI_API_KEY environment variable is not set")

        conn = http.client.HTTPSConnection("api.musicapi.ai")

        payload = json.dumps({
            "custom_mode": True,
            "prompt": str(translated_lyrics),
            "title": title,
            "tags": genre,
            "negative_tags": "piano",
            "gpt_description_prompt": "",
            "make_instrumental": False,
            "mv": "sonic-v3-5"
        })

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {musicapi_key}'
        }

        conn.request("POST", "/api/v1/sonic/create", payload, headers)
        res = conn.getresponse()
        data = res.read()

        response_json = json.loads(data)
        if response_json.get("code") == 200 and "data" in response_json:
            task_id = response_json["data"].get("task_id")
            if not task_id:
                return {"error": "Task ID not found in response"}
        else:
            return {"error": f"Invalid API response: {response_json}"}

        payload = ''
        conn.request("GET", f"/api/v1/sonic/task/{task_id}", payload, headers)
        musicapires = conn.getresponse()
        musicapidata = musicapires.read()

        musicapi_json = json.loads(musicapidata)

        if musicapi_json.get("code") == 200 and "data" in musicapi_json:
            audio_url = next((item["audio_url"] for item in musicapi_json["data"] if "audio_url" in item), None)
            if audio_url:
                return {"audio_url": audio_url, "success": True}

        return {"error": "Could not retrieve audio URL from API"}

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in generate_music_from_file: {error_details}")
        return {"error": str(e), "details": error_details}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle agent chat messages and manage conversation state.
    """
    try:
        message = request.form.get('message', '')
        state_str = request.form.get('state', '{}')
        file = request.files.get('file')

        try:
            state = json.loads(state_str)
        except json.JSONDecodeError:
            state = {}

        if 'history' not in state:
            state['history'] = []

        confirm_action = request.form.get('confirm_action')
        confirm_param = request.form.get('confirm_param')
        confirm_value = request.form.get('confirm_value')
        if confirm_action and confirm_param:
            if confirm_action.lower() == 'yes' and confirm_value:
                state[confirm_param] = {
                    'value': confirm_value,
                    'source': 'user',
                    'confidence': 0.99,
                    'confirmed': True
                }
            state.pop('pending_confirmation', None)

        file_path = None
        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            state['file_uploaded'] = True
            state['filename'] = filename

        state = agent.update_state(state, message)

        pending = state.get('pending_confirmation')
        if pending:
            param = pending.get('param')
            val = pending.get('value')
            conf = pending.get('confidence', 0.0)
            state['confirmation_asked_for'] = {'param': param, 'value': val, 'confidence': conf}
            human_msgs = {
                'language': f"It sounds like you might want the language to be '{val}'. Is that right?",
                'title': f"Do you want to use the title '{val}'?",
                'genre': f"Would you like the genre to be '{val}'?"
            }
            reply = human_msgs.get(param, f"I think the {param} could be '{val}'. Should I use this?")
            return jsonify({
                'reply': reply,
                'action': 'request_confirmation',
                'state': state,
                'confirm': {
                    'param': param,
                    'value': val,
                    'confidence': conf
                }
            })
        response_data = agent.chat(message, state.get('history', []))

        state['history'].append({
            "role": "user",
            "content": message
        })
        state['history'].append({
            "role": "assistant",
            "content": response_data.get('reply', '')
        })

        action = response_data.get('action', 'chat')

        def has_param(s, p):
            v = s.get(p)
            if not v:
                return False
            if isinstance(v, dict):
                return bool(v.get('value'))
            return True

        missing = [p for p in ('title', 'genre', 'language') if not has_param(state, p)]
        if missing and action != 'generate_music' and response_data.get('action') != 'request_param':
            ask_for = missing[0]
            friendly = {
                'title': "the song title",
                'genre': "the music genre",
                'language': "the target language for lyrics"
            }
            return jsonify({
                'reply': f"I don't have {friendly.get(ask_for)} yet. What should {friendly.get(ask_for)} be?",
                'action': 'request_param',
                'state': state,
                'params': {
                    'title': state.get('title'),
                    'language': state.get('language'),
                    'genre': state.get('genre')
                }
            })

        if action == 'generate_music' and file_path:
            def extract_value(s, default):
                if not s:
                    return default
                if isinstance(s, dict):
                    return s.get('value') or default
                return s

            title = extract_value(state.get('title'), 'Generated Song')
            language = extract_value(state.get('language'), 'EN')
            genre = extract_value(state.get('genre'), 'pop')

            generation_result = generate_music_from_file(file_path, title, language, genre)

            if 'audio_url' in generation_result:
                return jsonify({
                    'reply': f"🎵 Success! Your song '{title}' has been generated! Here it is: {generation_result['audio_url']}",
                    'action': 'music_generated',
                    'state': state,
                    'audio_url': generation_result['audio_url']
                })
            else:
                return jsonify({
                    'reply': f"Sorry, there was an error generating the music: {generation_result.get('error', 'Unknown error')}",
                    'action': 'error',
                    'state': state
                })

        return jsonify({
            'reply': response_data.get('reply', 'I did not understand. Could you rephrase?'),
            'action': action,
            'state': state,
            'params': {
                'title': state.get('title'),
                'language': state.get('language'),
                'genre': state.get('genre')
            }
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in chat endpoint: {error_details}")
        return jsonify({
            'reply': f"An error occurred: {str(e)}",
            'action': 'error',
            'state': state if 'state' in locals() else {}
        }), 500


def legacy_generate():
    try:
        title = request.form.get('title')
        language = request.form.get('language')
        genre = request.form.get('genre')
        file = request.files.get('file')

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)
        else:
            return jsonify({"error": "No file uploaded"}), 400

        print(f"Received file: {file_path}")

        extracted_lyrics = whisper_model.transcribe(file_path)
        original_lyrics = extracted_lyrics["text"]

        print(f"Extracted lyrics.")
        print(extracted_lyrics)

        translated_lyrics = translator.translate_text(original_lyrics, target_lang=language)

        print(f"Translated lyrics.")
        print(translated_lyrics)

        musicapi_key = os.getenv("MUSICAPI_API_KEY")
        if not musicapi_key:
            raise ValueError("MUSICAPI_API_KEY environment variable is not set")

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
            'Authorization': f'Bearer {musicapi_key}'
        }
        conn.request("POST", "/api/v1/sonic/create", payload, headers)
        res = conn.getresponse()
        data = res.read()

        response_json = json.loads(data)
        if response_json.get("code") == 200 and "data" in response_json:
            task_id = response_json["data"].get("task_id")
            if not task_id:
                return jsonify({"error": "Task ID not found in response"}), 400
        else:
            return jsonify({"error": "Invalid API response"}), 400

        payload = ''
        headers = {
            'Authorization': f'Bearer {musicapi_key}'
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
        return jsonify({"error": "Unknown error retrieving audio URL"}), 500

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in legacy_generate: {error_details}")
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if 'file_path' in locals() and file_path:
                os.remove(file_path)
        except:
            pass

@app.route('/generate', methods=['POST'])
def generate():
    """Legacy generate endpoint - use /chat for the new agentic interface."""
    return legacy_generate()

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)