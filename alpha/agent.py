import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import re

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=gemini_api_key)

class MusicAgentChat:
    """
    An agentic chat interface that allows users to control the music generator
    through natural language conversation with Gemini API.
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.system_prompt = """You are an AI Music Generator Assistant. You help users create music by guiding them through a conversation.

Your capabilities:
1. Generate music from audio files with translated lyrics
2. Extract lyrics from audio using speech recognition
3. Translate lyrics to different languages
4. Create music based on title, genre, and custom lyrics

When users want to generate music, you need to collect:
- title: The title of the song
- language: The target language for lyrics (e.g., 'EN' for English, 'DE' for German, etc.)
- genre: The music genre (e.g., 'rock', 'pop', 'jazz', etc.)
- audio_file: An audio file for extracting/translating lyrics (optional if lyrics are provided directly)

Guidelines:
- Be friendly and conversational
- Guide the conversation naturally
- Ask for clarification when needed
- When you have ALL required information (title, language, genre), tell the user you're ready to generate
- If a user uploads a file, acknowledge it and proceed with music generation
- Be enthusiastic about music creation
- Keep responses concise (2-3 sentences for regular chat)

When ready to generate music (have title, language, and genre), respond naturally explaining you have what you need.

For other conversations, respond conversationally without any special formatting."""

    def chat(self, user_message, conversation_history=None):
        """
        Process user message and return agent response with action.
        
        Args:
            user_message: The user's input message
            conversation_history: List of previous messages for context
            
        Returns:
            dict: Contains 'reply', 'action', and extracted parameters
        """
        if conversation_history is None:
            conversation_history = []

        try:
            messages = []
            for msg in conversation_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "parts": [msg.get("content", "")]
                })

            full_system = f"{self.system_prompt}\n\nNote: Be direct and conversational. If the user has provided a file, acknowledge it and ask if you should proceed."

            content = [
                {"role": "user", "parts": [full_system]},
            ]

            if messages:
                content.extend(messages)

            content.append({
                "role": "user",
                "parts": [user_message]
            })

            response = self.model.generate_content(content)
            response_text = response.text

            result = self._parse_response(response_text)

            return result

        except Exception as e:
            return {
                "reply": f"I encountered an error processing your message. Please try again.",
                "action": "error",
                "error": str(e)
            }
    
    def _parse_response(self, response_text):
        """Parse the model response to extract action and parameters."""
        json_patterns = [
            r'```json\n(.*?)\n```',
            r'```(.*?)```',
            r'\{.*?"action".*?\}'
        ]

        for pattern in json_patterns:
            json_match = re.search(pattern, response_text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(1) if '```' in pattern else json_match.group(0)
                    json_str = json_str.replace('```json\n', '').replace('```', '').strip()
                    action_data = json.loads(json_str)
                    if isinstance(action_data, dict) and 'reply' in action_data:
                        return action_data
                except (json.JSONDecodeError, IndexError):
                    pass

        response_lower = response_text.lower()

        if "generate" in response_lower and any(
            phrase in response_lower for phrase in 
            ["ready", "let's create", "time to make", "music", "time to generate"]
        ):
            return {
                "action": "request_file",
                "reply": response_text
            }

        if any(phrase in response_lower for phrase in ["upload", "upload your", "provide your", "share your audio"]):
            return {
                "action": "request_file",
                "reply": response_text
            }

        return {
            "action": "chat",
            "reply": response_text
        }
    
    def extract_generation_params(self, state_data):
        """
        Extract music generation parameters from conversation state.
        
        Returns:
            dict: Contains title, language, genre, or None if incomplete
        """
        required_fields = ["title", "language", "genre"]

        params = {}
        for field in required_fields:
            if field in state_data:
                params[field] = state_data[field]

        if len(params) == len(required_fields):
            return params

        return None
    
    def update_state(self, current_state, user_message):
        """
        Update conversation state based on user message.
        Tries to extract title, language, and genre.
        """
        updated_state = current_state.copy()

        text = user_message.strip()
        if not text:
            return updated_state

        def set_field(field_name, value, source, confidence, confirmed=False):
            updated_state[field_name] = {
                'value': value,
                'source': source,
                'confidence': float(confidence),
                'confirmed': bool(confirmed)
            }

        explicit_title = re.search(r"(?:title)\s*[:\-]\s*(?P<t>.+)$", text, re.IGNORECASE) or \
            re.search(r"(?:title)\s+is\s+(?P<t>.+)$", text, re.IGNORECASE) or \
            re.match(r'^["\'](?P<t>.+?)["\']$', text)
        if explicit_title:
            title_val = explicit_title.group('t').strip()
            if title_val:
                set_field('title', title_val, 'user', 0.99, confirmed=True)

        explicit_lang = re.search(r"(?:language)\s*[:\-]\s*(?P<l>\w+)", text, re.IGNORECASE) or \
            re.search(r"(?:language)\s+is\s+(?P<l>\w+)", text, re.IGNORECASE)
        if explicit_lang:
            l = explicit_lang.group('l').strip().lower()
            language_map = {
                "english": "EN", "en": "EN",
                "german": "DE", "de": "DE",
                "deutsch": "DE",
                "french": "FR", "fr": "FR",
                "spanish": "ES", "es": "ES",
                "italian": "IT", "it": "IT",
                "portuguese": "PT", "pt": "PT",
                "russian": "RU", "ru": "RU",
                "japanese": "JA", "ja": "JA",
                "chinese": "ZH", "zh": "ZH",
                "korean": "KO", "ko": "KO",
                "hungarian": "HU", "hu": "HU",
                "magyar": "HU",
                "angol": "EN",
            }
            if l in language_map:
                set_field('language', language_map[l], 'user', 0.99, confirmed=True)

        explicit_genre = re.search(r"(?:genre)\s*[:\-]\s*(?P<g>.+)$", text, re.IGNORECASE) or \
            re.search(r"(?:genre)\s+is\s+(?P<g>.+)$", text, re.IGNORECASE)
        if explicit_genre:
            g = explicit_genre.group('g').strip().lower()
            genres_map = {
                "rock": "rock",
                "pop": "pop",
                "jazz": "jazz",
                "classical": "classical",
                "metal": "metal",
                "hip-hop": "hip-hop",
                "hip hop": "hip-hop",
                "hiphop": "hip-hop",
                "r&b": "r&b",
                "rnb": "r&b",
                "electronic": "electronic",
                "edm": "electronic",
                "folk": "folk",
                "country": "country",
                "reggae": "reggae",
                "blues": "blues",
                "indie": "indie",
                "alternative": "alternative",
                "soul": "soul",
                "funk": "funk",
                "disco": "disco",
                "techno": "techno",
                "house": "house",
                "ambient": "ambient",
                "experimental": "experimental",
            }
            if g in genres_map:
                set_field('genre', genres_map[g], 'user', 0.99, confirmed=True)

        def infer_title_from_text(s):
            q = re.search(r'"([^"]{2,200})"|\'([^\']{2,200})\'', s)
            if q:
                return q.group(1) or q.group(2), 0.95
            m = re.search(r"I (?:think|feel|believe|guess)\s*,?\s*(?:the\s+)?(?P<t>.+?)\s+(?:would be|is|could be|might be)\s+(?:a|the)?\s*title", s, re.IGNORECASE)
            if m:
                return m.group('t').strip().strip('"\''), 0.9
            m2 = re.search(r"(?:call it|let's call it|let us call it)\s+(?P<t>.+)$", s, re.IGNORECASE)
            if m2:
                return m2.group('t').strip(), 0.85
            m3 = re.search(r"(?:title|name)\s+(?:could be|is|:)?\s*(?P<t>[A-Z][A-Za-z0-9 \-]{2,80})", s)
            if m3:
                return m3.group('t').strip(), 0.75
            return None, 0.0

        def infer_genre_from_text(s):
            for g in ["rock","pop","jazz","classical","metal","hip hop","r&b","electronic","edm","folk","country","reggae","blues","indie","alternative","soul","funk","disco","techno","house","ambient","experimental"]:
                if re.search(r"\b" + re.escape(g) + r"\b", s, re.IGNORECASE):
                    return g, 0.85
            return None, 0.0

        def infer_language_from_text(s):
            for k,v in [("english","EN"),("german","DE"),("french","FR"),("spanish","ES"),("italian","IT"),("hungarian","HU"),("japanese","JA"),("chinese","ZH"),("korean","KO")]:
                if re.search(r"\b"+k+r"\b", s, re.IGNORECASE):
                    return v, 0.88
            # Important: don't treat arbitrary two-letter words (like 'Hi') as language codes.
            # Only rely on explicit language names above or explicit user fields.
            return None, 0.0

        if 'title' not in updated_state:
            val, conf = infer_title_from_text(text)
            if val:
                if conf >= 0.8:
                    set_field('title', val, 'inferred', conf, confirmed=True)
                elif conf >= 0.5:
                    # Only set pending confirmation if not already pending or previously asked
                    pending = updated_state.get('pending_confirmation')
                    asked = updated_state.get('confirmation_asked_for')
                    new_pending = {'param': 'title', 'value': val, 'confidence': conf}
                    if not (pending and pending.get('param') == 'title' and pending.get('value') == val) and not (asked and asked.get('param') == 'title' and asked.get('value') == val):
                        updated_state['pending_confirmation'] = new_pending

        if 'genre' not in updated_state:
            val, conf = infer_genre_from_text(text)
            if val:
                if conf >= 0.8:
                    set_field('genre', val, 'inferred', conf, confirmed=True)
                elif conf >= 0.5:
                    pending = updated_state.get('pending_confirmation')
                    asked = updated_state.get('confirmation_asked_for')
                    new_pending = {'param': 'genre', 'value': val, 'confidence': conf}
                    if not (pending and pending.get('param') == 'genre' and pending.get('value') == val) and not (asked and asked.get('param') == 'genre' and asked.get('value') == val):
                        updated_state['pending_confirmation'] = new_pending

        if 'language' not in updated_state:
            val, conf = infer_language_from_text(text)
            if val:
                if conf >= 0.8:
                    set_field('language', val, 'inferred', conf, confirmed=True)
                elif conf >= 0.5:
                    pending = updated_state.get('pending_confirmation')
                    asked = updated_state.get('confirmation_asked_for')
                    new_pending = {'param': 'language', 'value': val, 'confidence': conf}
                    if not (pending and pending.get('param') == 'language' and pending.get('value') == val) and not (asked and asked.get('param') == 'language' and asked.get('value') == val):
                        updated_state['pending_confirmation'] = new_pending

        return updated_state


def create_agent():
    """Factory function to create and return a MusicAgentChat instance."""
    return MusicAgentChat()
