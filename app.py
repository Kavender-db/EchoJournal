import os
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import traceback

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("⚠️ Please set your OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Flask Setup
# ---------------------------
app = Flask(__name__)
CORS(app)  # Allow all origins for testing; you can restrict to your frontend domain

# ---------------------------
# Helper Functions
# ---------------------------
def transcribe_audio(file_path: str) -> str:
    """Transcribe audio using OpenAI Whisper API"""
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        )
    return transcript.text

def analyze_journal(transcription: str) -> str:
    """Send transcription to GPT for summary, mood analysis, and PST-based guidance"""
    today = datetime.date.today().strftime("%Y-%m-%d")

    template = """
You are an empathetic AI psychologist assistant. 
The user has shared a journal entry (transcribed from speech). 
Perform the following steps:

Prompt 1 - Acknowledge & Summarize: Briefly summarize what the user expressed in a warm, empathetic tone. 1-2 sentences.
Prompt 2 - Mood Analysis: Identify the emotional state(s) the user might be experiencing, based on their description. Be specific (e.g., anxious, frustrated, relieved, overwhelmed).
Prompt 3 - Therapeutic Response: Use evidence-based therapeutic frameworks—primarily Problem-Solving Therapy (PST)—to guide your response. 
This includes:
- Clarifying the stressor
- Breaking it into manageable parts
- Suggesting actionable coping strategies
- Encouraging reflection and positive reinforcement

Tone & Style: Always maintain empathy, non-judgment, and encouragement. 
Use simple, supportive language rather than clinical jargon. 

Format your output exactly like this:

📅 Journal Date: {today}
📝 Journal Entry Summary: ...
😊 Mood Analysis: ...
🧠 Therapeutic Guidance (PST-based): ...
🌱 Encouragement: ...
    
User Journal Entry (transcribed):
\"\"\"{transcription}\"\"\"
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = prompt | llm
    response = chain.invoke({"transcription": transcription, "today": today})
    return response.content

# ---------------------------
# Flask Routes
# ---------------------------
@app.route("/upload", methods=["POST"])
def upload_audio():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Save file temporarily
        file_path = os.path.join("temp_audio", file.filename)
        os.makedirs("temp_audio", exist_ok=True)
        file.save(file_path)

        # Transcribe
        transcription = transcribe_audio(file_path)

        # Analyze
        analysis = analyze_journal(transcription)

        # Cleanup
        os.remove(file_path)

        return jsonify({
            "transcription": transcription,
            "analysis": analysis
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Run Flask
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
