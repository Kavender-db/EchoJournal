import os
import datetime
from flask import Flask, request, jsonify
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# ---------------------------
# Setup
# ---------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("⚠️ Please set your OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

# ---------------------------
# Step 1 - Transcribe Audio
# ---------------------------
def transcribe_audio(file_path: str) -> str:
    """Transcribe audio using OpenAI Whisper API"""
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",  # Whisper model
            file=f
        )
    return transcript.text

# ---------------------------
# Step 2 - Analyze with GPT
# ---------------------------
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
@app.route("/process", methods=["POST"])
def process_audio():
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        transcription = transcribe_audio(file_path)
        analysis = analyze_journal(transcription)

        return jsonify({
            "transcription": transcription,
            "analysis": analysis
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Run Flask App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
