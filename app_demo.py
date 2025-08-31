import os
import datetime
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
# ---------------------------
# Setup
# ---------------------------
# Make sure you set your OpenAI API key in the environment:
#   export OPENAI_API_KEY="your_api_key"
# or on Windows (Powershell):
#   setx OPENAI_API_KEY "your_api_key"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("⚠️ Please set your OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

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

    # Build prompt
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
# Run Example
# ---------------------------
if __name__ == "__main__":
    # Change path to your audio file
    audio_file = "journal.m4a"  

    print("🎙️ Transcribing audio...")
    transcription = transcribe_audio(audio_file)
    print(f"\n🔹 Transcription:\n{transcription}\n")

    print("🤖 Analyzing audio with GPT...")
    analysis = analyze_journal(transcription)
    print(f"\n✅ Final Output:\n{analysis}\n")
