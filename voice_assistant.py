import os
import time
from dotenv import load_dotenv
from speech_recognition import Microphone, Recognizer, UnknownValueError, WaitTimeoutError
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# --- Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
recognizer = Recognizer()
mic = Microphone()
chat_history = ChatMessageHistory()

# --- Prompt Chains ---
interview_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are a professional medical assistant conducting a pre-visit voice interview with a patient.

Your goal is to gather important clinical details:
- Main symptom(s)
- Onset (when it started)
- Symptom quality/severity
- What makes it better or worse
- Associated symptoms
- Relevant medical history

Speak clearly and professionally — like a trained nurse.
Avoid small talk. Ask only one question at a time, tailored to the patient’s last response.

Once enough has been gathered, say:
"I believe I understand your situation. Would you like to add anything else?"

If the patient says no, thank them and end the interview.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
interview_chain = interview_prompt | llm | StrOutputParser()

summary_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are a medical assistant. Based on the conversation so far, generate a structured summary for the doctor.

If any detail is missing, just write 'Not specified'. Do not ask for more input.

Format:
- Chief Complaint:
- Onset/Duration:
- Quality/Severity:
- Aggravating/Relieving Factors:
- Associated Symptoms:
- Relevant History:
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Summarize the patient case."),
])
summary_chain = summary_prompt | llm | StrOutputParser()

chain_with_history = RunnableWithMessageHistory(
    interview_chain,
    lambda _: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# --- Voice Transcription ---
with mic as source:
    recognizer.adjust_for_ambient_noise(source)

def listen_and_transcribe():
    with mic as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            text = recognizer.recognize_whisper(audio, model="base", language="english")
            print("You said:", text)
            return text.strip()
        except (UnknownValueError, WaitTimeoutError):
            print("Didn't catch that. Please try again.")
        except Exception as e:
            print("Error:", e)
        return ""

def is_exit_command(text):
    return any(phrase in text.lower() for phrase in [
        "no", "nothing else", "that's it", "i'm done", "no thank you", "nope", "nah", "that's all"
    ])

# --- Interview Flow ---
print("\nAssistant: Hello. Let's begin your medical interview.\n")

while True:
    user_input = ""
    for _ in range(3):
        user_input = listen_and_transcribe()
        if user_input:
            break
    if not user_input:
        print("Too many failed attempts. Exiting interview.\n")
        break

    if is_exit_command(user_input):
        print("\nAssistant: Thank you. I’ll now prepare your summary for the doctor.\n")
        break

    try:
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "1"}}
        )
        print("\nAssistant:", response, "\n")
    except Exception as e:
        print("Error generating assistant response:", e)

# --- Summary Generation ---
try:
    time.sleep(1)
    summary = summary_chain.invoke({"chat_history": chat_history.messages})
    print("\n--- Pre-Medical Summary ---\n")
    print(summary)
except Exception as e:
    print("Failed to generate summary:", e)
