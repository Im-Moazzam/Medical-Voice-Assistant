import openai
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()

# Helper: Speak text using OpenAI's TTS
def speak(text):
    player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
    with openai.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        response_format="pcm",
        input=text,
    ) as stream:
        for chunk in stream.iter_bytes(chunk_size=1024):
            player.write(chunk)

# Check if the user is signaling to end
def is_exit_command(text):
    exit_phrases = [
        "no", "nothing else", "that's it", "i'm done", "no thank you", "nope"
    ]
    return any(phrase in text.lower() for phrase in exit_phrases)

# Set up GPT-4o with LangChain + memory
chat_history = ChatMessageHistory()

SYSTEM_PROMPT = """
You are a professional medical assistant conducting a pre-visit voice interview with a patient.

Your goal is to gather all important clinical details, including:
- The main symptom(s)
- Onset (when it started)
- Symptom quality/severity
- What makes it better or worse
- Any associated symptoms
- Relevant medical history

Speak clearly, concisely, and with a professional tone — like a trained nurse.
Avoid unnecessary friendliness, emotions, or small talk.
Ask only one question at a time, and tailor your next question based on the patient’s previous answer.

Once you believe enough has been collected, say:
"I believe I understand your situation. Would you like to add anything else?"

If the patient says no, thank them and end the interview.
"""

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

model_chain = prompt_template | ChatOpenAI(model="gpt-4o") | StrOutputParser()

chain = RunnableWithMessageHistory(
    model_chain,
    lambda _: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Microphone & recognizer
recognizer = Recognizer()
mic = Microphone()
with mic as source:
    recognizer.adjust_for_ambient_noise(source)

# Start the conversation
speak("Hello. Let's begin your medical interview.")
first_turn = True
interview_over = False

while not interview_over:
    try:
        print("Listening...")
        audio = recognizer.listen(mic, timeout=10, phrase_time_limit=15)
        user_input = recognizer.recognize_whisper(audio, model="base", language="english")
        print("You said:", user_input)

        # Check if user is signaling to end
        if is_exit_command(user_input):
            speak("Thank you. I’ll now prepare your summary for the doctor.")
            interview_over = True
            break

        # Get assistant's reply based on user input
        response = chain.invoke({"input": user_input}, config={"configurable": {"session_id": "1"}})
        print("Assistant:", response)
        speak(response)

    except UnknownValueError:
        print("Could not understand audio.")
        speak("I'm sorry, could you please repeat that?")
    except Exception as e:
        print("Error:", e)
        speak("Something went wrong. Please try again.")

# After interview is done, generate structured summary
summary_input = "Please summarize the patient's condition in a structured medical format suitable for doctors."
final_summary = chain.invoke({"input": summary_input}, config={"configurable": {"session_id": "1"}})

print("\n--- Pre-Medical Summary ---\n")
print(final_summary)
print("nothing")