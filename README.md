# Medical Voice Assistant 🩺

**Medical Voice Assistant** is an AI-powered voice interface that helps medical professionals interact with patient records, reference clinical protocols, or obtain expert guidance—all hands-free. Whether you're navigating EHR systems, conducting tele-consultations, or keeping up with patient data, this assistant makes it searchable *and speakable* via voice commands.

---

## Features

* **Voice-to-Text Conversion**
  Capture spoken queries or dictations using speech recognition.

* **Contextual AI Responses**
  Tap into GPT-powered insights, medical information, or patient data summaries.

* **Voice Output**
  Hear AI-generated responses via text-to-speech for a hands-free experience.

* **Multiple Interaction Modes**
  Supports clinician dictation, data lookup, conversational guidance, or protocol queries.

* **Lightweight & Easy to Run**
  Powered by Python with minimal dependencies for quick prototyping and deployment.

---

## Project Structure

```
medical_voice_assistant/
├── README.md
├── requirements.txt
├── assistant.py                # Core voice assistant logic
├── voice_assistant.py         # CLI interface for voice I/O
├── v2v.py                      # Voice-to-voice interaction loop
├── voice_to_voice_assistant.py# Speech recognition integration example
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/Im-Moazzam/medical_voice_assistant.git
cd medical_voice_assistant

python3 -m venv venv
source venv/bin/activate   # or `.\venv\Scripts\activate` on Windows
pip install -U pip
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the root directory and add your API keys:

```
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-cloud-text-to-speech-key
```

* Keys are required for GPT-based responses and text-to-speech audio output.
* For Mac (Apple Silicon) users, if you encounter audio issues, install PortAudio:

  ```bash
  brew install portaudio
  ```

---

## Quickstart

Run the voice assistant with:

```bash
python assistant.py
```

Depending on your setup:

* **`voice_assistant.py`**: launch a CLI that listens and responds.
* **`v2v.py`**: enter a continuous voice-to-voice interaction.
* **`voice_to_voice_assistant.py`**: an example of integrating speech recognition libraries.

---

## How It Works

1. **Listen**: Captures user voice using a microphone.
2. **Transcribe**: Converts speech to text.
3. **Process**: Sends the query to GPT or another LLM for processing.
4. **Respond**: Reads out the answer using text-to-speech.

---

## Requirements

* Python 3.8+
* `openai`
* `SpeechRecognition` (or compatible library)
* `pyttsx3` or Google Text-to-Speech (as configured)
* `dotenv` (for managing environment keys)

---

## Use Cases

* **Clinician Assistants** – Query medical guidelines or patient histories by voice.
* **Virtual Health Coordinators** – Voice-driven workflows for dictations and summaries.
* **Accessibility Tools** – Hands-free access to medical content for clinicians on the move.

---

## Future Improvements

* Support for **secure patient-specific data retrieval** (EHR integration)
* Integration with **medical knowledge bases** (e.g., PubMed, clinical decision support)
* Voice-driven **note-taking assistants** with structured outputs
* Multilingual voice support for diverse user populations

---

## License

Licensed under the **MIT License**—feel free to use, modify, and contribute!

---

## Contact

GitHub: [@Im-Moazzam](https://github.com/Im-Moazzam)
Email: [moazzamaleem786@gmail.com](mailto:moazzamaleem786@gmail.com)

---
