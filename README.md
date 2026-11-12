# 🤖 JARVIS

> **⚠️ Proof of Concept — This is not a production assistant.**
> It's a raw experiment: you type, it thinks, it talks back. That's it.

---

## What it actually does

```
You type  →  LLM thinks  →  Voice speaks
```

- Takes **text input** from your terminal
- Streams the response from any **OpenAI-compatible LLM endpoint**
- Speaks the reply aloud in real-time using **local TTS** cloned from a voice sample
- Sounds like JARVIS. Kind of.

---

## What it does NOT do

- ❌ No wake word / always-on listening
- ❌ No speech-to-text (you type, not speak)
- ❌ No memory beyond the current session
- ❌ No tool use, web search, or actions
- ❌ Not production-ready in any way

---

## Stack

| Layer | Tech |
|---|---|
| LLM | Any OpenAI-compatible endpoint (configured via `.env`) |
| TTS | `pocket_tts` — local, voice-cloned from a `.wav` sample |
| Audio playback | `sounddevice` |
| Streaming | Sentence-level chunking for low-latency speech |

---

## Setup

**1. Clone & install**
```bash
pip install -r requirements.txt
```

**2. Configure your LLM endpoint**

Create a `.env` file:
```env
LLM_ENDPOINT=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name
```

**3. Add a voice sample**

Place a `.wav` file of the voice you want to clone, then run `voice.py` to export a `.safetensors` state file. Update the `VOICE` path in `main.py`.

**4. Run**
```bash
python main.py
```

---

## How it works

`main.py` streams LLM output token-by-token, splits it into sentences, and feeds each sentence into a TTS queue. A separate thread plays audio chunks as they're generated — so it starts speaking before the full response is done.

```
LLM stream → sentence splitter → TTS queue → audio playback queue → speakers
```

---

## Project structure

```
JARVIS/
├── main.py               # Entry point — the whole pipeline
├── voice.py              # Voice cloning / state export utility
├── voice_faster.py       # Benchmarking experiments
├── first_voice.wav       # Source voice sample
├── first_voice.safetensors  # Exported voice state (fast loading)
├── requirements.txt
└── .env                  # Your LLM endpoint config (not committed)
```

---

> Built as a weekend experiment to see how close you can get to a real-time AI voice assistant with local TTS and a remote LLM. The answer: pretty close, actually.
