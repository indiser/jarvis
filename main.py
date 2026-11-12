import os
import re
import queue
import threading

import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv
from openai import OpenAI
from pocket_tts import TTSModel

load_dotenv()

MODEL = os.environ.get("LLM_MODEL")
BASE_URL = os.environ.get("LLM_ENDPOINT")
VOICE = "./first_voice.safetensors"

SYSTEM_PROMPT = (
    "You are JARVIS from Avengers, a helpful assistant. "
    "Your replies are spoken aloud. Keep them short, 1 to 3 sentences. "
    "Plain text only. No markdown, no lists, no emoji."
)

client = OpenAI(base_url=BASE_URL, api_key="no-api")

torch.set_num_threads(1)  # 1 thread was fastest in your benchmark
tts = TTSModel.load_model()
voice_state = tts.get_state_for_audio_prompt(VOICE)
tts.generate_audio(voice_state, "Warmup.")  # first call slow

text_q = queue.Queue()
audio_q = queue.Queue()


def tts_worker():
    while True:
        sentence = text_q.get()
        if sentence is None:
            audio_q.put(None)
            text_q.task_done()
            break
        try:
            for chunk in tts.generate_audio_stream(voice_state, sentence):
                audio_q.put(np.asarray(chunk.numpy(), dtype=np.float32).reshape(-1))
        except Exception as e:
            print(f"tts error: {e}")
        text_q.task_done()


def play_worker():
    with sd.OutputStream(samplerate=tts.sample_rate, channels=1, dtype="float32") as out:
        while True:
            chunk = audio_q.get()
            if chunk is None:
                break
            out.write(chunk.reshape(-1, 1))


threading.Thread(target=tts_worker, daemon=True).start()
threading.Thread(target=play_worker, daemon=True).start()

SENT_END = re.compile(r"(?<=[.!?])\s+")


def clean(text):
    return re.sub(r"[*#`_>]", "", text).strip()


def run_agent(messages):
    full = ""
    buf = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if not delta:
                continue
            print(delta, end="", flush=True)
            full += delta
            buf += delta
            parts = SENT_END.split(buf)
            for sentence in parts[:-1]:
                if clean(sentence):
                    text_q.put(clean(sentence))
            buf = parts[-1]
        if clean(buf):
            text_q.put(clean(buf))
        print()
    except Exception as e:
        return f"error: model call failed: {e}"
    return full

def main():
    print(f" [cfg] endpoint={BASE_URL!r} model={MODEL!r}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    try:
        while True:
            user = input("\nyou: ")
            if user.strip().lower() in ("exit", "quit"):
                break
            messages.append({"role": "user", "content": user})
            reply = run_agent(messages)
            if reply.startswith("error:"):
                print(f"\n{reply}")
                messages.pop()
                continue
            if not reply.strip():
                print("\n(empty reply from model)")
                messages.pop()
                continue
            messages.append({"role": "assistant", "content": reply})
            text_q.join()
    except KeyboardInterrupt:
        pass
    finally:
        text_q.put(None)


if __name__ == "__main__":
    main()