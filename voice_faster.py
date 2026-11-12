import time
import scipy.io.wavfile
from pocket_tts import TTSModel

model = TTSModel.load_model()
state = model.get_state_for_audio_prompt("./first_voice.safetensors")

texts = {
    "narration": "The old lighthouse stood at the edge of the cliff for more than two hundred years, watching ships pass through the narrow channel below. Every evening, as the sun dipped behind the western hills, the keeper climbed the spiral staircase, lit the great lamp, and waited. Fog often rolled in without warning, thick and silent, swallowing the coastline whole. On those nights the beam swept slowly across the water, steady and patient, guiding sailors home through the dark. Nobody remembered who had built it, but everybody remembered the first time its light had saved them.",
    "technical": "Modern processors contain between 8 and 64 cores, each running at roughly 3.5 to 5.2 gigahertz. A typical neural network with 100 million parameters needs about 400 megabytes of memory in 32-bit floating point, or only 100 megabytes when quantized to 8 bits. On a laptop CPU, inference can reach real-time speed if the model is small enough and the code avoids unnecessary copies. Engineers therefore measure latency in milliseconds, not seconds, and they care deeply about the difference between 45 and 90.",
    "emotion": "Wait, are you serious? You really built the whole thing in a single weekend? That is incredible! I spent three months on a similar project and never got past the login screen. Tell me everything. What broke first, what surprised you, and what would you do differently next time? Honestly, I am a little jealous, but mostly I am impressed. Let us celebrate properly, because this deserves more than a quick message.",
    "jarvis": "Good morning. The time is seven fifteen, and the weather today is partly cloudy with a high of twenty-eight degrees. You have three meetings scheduled, the first at nine thirty with the design team. I have also noticed two unread messages marked urgent, and one reminder to renew your domain before Friday. Shall I read the messages aloud, or would you prefer a short summary first?",
    "stress": "Language is the oldest technology humans ever invented, and perhaps the most powerful, because it allows one mind to place a picture inside another mind without ever touching it, and because every tool, every city, every law, and every story that followed depended on that first spoken exchange, which means that when a machine learns to speak, it is not merely producing sound, it is stepping into a conversation that has been running for hundreds of thousands of years, carrying with it all the weight, humor, confusion, and beauty that the conversation has gathered along the way.",
}

for name, text in texts.items():
    t0 = time.time()
    audio = model.generate_audio(state, text)
    gen = time.time() - t0
    dur = len(audio) / model.sample_rate
    print(name, "gen", round(gen, 2), "audio", round(dur, 2), "rtf", round(gen / dur, 2))
    scipy.io.wavfile.write(f"{name}.wav", model.sample_rate, audio.numpy())