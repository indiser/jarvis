from pocket_tts import TTSModel, export_model_state
import scipy.io.wavfile

model = TTSModel.load_model()

# Export a voice state for fast loading later
model_state = model.get_state_for_audio_prompt("ultron_best_vo.wav")
export_model_state(model_state, "./ultron_voice.safetensors")


model_state_copy = model.get_state_for_audio_prompt("./ultron_voice.safetensors")

audio = model.generate_audio(model_state_copy, "Language is the oldest technology humans ever invented, and perhaps the most powerful, because it allows one mind to place a picture inside another mind without ever touching it, and because every tool, every city, every law, and every story that followed depended on that first spoken exchange, which means that when a machine learns to speak, it is not merely producing sound, it is stepping into a conversation that has been running for hundreds of thousands of years, carrying with it all the weight, humor, confusion, and beauty that the conversation has gathered along the way.")

scipy.io.wavfile.write("output.wav", model.sample_rate, audio.numpy())

# import time
# import torch
# import scipy.io.wavfile
# from pocket_tts import TTSModel

# print(torch.get_num_threads())

# t0 = time.time()
# model = TTSModel.load_model()
# print("load", time.time() - t0)

# TEXT = "The old lighthouse stood at the edge of the cliff for more than two hundred years, watching ships pass through the narrow channel below. Every evening, as the sun dipped behind the western hills, the keeper climbed the spiral staircase, lit the great lamp, and waited. Fog often rolled in without warning, thick and silent, swallowing the coastline whole. On those nights the beam swept slowly across the water, steady and patient, guiding sailors home through the dark. Nobody remembered who had built it, but everybody remembered the first time its light had saved them."

# t0 = time.time()
# state = model.get_state_for_audio_prompt("./first_voice.safetensors")
# print("voice state", time.time() - t0)

# t0 = time.time()
# audio = model.generate_audio(state, TEXT)
# dur = len(audio) / model.sample_rate
# print("gen", time.time() - t0, "audio sec", dur)

# scipy.io.wavfile.write("output.wav", model.sample_rate, audio.numpy())

# import os
# import time
# import torch
# from pocket_tts import TTSModel

# print("cpu_count", os.cpu_count())

# model = TTSModel.load_model()
# state = model.get_state_for_audio_prompt("./first_voice.safetensors")

# TEXT = "Every evening, as the sun dipped behind the western hills, the keeper climbed the spiral staircase, lit the great lamp, and waited."

# for n in [1, 2, 4, 6, 8]:
#     torch.set_num_threads(n)
#     t0 = time.time()
#     audio = model.generate_audio(state, TEXT)
#     gen = time.time() - t0
#     dur = len(audio) / model.sample_rate
#     print(n, "threads", "gen", round(gen, 2), "audio", round(dur, 2), "rtf", round(gen / dur, 2))