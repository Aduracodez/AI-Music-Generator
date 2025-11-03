from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torchaudio

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

prompt = "relaxing lo-fi beat for studying"

inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt"
)

audio_values = model.generate(**inputs, max_new_tokens=256)
torchaudio.save("generated_music.wav", audio_values[0].cpu(), 32000)
from IPython.display import Audio
Audio("generated_music.wav")
