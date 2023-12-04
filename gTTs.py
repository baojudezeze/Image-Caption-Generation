import os

from gtts import gTTS


def run_gTTs(act: bool, text_to_say: str):
    if act:
        language = 'en'
        tts = gTTS(text=text_to_say, lang=language, slow=False)
        tts.save("output.mp3")
        os.system(f'afplay "output.mp3"')
