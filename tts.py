from google.cloud import texttospeech_v1beta1 as texttospeech

def synthesize_ssml(ssml_text, output_filename):
    """Synthesizes speech from the input SSML and saves it to an MP3 file."""

    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(ssml=ssml_text)  # Use ssml here

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-D",  # You can set a default voice here
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to "{output_filename}"')

if __name__ == "__main__":
    dialogue_ssml = """
    <speak>
      <prosody rate="slow">
        Narrator: The old house stood on a hill overlooking the town.
      </prosody>
      <voice name="en-US-Wavenet-D">
        <prosody pitch="+5st">
          Character A: Hello there!
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody pitch="-2st">
          Character B: Oh, hi! What brings you here?
        </prosody>
      </voice>
      <prosody rate="medium">
        Narrator: A moment of silence hung in the air.
      </prosody>
      <voice name="en-US-Wavenet-D">
        Character A: I was just passing by...
      </voice>
    </speak>
    """
    output_file = "dialogue.mp3"
    synthesize_ssml(dialogue_ssml, output_file)