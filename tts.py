from google.cloud import texttospeech

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
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium" duration="5s">
          Good morning, Sophia. Thanks for coming in. What can I help you with today?
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody rate="medium" duration="13s">
          Morning, Doctor. I've been feeling quite weak and tired lately, and I've lost my appetite.
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium" duration="12s">
          Weakness, tiredness, and loss of appetite. When did these symptoms begin? Have they come on suddenly or gradually?
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody rate="medium" duration="15s">
          It's been about two weeks now, starting gradually but getting worse. I just don't feel like eating much.
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium" duration="15s">
          I see. Have you had any changes in your weight because of the reduced appetite? How is the weakness impacting your ability to do things around the house?
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody rate="medium" duration="10s">
          Yes, I think I've lost a little weight. The weakness makes me feel unsteady on my feet.
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium" duration="10s">
          Thank you. You're taking Metoprolol – is that for blood pressure, and are you taking it regularly?
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody rate="medium" duration="5s">
          Yes, for blood pressure. I take it every day.
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium" duration="10s">
          I need to update my records regarding smoking status. Have you ever smoked?
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody rate="medium" duration="3s">
          No, I've never smoked.
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium" duration="12s">
          Okay. Have you experienced any other symptoms, like fever, nausea, or changes in your bowel movements?
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody rate="medium" duration="5s">
          No, just the weakness and no appetite.
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium">
          Alright, these symptoms can be important, especially the unexplained weight loss. Let's do an exam and discuss further.
        </prosody>
      </voice>
    </speak>
    """
    output_file = "dialogue10.mp3"
    synthesize_ssml(dialogue_ssml, output_file)