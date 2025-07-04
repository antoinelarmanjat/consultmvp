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
        <prosody rate="medium">
          Hello James, thanks a lot for coming in. Please have a seat. What can I help you with today?
          <break time="500ms"/>
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody rate="medium">
          Hi Doctor. I've been having this weird feeling in my chest, like a fluttering or skipping beat sometimes. It's happening more often now and it's making me a bit worried.
          <break time="1500ms"/>
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium">
          A fluttering or skipping feeling in your chest. I understand why that would be concerning. How often does this happen per day or week? Does it occur more at rest, when you're active, or is it random?
          <break time="1500ms"/>
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody rate="medium">
          It's pretty random, maybe five or six times a day now. Sometimes when I'm just sitting around, other times when I'm walking.
          <break time="1000ms"/>
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium">
          Okay. Are you feeling short of breath when it happens, or dizzy at all? Do you get chest pain with it?
          <break time="1000ms"/>
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody rate="medium">
          Not really short of breath, maybe a tiny bit lightheaded once or twice, but mostly just the weird feeling. No chest pain.
          <break time="1000ms"/>
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium">
          Thanks. You're still taking Omeprazole for stomach issues, correct? Are those symptoms well-controlled?
          <break time="800ms"/>
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody rate="medium">
          Yes, still taking it. My stomach has been okay, it's just this new heart thing.
          <break time="800ms"/>
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium">
          Got it. And just for my records, do you know what your current weight is? We don't have it listed accurately from your last visit.
          <break time="1000ms"/>
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-E">
        <prosody rate="medium">
          Uh, I'm not sure exactly, maybe around 85kg?
          <break time="500ms"/>
        </prosody>
      </voice>
      <voice name="en-US-Wavenet-D">
        <prosody rate="medium">
          Let's discuss these palpitations further and see what might be causing them.
        </prosody>
      </voice>
    </speak>
    """
    output_file = "dialogue1.mp3"
    synthesize_ssml(dialogue_ssml, output_file)