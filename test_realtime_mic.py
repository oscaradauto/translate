from RealtimeSTT import AudioToTextRecorder


def on_partial_text(text):
    print(f"\r[Parcial] {text}", end="", flush=True)


def on_stable_text(text):
    print(f"\n[ESTABLE] {text}\n")


if __name__ == "__main__":

    print("Cargando modelo... (puede tardar la primera vez)")

    recorder = AudioToTextRecorder(

        model="small",

        language="en",

        spinner=False,

        enable_realtime_transcription=True,

        on_realtime_transcription_update=on_partial_text,

        on_realtime_transcription_stabilized=on_stable_text,

        use_microphone=True,

        input_device_index=1,

        post_speech_silence_duration=0.4,

    )

    print("Habla en inglés, Ctrl+C para salir...")

    try:

        while True:
            sentence = recorder.text()  # bloquea hasta obtener UNA oración completa

            print(f"\n[FRASE COMPLETA] {sentence}\n")

    except KeyboardInterrupt:

        recorder.stop()

        print("\nDetenido.")
