import soundcard as sc
import sounddevice as sd

print("=== Dispositivos de SALIDA (soundcard) ===")
for spk in sc.all_speakers():
    print(f"- {spk.name}")

print("\n=== Altavoz por defecto ===")
print(sc.default_speaker().name)

print("\n=== Dispositivos de ENTRADA (sounddevice) ===")
print(sd.query_devices())