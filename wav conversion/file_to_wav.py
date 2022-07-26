from pydub import AudioSegment
import os
import fleep

dir_1 = input("please enter the path to m4a or mp3 (local)")
files_from_dir = os.listdir(dir_1)

for i in files_from_dir:
    os.chdir(dir_1)
    with open(i, "rb") as file:
        info = fleep.get(file.read(128))
        print(info.mime)
        if info.mime == ["video/mp4"]:
            print("converting ", i)
            wav_filename = (i[:-4] + ".wav")
            track = AudioSegment.from_file(i,  format="m4a")
            os.chdir(r"C:\Users\EliKurtz\Desktop\AD_Audio\exported wavs")
            file_handle = track.export(wav_filename, format='wav')
        if not info.mime:
            print("converting ", i)
            wav_filename = (i[:-4] + ".wav")
            track = AudioSegment.from_file(i,  format="mp3")
            os.chdir(r"C:\Users\EliKurtz\Desktop\AD_Audio\exported wavs")
            file_handle = track.export(wav_filename, format='wav')
        if info.mime == ['audio/wav']:
            print("converting ", i)
            wav_filename = (i[:-4] + ".wav")
            track = AudioSegment.from_file(i,  format="wav")
            os.chdir(r"C:\Users\EliKurtz\Desktop\AD_Audio\exported wavs")
            file_handle = track.export(wav_filename, format='wav')
