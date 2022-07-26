from pydub import AudioSegment

m4a_file = input("please enter the path to m4a or mp3 (local)")
wav_filename = m4a_file + ".wav"
track = AudioSegment.from_file(m4a_file,  format= input("please enter the format of this file -->"))
file_handle = track.export(wav_filename, format='wav')

