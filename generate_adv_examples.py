import os 
import numpy as np 
import csv 
from scipy.io import wavfile
import random 
from augment import Augment_wave
import argparse

parser = argparse.ArgumentParser(description = "generate_parameter")
parser.add_argument('--rir', dest='rir', action='store_true', help='Whether do reverberation')
parser.add_argument('--rir_root', type=str, default='/data/zhangweiyi/rir/IRs_release/BUT_IRs/test', help='Test rir wavs path')
parser.add_argument('--wav_root', type=str, default='/data/zhangweiyi/LibriSpeech_dataset/test_clean/wav', help='Wav root path')
parser.add_argument('--wav_file', type=str, default='./datas/splits/test.txt', help='Wav file contains the speaker and its audio names')
parser.add_argument('--noise_root', type=str, default='./out_intra_rir_norm2/step2')
parser.add_argument('--out_root', type=str, default='./test_data_rir', help='Output wav save path')
args = parser.parse_args()

if args.rir:
    a = Augment_wave(fs=16000, rir_path=args.rir_root, split='test')

def loadWAV(filename):
    sample_rate, audio  = wavfile.read(filename)
    if len(audio.shape) == 2:
        audio = audio[:,0]
    feat = np.asarray(audio).astype(np.float)
    return feat

# speaker wavs list
with open(args.wav_file, 'r') as f:
    lines = f.readlines()
speaker_wavs ={}
for line in lines:
    line = line.split()
    speaker, wav = line[0], line[1].strip()
    if speaker not in speaker_wavs:
        speaker_wavs[speaker] = []
    speaker_wavs[speaker].append(wav)

if not os.path.exists(args.out_root):
    os.mkdir(args.out_root)

# adversarial perturbation for every set: speaker and targeted speaker
speaker_noise = {}
original = []
target = []
noise_step2 = os.listdir(args.noise_root)
for noise in noise_step2:
    tmp = noise[:-4].split('_')
    original_speaker, target_speaker = tmp[0], tmp[1]
    original.append(original_speaker)
    target.append(target_speaker)
    speaker_noise[original_speaker] = noise

# generate the data
for speaker in speaker_wavs:
    wavs = speaker_wavs[speaker]
    noise = speaker_noise[speaker]
    noise_data = loadWAV(os.path.join(args.noise_root, noise))
    for wav in wavs:
        wav_path = os.path.join(args.wav_root, wav)
        wav_path_new = os.path.join(args.out_root, wav)
        wav_data = loadWAV(wav_path)
        noise_data_tmp = np.tile(noise_data, wav_data.shape[0]//noise_data.shape[0] + 1)[0:wav_data.shape[0]]
        wav_data = wav_data + noise_data_tmp
        wav_data = np.clip(wav_data, -2**15, 2**15-1)
        if args.rir:
            wav_data = a.inject(wav_data)
            wav_data = np.clip(wav_data, -2**15, 2**15-1)
        wavfile.write(wav_path_new, 16000, np.asarray(wav_data, dtype=np.int16))