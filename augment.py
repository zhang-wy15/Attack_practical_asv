import numpy as np 
import soundfile as sf 
from scipy import signal 
import random
import os 

def load_audio(path):
    sound, sample_rate = sf.read(path, dtype='int16')
    # TODO this should be 32768.0 to get twos-complement range.
    # TODO the difference is negligible but should be fixed for new models.
    sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound

class Augment_wave(object):

    def __init__(self, fs, rir_path, split='train', shift_output=True, normalize=True):
        self.fs = fs 
        self.shift_output = shift_output
        self.normalize = normalize
        assert split in ['train', 'dev', 'test']
        self.rir_path = rir_path
        self.rir = None
        self.init()

    def init(self):
        # construct rir dict
        self.rooms = ['ConferenceRoom2', 'C236', 'D105', 'E112', 'L207', 'L212', 'Q301']
        rir_all = {}
        for room in self.rooms:
            rir_all[room] = []
        for _, _, files in os.walk(self.rir_path):
            for rir in files:
                if rir.endswith('.wav'):
                    rir_room = rir.split('_')[2].split('-')[0]
                    assert rir_room in self.rooms
                    rir_all[rir_room].append(rir)
        for room in self.rooms:
            random.shuffle(rir_all[room])
        self.rir_dict = rir_all

    def compute_early_reverb_power(self, wave, rir):
        peak_index = rir.argmax()
        before_peak = int(0.001 * self.fs)
        after_peak = int(0.05 * self.fs)
        early_rir_start = max(0, peak_index - before_peak)
        early_rir_end = min(rir.shape[0], peak_index + after_peak)
        early_rir = rir[early_rir_start: early_rir_end]
        early_reverb = signal.fftconvolve(wave, early_rir, mode="full")
        early_power = np.dot(early_reverb, early_reverb) / early_reverb.shape[0]
        return early_power

    def do_reverb(self, wave, rir):
        rir = rir.astype("float32")
        rir = rir / np.max(np.abs(rir))
        early_power = self.compute_early_reverb_power(wave, rir)
        wave = signal.fftconvolve(wave, rir)
        return wave, early_power

    def augment(self, wave, rir=None):
        if rir is None:
            return wave 
        wave = wave.astype("float32")
        if self.shift_output is True:
            dur2len = len(wave)
        else:
            dur2len = len(wave) + len(rir) - 1
        before_power = np.dot(wave, wave) / wave.shape[0]
        peak_index = 0 
        early_power = before_power
        # reverberate the wave
        if rir is not None:
            wave, early_power = self.do_reverb(wave, rir)
            if self.shift_output is True:
                peak_index = rir.argmax()

        # normalize wave
        if self.normalize is True:
            # compute the power of wave after reverberation and possibly noise
            after_power = np.dot(wave, wave) / wave.shape[0]
            wave = wave * (np.sqrt(before_power / after_power))
        # extend wave if duration is larger than the length of wave
        if dur2len > wave.shape[0]:
            wave = np.pad(wave, pad_width=(0, dur2len - wave.shape[0]), mode="wrap")
        # shift the output wave by shift_output
        wave = wave[peak_index: peak_index + dur2len]
        return wave

    def inject(self, wave):
        # sample a room and a rir of the room based on P(r): uniform distribution
        room = np.random.choice(self.rooms)
        rir_name = np.random.choice(self.rir_dict[room])
        rir = load_audio(os.path.join(self.rir_path, rir_name))

        return self.augment(wave, rir=rir)

