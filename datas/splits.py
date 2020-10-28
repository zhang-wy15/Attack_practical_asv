import os 
import random 

# split LibriSpeech test clean dataset for adversarial attack
# Three parts: enroll, train, test
def split(wav_path, libri_speaker_file, save_path, enroll_num, train_num):
    speaker = {}
    with open(libri_speaker_file, 'r') as f:
        for line in f.readlines():
            speaker[line.split(" ")[0]] = []
    wavs = os.listdir(wav_path)
    for wav in wavs:
        wav_speaker = wav.split('-')[0]
        speaker[wav_speaker].append(wav)
    
    ## enroll
    with open(os.path.join(save_path, 'enroll.txt'), 'w') as f:
        for k in speaker:
            random.shuffle(speaker[k])
            for _ in range(enroll_num):
                f.write('%s %s\n' % (k, speaker[k][0]))
                speaker[k].pop(0)
    
    ## train
    with open(os.path.join(save_path, 'train.txt'), 'w') as f:
        for k in speaker:
            random.shuffle(speaker[k])
            for _ in range(train_num):
                f.write('%s %s\n' % (k, speaker[k][0]))
                speaker[k].pop(0)
    
    ## test
    with open(os.path.join(save_path, 'test.txt'), 'w') as f:
        for k in speaker:
            random.shuffle(speaker[k])
            for _ in range(len(speaker[k])):
                f.write('%s %s\n' % (k, speaker[k][0]))
                speaker[k].pop(0)
    
    for k,v in speaker.items():
        assert len(v) == 0

if __name__ == '__main__':
    wav_path = '/data/zhangweiyi/LibriSpeech_dataset/test_clean/wav'
    libri_speaker_file = './datas/gender_librispeech_test_clean.txt'
    save_path = './datas/splits'
    enroll_num = 1
    train_num = 20
    split(wav_path, libri_speaker_file, save_path, enroll_num, train_num)