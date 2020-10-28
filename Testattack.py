import sys, time, os, importlib, random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from config.args import parser
from tools import load_yaml_config, load_speaker_model_parameters, reverb, reverb_np

args = parser.parse_args()
args = load_yaml_config(args, parser)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# set random seed for reproducibility
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(args.seed)

# load wav data with shape (1, SampleNumbers)
def loadWAV(filename):
    sample_rate, audio  = wavfile.read(filename)
    if len(audio.shape) == 2:
        audio = audio[:,0]
    feat = np.stack([audio], axis=0).astype(np.float)
    return feat

# Evaluation the attack success rate of existed audio adversarial examples
def Test(speaker_model, enroll_speaker_feat, test_speaker, speaker, target, args):
    speaker_model.eval()
    wavs = test_speaker[speaker]
    target_feat = torch.FloatTensor(enroll_speaker_feat[target]).cuda()
    success = 0
    if args.rir:
        rir_wavs_test = os.listdir(os.path.join(args.attack_rir_path, 'test'))
    for wav in wavs:
        wav = os.path.join(args.wav_path, wav)
        wav_data = loadWAV(wav)
        wav_data = np.clip(wav_data, -2**15, 2**15-1)
        wav_data_tensor = torch.FloatTensor(wav_data).cuda()
        wav_feat = speaker_model.forward(wav_data_tensor).detach()
        dist = F.cosine_similarity(wav_feat, target_feat).cpu().numpy()
        score = 1 * np.mean(dist)
        if score > args.thresh:
            success += 1
        
    print('%s to %s attack success rate: %2.1f' % (speaker, target, success * 1.0 / len(wavs) * 100))
    return success, len(wavs), success * 1.0 / len(wavs) * 100

####################################################################################################################
####################################################################################################################
### step 1: Set Parameters ###
enroll_file = args.enroll_file
test_file = args.test_file
wav_path = args.wav_path
enroll_path = args.enroll_path

enroll_speaker = {}
with open(enroll_file, 'r') as f:
    for line in f.readlines():
        speaker = line.split(" ")[0]
        if speaker not in enroll_speaker:
            enroll_speaker[speaker] = []
        enroll_speaker[speaker].append(line.split(" ")[1].strip())

test_speaker = {}
with open(test_file, 'r') as f:
    for line in f.readlines():
        speaker = line.split(" ")[0]
        if speaker not in test_speaker:
            test_speaker[speaker] = []
        test_speaker[speaker].append(line.split(" ")[1].strip())

### step 2: Load speaker model ###
print('Loading speaker model...')
speaker_model = importlib.import_module('models.' + args.model).__getattribute__('MainModel')
speaker_model = speaker_model(**vars(args)).cuda()
if args.initial_model != "":
    speaker_model = load_speaker_model_parameters(speaker_model, args.initial_model)
    print("Model %s loaded!"%args.initial_model)
speaker_model.eval()

### step 3: enroll speakers ###
print('Enrolling...')
enroll_speaker_feat = {}
with torch.no_grad():
    for k,v in enroll_speaker.items():
        enroll_speaker_feat[k] = 0
        for i in range(len(v)):
            wav = os.path.join(enroll_path, v[i])
            wav_data = torch.FloatTensor(loadWAV(wav)).cuda()
            wav_feat = speaker_model.forward(wav_data)
            enroll_speaker_feat[k] += wav_feat.detach().cpu().numpy()
            del wav_feat, wav_data
        enroll_speaker_feat[k] /= len(v)
print('Enrolling ok!')

### step 4: match original and target speaker ###
original_speaker = []
target_speaker = []
if args.intra_gender:
    with open('./datas/intra_match.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            o,t = line[0], line[1].strip()
            original_speaker.append(o)
            target_speaker.append(t)

if args.inter_gender:
    with open('./datas/inter_match.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            o,t = line[0], line[1].strip()
            original_speaker.append(o)
            target_speaker.append(t)

### step 5: test ###
success = 0
number = 0
for i in range(len(original_speaker)):
    o = original_speaker[i]
    t = target_speaker[i]
    a, b, _ = Test(speaker_model, enroll_speaker_feat, test_speaker, o, t, args)
    success += a
    number += b
print(success/number)