import sys, time, os, importlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from config.args import parser
from tools import load_yaml_config, load_speaker_model_parameters, reverb, reverb_np
sys.path.append('voxceleb_trainer')
from SpeakerNet import SpeakerNet
import random

args = parser.parse_args()
args = load_yaml_config(args, parser)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(args.seed)

def loadWAV(filename):
    sample_rate, audio  = wavfile.read(filename)
    feat = np.stack([audio], axis=0).astype(np.float)
    return feat

def Attack(speaker_model, speech_model, enroll_speaker_feat, train_speaker, speaker, target, args):
    # Parameter
    num_train = args.num_train
    adv_len = int(args.noise_len * 16000)
    max_steps = args.max_steps 
    lr = args.lr 
    momentum = args.momentum 
    max_delta = args.max_delta 
    e1 = args.e1
    e2 = args.e2

    # Load wav data
    wavs = train_speaker[speaker]
    wav_input_np = []
    max_len = 0
    for i in range(num_train):
        wav = os.path.join(args.wav_path, wavs[i])
        wav_data = loadWAV(wav)
        if wav_data.shape[1] > max_len:
            max_len = wav_data.shape[1]
        wav_input_np.append(wav_data)
    
    # repeat wav data
    for i in range(num_train):
        wav_input_np[i] = np.tile(wav_input_np[i], (1, max_len // wav_input_np[i].shape[1] + 1))[:, 0:max_len]
    wav_input_np = np.asarray(wav_input_np).squeeze(axis=1)

    if args.rir:
        rir_wavs_train = os.listdir(os.path.join(args.attack_rir_path, 'train'))
    # initialize tensors
    adv_noise_np = np.random.uniform(-max_delta, max_delta, adv_len).astype(np.int16)
    target_feat = torch.FloatTensor(enroll_speaker_feat[target]).cuda()
    wav_input = torch.FloatTensor(wav_input_np).cuda()
    
    # step1 maximize the attack on ASV model
    batch_num = num_train // args.batch
    grad_pre = 0
    result = adv_noise_np
    for i in range(15000):
        loss = 0
        grad = 0
        dist = []
        for b in range(batch_num):
            wav_input_batch = wav_input[b*args.batch:(b+1)*args.batch]
            adv_noise = torch.FloatTensor(adv_noise_np).cuda()
            adv_noise.requires_grad = True
            adv_noise_tmp = adv_noise.repeat(max_len // adv_len + 1)[:max_len]
            if not args.rir:
                wav_adv_input = wav_input_batch + adv_noise_tmp.unsqueeze(0)
            else:
                rir_tensor = torch.FloatTensor(loadWAV(os.path.join(args.attack_rir_path, 'train', random.choice(rir_wavs_train)))).cuda()
                rir_tensor = rir_tensor / torch.norm(rir_tensor)
                adv_noise_tmp = reverb(adv_noise_tmp.unsqueeze(0), rir_tensor)
                wav_input_batch = reverb(wav_input_batch, rir_tensor)
                wav_adv_input = wav_input_batch + adv_noise_tmp
            wav_adv_input = torch.clamp(wav_adv_input, -2**15, 2**15-1)
            wav_feat = speaker_model.forward(wav_adv_input)
            dist_batch = F.cosine_similarity(wav_feat, target_feat)
            loss1 = torch.sum(torch.clamp(args.thresh - dist_batch + args.margine, min=0.))
            loss1.backward()

            loss += loss1.item()
            grad += adv_noise.grad.data.cpu().numpy()
            dist.extend(list(dist_batch.detach().cpu().numpy()))
        loss /= batch_num
        grad /= batch_num
        print('step1: %02d, %s to %s, loss: % 2.4f' % (i, speaker, target, loss))

        # if convergence break
        result = adv_noise_np
        if loss < e1:
            break

        # PGD can also be interpreted as an iterative algorithm to solve the following problem:
        grad_new = momentum * grad_pre + grad
        grad_pre = grad_new
        adv_noise_np = adv_noise_np - lr * np.sign(grad_new)
        adv_noise_np = np.clip(adv_noise_np, -max_delta, max_delta)
    step1 = i
    result1 = result
    #step2
    grad_pre = 0
    for i in range(max_steps):
        loss1_value = 0
        loss2_value = 0
        grad1 = 0
        grad2 = 0
        dist = []
        for b in range(batch_num):
            wav_input_batch = wav_input[b*args.batch:(b+1)*args.batch]
            adv_noise = torch.FloatTensor(adv_noise_np).cuda()
            adv_noise.requires_grad = True
            adv_noise_tmp = adv_noise.repeat(max_len // adv_len + 1)[:max_len]
            if not args.rir:
                wav_adv_input = wav_input_batch + adv_noise_tmp.unsqueeze(0)
            else:
                rir_tensor = torch.FloatTensor(loadWAV(os.path.join(args.attack_rir_path, 'train', random.choice(rir_wavs_train)))).cuda()
                rir_tensor = rir_tensor / torch.norm(rir_tensor)
                adv_noise_tmp = reverb(adv_noise_tmp.unsqueeze(0), rir_tensor)
                wav_input_batch = reverb(wav_input_batch, rir_tensor)
                wav_adv_input = wav_input_batch + adv_noise_tmp
            wav_adv_input = torch.clamp(wav_adv_input, -2**15, 2**15-1)
            wav_feat = speaker_model.forward(wav_adv_input)
            dist_batch = F.cosine_similarity(wav_feat, target_feat)
            loss1 = torch.sum(torch.clamp(args.thresh - dist_batch + args.margine, min=0.))
            spec_stft = torch.stft(adv_noise/2**15, n_fft=320, hop_length=160, win_length=320, window=torch.hamming_window(320).cuda())
            loss2 = torch.mean(torch.sqrt(torch.square(spec_stft[:,:,0]) + torch.square(spec_stft[:,:,1])))

            loss1_value += loss1.item()
            loss2_value += loss2.item()

            grad1 += torch.autograd.grad(loss1, adv_noise)[0].detach().cpu().numpy()
            grad2 += torch.autograd.grad(loss2, adv_noise)[0].detach().cpu().numpy()
            dist.extend(list(dist_batch.detach().cpu().numpy()))

            del loss1, loss2, dist_batch, wav_feat, wav_adv_input, wav_input_batch, adv_noise_tmp
         
        loss1_value /= batch_num
        loss2_value /= batch_num
        grad1 /= batch_num 
        grad2 /= batch_num
        print('step2: %02d, %s to %s, loss1: %2.4f, loss2: %2.4f' % (i, speaker, target, loss1_value, loss2_value))

        # if convergence break
        result = adv_noise_np
        if loss1_value <= e1 and loss2_value <= e2:
            break
        # PGD can also be interpreted as an iterative algorithm to solve the following problem:
        alpha = args.gamma if loss1_value > e1 else 0
        beta = 1 if loss2_value > e2 else 0
        grad_new = momentum * grad_pre + alpha * grad1 / (np.linalg.norm(grad1, 1) + 1e-12) + beta * grad2 / np.linalg.norm(grad2, 1)
        grad_new = np.nan_to_num(grad_new)
        grad_pre = grad_new
        adv_noise_np = adv_noise_np - lr * np.sign(grad_new)
        adv_noise_np = np.clip(adv_noise_np, -max_delta, max_delta)
    step2 = i
    return result1, step1, loss, result, step2, loss1_value, loss2_value

def Test(speaker_model, speech_model, enroll_speaker_feat, test_speaker, speaker, target, adv_noise, args):
    speaker_model.eval()
    wavs = test_speaker[speaker]
    target_feat = torch.FloatTensor(enroll_speaker_feat[target]).cuda()
    success = 0
    if args.rir:
        rir_wavs_test = os.listdir(os.path.join(args.attack_rir_path, 'test'))
    for wav in wavs:
        wav = os.path.join(args.wav_path, wav)
        wav_data = loadWAV(wav)
        adv_noise_tmp = np.tile(adv_noise, wav_data.shape[1] // adv_noise.shape[0] + 1)[0:wav_data.shape[1]]
        if not args.rir:
            wav_data = wav_data + np.expand_dims(adv_noise_tmp, axis=0)
        else:
            rir_np = loadWAV(os.path.join(args.attack_rir_path, 'test', random.choice(rir_wavs_test)))
            rir_np = rir_np / np.linalg.norm(rir_np)
            adv_noise_tmp = reverb_np(np.expand_dims(adv_noise_tmp, axis=0), rir_np)
            wav_data = reverb_np(wav_data, rir_np)
            wav_data = wav_data + adv_noise_tmp
        wav_data = np.clip(wav_data, -2**15, 2**15-1)
        wav_data_tensor = torch.FloatTensor(wav_data).cuda()
        wav_feat = speaker_model.forward(wav_data_tensor).detach()
        if args.cosine_similarity:
            dist = F.cosine_similarity(wav_feat, target_feat).cpu().numpy()
            score = 1 * np.mean(dist)
        else:
            wav_feat = F.normalize(wav_feat, p=2, dim=1)
            target_feat = F.normalize(target_feat, p=2, dim=1)
            dist = F.pairwise_distance(wav_feat.unsqueeze(-1), target_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()
            score = -1 * np.mean(dist)
        print(score)
        if score > args.thresh:
            success += 1
        del wav_feat, wav_data_tensor
    del target_feat
    torch.cuda.empty_cache()
    print('%s to %s attack success rate: %2.1f' % (speaker, target, success * 1.0 / len(wavs) * 100))
    return success, len(wavs), success * 1.0 / len(wavs) * 100

####################################################################################################################
####################################################################################################################
### step 1: Set Parameters ###
enroll_file = 'datas/splits/enroll.txt'
train_file = 'datas/splits/train.txt'
test_file = 'datas/splits/test.txt'
wav_path = args.wav_path
libri_speaker_file = 'datas/gender_librispeech_test_clean.txt'

enroll_speaker = {}
with open(enroll_file, 'r') as f:
    for line in f.readlines():
        speaker = line.split(" ")[0]
        if speaker not in enroll_speaker:
            enroll_speaker[speaker] = []
        enroll_speaker[speaker].append(line.split(" ")[1].strip())

train_speaker = {}
with open(train_file, 'r') as f:
    for line in f.readlines():
        speaker = line.split(" ")[0]
        if speaker not in train_speaker:
            train_speaker[speaker] = []
        train_speaker[speaker].append(line.split(" ")[1].strip())

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
speech_model = 0

### step 3: enroll speakers ###
print('Enrolling...')
enroll_speaker_feat = {}
with torch.no_grad():
    for k,v in enroll_speaker.items():
        enroll_speaker_feat[k] = 0
        for i in range(len(v)):
            wav = os.path.join(wav_path, v[i])
            wav_data = torch.FloatTensor(loadWAV(wav)).cuda()
            wav_feat = speaker_model.forward(wav_data)
            enroll_speaker_feat[k] += wav_feat.detach().cpu().numpy()
            del wav_feat, wav_data
        enroll_speaker_feat[k] /= len(v)
print('Enrolling ok!')

### step 4: match original and target speaker ###
female_speaker = []
male_speaker = []
with open(libri_speaker_file, 'r') as f:
    for line in f.readlines():
        if line.split(" ")[1].strip() == 'M':
            male_speaker.append(line.split(" ")[0])
        else:
            female_speaker.append(line.split(" ")[0])

original_speaker = []
target_speaker = []
# genter-intra
for speaker in female_speaker:
    original_speaker.append(speaker)
    while True:
        idx = random.randint(0, len(female_speaker)-1)
        if female_speaker[idx] != speaker:
            target_speaker.append(female_speaker[idx])
            break
for speaker in male_speaker:
    original_speaker.append(speaker)
    while True:
        idx = random.randint(0, len(male_speaker)-1)
        if male_speaker[idx] != speaker:
            target_speaker.append(male_speaker[idx])
            break
# genter-inter
for speaker in female_speaker:
    original_speaker.append(speaker)
    target_speaker.append(male_speaker[random.randint(0, len(male_speaker)-1)])
for speaker in male_speaker:
    original_speaker.append(speaker)
    target_speaker.append(female_speaker[random.randint(0, len(female_speaker)-1)])
print('Matching attack targets')
for i in range(len(original_speaker)):
    print(i, original_speaker[i], target_speaker[i])

### step 5: attack and get universal adversarial noise ###
if not args.test:
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
        os.mkdir(os.path.join(args.out_path, 'step1'))
        os.mkdir(os.path.join(args.out_path, 'step2'))

    f = open(os.path.join(args.out_path, 'result.txt'), 'w')

    if args.gender_intra:
        for i in range(len(original_speaker)//2):
            print('Num: %02d' % i)
            speaker = original_speaker[i]
            target = target_speaker[i]
            noise1, step1, loss, noise, step2, loss1_value, loss2_value = Attack(speaker_model, speech_model, enroll_speaker_feat, train_speaker, speaker, target, args)
            with torch.no_grad():
                success, number, rate = Test(speaker_model, speech_model, enroll_speaker_feat, test_speaker, speaker, target, noise, args)
            torch.cuda.empty_cache()
            out = 'Num: %02d, %5s to %5s, step1: %4d, loss: %.4f, step2: %4d, loss1: %.4f, loss2: %.4f, success number: %3d, total number: %3d, success rate: %3.2f\n' % (i+1, speaker, target, step1, loss, step2, loss1_value, loss2_value, success, number, rate)
            f.write(out)
            f.flush()
            wavfile.write(os.path.join(args.out_path, 'step1', speaker +'_' + target + '.wav'), 16000, np.asarray(noise1, dtype=np.int16))
            wavfile.write(os.path.join(args.out_path, 'step2', speaker +'_' + target + '.wav'), 16000, np.asarray(noise, dtype=np.int16))

    if args.gender_inter:
        for i in range(len(original_speaker)//2):
            print('Num: %02d' % i)
            speaker = original_speaker[i+40]
            target = target_speaker[i+40]
            noise1, step1, loss, noise, step2, loss1_value, loss2_value = Attack(speaker_model, speech_model, enroll_speaker_feat, train_speaker, speaker, target, args)
            with torch.no_grad():
                success, number, rate = Test(speaker_model, speech_model, enroll_speaker_feat, test_speaker, speaker, target, noise, args)
            torch.cuda.empty_cache()
            out = 'Num: %02d, %5s to %5s, step1: %4d, loss: %.4f, step2: %4d, loss1: %.4f, loss2: %.4f, success number: %3d, total number: %3d, success rate: %3.2f\n' % (i+1, speaker, target, step1, loss, step2, loss1_value, loss2_value, success, number, rate)
            f.write(out)
            f.flush()
            wavfile.write(os.path.join(args.out_path, 'step1', speaker +'_' + target + '.wav'), 16000, np.asarray(noise1, dtype=np.int16))
            wavfile.write(os.path.join(args.out_path, 'step2', speaker +'_' + target + '.wav'), 16000, np.asarray(noise, dtype=np.int16))
