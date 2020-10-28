import sys, time, os, importlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from config.args import parser
from tools import compute_min_cost, load_yaml_config, load_speaker_model_parameters

args = parser.parse_args()
args = load_yaml_config(args, parser)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240
    # Read wav file and convert to torch tensor
    sample_rate, audio  = wavfile.read(filename)
    audiosize = audio.shape[0]
    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = np.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]
    if evalmode:
        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
    feat = np.stack(feats,axis=0).astype(np.float)

    return feat

def evaluateFromList(model, listfilename, print_interval=100, test_path='', num_eval=10, eval_frames=None):
    model.eval()
    lines       = []
    files       = []
    feats       = {}
    tstart      = time.time()
    ## Read all lines
    with open(listfilename) as listfile:
        while True:
            line = listfile.readline()
            if (not line):
                break
            data = line.split()
            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data
            files.append(data[1])
            files.append(data[2])
            lines.append(line)
    setfiles = list(set(files))
    setfiles.sort()
    ## Save all features to file
    for idx, file in enumerate(setfiles):
        inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()
        ref_feat = model.forward(inp1).detach().cpu()
        feats[file] = ref_feat
        telapsed = time.time() - tstart
        if idx % print_interval == 0:
            sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]))
    print('')
    all_scores = []
    all_labels = []
    all_trials = []
    tstart = time.time()
    ## Read files and compute all scores
    for idx, line in enumerate(lines):
        data = line.split()
        ## Append random label if missing
        if len(data) == 2: data = [random.randint(0,1)] + data
        ref_feat = feats[data[1]].cuda()
        com_feat = feats[data[2]].cuda()
        if False:
            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            com_feat = F.normalize(com_feat, p=2, dim=1)
        dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()
        score = 1 * np.mean(dist)
        all_scores.append(score)
        all_labels.append(int(data[0]))
        all_trials.append(data[1]+" "+data[2])
        if idx % print_interval == 0:
            telapsed = time.time() - tstart
            sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed))
            sys.stdout.flush()
    print('\n')

    return (all_scores, all_labels, all_trials)

def main():
    # Load model
    model = importlib.import_module('models.' + args.model).__getattribute__('MainModel')
    model = model(**vars(args)).cuda()
    if args.initial_model != "":
        model = load_speaker_model_parameters(model, args.initial_model)
        print("Model %s loaded!"%args.initial_model)
    
    # Runing eval
    sc, lab, trials = evaluateFromList(model, args.test_list, print_interval=100, test_path=args.test_path, eval_frames=args.eval_frames)
    eer, eer_thresh, min_c_det, min_dcf, min_c_det_threshold = compute_min_cost(sc, lab, p_target=0.01)
    print('EER %2.4f, EER_Thresh %2.4f' % (eer, eer_thresh))
    print('min_c_det %2.4f, minDCF %2.4f, min_c_det_threshold %2.4f' % (min_c_det, min_dcf, min_c_det_threshold))

if __name__ == '__main__':
    main()