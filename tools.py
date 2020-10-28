import sys
import os
import os.path
import numpy as np
from operator import itemgetter
import yaml
import torch

########################################################################
# Tool 1: Compute EER and minDCF
#computes the equal error rate (EER) given FNR and FPR values calculated
#for a range of operating points on the DET curve
def compute_eer(fnrs, fprs, thresholds):
    diff_pm_fa = np.asarray(fnrs) - np.asarray(fprs)
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = 1.0 / (fprs[x2] - fprs[x1] - (fnrs[x2] - fnrs[x1]))
    eer = a * (fprs[x2] * fnrs[x1] - fprs[x1] * fnrs[x2])
    thresh_index = a * (x2 * (fnrs[x1] - fprs[x1]) + x1 * (fprs[x2] - fnrs[x2]))
    thresh = thresholds[int(thresh_index)]
    return eer*100, thresh 

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_c_det, min_dcf, min_c_det_threshold

def compute_min_cost(scores, labels, p_target=0.01):
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    eer, eer_thresh = compute_eer(fnrs, fprs, thresholds)
    min_c_det, min_dcf, min_c_det_threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss=10, c_fa=1)

    return eer, eer_thresh, min_c_det, min_dcf, min_c_det_threshold


########################################################################
# Tool 2: Parse Yaml file to config
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

def load_yaml_config(args, parser):
    if args.config is not None:
        with open(args.config, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            if k in args.__dict__:
                typ = find_option_type(k, parser)
                args.__dict__[k] = typ(v)
            else:
                sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))
    return args

########################################################################
# Tool 3: Load parameters for speaker model
def load_speaker_model_parameters(model, path):
    model_state = model.state_dict()
    loaded_state = torch.load(path)
    for name, param in loaded_state.items():
        origname = name
        if name not in model_state:
            name = name.replace("__S__.", "")
            if name not in model_state:
                print("%s is not in the model." % origname)
                continue
        if model_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            continue
        model_state[name].copy_(param)

    return model

#########################################################################
# Tool 4: Calculate the convolution of two signals using fft method
def complex_multiplication(t1, t2):
    real1, imag1 = t1[:,:,0], t1[:,:,1]
    real2, imag2 = t2[:,:,0], t2[:,:,1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)

def reverb(s1, s2):
    b = torch.nn.functional.pad(s2, (0, s1.shape[1]-s2.shape[1]))
    a = torch.rfft(s1, 2)
    b = torch.rfft(b, 2)
    return torch.irfft(complex_multiplication(a,b), 2, onesided=True, signal_sizes=s1.shape)

def reverb_np(s1, s2):
    n = s1.shape[1]
    a = np.fft.rfft(s1, n)
    b = np.fft.rfft(s2, n)
    return np.fft.irfft(a*b, n)
