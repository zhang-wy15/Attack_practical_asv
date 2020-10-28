import argparse

parser = argparse.ArgumentParser(description = "Hyperparameters")
parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')
parser.add_argument('--gpu',            type=str,   default="0",   help='Gpu number')
## Data loader
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing; 0 uses the whole files')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--test_path',      type=str,   default="",     help='Root path of test audios to evaluate the ASV model')
parser.add_argument('--test_list',      type=str,   default="",     help='List file of test audios to evaluate the ASV model')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')

## Attack path set
parser.add_argument('--wav_path',       type=str,   default="",     help='Root path of train or test wav files to load')
parser.add_argument('--enroll_path',    type=str,   default="",     help='Root path of enroll wav files to load')
parser.add_argument('--enroll_file',    type=str,   default="",     help='Enroll wav files list')
parser.add_argument('--train_file',     type=str,   default="",     help='Train wav files list')
parser.add_argument('--test_file',      type=str,   default="",     help='Test wav files list')
parser.add_argument('--out_path',       type=str,   default="./output", help='Output path to save adversarial perturbations and results')
parser.add_argument('--seed',           type=int,   default=123456, help='Random seed')

## Attack parameters
parser.add_argument('--num_train',      type=int,   default=15,     help='Number of train audios for every adversary')
parser.add_argument('--thresh',         type=float, default=0.0,    help='The threshold for ASV model')
parser.add_argument('--margine',        type=float, default=0.0,    help='Attack confidence for loss1')
parser.add_argument('--noise_len',      type=float, default=1.0,    help='Wav time length for adversarial perturbation')
parser.add_argument('--max_steps',      type=int,   default=1000,   help='Max iteration steps')
parser.add_argument('--lr',             type=int,   default=10,     help='Attack step size')
parser.add_argument('--momentum',       type=float, default=0.8,    help='Momentum factor')
parser.add_argument('--max_delta',      type=int,   default=1000,   help='Attack strength')
parser.add_argument('--batch',          type=int,   default=5,      help='Batch size')
parser.add_argument('--e1',             type=float, default=1e-3,   help='Convergence condition factor of loss1')
parser.add_argument('--e2',             type=float, default=1e-3,   help='Convergence condition factor of loss2')
parser.add_argument('--intra_gender',   type=bool,  default=True,   help='Whether doing intra-gender attack')
parser.add_argument('--inter_gender',   type=bool,  default=False,  help='Whether doing inter-gender attack')
parser.add_argument('--gamma',          type=float, default=10.0,   help='Balance factor in Loss')
parser.add_argument('--cosine_similarity', type=bool, default=True, help='Whether use cosine similarity as distance metric')

## Attack RIR simulation set
parser.add_argument('--rir',            type=bool,  default=False,  help='Whether doing rir simulation')
parser.add_argument('--attack_rir_path',type=str,   default="",     help='Root path of train rirs to load')
parser.add_argument('--test_rir_path',  type=str,   default="",     help='Root path of test rirs to load')
parser.add_argument('--physical',       type=bool,  default=False,  help='Whether doing physical attack')

