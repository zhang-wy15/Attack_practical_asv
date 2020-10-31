# Attack_practical_asv
This is the code and note for our ICASSP 2021 submission: "Attack on practical speaker verification system using universal adversarial perturbations"

<div align="center">  
<img src="./images/practical.jpg" width = "575" height = "184"/>
</div>

## How to run the code
### 1. Data preparation
You should download [LibriSpeech test-clean data set](http://www.openslr.org/resources/12/test-clean.tar.gz) and [BUT Speech@FIT Reverb Database](https://obj.umiacs.umd.edu/gammadata/dataset/eq/IRs_release.zip). Remember the path where they are saved.

### 2. Pretrained models
Their [model](https://github.com/clovaai/voxceleb_trainer) is used as our speaker embedding encoder. Please download the pretrained model [here](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model) and put it in ``./checkpoint/`` folder. To evaluate EER and minDCF for the pretrained model, you can change the ```path config``` in ```./config/eval_libri_speaker.yaml``` and run the following command. You will a **1.36% EER** with **0.4321 score threshold** on LibriSpeech test-clean data set.

```
python Testspeaker.py --config config/eval_libri_speaker.yaml
```

### 3. Generate splits
There are 40 speaker in LibriSpeech test clean set. We have to select enrolling, training, and testing audios for each speaker. You can set the number of enrolling and training audios in ``./datas/splits.py``. Also you have to change the ``wav_path`` in it to your save path. Just run ``python ./datas/splits.py`` to generate the split files in ``./datas/splits/`` folder.

### 4. Perform two-step attack
Change configs in ```./config/attack_config.yaml``` and then run the following command. It has four main parts: enrolling for every speaker, match differen adversary and targeted speaker pairs including intra-gender and inter-gender matchs, generate adversarial perturbation for each pair on train audios, evaluate the perturbation on test audios. The results will be written into a txt file in your config ```out_path```.

```
python attack.py --config config/attack_config.yaml
```

### 5. Evaluate adversarial examples

```
python Testattack.py --config config/test_config.yaml
```

## Experimental results
Coming soon.

Physical attack:

<div align="center">  
<img src="./images/physical_attack.jpg" width = "474" height = "355"/>
</div>

## References
Our ASV model code is cloned from their project.

```
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Interspeech},
  year={2020}
}
```

## Cite
If you find our paper is useful for your work, please cite the following.

```
@unpublished{wei2021practical,
  title={Attack on practical speaker verification system using universal adversarial perturbations},
  author={Weiyi, Zhang and Shuning, Zhao and Le, Liu and Jianmin, Li and Xingliang, Cheng and Thomas, Fang Zheng and Xiaolin, Hu},
  note = {Submitted},
  year={2021}
}
```
