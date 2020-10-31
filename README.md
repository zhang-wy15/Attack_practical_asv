# Attack_practical_asv
This is the code and note for our ICASSP 2021 submission: "Attack on practical speaker verification system using universal adversarial perturbations"

## How to run the code
### 1. Data preparation
You should download [LibriSpeech test-clean data](http://www.openslr.org/resources/12/test-clean.tar.gz) and [BUT Speech@FIT Reverb Database](https://obj.umiacs.umd.edu/gammadata/dataset/eq/IRs_release.zip). Remember the path where they are saved.

### 2. Pretrained models
Their [model](https://github.com/clovaai/voxceleb_trainer) is used as our speaker embedding encoder. Please download the pretrained model [here](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model) and put it in ``./checkpoint/`` folder.

### 3. Generate splits
There are 40 speaker in LibriSpeech test clean set. We have to select enrolling, training, and testing audios for each speaker. You can set the number of enrolling and training audios in ``./datas/splits.py``. Also you have to change the ``wav_path`` in it to your save path. Just run ``python ./datas/splits.py`` to generate the split files in ``./datas/splits/`` folder.

## Experimental results
Coming soon.

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
