# 1st place solution for PBVS 2025 Thermal Image Super-Resolution Challenge (TISR) - Track1

## Environment

- [PyTorch >= 1.12.1](https://pytorch.org/)
- Python 3.8

### Installation

```
git clone https://github.com/Raojiyong/PBVS_TSR.git
conda create --name py38 python=3.8 -y
conda activate py38
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
cd ThermalSR
pip install -r requirements.txt
python setup.py develop
```

## Preparation

- Download the dataset from [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/21247#participate-get_data) and put it in the `./datasets` folder.

```
${PROJECT_ROOT}
|── datasets
    │── track1
        │-- thermal
        │   │-- test
            │   │-- sisr_x8
        │   |-- train
            │   │-- GT
            │   │-- LR_x8
        │   │-- val
            │   │-- GT
            │   │-- LR_x8
```

- Download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/16x-lAvfyrAqbwAMBYsGizj0Erk0Rp-k6?usp=sharing) and put it in
  the `./experiments/pretrain_models` folder.
- And the `DRCT-L-X4.pth` pretrained model is from [GoogleDrive](https://drive.google.com/file/d/1bVxvA6QFbne2se0CQJ-jyHFy94UOi3h5/view), and used for initializing the model.
```
${PROJECT_ROOT}
|-- experiments
     |-- pretrained_models
     |   |-- pretrained_model.pth
     |   |-- net_g_model_best.pth
     |   |-- DRCT-L-X4.pth
```

Note that the `pretrained_model.pth` is trained on the combination dataset, so you need to follow `utils/merge_image.py` and merge the training dataset first in the
training stage.

And the `net_g_model_best.pth` is the **best performance** model after finetuning on the training set.

## How To Test

- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.
- Then run the following codes (taking `options/test/test_DRCT-L_SRx8_finetune_from_merged.yml` as an example):

```
python drct/test.py -opt options/test/DRCT_SRx4_ImageNet-pretrain.yml
# or
bash test.sh
```

The testing results will be saved in the `./results` folder.

## How To Train

- Refer to `./options/train` for the configuration file of the model to train.
- The training command is like

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29521 drct/train.py -opt options/train/train_coslr_multiloss_multiscale_DRCT-L_SRx8_finetune_from_merged.yml --launcher pytorch 
# or
bash train.sh
```

The training logs and weights will be saved in the `./experiments` folder.

## Thanks

A part of our work has been facilitated by [DRCT](https://github.com/ming053l/DRCT), [TISR](https://github.com/upczww/TISR/tree/master) framework, and we are
grateful for their outstanding contributions.

## Contact

If you have any question, please email raojyon@gmail.com to discuss with the author.
