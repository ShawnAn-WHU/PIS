# Codes for “Pretrain A Remote Sensing Foundation Model by **P**romoting **I**ntra-instance **S**imilarity"

This is a code demo for the paper: Pretrain A Remote Sensing Foundation Model by ***P***romoting ***I***ntra-instance ***S***imilarity.

we have realsed our pre-trained PIS models and related materials:
- [x] PIS-R50-IV16-E30-b128+SP & PIS-SwinB-IV16-E30-B48+SP
- [x] Codes for pretraining
- [ ] Codes for downstream fine-tuning
  - [x] Scene classification
  - [ ] Semantic segmentation
  - [ ] Change detection
     
## Dataset
- Pretraining dataset: the compressed 8-bit RGB-version [SSL4EO-S12](https://mediatum.ub.tum.de/1702379) dataset.
- Downstream datasets:
  - Scene classification: [UC Merced (UCM)](http://weegee.vision.ucmerced.edu/datasets/landuse.html), [Aerial image dataset (AID)](https://captain-whu.github.io/AID/), [PatternNet](https://sites.google.com/view/zhouwx/dataset), [NWPU-RESISC45 (NR)](https://gcheng-nwpu.github.io/#Datasets) and [EuroSAT (RGB version)](https://github.com/phelber/eurosat).
  - Semantic segmentation: [DLRSD](https://sites.google.com/view/zhouwx/dataset#h.p_hQS2jYeaFpV0), [ISPRS Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/), [LoveDA](https://doi.org/10.5281/zenodo.5706578).
  - Change detection: [CDD](https://gitlab.citius.usc.es/hiperespectral/ChangeDetectionDataset).
 
## Pretraining
1. Pretrain with ResNet-50 backbone.

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5' python pretrain.py --arch resnet50 --bs 128 --lr 0.3 --epoch 30 --data SSL4EO_RGB_MIX --num_patch 16 --tcr 1 --patch_sim 400
```

2. Pretrain with Swin Transforemr-Base backbone.

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5' python pretrain.py --arch swin_b --bs 48 --lr 3e-4 --epoch 30 --data SSL4EO_RGB_MIX --num_patch 16 --tcr 4 --patch_sim 200
```

## Fine-tuning
1. Scene classification

e.g.,
```bash
cd transfer_classification
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='0' python finetune.py --arch resnet50 --bs 12 --lr 5e-4 --epoch 100 --data ucm --num_patch 16 --num_sampels 5 --model_path <your pretrained model path>
```

2. Semantic segmentation

e.g.,
```bash
cd transfer_segmentation
```

3. Change detection

e.g.,
```bash
cd transfer_detection
```
