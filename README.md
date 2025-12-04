# :rocket: MVSMamba (NeurIPS 2025)

## [Arxiv](https://arxiv.org/abs/2511.01315) | [Openreview](https://openreview.net/forum?id=DLVn11YIHx)

> MVSMamba: Multi-View Stereo with State Space Model  
> Authors: Jianfei Jiang, Qiankun Liu*, Hongyuan Liu, Haochen Yu, Liyong Wang, Jiansheng Chen, Huimin Ma*   
> Institute: University of Science and Technology Beijing  
> NeurIPS 2025  

## 📢 News
* 2025-12-04: Code and pre-trained model release !

## Installation

```bash
conda create -n mvsmamba python=3.10.8
conda activate mvsmamba
pip install -r requirements.txt
```

## Data Preparation

Please refer to [RRT-MVS](https://github.com/JianfeiJ/RRT-MVS).

You need to download extra [Rectified_raw](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip) data for high-resolution training.

## Training

### Training on DTU

To train the model on DTU, specify ``DTU_TRAINING`` in ``./scripts/train_dtu.sh`` first and then run:
```
bash scripts/train_dtu.sh
```
After training, you will get model checkpoints in `./checkpoints/dtu`.

## Testing

### Testing on DTU

For DTU testing, just run:
```
bash scripts/test_dtu.sh
```

### Testing on Tanks and Temples
For TNT evaluation, just run:
```
bash scripts/test_tnt_inter.sh
```
```
bash scripts/test_tnt_adv.sh
``` 
For quantitative evaluation, you can upload your point clouds to [Tanks and Temples benchmark](https://www.tanksandtemples.org/).


## Citation
If you find this work useful in your research, please consider citing the following:
```bibtex
@article{jiang2025mvsmamba,
  title={MVSMamba: Multi-View Stereo with State Space Model},
  author={Jiang, Jianfei and Liu, Qiankun and Liu, Hongyuan and Yu, Haochen and Wang, Liyong and Chen, Jiansheng and Ma, Huimin},
  journal={arXiv preprint arXiv:2511.01315},
  year={2025}
}
```

## Acknowledgements
Our work is partially based on these opening source works [ET-MVSNet](https://github.com/TQTQliu/ET-MVSNet), [JamMa](https://github.com/leoluxxx/JamMa), and [EfficientVMamba](https://github.com/TerryPei/EfficientVMamba). We appreciate their contributions to the MVS community.
