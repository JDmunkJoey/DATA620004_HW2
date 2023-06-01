# FCOS on VOC

This project is the homework of deeplearning，training FCOS on VOC2012

Based on the paper: [https://arxiv.org/abs/1904.01355](https://arxiv.org/abs/1904.01355).

Implementation based on [tianzhi0549/FCOS: FCOS: Fully Convolutional One-Stage Object Detection (ICCV&#39;19) (github.com)](https://github.com/tianzhi0549/FCOS)

The environment of this project based on pytorch1.12

Please follow the instruction of `INSTALL.md` to install the environment, and place VOC2012 in ./

use command : `tools/train_net.py  --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml  DATALOADER.NUM_WORKERS 2  OUTPUT_DIR trained_model/fcos_imprv_R_50_FPN_1x` for train

you can change the directory of data by changing `fcos_imprv_R_50_FPN_1x.yaml` above and `./fcos_core/config/path_catalog.py`.
