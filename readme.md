# DSRL: Dual Super-Resolution Learning for Semantic Segmentation

The Code is modified from [EdgeNet in pytorch](https://github.com/sacmehta/EdgeNets), you can follow the procedure in it to prepare the datasets and model directory files.


# Testing

  * The first step aims to save the gray prediction mask
  * The second step aims to evaluate the mIoU with prediction mask and groundtruth

```
# To evaluate ESPNetv2_DSRL, use below command:
sh run_eval_256x512.sh   
# sh run_eval_512x1024.sh
```

# Main results

| Method | s | Image Size | FLOPs | Params | mIOU (class-wise) | Link |
|---|---|---|---|---|---|---|
| ESPNetv2 | 2. 0 | 512x256 | 674.78M | 0.79M | 54.83% (val) | N/A |
| ESPNetv2 + DSRL | 2.0 | 512x256 | 674.78M | 0.79M | 60.61% (val)  | [here](ckpt-segmentation/espnetv2_dsrl/256x512/espnetv2_2.0_1024_best.pth) |
| ESPNetv2 | 2. 0 | 1024x512 | 2.7G | 0.79M | 64.44 (val) | N/A |
| ESPNetv2 + DSRL | 2.0 | 1024x512 | 2.7G | 0.79M | 66.50% (val) | [here](ckpt-segmentation/espnetv2_dsrl/256x512/espnetv2_2.0_2048_best.pth) |


# Citation
If you find this repository helpful, please feel free to cite below work:
```
@InProceedings{Wang_2020_CVPR,
author = {Wang, Li and Li, Dong and Zhu, Yousong and Tian, Lu and Shan, Yi},
title = {Dual Super-Resolution Learning for Semantic Segmentation},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}


```




