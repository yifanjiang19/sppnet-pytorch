# sppnet-pytorch
SPP layer could be added in CNN model between convolutional layer and fully-connected lay, so that you can input multi-size images into your CNN model. We use this structure in the paper <a href="https://arxiv.org/abs/1804.02047">Pedestrian-Synthesis-GAN: Generating Pedestrian Data in Real Scene and Beyond</a>
</br>
</br>
The function `spatial_pyramid_pool()` in file `spp_layer.py` is independent. It could be added in your own models.
</br>
</br>
See this:<a href="https://arxiv.org/abs/1406.4729">Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition</a>


## Citation
If you find this work useful for your research, please cite:
```
@article{ouyang2018pedestrian,
  title={Pedestrian-Synthesis-GAN: Generating Pedestrian Data in Real Scene and Beyond},
  author={Ouyang, Xi and Cheng, Yu and Jiang, Yifan and Li, Chun-Liang and Zhou, Pan},
  journal={arXiv preprint arXiv:1804.02047},
  year={2018}
}
```
and
```
@inproceedings{he2014spatial,
  title={Spatial pyramid pooling in deep convolutional networks for visual recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={European conference on computer vision},
  pages={346--361},
  year={2014},
  organization={Springer}
}
```