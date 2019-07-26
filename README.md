# sppnet-pytorch
SPP layer could be added in CNN model between convolutional layer and fully-connected lay, so that you can input multi-size images into your CNN model.
</br>
</br>
The function `spatial_pyramid_pool()` in file `spp_layer.py` is independent. It could be added in your own models.
</br>
</br>
See this:<a href="https://arxiv.org/abs/1406.4729">Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition</a>



`SPP_Layer.py` provides a torch.nn.Module of spp_layer which can be inserted into any models very easily.
