def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_strd = previous_conv_size[0] / out_pool_size[i]
        w_strd = previous_conv_size[1] / out_pool_size[i]
        h_wid = previous_conv_size[0] - h_strd * out_pool_size[i] + 1
        w_wid = previous_conv_size[1] - w_strd * out_pool_size[i] + 1
        maxpool = nn.MaxPool2d((h_wid,w_wid),stride=(h_strd,w_strd))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp