model_path = "./vgg16.caffemodel"
# add  'force_backward: true' in the prototxt file otherwise the caffe does not do backward computation and gradient is 0
model_definition   = './VGG_ILSVRC_16_layers_deploy.prototxt.txt'
gpu = False