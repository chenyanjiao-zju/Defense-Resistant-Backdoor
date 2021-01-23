
model_path = "model.caffemodel"
# add  'force_backward: true' in the prototxt file otherwise the caffe does not do backward computation and gradient is 0
model_definition   = 'deploy.prototxt'
gpu = False
