# Defense-Resistant Backdoor Attacks against DNN

## Requirements
* caffe 

## Neuron Selection
Get into a dataset dict, such as `backdoor/mnist`. 
```
python select*.py
```
Get the array/file of the index of neurons' `activation`, and the array/file of the corresponding `weight` sequence of that index. Pick an index for trigger generation.

## Trigger Generation  
Modify the path in `setting.py` and params, especially `neuron` and `layer`, in `gen_ad.sh`, then run:
```
bash gen_ad.sh
```
and get the trigger.

## Data Poisoning
```
python filter.py
```
Attach the trigger image to the clean data in order to get the poisoned data. 

## Model Retraining
- First, run `python load_data.py`, transform the image to pickle file.
- For single-trigger and single-target scheme.Run `retrain.py` to retrain the benign neural network with the combined clean and poisoned datasets to obtain the backdoored model.
  - To change the target label, modify `inpersonate` in func `load_data_trend()`.

- For multi-location retraining, refer to `mnist/mul_location_datahandler.py`;For multi-trigger retraining, refer to `mnist/mul_trigger_datahandler.py`. Then, update the code in `retrain.py` to poison test data and  `filter.py` for training data.

- Run `python read_caffe_param.py read` to get the layer params stored in the pkl file; `python read_caffe_param.py save` to get the caffemodel file from pkl file.

- Run `PA.py` to get models' prediction accuracy with clean test data; and `ASR.py` to check the backdoored models' Attack Success Rate with poisoned data.
## Attack Methods Comparsion
- For `Badnets`, we generate random trigger and obey the following steps
- For `Hidden-Trigger`, we refer to: https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks.


## Defenses
- Pruning
  - We prune a certain `proportion` of neurons in the convolution layer and test PA and ASR after pruning.

- NeuralCleanse
    - NC codes refer to: https://github.com/bolunwang/backdoor. 
    - Some of our models are `channel_first`, but the source codes are only for `channel_last` model. Therefore, we provide the corresponding modified version.
    - The source code relies on `tensorflow-gpu==1.10.1`, which is out-of-date. Thus we use `tensorflow-gpu==2.4.0`, and modify the code in  `utils_backdoor.py`.

    - `gtsrb_visualize_example.py` is the sample code for the cifar10 dataset
- ABS 
  - ABS codes refer to: https://github.com/naiyeleo/ABS. 
  - Our image processing method is shown in `preprocess.py`.
- Strip
  - STRIP source codes refer to：https://github.com/garrisongys/STRIP.
