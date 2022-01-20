# channels-pruning
tensorflow channels pruning

### API

* Keras                     2.3.1
* kerassurgeon       0.2.0
* tensorflow             2.3.0
* tensorflow-gpu     2.2.0
* python                    3.7.0

### 过程

* 权重稀疏化训练-添加L1正则项
* 根据验证集及现有API-kerassurgeon，确定需要减去的通道
  * conv/dense等可以直接使用现有API
  * DepthwiseConv2D
    * 需要在kerassurgeon源码中添加如下代码，用于确定需要减去通道
* 加载原先网络参数，新建新的模型，只取原先的网络参数中的通道
  * shortcut-add：在得到相加的两层conv1/conv2，利用现有API确定需要减去的通道，以list的形式，例如conv1-[ 4  6  9 11 12 14 16 26 30 36 38 42 45]，conv2-[ 6  7  9 11 12 13 14 16 24 29 34]
    * 两个list取交集 [6,9,11,12,14]，两个conv都以交集的list为最终结果
    * 以conv2-层数较多/相对于conv1处于整个网络后面，提取的特征相对是高维特征，以conv2的list为两个conv的最终结果
  * MobileNet中一个conv = pw * dw * pw， 通过API可以得到三层，需要减去的通道list，例如pw1-[ 0  4  6  8 12 13 15 16 17 19 20 23 24 30 39 40 42 44]，dw-[ 1  5  8 15 16 17 19 20 21 24 32 34 45 47]，pw2-[ 4  5 16 17 19 20 23 24 28 30 32 34 46]
    * 三个list取交集，例子中不存在
    * 以dw的list为三层的最终结果，因为两个pw都是1 * 1的卷积
    * 理论上不需要在剪枝，因为本身MobileNet参数相比于之前的一层conv就比较少了
* 新模型加载减去通道之后的参数，验证当前的结果，重新训练及微调新模型

~~~python
### ps
### 稀疏训练-API-tensorflow_model_optimization，官网存在教程
### short-cut和DW的剪裁方法二，都是个人的选择，同时因为没有API处理，需要对自己的模型很熟悉，一般整个模型的weights里面，len(conv/dw)==2,权重+偏置；len(BN)==4,权重+偏置+均值+方差
### 训练-减去通道-再训练的过程中，有预热-回温，具体可以在参考当年的论文
~~~

~~~python
### 添加的部分，针对dw的定位得到mask
### vi ../anaconda3/envs/py37/lib/python3.7/site-packages/kerassurgeon/surgeon.py
588         elif layer_class == 'DepthwiseConv2D':
589             if np.all(inbound_masks):
590                 new_layer = layer
591                 outbound_mask = None
592             else:
593                 if data_format == 'channels_first':
595                     inbound_masks = np.swapaxes(inbound_masks, 0, -1)
597                 k_size = layer.kernel_size
601                 my_mask = np.swapaxes(inbound_masks, 0, -1)
603
604                 channel_indices = []
605                 for i in range(my_mask.shape[0]):
606                     if np.where(my_mask[i] == False)[0].size != 0:
607                         channel_indices.append(i)
610                 weights = [np.delete(layer.get_weights()[0], channel_indices, axis=-2)]
613                 if len(layer.get_weights()) == 2:
614                     weights.append(layer.get_weights()[1])
615
616                 config = layer.get_config()
617                 print("Config", config)
618                 config['weights'] = weights
619                 new_layer = type(layer).from_config(config)
620                 output_shape = layer.output_shape[1:]
621                 outbound_mask = inbound_masks[:output_shape[0], :output_shape[1], :]
622                 print("Outputmask Shape", outbound_mask.shape)
623     
### 源码
625         else:
626             # Not implemented:
627             # - Lambda
628             # - SeparableConv2D
629             # - Conv2DTranspose
630             # - LocallyConnected1D
631             # - LocallyConnected2D
632             # - TimeDistributed
633             # - Bidirectional
634             # - Dot
635             # - PReLU
636             # Warning/error needed for Reshape if channels axis is split
637             raise ValueError('"{0}" layers are currently '
638                              'unsupported.'.format(layer_class))
~~~

### 案例

* 以识别的网络为例-MobileNet V2
  * model.py：MobileNet V2
  * model_res.py：将MobileNet V2中的 pw * dw * pw 转回原先的一层conv
  * model_vgg.py：去掉shortcut
* 数据
  * data_generator.py
  * image_array.npy/label_one_hot.npy
* 训练
  * main_dataset_fit_generator.py
* 剪枝+再训练
  * model_pruning_1.py：model_vgg.py
  * model_pruning_2.ipynb：model_res.py
  * model_pruning_3.ipynb：model.py

* 案例中训练周期及数据的量级较少
