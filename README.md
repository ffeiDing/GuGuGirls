# *GuGuGirls*: Masked Face Generation and Recognition

By Feifei Ding, Xiaoyu Wang and Kejing Yang

### Privacy

The test dataset MTCNN_600_id is our private dataset and should not be released.


### Contents
0. [Mased Face Generation](#Mased Face Generation)
0. [Mased Face Recognition](#Mased Face Recognition)
0. [Video Demo](#video-demo)


### Mased Face Generation

#### Step 1: Data Preparation
Download the training set (`CASIA-WebFace`). Detect faces and facial landmarks in CAISA-WebFace using `MTCNN`. Align faces to a canonical pose using similarity transformation. (see: [MTCNN - face detection & alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)). 

#### Part 2: Model Preparation
Download the Dlib model from 

#### Part 3: Generate Masked faces
**Note:** In this part, we assume you are in the directory **`$GuGuGirls/MaskGeneration/`**
Change the paths in generate_mask.py and run it.
	```Shell&Matlab
	python3 generate_mask.py
	```


2. Train the sphereface model.

	```Shell
	./code/sphereface_train.sh 0,1
	```
    After training, a model `sphereface_model_iter_28000.caffemodel` and a corresponding log file `sphereface_train.log` are placed in the directory of `result/sphereface/`.

#### Part 3: Test
**Note:** In this part, we assume you are in the directory **`$SPHEREFACE_ROOT/test/`**

1. Get the pair list of LFW ([view 2](http://vis-www.cs.umass.edu/lfw/#views)).

	```Shell
	mv ../preprocess/result/lfw-112X96 data/
	./code/get_pairs.sh
	```
	Make sure that the LFW dataset and`pairs.txt` in the directory of **`data/`**

1. Extract deep features and test on LFW.

	```Matlab
	# In Matlab Command Window
	run code/evaluation.m
	```
    Finally we have the `sphereface_model.caffemodel`, extracted features `pairs.mat` in folder **`result/`**, and accuracy on LFW like this:

	fold|1|2|3|4|5|6|7|8|9|10|AVE
	:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
	ACC|99.33%|99.17%|98.83%|99.50%|99.17%|99.83%|99.17%|98.83%|99.83%|99.33%|99.30%

### Models
1. Visualizations of network architecture (tools from [ethereon](http://ethereon.github.io/netscope/quickstart.html)):
	- SphereFace-20: [link](http://ethereon.github.io/netscope/#/gist/20f6ddf70a35dec5019a539a502bccc5)
2. Model file
	- SphereFace-20: [Google Drive](https://drive.google.com/open?id=0B_geeR2lTMegb2F6dmlmOXhWaVk) | [Baidu](http://pan.baidu.com/s/1qY5FTF2)
	- Third-party SphereFace-4 & SphereFace-6: [here](https://github.com/wy1iu/sphereface/issues/81) by [zuoqing1988](https://github.com/zuoqing1988)


### Results
1. Following the instruction, we go through the entire pipeline for 5 times. The accuracies on LFW are shown below. Generally, we report the average but we release the [model-3](#models) here.

	Experiment |#1|#2|#3 (released)|#4|#5
	:---:|:---:|:---:|:---:|:---:|:---:
	ACC|99.24%|99.20%|**99.30%**|99.27%|99.13%

2. Other intermediate results:
    - LFW features: [Google Drive](https://drive.google.com/open?id=0B_geeR2lTMegenU0cGJYZmlRUlU) | [Baidu](http://pan.baidu.com/s/1o8QIMUY)
    - Training log: [Google Drive](https://drive.google.com/open?id=0B_geeR2lTMegcWkxdVV4X1FOaFU) | [Baidu](http://pan.baidu.com/s/1i5QmXrJ)



### Video Demo
[![SphereFace Demo](https://img.youtube.com/vi/P6jEzzwoYWs/0.jpg)](https://www.youtube.com/watch?v=P6jEzzwoYWs)

Please click the image to watch the Youtube video. For Youku users, click [here](http://t.cn/RCZ0w1c).

Details:
1. It is an **open-set** face recognition scenario. The video is processed frame by frame, following the same pipeline in this repository.
2. Gallery set consists of 6 identities. Each main character has only 1 gallery face image. All the detected faces are included in probe set.
3. There is no overlap between gallery set and training set (CASIA-WebFace).
4. The scores between each probe face and gallery set are computed by cosine similarity. If the maximal score of a probe face is smaller than a pre-definded threshold, the probe face would be considered as an outlier.
5. Main characters are labeled by boxes with different colors. (
![#ff0000](https://placehold.it/15/ff0000/000000?text=+)Rachel,
![#ffff00](https://placehold.it/15/ffff00/000000?text=+)Monica,
![#ff80ff](https://placehold.it/15/ff80ff/000000?text=+)Phoebe,
![#00ffff](https://placehold.it/15/00ffff/000000?text=+)Joey,
![#0000ff](https://placehold.it/15/0000ff/000000?text=+)Chandler,
![#00ff00](https://placehold.it/15/00ff00/000000?text=+)Ross)


### Note
1. **Backward gradient**
	- In this implementation, we did not strictly follow the equations in paper. Instead, we normalize the scale of gradient. It can be interpreted as a varying strategy for learning rate to help converge more stably. Similar idea and intuition also appear in [normalized gradients](https://arxiv.org/pdf/1707.04822.pdf) and [projected gradient descent](https://www.stats.ox.ac.uk/~lienart/blog_opti_pgd.html).
	- More specifically, if the original gradient of ***f*** w.r.t ***x*** can be written as **df/dx = coeff_w \*  w + coeff_x \* x**, we use the normalized version **[df/dx] = (coeff_w \* w + coeff_x \* x) / norm_wx** to perform backward propragation, where **norm_wx** is **sqrt(coeff_w^2 + coeff_x^2)**. The same operation is also applied to the gradient of ***f*** w.r.t ***w***.
	- In fact, you do not necessarily need to use the original gradient, since the original gradient sometimes is not an optimal design. One important criterion for modifying the backprop gradient is that the new "gradient" (strictly speaking, it is not a gradient anymore) need to make the objective value decrease stably and consistently. (In terms of some failure cases for gradient-based back-prop, I recommand [a great talk](https://www.youtube.com/watch?v=jWVZnkTfB3c) by [Shai Shalev-Shwartz](https://www.cs.huji.ac.il/~shais/))
	- If you use the original gradient to do the backprop, you could still make it work but may need different lambda settings, iteration number and learning rate decay strategy. 

2. **Lambda** and **Note for training (When the loss becomes 87)**
	- Please refer to our previous [note and explanation](https://github.com/wy1iu/LargeMargin_Softmax_Loss#notes-for-training).
	
3. **According to recent advances, using feature normalization with a tunable scaling parameter s can significantly improve the performance of SphereFace on MegaFace challenge**
	- This is supported by the experiments done by [CosFace](https://arxiv.org/abs/1801.09414). Similar idea also appears in [additive margin softmax](https://arxiv.org/abs/1801.05599).

4. **Difficulties in convergence**
        - When you encounter difficulties in convergence (it may appear if you use *SphereFace* in another dataset), usually there are a few easy ways to address it.
	- First, try to use large mini-batch size. 
	- Second, try to use PReLU instead of ReLU. 
	- Third, increase the width and depth of our network. 
	- Fourth, try to use better initialization. For example, use the pretrained model from the original softmax loss (it is also equivalent to finetuning).
	- Last and the most effective thing you could try is to change the hyper-parameters for lambda_min, lambda and its decay speed.


### Third-party re-implementation
- PyTorch: [code](https://github.com/clcarwin/sphereface_pytorch) by [clcarwin](https://github.com/clcarwin).
- PyTorch: [code](https://github.com/Joyako/SphereFace-pytorch) by [Joyako](https://github.com/Joyako).
- TensorFlow: [code](https://github.com/pppoe/tensorflow-sphereface-asoftmax) by [pppoe](https://github.com/pppoe).
- TensorFlow (with awesome animations): [code](https://github.com/YunYang1994/SphereFace) by [YunYang1994](https://github.com/YunYang1994).
- TensorFlow: [code](https://github.com/hujun100/tensorflow-sphereface) by [hujun100](https://github.com/hujun100).
- TensorFlow: [code](https://github.com/HiKapok/tf.extra_losses) by [HiKapok](https://github.com/HiKapok).
- TensorFlow: [code](https://github.com/andrewhuman/sphereloss_tensorflow) by [andrewhuman](https://github.com/andrewhuman).
- MXNet: [code](https://github.com/deepinsight/insightface) by [deepinsight](https://github.com/deepinsight) (by setting loss-type=1: SphereFace).
- Model compression for SphereFace: [code](https://github.com/isthatyoung/Sphereface-prune) by [Siyang Liu](https://github.com/isthatyoung) (useful in practice)
- Caffe2: [code](https://github.com/tpys/face-recognition-caffe2) by [tpys](https://github.com/tpys).
- Trained on MS-1M: [code](https://github.com/KaleidoZhouYN/Sphereface-Ms-celeb-1M) by [KaleidoZhouYN](https://github.com/KaleidoZhouYN).
- System: [A cool face demo system](https://github.com/tpys/face-everthing) using SphereFace by [tpys](https://github.com/tpys).
- Third-party pretrained models: [code](https://github.com/goodluckcwl/Sphereface-model) by [goodluckcwl](https://github.com/goodluckcwl).

### Resources for angular margin learning

[L-Softmax loss](https://github.com/wy1iu/LargeMargin_Softmax_Loss) and [SphereFace](https://github.com/wy1iu/sphereface) present a promising framework for angular representation learning, which is shown very effective in deep face recognition. We are super excited that our works has inspired many well-performing methods (and loss functions). We list a few of them for your potential reference (not very up-to-date):

- Additive margin softmax: [paper](https://arxiv.org/abs/1801.05599) and [code](https://github.com/happynear/AMSoftmax)
- CosFace: [paper](https://arxiv.org/abs/1801.09414)
- ArcFace/InsightFace: [paper](https://arxiv.org/abs/1801.07698) and [code](https://github.com/deepinsight/insightface)
- NormFace: [paper](https://arxiv.org/abs/1704.06369) and [code](https://github.com/happynear/NormFace)
- L2-Softmax: [paper](https://arxiv.org/abs/1703.09507)
- von Mises-Fisher Mixture Model: [paper](https://arxiv.org/abs/1706.04264)
- COCO loss: [paper](https://arxiv.org/pdf/1710.00870.pdf) and [code](https://github.com/sciencefans/coco_loss)
- Angular Triplet Loss: [code](https://github.com/KaleidoZhouYN/Angular-Triplet-Loss)

To evaluate the effectiveness of the angular margin learning method, you may consider to use the **angular Fisher score** proposed in [the Appendix E of our SphereFace Paper](http://wyliu.com/papers/LiuCVPR17v3.pdf).

Disclaimer: Some of these methods may not necessarily be inspired by us, but we still list them due to its relevance and excellence.

### Contact

  [Weiyang Liu](https://wyliu.com) and [Yandong Wen](https://ydwen.github.io)

  Questions can also be left as issues in the repository. We will be happy to answer them.


# Mask faces Generation



# Masked Face Recognition
