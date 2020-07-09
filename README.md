# *GuGuGirls*: Masked Face Recognition with Mask Generation

By Feifei Ding, Xiaoyu Wang and Kejing Yang

### Privacy

The test dataset `MTCNN_align_600id` is our private dataset and should not be released. It can only be used in our project.


### Contents
0. [Mask Generation](#mask-generation)
0. [Masked Face Recognition](#masked-face-recognition)
0. [Video Demo](#video-demo)


### Mask Generation

#### Step 1: Data Preparation
Download the training set (`CASIA-WebFace`). Detect faces and facial landmarks in CAISA-WebFace using `MTCNN`. Align faces to a canonical pose using similarity transformation. (see: [MTCNN - face detection & alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)). 

#### Step 2: Model Preparation
Download the Dlib model from [Google Drive](https://drive.google.com/file/d/16Zv5y2MJUShO6xNE_hV45WdzN-zesMJ5/view?usp=sharing) and put it in the directory **`$GuGuGirls/MaskGeneration/models/`**


#### Step 3: Generate Masked Faces
**Note:** In this step, we assume you are in the directory **`$GuGuGirls/MaskGeneration/`**
Change the paths in generate_mask.py and run it. TWe get masked faces based on WebFace.

```Shell
python3 generate_mask.py
``` 
	

### Masked Face Recognition
**Note:** In this part, we assume you are in the directory **`$GuGuGirls/FaceRecognition/`**

#### Step 1: Data Preparation
We combine original images without masks with our generated masked faces as training data.
Change the paths of datasets and run:

```Shell
python3 script/dataset/transform.py
```

The test data is our private `MTCNN_align_600id`. You can download from [Google Drive](https://drive.google.com/drive/folders/1e5AHQ7qNPZZ6QldWfs-cfYJyj7mkQ4JM?usp=sharing). 
Change the paths of datasets and run:

```Shell
python3 script/dataset/transform_test.py
```

#### Step 2: Training
Change the paths in the file and run:
	
```Shell
python3 script/experiment/train.py
```
You can download the pretrained model from [Google Drive](https://drive.google.com/file/d/1BNDbwM_SS9GX7g2kSaStllN8it8FRtJt/view?usp=sharing). 

#### Step 3: Test
Change the paths in the file and run:
	
```Shell
python3 script/experiment/test.py
```

The results on `MTCNN_align_600id` are:

	Experiment |#1|#2|#3 (released)|#4|#5
	:---:|:---:|:---:|:---:|:---:|:---:
	ACC|99.24%|99.20%|**99.30%**|99.27%|99.13%


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







