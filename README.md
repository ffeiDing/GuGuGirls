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

rank1 |rank5|rank10|mAP
:---:|:---:|:---:|:---:
85.39%|92.54%|94.21%|45.61%



### Video Demo
You can download [Demo](demo.mov) here.
We use google [colab](https://colab.research.google.com/drive/1esy30GNED8aHe7NWjE2GHZUT-wkUMHw5?usp=sharing) to show our demo. You can download [GuGuGirls](https://drive.google.com/drive/folders/16SdaTUtBLBU7arjZ_qem0SeW232hGi4I?usp=sharing) and [MTCNN_align_600id](https://drive.google.com/drive/folders/1e5AHQ7qNPZZ6QldWfs-cfYJyj7mkQ4JM?usp=sharing) to your own google drive. Follow the instructions in our colab to test our pretrained model interactively!








