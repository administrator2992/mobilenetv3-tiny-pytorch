## YOLOV4-Tiny: Implementation of You Only Look Once-Tiny target detection model in Pytorch
---

## Table of contents
1. [Warehouse update Top News](#top-news)
2. [Related warehouse Related code](#related-warehouses)
3. [Performance Status](#performance)
4. [Required Environment Environment](#required-environment)
5. [File Download Download](#file-download)
6. [Training Steps How2train](#training-steps)
7. [Prediction step How2predict](#prediction-steps)
8. [Evaluation Steps How2eval](#evaluation-steps)
9. [Reference](#reference)

## Top News
**`2022-04`**:** Supports multi-GPU training, adds counting of number of targets for each category, adds heatmap.**  

**`2022-03`**:** Significantly updated, modified the composition of the loss, so that the proportion of classification, target, regression loss is appropriate, support for STEP, cos learning rate descent method, support for adam, sgd optimizer selection, support for adaptive adjustment of the learning rate according to the batch_size, new image cropping. **
The original repository address in the BiliBili video is: https://github.com/bubbliiiing/yolov4-tiny-pytorch/tree/bilibili

**`2021-10`**:** Significantly updated, added a large number of comments, added a large number of adjustable parameters, modified the code's constituent modules, added fps, video prediction, batch prediction and other functions. **    

## Related Warehouses
| Models | Paths |
| :----- | :----- |
YoloV3 | https://github.com/bubbliiiing/yolo3-pytorch  
Efficientnet-Yolo3 | https://github.com/bubbliiiing/efficientnet-yolo3-pytorch  
YoloV4 | https://github.com/bubbliiiing/yolov4-pytorch
YoloV4-tiny | https://github.com/bubbliiiing/yolov4-tiny-pytorch
Mobilenet-Yolov4 | https://github.com/bubbliiiing/mobilenet-yolov4-pytorch
YoloV5-V5.0 | https://github.com/bubbliiiing/yolov5-pytorch
YoloV5-V6.1 | https://github.com/bubbliiiing/yolov5-v6.1-pytorch
YoloX | https://github.com/bubbliiiing/yolox-pytorch
YoloV7 | https://github.com/bubbliiiing/yolov7-pytorch
YoloV7-tiny | https://github.com/bubbliiiing/yolov7-tiny-pytorch

## Performance
| training dataset | weights file name | test dataset | input image size | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12+COCO | [yolov4_tiny_weights_voc.pth](https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_voc.pth) | VOC-Test07 | 416x416 | - | 77.8
| VOC07+12+COCO | [yolov4_tiny_weights_voc_SE.pth](https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_voc_SE.pth) | VOC-Test07 | 416x416 | - | 78.4
| VOC07+12+COCO | [yolov4_tiny_weights_voc_CBAM.pth](https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_voc_CBAM.pth) | VOC-Test07 | 416x416 | - | 78.6
| VOC07+12+COCO | [yolov4_tiny_weights_voc_ECA.pth](https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_voc_ECA.pth) | VOC-Test07 | 416x416 | - | 77.6
| COCO-Train2017 | [yolov4_tiny_weights_coco.pth](https://github.com/bubbliiiing/yolov4-tiny-pytorch/releases/download/v1.0/yolov4_tiny_weights_coco.pth) | COCO-Val2017 | 416x416 | 21.5 | 41.0

## Required environment
torch==1.2.0

## File Download
All kinds of weights needed for training can be downloaded from Baidu.com.    
Link: https://pan.baidu.com/s/1ABR6lOd0_cs5_2DORrMSRw      
Extract code: iauv    

The VOC dataset can be downloaded from the following address, which already includes the training set, test set and validation set (same as the test set), so there is no need to divide it again:  
Link: https://pan.baidu.com/s/19Mw2u_df_nBzsC2lg20fQA    
Extract code: j5ge

## Training steps
### a. Training VOC07+12 dataset
1. Preparation of dataset   
**This paper uses VOC format for training, you need to download the VOC07+12 dataset before training, unzip it and put it in the root directory**.  

2. Processing of the dataset   
Modify the annotation_mode=2 in voc_annotation.py, run voc_annotation.py to generate 2007_train.txt and 2007_val.txt in the root directory.   

3. Start network training   
The default parameters of train.py are used to train the VOC dataset, run train.py directly to start training.   

4. Prediction of training results   
To predict the training results, we need to use two files, yolo.py and predict.py. First, we need to go to yolo.py to modify the model_path and classes_path, which must be modified.   
**model_path points to the trained weights file, which is in the logs folder.   
classes_path points to the txt corresponding to the detected categories.**   
After completing the modifications you can run predict.py for detection. After running, enter the path of the image to be detected. 

### b. Training your own dataset
1. Preparation of dataset  
**This paper uses VOC format for training, you need to make your own dataset before training, **    
Before training, put the label file in Annotation under VOC2007 folder in VOCdevkit folder.   
Put the image files in JPEGImages under VOC2007 folder in VOCdevkit folder before training.   

2 Processing of the dataset  
After finishing the placement of the dataset, we need to utilize voc_annotation.py to obtain 2007_train.txt and 2007_val.txt for training.   
Modify the parameters inside voc_annotation.py. For the first training you can modify only the classes_path, which is used to point to the txt corresponding to the detected classes.   
When you train your own dataset, you can create a cls_classes.txt with the classes you want to distinguish.   
The contents of the model_data/cls_classes.txt file are:      
```python
cat
dog
...
```
Modify the classes_path in voc_annotation.py to correspond to cls_classes.txt and run voc_annotation.py.

3. Start network training  
**There are many parameters for training, they are all in train.py, you can read the comments carefully after downloading the library, the most important part is still the classes_path in train.py.  
**classes_path is used to point to the txt corresponding to the detection category, which is the same as the txt inside voc_annotation.py! The dataset for training yourself must be modified! **  
After modifying the classes_path you can run train.py to start training, after training multiple epochs the weights will be generated in the logs folder.  

4. Prediction of training results  
Predicting training results requires two files, yolo.py and predict.py. In yolo.py, modify model_path and classes_path.  
Inside yolo.py, modify model_path and classes_path. **model_path points to the trained weights file, which is in the logs folder.  
classes_path points to the txt corresponding to the detected classes.**  
After completing the modifications you can run predict.py for detection. After running it, enter the path of the image to be detected.

## Prediction steps
### a. Use pre-trained weights
1. Unzip the library after downloading, download yolo_weights.pth from Baidu.com, put it into model_data, run predict.py, and enter  
`` python
img/street.jpg
```
2. Setup inside predict.py can do fps test and video video detection.  
### b. Use your own trained weights
1. Follow the training steps.  
2. Inside the yolo.py file, change model_path and classes_path to correspond to the trained files in the following section; **model_path corresponds to the weights file under the logs folder, and classes_path corresponds to the classes in model_path**.  
``python
_defaults = {
    #--------------------------------------------------------------------------#
    # Use your own trained model for prediction be sure to modify model_path and classes_path!
    # model_path points to the weights file in the logs folder, and classes_path points to the txt in model_data.
    # If there is a shape mismatch, also note that the model_path and classes_path parameters were changed during training.
    # --------------------------------------------------------------------------#
    "model_path" : 'model_data/yolov4_tiny_weights_coco.pth',
    "classes_path" : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    # anchors_path represents the txt file corresponding to the a priori box, which is generally not modified.
    # anchors_mask is used to help the code find the corresponding a priori box, generally not modified.
    # ---------------------------------------------------------------------#
    "anchors_mask" : [[3,4,5], [1,2,3]], ...
    #-------------------------------#
    # Type of attention mechanism used
    # phi = 0 for no attention mechanism used
    # phi = 1 for SE
    # phi = 2 for CBAM
    # phi = 3 for ECA
    # -------------------------------#
    "phi"               : 0,  
    #---------------------------------------------------------------------#
    # Enter the size of the image, must be a multiple of 32.
    # ---------------------------------------------------------------------#
    "input_shape" : [416, 416].
    #---------------------------------------------------------------------#
    # Only prediction frames with scores greater than the confidence level will be retained
    #---------------------------------------------------------------------#
    "confidence" : 0.5, ##
    #---------------------------------------------------------------------#
    # nms_iou size used for non-great suppression
    #---------------------------------------------------------------------#
    "nms_iou" : 0.3, #
    #---------------------------------------------------------------------#
    # This variable is used to control whether or not to use letterbox_image to resize the input image without distortion.
    # After several tests, it was found that turning off letterbox_image directly resizes the image better.
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
    #-------------------------------#
    # Whether to use Cuda
    # Set to False if you don't have a GPU.
    #-------------------------------#
    "cuda"              : True,
}
```
3. Run predict.py and type  
```python
img/street.jpg
```
4. Inside predict.py, you can set up fps testing and video detection.  

## Evaluation steps 
### a. Evaluate the VOC07+12 test set
1. In this paper, we use VOC format for evaluation, VOC07+12 has been divided into test sets, no need to utilize voc_annotation.py to generate the txt under ImageSets folder. 2.
2. modify model_path as well as classes_path inside yolo.py. **model_path points to the trained weights file, in the logs folder. classes_path points to the txt corresponding to the test category. **  
3. Run get_map.py to get the evaluation results, which are saved in the map_out folder.

### b. Evaluate your own dataset
1. This article uses the VOC format for evaluation.  
2. If you have run the voc_annotation.py file before training, the code will automatically divide the dataset into training set, validation set and test set. If you want to modify the ratio of the test set, you can modify the trainval_percent under the voc_annotation.py file. trainval_percent is used to specify the ratio of (training set + validation set) to the test set, by default (training set + validation set):test set = 9:1. train_percent is used to specify the ratio of training set to validation set in (training set + validation set), by default training set:validation set = 9:1.
3. After dividing the test set using voc_annotation.py, go to the get_map.py file and modify classes_path. classes_path is used to point to the txt corresponding to the detection category, which is the same as the txt used for training. Evaluating your own dataset must be modified.
4. Modify model_path and classes_path in yolo.py. **model_path points to the trained weights file in the logs folder. classes_path points to the txt that corresponds to the detected category.  
5. Run get_map.py to get the evaluation results, which are saved in the map_out folder.

## Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  
