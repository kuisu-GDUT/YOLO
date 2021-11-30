# README

implementation of YOLOv1

## 结构
|-- checkpoint #存放保持的模型\
|-- test_img #待测试的图像\
|-- data.py #读取以处理好的YOLO v1格式的图像以及标注数据\
|-- model.py #构建YOLO v1的模型以及损失函数的训练\
|-- test.py #测试模\
|-- train.py #训练模型\
|-- utils.py #实现将labels[7,7,30]转为bbox[n,6]->[x1,y1,x2,y2,confidence,cls]


**Required:**

* pytorch
* torchvision
* numpy
* opencv
* VOC2012 Dataset

**What can this repo do now**

* train with VOC2012 Dataset
* inference(test images)

## structure

![image-20211130114217237](E:\Git\YOLO\YOLOV1\README.assets\YOLOV1_ResNet38.png)
