import os
import torch
from util import labels2bbox
from prepare_data import GL_CLASSES
import torchvision.transforms as transforms
from PIL import Image
import cv2
import argparse


COLOR = [(255,0,0),(255,125,0),(255,255,0),(255,0,125),(255,0,250),
         (255,125,125),(255,125,250),(125,125,0),(0,255,125),(255,0,0),
         (0,0,255),(125,0,255),(0,125,255),(0,255,255),(125,125,255),
         (0,255,0),(125,255,125),(255,255,255),(100,100,100),(0,0,0),]  # 用来标识20个类别的bbox颜色，可自行设定


class TestInterface(object):
    """
    网络测试接口，
    main(): 网络测试主函数
    """
    def __init__(self, opts):
        self.opts = opts
        print("=======================Start inferring.=======================")

    def main(self):
        """
        具体测试流程根据不同项目有较大区别，需要自行编写代码，主要流程如下：
        1. 获取命令行参数
        2. 获取测试集
        3. 加载网络模型
        4. 用网络模型对测试集进行测试，得到测试结果
        5. 根据不同项目，计算测试集的评价指标， 或者可视化测试结果
        """
        opts = self.opts
        img_list = os.listdir(opts.dataset_dir)
        trans = transforms.Compose([
            # transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
        model = torch.load(opts.weight_path)
        if opts.use_GPU:
            model.to(opts.GPU_id)
        for img_name in img_list:
            img_path = os.path.join(opts.dataset_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = trans(img)
            img = torch.unsqueeze(img, dim=0)
            print(img_name, img.shape)
            if opts.use_GPU:
                img = img.to(opts.GPU_id)
            preds = torch.squeeze(model(img), dim=0).detach().cpu()
            preds = preds.permute(1,2,0)
            bbox = labels2bbox(preds)
            draw_img = cv2.imread(img_path)
            self.draw_bbox(draw_img, bbox)

    def draw_bbox(self, img, bbox):
        """
        根据bbox的信息在图像上绘制bounding box
        :param img: 绘制bbox的图像
        :param bbox: 是(n,6)的尺寸，0:4是(x1,y1,x2,y2), 4是conf， 5是cls
        """
        h, w = img.shape[0:2]
        n = bbox.shape[0]
        for i in range(n):
            confidence = bbox[i, 4]
            if confidence<0.2:#todo 控制显示box的置信度
                continue
            p1 = (int(w * bbox[i, 0]), int(h * bbox[i, 1]))
            p2 = (int(w * bbox[i, 2]), int(h * bbox[i, 3]))
            cls_name = GL_CLASSES[int(bbox[i, 5])]
            print(cls_name, p1, p2)
            cv2.rectangle(img, p1, p2, COLOR[int(bbox[i, 5])])
            cv2.putText(img, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(img, str(confidence), (p1[0],p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.imshow("bbox", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    # 网络测试代码

    parser = argparse.ArgumentParser()
    """options for inference"""
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_GPU", action="store_true", help="identify whether to use gpu")
    parser.add_argument("--GPU_id", type=int, default=None, help="device id")
    parser.add_argument("--dataset_dir", type=str, default=r"E:\kuisu\Datasets\voc\VOC2012\voc2012_forYolov1\img")
    parser.add_argument("--weight_path", type=str,
                             default=r"checkpoints/epoch60.pkl",
                             help="load path for model weight")

    opts = parser.parse_args()
    if torch.cuda.is_available():
        opts.use_GPU = True
        opts.GPU_id = torch.cuda.current_device()
        print("use GPU %d to train." % (opts.GPU_id))
    else:
        print("use CPU to train.")

    test_interface = TestInterface(opts)
    test_interface.main()  # 调用测试接口