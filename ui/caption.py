import re
import torch
import torchvision
from PIL import Image
from model_pre import CaptionModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Config:
    caption_path   = "C:\\Users\\lwh20\\Desktop\\LF\\Test\\caption.pth"
    model_ckpt     = "C:\\Users\\lwh20\\Desktop\\LF\\Test\\caption_0914_1947"
    scale_size     = 300
    img_size       = 224
    rnn_hidden     = 256
    embedding_dim  = 256
    num_layers     = 2
    test_img       = None

def description(image_path):
    # 基础配置
    opt = Config()
    opt.test_img = image_path

    # 数据预处理
    data = torch.load(opt.caption_path)
    word2ix,ix2word = data['word2ix'],data['ix2word']
    IMAGENET_MEAN   =  [0.485, 0.456, 0.406]
    IMAGENET_STD    =  [0.229, 0.224, 0.225]
    normalize =  torchvision.transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
    transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(opt.scale_size),
                torchvision.transforms.CenterCrop(opt.img_size),
                torchvision.transforms.ToTensor(),
                normalize
                ])
    img_ = Image.open(opt.test_img)
    img = transforms(img_).unsqueeze(0)
    img_.resize((int(img_.width*256/img_.height),256))

    # 提取图片特征
    resnet50 = torchvision.models.resnet50(True).eval()
    del resnet50.fc
    resnet50.fc = lambda x:x
    with torch.no_grad():
        img_feats = resnet50(img)

    # 加载模型
    model = CaptionModel(opt,word2ix,ix2word)
    model = model.load(opt.model_ckpt).eval()
    model.cuda()

    # 结果正则化
    results = model.generate(img_feats.data[0])
    """ 
    normalized_results = [''.join(result.split()).replace('</EOS>', '') for result in results]
    normalized_output = '\r\n'.join(normalized_results)
    print(normalized_output)

    

    return normalized_output """

    #return results



    print('\r\n'.join(results)) 


image_path = "E:\\Ai_Challenger_Caption_2017\\ai_challenger_caption_train_20170902\\caption_train_images_20170902\\0000e06c1fc586992dc2445e9e102899ccb5e3fc.jpg"
description(image_path)
