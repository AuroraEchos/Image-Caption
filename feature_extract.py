import tqdm
import torch
import torchvision
from torch.utils import data
import os
from PIL import Image

class Config:
    caption_data_path = 'files/data_file/caption.pth'
    img_path = 'E:\\Ai_Challenger_Caption_2017\\ai_challenger_caption_train_20170902\\caption_train_images_20170902'

    batch_size = 8
    shuffle = True
    num_workers = 4 


torch.set_grad_enabled(False)
opt = Config()

IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
normalize = torchvision.transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)

class CaptionDataset(data.Dataset):
    def __init__(self, caption_data_path):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor(),
            normalize
        ])

        data = torch.load(caption_data_path)
        self.ix2id = data['ix2id']
        self.imgs = [os.path.join(opt.img_path, self.ix2id[_]) \
                     for _ in range(len(self.ix2id))]
        
    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transforms(img)
        
        return img, index
    
    def __len__(self):
        return len(self.imgs)


def get_dataloader(opt):
    dataset = CaptionDataset(opt.caption_data_path)
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers,
                                 )
    
    return dataloader

def main():
    # 数据
    opt.batch_size = 256
    dataloader = get_dataloader(opt)
    results = torch.Tensor(len(dataloader.dataset), 2048).fill_(0)
    batch_size = opt.batch_size

    # 模型
    resnet50 = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1)
    del resnet50.fc
    resnet50.fc = lambda x: x
    resnet50.cuda()

    # 前向传播，计算分数
    for ii, (imgs, indexs) in tqdm.tqdm(enumerate(dataloader)):
        assert indexs[0] == batch_size * ii
        
        imgs = imgs.cuda()
        features = resnet50(imgs)
        results[ii * batch_size:(ii + 1) * batch_size] = features.data.cpu()

    torch.save(results, 'files/data_file/results.pth')

if __name__ == '__main__':
    main()