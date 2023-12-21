import torch
import numpy as np
from torch.utils import data

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Config:
    caption_data_path = 'caption.pth'
    img_feature_path = 'results.pth'
    batch_size = 8
    shuffle = True
    num_workers = 4 # 用于指定数据加载时使用的子进程数目


def create_collate_fn(padding, eos, max_length=50):
    def collate_fn(img_cap):
        # 将多个样本拼接在一起成一个batch
        img_cap.sort(key=lambda p: len(p[1]), reverse=True)
        imgs, caps, indexs = zip(*img_cap)
        imgs = torch.cat([img.unsqueeze(0) for img in imgs], 0)
        lengths = [min(len(c) + 1, max_length) for c in caps]
        batch_length = max(lengths)
        cap_tensor = torch.LongTensor(batch_length, len(caps)).fill_(padding)
        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[end_cap, i] = eos
            cap_tensor[:end_cap, i].copy_(c[:end_cap])
        return (imgs, (cap_tensor, lengths), indexs)

    return collate_fn


class CaptionDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        data = torch.load(opt.caption_data_path)
        word2ix = data['word2ix']
        self.captions = data['caption']
        self.padding = word2ix.get(data.get('padding'))
        self.end = word2ix.get(data.get('end'))
        self._data = data
        self.ix2id = data['ix2id']
        self.all_imgs = torch.load(opt.img_feature_path)

    def __getitem__(self, index):
        img = self.all_imgs[index]
        caption = self.captions[index]
        rdn_index = np.random.choice(len(caption), 1)[0]
        caption = caption[rdn_index]

        return img, torch.LongTensor(caption), index

    def __len__(self):
        return len(self.ix2id)


def get_dataloader(opt):
    dataset = CaptionDataset(opt)
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=opt.shuffle,
                                 num_workers=0, # 单进程加载数据
                                 collate_fn=create_collate_fn(dataset.padding, dataset.end))

    return dataloader




if __name__ == '__main__':
    opt = Config()
    dataloader = get_dataloader(opt)
     
    for ii, data in enumerate(dataloader):
        print(ii, data)
        break
 