import torch
import torchvision
import tqdm
from torchnet import meter
import os
import ipdb
from utils import Visualizer

from torch.nn.utils.rnn import pack_padded_sequence
from model import CaptionModel
from data import get_dataloader
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class Config:
    caption_data_path = 'files/data_file/caption.pth'
    img_feature_path = 'files/data_file/results.pth'
    img_path = 'E:\\Ai_Challenger_Caption_2017\\ai_challenger_caption_train_20170902\\caption_train_images_20170902'

    scale_size = 300
    img_size = 224
    test_img = None
    shuffle = True
    num_workers = 4
    batch_size = 8
    rnn_hidden = 256
    embedding_dim = 256
    num_layers = 2
    share_embedding_weights = False

    prefix = 'files/model_file/caption'

    use_gpu = True
    model_ckpt = None
    env = 'caption'

    lr = 1e-3
    #epoch = 1
    epoch = 1
    plot_every = 10
    debug_file = None

    test_img = 'example.jpeg'

    debug_file = 'debug/file'

def generate(**kwargs):
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    device=torch.device('cuda') if opt.use_gpu else torch.device('cpu')

    # 数据预处理
    data = torch.load(opt.caption_data_path, map_location=lambda s, l: s)
    word2ix, ix2word = data['word2ix'], data['ix2word']

    normalize = torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.scale_size),
        torchvision.transforms.CenterCrop(opt.img_size),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    img = Image.open(opt.test_img)
    img = transforms(img).unsqueeze(0)

    # 提取图片特征
    resnet50 = torchvision.models.resnet50(True).eval()
    del resnet50.fc
    resnet50.fc = lambda x: x
    resnet50.to(device)
    img = img.to(device)
    img_feats = resnet50(img).detach()

    # Caption模型
    model = CaptionModel(opt, word2ix, ix2word)
    model = model.load(opt.model_ckpt).eval()
    model.to(device)

    # 结果
    results = model.generate(img_feats.data[0])
    print('\r\n'.join(results))

def train(**kwargs):
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    device=torch.device('cuda') if opt.use_gpu else torch.device('cpu')

    opt.caption_data_path = 'files/data_file/caption.pth'
    opt.test_img = ''

    # 数据
    #vis = Visualizer(env=opt.env)
    dataloader = get_dataloader(opt)
    _data = dataloader.dataset._data
    word2ix, ix2word = _data['word2ix'], _data['ix2word']

    # 模型
    model = CaptionModel(opt, word2ix, ix2word)
    if opt.model_ckpt:
        model.load(opt.model_ckpt)
    optimizer = model.get_optimizer(opt.lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    loss_meter = meter.AverageValueMeter()

    
    for epoch in range(opt.epoch):
        loss_meter.reset()
        for ii, (imgs, (captions, lengths), indexes) in tqdm.tqdm(enumerate(dataloader)):
            # 训练
            optimizer.zero_grad()
            imgs = imgs.to(device)
            captions = captions.to(device)
            input_captions = captions[:-1]
            target_captions = pack_padded_sequence(captions, lengths)[0]
            score,_=model(imgs, input_captions, lengths)
            loss = criterion(score, target_captions)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            """ 
            # 可视化
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot('loss', loss_meter.value()[0])

                # 可视化图片，可视化人工描述语句
                raw_img = _data['ix2id'][indexes[0]]
                img_path = opt.img_path + raw_img
                raw_img = Image.open(img_path).convert('RGB')
                raw_img = torchvision.transforms.ToTensor()(raw_img)

                raw_caption = captions.data[: , 0]
                raw_caption = ''.join([_data['ix2word'][ii] for ii in raw_caption])
                vis.text(raw_caption, u'raw_caption')
                vis.img('raw', raw_img, caption=raw_caption)

                # 可视化网络生成的描述语句
                results = model.generate(imgs.data[0])
                vis.text('</br>'.join(results), u'caption')
                 """
        model.save()

if __name__ == '__main__':
    train()