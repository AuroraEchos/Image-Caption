import torch
from torch import nn

import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)
from utils.beam_search import CaptionGenerator

class CaptionModel(nn.Module):
    def __init__(self, opt, word2ix, ix2word):
        super(CaptionModel, self).__init__()
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.opt = opt
        self.fc = nn.Linear(2048, opt.rnn_hidden)

        self.rnn = nn.LSTM(opt.embedding_dim, opt.rnn_hidden, num_layers=opt.num_layers)
        self.classifier = nn.Linear(opt.rnn_hidden, len(word2ix))
        self.embedding = nn.Embedding(len(word2ix), opt.embedding_dim)

    def generate(self, img, eos_token='</EOS>',
                 beam_size=3,
                 max_caption_length=30,
                 length_normalization_factor=0.0):
        # 根据图片生成描述,主要是使用beam search算法以得到更好的描述
        cap_gen = CaptionGenerator(embedder=self.embedding,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.word2ix[eos_token],
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)
        if next(self.parameters()).is_cuda:
            img = img.cuda()
        
        img =img.unsqueeze(0)
        img = self.fc(img).unsqueeze(0)

        sentences, scores = cap_gen.beam_search(img)
        sentences = [' '.join([self.ix2word[idx.item()] for idx in sent]) for sent in sentences]
        
        """ 
        scores_tensor = torch.tensor(scores)
        best_idx = torch.argmax(scores_tensor)
        best_sentence = [' '.join([self.ix2word[idx.item()] for idx in sentences[best_idx]])]
        
        print("Best Sentence Score:", scores[best_idx])
        print("Best Sentence:", best_sentence)
        """
        return sentences
    
    def load(self, path, load_opt=False):
        data = torch.load(path, map_location=lambda s, l: s)
        state_dict = data['state_dict']
        self.load_state_dict(state_dict)

        if load_opt:
            for k, v in data['opt'].items():
                setattr(self.opt, k, v)

        return self