import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils.beam_search import CaptionGenerator
import time

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

    def forward(self, img_feats, captions, lengths):
        embeddings = self.embedding(captions)
        img_feats = self.fc(img_feats).unsqueeze(0)
        embeddings = torch.cat([img_feats, embeddings], 0)
        packed_embeddings = pack_padded_sequence(embeddings, lengths)
        outputs, state = self.rnn(packed_embeddings)
        pred = self.classifier(outputs[0])

        return pred, state
    
    def generate(self, img, eos_token='</EOS>',
                 beam_size=3,
                 max_caption_length=30,
                 length_normalization_factor=0.0):
        cap_gen = CaptionGenerator(embedder=self.embedding,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.word2ix[eos_token],
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)
        if next(self.parameters()).is_cuda:
            img = img.cuda()
        img = img.unsqueeze(0)
        img = self.fc(img).unsqueeze(0)
        sentences, score = cap_gen.beam_search(img)
        sentences = [' '.join([self.ix2word[idx] for idx in sent])for sent in sentences]

        return sentences
    
    def states(self):
        opt_state_dict = {attr: getattr(self.opt, attr)
                          for attr in dir(self.opt)
                          if not attr.startswith('__')}
        
        return {'state_dict': self.state_dict(), 'opt': opt_state_dict}
    
    def save(self, path=None, **kwargs):
        if path is None:
            path = '{prefix}_{time}'.format(prefix=self.opt.prefix,
                                            time=time.strftime('%m%d_%H%M'))
        states = self.states()
        states.update(kwargs)
        torch.save(states, path)

        return path
    
    def load(self, path, load_opt=False):
        data = torch.load(path, map_location=lambda s, l: s)
        state_dict = data['state_dict']
        self.load_state_dict(state_dict)

        if load_opt:
            for k, v in data['opt'].items():
                setattr(self.opt, k, v)

        return self
    
    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)