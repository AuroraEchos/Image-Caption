import torch
import json
import jieba
import tqdm

class Config:
    annotation_file = 'json/caption_train_annotations_20170902.json'
    unknown     = '</UNKNOWN>'
    end         = '</EOS>'
    padding     = '</PAD>'
    max_words   = 10000
    min_appear  = 2
    save_path   = 'files/data_file/caption.pth'

def process(**kwargs):
    opt = Config()
    for k,v in kwargs.items():
        setattr(opt, k, v)
    
    with open(opt.annotation_file) as f:
        data = json.load(f)

    id2ix = {item['image_id']: ix for ix, item in enumerate(data)}
    ix2id = {ix: id for id, ix in (id2ix.items())}
    assert id2ix[ix2id[10]] == 10

    """ 
    # 打印 id2ix 字典
    print("id2ix Dictionary:")
    for image_id, index in id2ix.items():
        print(f"Image ID: {image_id}, Index: {index}") 
    
    # 打印 ix2id 字典
    print("\nix2id Dictionary:")
    for index, image_id in ix2id.items():
        print(f"Index: {index}, Image ID: {image_id}")
    """

    captions = [item['caption'] for item in data]
    cut_captions = [[list(jieba.cut(ii, cut_all=False)) for ii in item] for item in tqdm.tqdm(captions)]
    
    """ 
    # 打印 captions 列表
    print("\nCaptions List:")
    for caption in captions:
        print(caption)
    
    # 打印 cut_captions 列表
    print("\ncut_Captions List:")
    for caption in cut_captions:
        print(caption)
    
    """
    
    word_nums = {}

    def update(word_nums):
        def fun(word):
            word_nums[word] = word_nums.get(word, 0) + 1
            return None

        return fun
    
    lambda_ = update(word_nums)
    _ = {lambda_(word) for sentences in cut_captions for sentence in sentences for word in sentence}

    word_nums_list = sorted([(num, word) for word, num in word_nums.items()], reverse=True)

    # 以上操作为无损可逆操作
    # 以下操作删除一些信息 --丢弃词频不够的词 --丢弃长度过长的词

    words = [word[1] for word in word_nums_list[:opt.max_words] if word[0] >= opt.min_appear]
    words = [opt.unknown, opt.padding, opt.end] + words
    word2ix = {word: ix for ix, word in enumerate(words)}
    ix2word = {ix: word for word, ix in word2ix.items()}
    assert word2ix[ix2word[123]] == 123

    """
    # 打印 word2ix 字典
    print("\nword2ix Dictionary:")
    for word, index in word2ix.items():
        print(f"Word: {word}, Index: {index}")
    
    # 打印 ix2word 字典
    print("\nix2word Dictionary:")
    for index, word in ix2word.items():
        print(f"Index: {index}, Word: {word}")

    """ 
    
    ix_captions = [[[word2ix.get(word, word2ix.get(opt.unknown)) for word in sentence]
                    for sentence in item]
                   for item in cut_captions] 
    """ 
    # 打印 ix_captions
    print("\nix_captions List:")
    for item in ix_captions:
        for sentence in item:
            print(sentence)
    """

    readme = u"""
    word：词
    ix:index
    id:图片名
    caption: 分词之后的描述，通过ix2word可以获得原始中文词
    """

    results = {
        'caption': ix_captions,
        'word2ix': word2ix,
        'ix2word': ix2word,
        'ix2id'  : ix2id,
        'id2ix'  : id2ix,
        'padding': '</PAD>',
        'end'    : '</EOS>',
        'readme' : readme
    }

    torch.save(results, opt.save_path)
    print('save file in %s' % opt.save_path)

    def test(ix, ix2=4):
        results = torch.load(opt.save_path)
        ix2word = results['ix2word']
        examples = results['caption'][ix][4]
        sentences_p = (''.join([ix2word[ii] for ii in examples]))
        sentences_r = data[ix]['caption'][ix2]
        assert sentences_p == sentences_r, 'test failed'

    test(1000)
    print('test success')

process()