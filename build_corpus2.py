import random
random.seed(1337)

docs = [
    ('hongloumeng_s.txt','红楼梦'), ('sanguo_s.txt','三国演义'),
    ('shuihu_s.txt','水浒传'),      ('xiyouji_s.txt','西游记'),
    ('rulin_s.txt','儒林外史'),      ('fengshen_s.txt','封神演义'),
    ('liaozhai_s.txt','聊斋志异'),   ('nahan_s.txt','鲁迅'),
    ('panghuang_s.txt','鲁迅'),
]
P = 100   # characters per tagged passage

def strip_gutenberg(t):
    a = t.find('*** START OF')
    if a != -1: t = t[t.find('\n', a)+1:]
    b = t.find('*** END OF')
    if b != -1: t = t[:b]
    return t.strip()

train, val = [], []
for fname, tag in docs:
    body = strip_gutenberg(open(fname, encoding='utf-8').read())
    passages = [body[i:i+P] for i in range(0, len(body), P)]
    tagged = ['=== %s ===\n%s' % (tag, p) for p in passages]   # tag EVERY passage
    cut = int(0.9*len(tagged))
    train += tagged[:cut]; val += tagged[cut:]                 # per-source 90/10

random.shuffle(train)   # mix styles so every batch must read the tag
open('train.txt','w',encoding='utf-8').write('\n'.join(train))
open('val.txt','w',encoding='utf-8').write('\n'.join(val))
print('train passages:', len(train), ' val passages:', len(val))
print('train chars:', sum(len(p) for p in train), ' val chars:', sum(len(p) for p in val))
