import random
random.seed(0)

docs = [
    ('hongloumeng_s.txt','红楼梦'), ('sanguo_s.txt','三国演义'),
    ('shuihu_s.txt','水浒传'),      ('xiyouji_s.txt','西游记'),
    ('rulin_s.txt','儒林外史'),      ('fengshen_s.txt','封神演义'),
    ('liaozhai_s.txt','聊斋志异'),   ('nahan_s.txt','鲁迅'),
    ('panghuang_s.txt','鲁迅'),
]
END = '◆'
TEMPLATES = ['请用%s的风格写一段。', '模仿%s写几句。', '以%s的笔法写一段话。']

def strip_gutenberg(t):
    a=t.find('*** START OF')
    if a!=-1: t=t[t.find('\n',a)+1:]
    b=t.find('*** END OF')
    if b!=-1: t=t[:b]
    return t.strip()

examples=[]
for fname, style in docs:
    body = strip_gutenberg(open(fname,encoding='utf-8').read())
    for _ in range(600):
        i = random.randint(0, len(body)-80)
        passage = body[i:i+random.randint(40,70)].replace('\n','')
        q = random.choice(TEMPLATES) % style       # vary the wording so it learns intent
        examples.append('问：%s\n答：%s%s' % (q, passage, END))

random.shuffle(examples)
open('sft.txt','w',encoding='utf-8').write('\n'.join(examples))
print('SFT examples:', len(examples), ' chars:', sum(len(e) for e in examples))
print('\nexample:\n' + examples[0])
