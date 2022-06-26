import json
import pandas as pd
import jieba
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *


def tokenize(text):
    return [_i for _i in jieba.cut(text)]


if __name__ == '__main__':
    sentence_lens = []
    df = pd.DataFrame()
    with open('data/train.json', 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
        titles = []
        title_lens = []
        title_in_contents = []
        content_lens = []
        frequency = dict()
        index_loc = []

        for _passage in data:
            _title = _passage['title']
            _content = _passage['content']

            titles.append(_title)

            _title_tokens = tokenize(_title)
            _content_tokens = tokenize(_content)

            title_lens.append(len(_title_tokens))
            content_lens.append(len(_content_tokens))
            for sen in re.split("[;,!，。；？！\n]", _content):
                sentence_lens.append(len(sen))

            _token_in_content = 0
            for _c in _title_tokens:
                if _c in frequency:
                    frequency[_c] += 1
                else:
                    frequency[_c] = 1
                if _c in _content_tokens:
                    _token_in_content += 1
                    index_loc.append(_content_tokens.index(_c) / len(_content_tokens))
            title_in_contents.append(_token_in_content)

        df['title'] = titles
        df['title_len'] = title_lens
        df['title_in_content'] = title_in_contents
        df['content_len'] = content_lens

    df.info()
    df['include_rate'] = df['title_in_content'] / df['title_len']

    """
    ---------- 标题在原文中的比例 ----------
    """
    s = pd.Series(df['include_rate'])

    plt.figure(figsize=(6, 4))
    sns.distplot(s, bins=10, hist=True, kde=False, norm_hist=False,
                 rug=True, vertical=False, label='num of passages',
                 axlabel='include rate',
                 hist_kws={'color': 'b', 'edgecolor': 'k'})

    plt.legend()
    plt.show()
    """
    ---------- end ----------
    """

    """
    ---------- 标题的词在原文中的相对位置 ----------
    """
    s = pd.Series(index_loc)

    plt.figure(figsize=(6, 4))
    sns.distplot(s, bins=10, hist=True, kde=False, norm_hist=False,
                 rug=False, vertical=False, label='num of tokens',
                 axlabel='title tokens location',
                 hist_kws={'color': 'b', 'edgecolor': 'k'})

    plt.legend()
    plt.show()
    """
    ---------- end ----------
    """

    """
    ---------- 标题长度 分词计算 ----------
    """
    s = pd.Series(df['title_len'])
    print(min(df['title_len']), max(df['title_len']))

    plt.figure(figsize=(6, 4))
    fig = sns.distplot(s, bins=18, hist=True, kde=False, norm_hist=False,
                 rug=False, vertical=False, label='num of titles',
                 axlabel='title tokens length',
                 hist_kws={'color': 'b', 'edgecolor': 'k'})
    fig.set_xlim(1, 20)
    fig.set_xticks(range(1, 20))
    plt.legend()
    plt.show()
    """
    ---------- end ----------
    """

    """
    ---------- 文章长度 ----------
    """
    s = pd.Series(df['content_len'])

    plt.figure(figsize=(6, 4))
    fig = sns.distplot(s, bins=18, hist=True, kde=False, norm_hist=False,
                 rug=False, vertical=False, label='num of news',
                 axlabel='content length',
                 hist_kws={'color': 'b', 'edgecolor': 'k'})
    # fig.set_xlim(1, 20)
    # fig.set_xticks(range(1, 20))
    plt.legend()
    plt.show()
    """
    ---------- end ----------
    """

    """
    ---------- 文章句子长度 ----------
    """
    s = pd.Series(sentence_lens)

    plt.figure(figsize=(6, 4))
    sns.distplot(s, bins=18, hist=True, kde=False, norm_hist=False,
                 rug=False, vertical=False, label='num of news',
                 axlabel='sentence len',
                 hist_kws={'color': 'b', 'edgecolor': 'k'})
    # fig.set_xlim(1, 20)
    # fig.set_xticks(range(1, 20))
    plt.legend()
    plt.show()
    """
    ---------- end ----------
    """

    # 词频统计
    ls = [(k, v) for k, v in frequency.items()]
    ls.sort(key=lambda x: x[1], reverse=True)
    words = [x for (x, y) in ls]
    cnts = [y for (x, y) in ls]

    """
    ---------- 词频分布(top 20) ----------
    """
    # 设置中文字体和负号正常显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    cnts_20 = cnts[:20]
    words_20 = words[:20]
    x = range(len(cnts_20))

    rects1 = plt.bar(x, height=cnts_20, width=0.4, alpha=0.8, color='blue', label="Frequency statistics")
    # plt.ylim(0, 10)  # y轴取值范围
    plt.ylabel("Frequency")

    plt.xticks([index + 0.2 for index in x], words_20)
    plt.xlabel("word")
    plt.title("")
    plt.legend()  # 设置题注

    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")

    plt.show()
    """
    ---------- end ----------
    """

    """
    ---------- 使用词数量-覆盖标题的比例 ----------
    """
    x_axis_data = [_i for _i in range(len(words))]
    y_axis_data = []
    part = 0
    cnt_sum = sum(cnts)
    for _v in cnts:
        part += _v
        y_axis_data.append(part / cnt_sum)

    plt.plot(x_axis_data, y_axis_data, color='#4169E1', alpha=0.8, linewidth=1)
    # plt.xticks(x, x[::1])

    plt.xlabel('使用的词个数')
    plt.ylabel('累计覆盖标题占比')

    plt.show()
    """
    ---------- end ----------
    """

    """
    ---------- 标题长度-文章长度 ----------
    """
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df['content_len'], df['title_len'],
                cbar=True,  # 是否显示颜色图例
                shade=True,  # 是否填充
                cmap='YlGnBu',  # 设置调色盘
                shade_lowest=True,  # 最外围颜色是否显示
                n_levels=8,  # 曲线个数（越大，越密集）
                bw=.3,
                clip=((0, 1500), (2, 15))
                )
    # 两个维度数据生成曲线密度图，以颜色作为密度衰减显示

    plt.show()
    """
    ---------- end ----------
    """

    df.to_csv('data\\title2content.csv')
