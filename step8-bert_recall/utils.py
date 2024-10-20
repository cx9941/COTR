import numpy as np

import re

def simple_word_tokenize(text):
    # 使用正则表达式将字符串中的单词提取出来
    # \w+ 匹配一个或多个字母、数字或下划线
    # [a-zA-Z]+ 匹配一个或多个字母（忽略数字和特殊符号）
    words = re.findall(r'\b\w+\b', text)
    return words

def word_level_edit_distance(sent1, sent2):
    # 将句子按词拆分
    words1 = simple_word_tokenize(sent1)
    words2 = simple_word_tokenize(sent2)
    
    len1 = len(words1)
    len2 = len(words2)

    # 创建一个(len1+1) x (len2+1) 的矩阵，用来存储编辑距离
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)

    # 初始化第一列和第一行
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # 动态规划填充矩阵
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,    # 删除
                               dp[i][j - 1] + 1,    # 插入
                               dp[i - 1][j - 1] + 1)  # 替换

    return dp[len1][len2]

if __name__ == '__main__':
    # 测试函数
    sentence1 = "I am learning Python for to1 process natural language."
    sentence2 = "I am learning Python to process natural language."
    distance = word_level_edit_distance(sentence1, sentence2)
    print(f"Word-level edit distance: {distance}")

    # # 测试分词函数
    # sentence = "I am learning Python for natural language processing, and it's fun!"
    # tokens = simple_word_tokenize(sentence)
    # print(tokens)