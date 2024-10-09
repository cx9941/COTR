
import os
import qianfan

# 通过环境变量初始化认证信息
# 方式一：【推荐】使用安全认证AK/SK鉴权
# 替换下列示例中参数，安全认证Access Key替换your_iam_ak，Secret Key替换your_iam_sk，如何获取请查看https://cloud.baidu.com/doc/Reference/s/9jwvz2egb
os.environ["QIANFAN_ACCESS_KEY"] = "ALTAKAPLmViDOdQsF3NMPPQtx6"
os.environ["QIANFAN_SECRET_KEY"] = "61a662aab7314cecb20a122a77109fcd"

chat_comp = qianfan.ChatCompletion()

# 指定特定模型
resp = chat_comp.do(model="ERNIE-Speed-8K", messages=[{
    "role": "user",
    "content": "你好"
}])

print(resp["body"])
