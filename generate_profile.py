from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os
import pandas as pd
import numpy as np
import csv
import io

os.environ["OPENAI_API_KEY"] = "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"


def generate_names(llm, k):
    res = llm([HumanMessage(content=f"生成{k}个不重复的常用英文名字")]).content
    res = res.split("\n")
    names = []
    for name in res:
        name = name.strip()
        name = name.split(" ")[-1]
        names.append(name)
    with open("data/names.txt", "w") as f:
        for name in names:
            f.write(name + "\n")
    return names


def generate_profiles(llm, k):
    # data = pd.read_csv('data/user.csv')

    # # 提取前 10 行数据
    # top_10_rows = data.head(10)
    # profile_example = top_10_rows.apply(lambda x: ','.join(x.astype(str)), axis=1)
    # # 将结果拼接成一个字符串
    # #profile_example = top_10_rows.to_string(index=False,sep=',')
    # header_string = ','.join(data.columns)
    # profile_example='\n'.join(profile_example)

    # profile_example=header_string+"\n"+profile_example
    # print(profile_example)
    filename = "data/user.csv"

    # 以文本形式读取CSV文件
    header = ""
    with open(filename, "r") as file:
        csv_text = ""
        for i, line in enumerate(file):
            if i == 0:
                header = line
            if i < 10:
                csv_text += line
            else:
                break
    print(csv_text)
    id = 300
    prompt = csv_text + f"\n\n根据上述信息，再续写新的{k}个人的信息，同时注意所有人之间应该存在一定的社交关系。注意：id从{id}开始"
    print(prompt)
    res = llm([HumanMessage(content=prompt)]).content
    res = res.split("\n")
    print(res)
    data_list = []
    data_list.append(header)
    cnt = 0
    for i in range(len(res)):
        if res[i] == "":
            continue
        # row = next(csv.reader(io.StringIO(res[i])))
        # row[0]=id+cnt
        data_list.append(res[i])
        # cnt+=1

    with open("data/new_user.csv", "w") as file:
        for item in data_list:
            file.write(str(item) + "\n")


# 写入CSV文件
def change_id(start_id):
    data = pd.read_csv("data/new_user.csv")
    # data['id'] should be the index + start_id
    data["id"] = data.index + start_id
    data.to_csv("data/new_user.csv", index=False)


def generate_relationship():
    pass


def generate_feature():
    # 读取CSV文件
    df = pd.read_csv("user_1000.csv")

    # 定义用户特性及其概率
    feature_probs = {
        "Watcher": 0.5,
        "Explorer": 0.3,
        "Critic": 0.3,
        "Chatter": 0.3,
        "Poster": 0.4,
    }

    # 为每一行添加新的特性列
    def add_features(row):
        # 对每个特性进行独立采样，如果采样出则添加到feature列
        row_features = []
        while len(row_features) == 0:
            row_features = [
                feature
                for feature, prob in feature_probs.items()
                if np.random.choice([True, False], p=[prob, 1 - prob])
            ]

        return ";".join(row_features)

    df["feature"] = df.apply(add_features, axis=1)

    # 将结果写回CSV
    df.to_csv("user_1000_2.csv", index=False)


def main():
    # llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo")

    # generate_profiles(llm,2)
    # change_id(300)
    # generate_relationship()
    generate_feature()


if __name__ == "__main__":
    main()
