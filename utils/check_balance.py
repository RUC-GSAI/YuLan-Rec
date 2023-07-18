import datetime
import requests
from typing import List
import argparse
from yacs.config import CfgNode
def check_balance(api_keys: List[str]):
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=90)
    end_date = now + datetime.timedelta(days=1)

    # 设置API请求URL和请求头
    url_subscription = "https://api.openai.com/v1/dashboard/billing/subscription"  # 查是否订阅
    url_usage = f"https://api.openai.com/v1/dashboard/billing/usage?start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}"  # 查使用量

    for key in api_keys:
        headers = {
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json"
        }

        # 获取API限额
        response = requests.get(url_subscription, headers=headers)
        subscription_data = response.json()
        total_granted = subscription_data["hard_limit_usd"]

        # 获取已使用量
        response = requests.get(url_usage, headers=headers)
        usage_data = response.json()
        total_used = usage_data["total_usage"] / 100

        # 计算剩余额度
        total_available = total_granted - total_used
        print(f"已使用：{total_used}\n剩余：{total_available}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()
    # create config
    config = CfgNode(new_allowed=True)
    config.merge_from_file(args.config_file)
    check_balance(config.api_keys)
    