import abc
from recommender.model import *
import importlib
import torch
import pandas as pd
from utils import utils, message
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class Recommender:
    """
    Recommender System class
    """

    def __init__(self, config, logger, data):
        self.data = data
        self.config = config
        self.logger = logger
        self.page_size = config["page_size"]
        self.random_k = config["rec_random_k"]
        module = importlib.import_module("recommender.model")
        self.model = getattr(module, config["rec_model"])(
            config, self.data.get_user_num(), self.data.get_item_num()
        )

        self.criterion = nn.BCEWithLogitsLoss()
        if config["rec_model"]=='Random':
            self.optimizer = None
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.epoch_num = config["epoch_num"]
        self.train_data = []

        self.record = {}
        self.round_record = {}
        self.positive = {}
        self.inter_df = None
        self.inter_num = 0
        for user in self.data.get_full_users():
            self.record[user] = []
            self.positive[user] = []
            self.round_record[user] = []

    def train(self):
        if len(self.train_data) == 0:
            return
        users = [x[0] for x in self.train_data]
        items = [x[1] for x in self.train_data]
        labels = [x[2] for x in self.train_data]

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(users), torch.tensor(items), torch.tensor(labels)
        )

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=True
        )

        self.model.train()

        for epoch in tqdm(range(self.epoch_num)):
            epoch_loss = 0.0
            for user, item, label in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(user, item)
                loss = self.criterion(outputs, label.float())
                print(f"epoch:{epoch}\n loss:{loss}\n")
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.logger.info(
                f"Epoch {epoch+1}/{self.epoch_num}, Loss: {epoch_loss/len(train_loader)}"
            )

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def swap_items(self, lst, page_size, random_k):
        total_pages = len(lst) // page_size
        lst = lst[: total_pages * page_size]
        for page in range(1, total_pages // 2 + 1):  # 只需要迭代前一半的页面
            # 计算当前页面和对称页面的开始和结束索引
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size - 1
            symmetric_start_idx = (total_pages - page) * page_size
            symmetric_end_idx = symmetric_start_idx + page_size

            # 交换random_k个item
            for k in range(1, random_k + 1):
                lst[end_idx - k], lst[symmetric_end_idx - k] = (
                    lst[symmetric_end_idx - k],
                    lst[end_idx - k],
                )

        return lst

    def add_random_items(self, user, item_ids):
        item_ids = self.swap_items(item_ids, self.page_size, self.random_k)
        return item_ids

    def get_full_sort_items(self, user, random=False):
        """
        Get a list of sorted items for a given user.
        """
        items = self.data.get_full_items()
        user_tensor = torch.tensor(user)
        items_tensor = torch.tensor(items)
        sorted_items = self.model.get_full_sort_items(user_tensor, items_tensor)
        if self.random_k > 0 and random == True:
            sorted_items = self.add_random_items(user, sorted_items)
        sorted_items = [item for item in sorted_items if item not in self.record[user]]
        sorted_item_names = self.data.get_item_names(sorted_items)
        description = self.data.get_item_description_by_id(sorted_items)
        items = [
            sorted_item_names[i]
            + ";;"
            + description[i]
            + ";; Genre: "
            + self.data.get_genres_by_id([sorted_items[i]])[0]
            for i in range(len(sorted_item_names))
        ]
        return sorted_items, items

    def get_item(self, idx):
        item_name = self.data.get_item_names([idx])[0]
        description = self.data.get_item_description_by_id([idx])[0]
        item = item_name + ";;" + description
        return item

    def get_search_items(self, item_name):
        return self.data.search_items(item_name)

    def get_inter_num(self):
        return self.inter_num

    def update_history_by_name(self, user_id, item_names):
        """
        Update the history of a given user.
        """
        item_names = [item_name.strip(" <>'\"") for item_name in item_names]
        item_ids = self.data.get_item_ids(item_names)
        self.record[user_id].extend(item_ids)

    def update_history_by_id(self, user_id, item_ids):
        """
        Update the history of a given user.
        """
        self.record[user_id].extend(item_ids)

    def update_positive(self, user_id, item_names):
        """
        Update the positive history of a given user.
        """
        item_ids = self.data.get_item_ids(item_names)
        if len(item_ids) == 0:
            return
        self.positive[user_id].extend(item_ids)
        self.inter_num += len(item_ids)

    def update_positive_by_id(self, user_id, item_id):
        """
        Update the history of a given user.
        """
        self.positive[user_id].append(item_id)

    def save_interaction(self):
        """
        Save the interaction history to a csv file.
        """
        inters = []
        users = self.data.get_full_users()
        for user in users:
            for item in self.positive[user]:
                new_row = {"user_id": user, "item_id": item, "rating": 1}
                inters.append(new_row)

            for item in self.record[user]:
                if item in self.positive[user]:
                    continue
                new_row = {"user_id": user, "item_id": item, "rating": 0}
                inters.append(new_row)

        df = pd.DataFrame(inters)
        df.to_csv(
            self.config["interaction_path"],
            index=False,
        )

        self.inter_df = df

    def add_train_data(self, user, item, label):
        self.train_data.append((user, item, label))

    def clear_train_data(self):
        self.train_data = []

    def get_entropy(
        self,
    ):
        tot_entropy = 0
        for user in self.record.keys():
            inters = self.record[user]
            genres = self.data.get_genres_by_id(inters)
            entropy = utils.calculate_entropy(genres)
            tot_entropy += entropy

        return tot_entropy / len(self.record.keys())


