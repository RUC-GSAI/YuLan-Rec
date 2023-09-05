import abc
from recommender.model import *
import importlib
import torch
import pandas as pd
from utils import utils,message


class Recommender:
    """
    Recommender System class
    """

    def __init__(self, config, data):
        self.data = data
        self.config = config
        self.page_size = config["page_size"]
        module = importlib.import_module("recommender.model")
        self.model = getattr(module, config["rec_model"])(config)
        self.record = {}
        self.positive = {}
        self.inter_df = None
        self.inter_num=0
        for user in self.data.get_full_users():
            self.record[user] = []
            self.positive[user] = []

    def get_full_sort_items(self, user):
        """
        Get a list of sorted items for a given user.
        """
        items = self.data.get_full_items()
        user_tensor = torch.tensor(user)
        items_tensor = torch.tensor(items)
        sorted_items = self.model.get_full_sort_items(user_tensor, items_tensor)
        sorted_items = [item for item in sorted_items if item not in self.record[user]]
        sorted_item_names = self.data.get_item_names(sorted_items)
        description = self.data.get_item_description_by_id(sorted_items)
        items = [sorted_item_names[i] + ";;" + description[i] for i in range(len(sorted_item_names))]
        return items

    def get_search_items(self, item_name):
        return self.data.search_items(item_name)
    
    def get_inter_num(self):
        return self.inter_num

    def update_history(self, user_id, item_names):
        """
        Update the history of a given user.
        """
        item_names = [item_name.strip(" <>'\"") for item_name in item_names]
        item_ids = self.data.get_item_ids(item_names)
        self.record[user_id].extend(item_ids)

    def update_positive(self, user_id, item_names):
        """
        Update the positive history of a given user.
        """
        item_ids = self.data.get_item_ids(item_names)
        if len(item_ids) == 0:
            return
        self.positive[user_id].extend(item_ids)
        self.inter_num+=len(item_ids)

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
