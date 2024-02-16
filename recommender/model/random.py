import importlib
from typing import Union, List, Optional
from recommender.model import BaseModel
import random
import torch


class Random(BaseModel):
    def __init__(self, config, n_users, n_items):
        self.config = config

    def get_full_sort_items(self, users, items):
        """Get a list of sorted items for a given user."""

        sorted_items = self._sort_full_items(users, items)
        return sorted_items

    def _sort_full_items(self, user, items):
        """Return a random list of items for a given user."""
        random_items = torch.randperm(items.size(0)).tolist()
        return random_items
