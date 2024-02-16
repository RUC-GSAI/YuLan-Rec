import importlib
from typing import Union, List, Optional


class BaseModel(object):
    """Base class for all models."""

    def __init__(self, config, n_users, n_items):
        self.config = config
        self.items = None

    def get_full_sort_items(self, user_id, *args, **kwargs):
        """Get a list of sorted items for a given user."""
        raise NotImplementedError

    def _sort_full_items(self, user_id, *args, **kwargs):
        """Sort a list of items for a given user."""
        raise NotImplementedError
