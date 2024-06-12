import csv
from langchain.vectorstores import FAISS
from utils import utils


class Data:
    """
    Data class for loading data from local files.
    """

    def __init__(self, config):
        self.config = config
        self.items = {}
        self.users = {}
        self.db = None
        self.tot_relationship_num = 0
        self.netwerk_density = 0.0
        self.role_id = -1
        self.load_items(config["item_path"])
        self.load_users(config["user_path"])
        self.load_relationship(config["relationship_path"])
        self.load_faiss_db(config["index_name"])

    def load_faiss_db(self, index_name):
        """
        Load faiss db from local if exists, otherwise create a new one.

        """
        _,embeddings =utils.get_embedding_model()
        try:
            self.db = FAISS.load_local(index_name, embeddings)
            print("Load faiss db from local")
        except:
            titles = [item["name"] for item in self.items.values()]
            self.db = FAISS.from_texts(titles, embeddings)
            self.db.save_local(index_name)

    def load_relationship(self, file_path):
        """
        Load relationship of agents from local file.
        """
        with open(file_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                user_1, user_2, relationship, _ = row
                user_1 = int(user_1)
                user_2 = int(user_2)

                if user_1 not in self.users or user_2 not in self.users:
                    continue
                if "contact" not in self.users[user_1]:
                    self.users[user_1]["contact"] = dict()

                self.users[user_1]["contact"][user_2] = relationship
                self.tot_relationship_num += 1

    def load_items(self, file_path):
        """
        Load items from local file.
        """
        with open(file_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header line
            for row in reader:
                item_id, title, genre, description = row
                self.items[int(item_id)] = {
                    "name": title.strip(),
                    "genre": genre,
                    "description": description.strip(),
                    "inter_cnt": 0,
                    "mention_cnt": 0,
                }

    def load_users(self, file_path):
        """
        Load users from local file.
        """
        cnt = 0
        with open(file_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header line
            for row in reader:
                user_id, name, gender, age, traits, status, interest, feature = row
                self.users[cnt] = {
                    "name": name,
                    "gender": gender,
                    "age": int(age),
                    "traits": traits,
                    "status": status,
                    "interest": interest,
                    "feature": feature,
                }
                cnt += 1
                if self.get_user_num() == self.config["agent_num"]:
                    break

    def load_role(
        self, id, name, gender, age, traits, status, interest, feature, relationships
    ):
        """
        @ Zeyu Zhang
        Load the role user into this `Data` object. Then other agents can interact with the role.
        """
        self.role_id = id
        self.users[id] = {
            "name": name,
            "gender": gender,
            "age": int(age),
            "traits": traits,
            "status": status,
            "interest": interest,
            "feature": feature,
        }
        for id2, rel_value in relationships.items():
            if "contact" not in self.users[id]:
                self.users[id]["contact"] = dict()
            self.users[id]["contact"][id2] = rel_value

    def get_full_items(self):
        return list(self.items.keys())

    def get_inter_popular_items(self):
        """
        Get the most popular items based on the number of interactions.
        """
        ids = sorted(
            self.items.keys(), key=lambda x: self.items[x]["inter_cnt"], reverse=True
        )[:3]
        return self.get_item_names(ids)

    def add_inter_cnt(self, item_names):
        item_ids = self.get_item_ids(item_names)
        for item_id in item_ids:
            self.items[item_id]["inter_cnt"] += 1

    def add_mention_cnt(self, item_names):
        item_ids = self.get_item_ids(item_names)
        for item_id in item_ids:
            self.items[item_id]["mention_cnt"] += 1

    def get_mention_popular_items(self):
        """
        Get the most popular items based on the number of mentions.
        """
        ids = sorted(
            self.items.keys(), key=lambda x: self.items[x]["mention_cnt"], reverse=True
        )[:3]
        return self.get_item_names(ids)

    def get_item_names(self, item_ids):
        return ["<" + self.items[item_id]["name"] + ">" for item_id in item_ids]

    def get_item_ids(self, item_names):
        item_ids = []
        for item in item_names:
            for item_id, item_info in self.items.items():
                if item_info["name"] in item:
                    item_ids.append(item_id)
                    break
        return item_ids

    def get_item_ids_exact(self, item_names):
        item_ids = []
        for item in item_names:
            for item_id, item_info in self.items.items():
                if item_info["name"] == item:
                    item_ids.append(item_id)
                    break
        return item_ids

    def get_full_users(self):
        return list(self.users.keys())

    def get_user_names(self, user_ids):
        return [self.users[user_id]["name"] for user_id in user_ids]

    def get_user_ids(self, user_names):
        user_ids = []
        for user in user_names:
            for user_id, user_info in self.users.items():
                if user_info["name"] == user:
                    user_ids.append(user_id)
                    break
        return user_ids

    def get_user_num(self):
        """
        Return the number of users.
        """
        return len(self.users.keys())

    def get_item_num(self):
        """
        Return the number of items.
        """
        return len(self.items.keys())

    def get_relationship_num(self):
        """
        Return the number of relationships.
        """
        return self.tot_relationship_num

    def get_role_id(self):
        """
        Return the number of relationships.
        """
        return self.role_id

    def search_items(self, item, k=50):
        """
        Search similar items from faiss db.
        Args:
            item: str, item name
            k: int, number of similar items to return
        """
        docs = self.db.similarity_search(item, k)
        item_names = [doc.page_content for doc in docs]
        return item_names

    def get_all_contacts(self, user_id):
        """
        Get all contacts of a user.
        """
        if "contact" not in self.users[user_id]:
            print(f"{user_id} has no contact.")
            return []
        ids = []
        for id in self.users[user_id]["contact"]:
            if id < self.get_user_num():
                ids.append(id)
        return self.get_user_names(ids)

    def get_all_contacts_id(self, user_id):
        """
        Get all contacts of a user.
        """
        if "contact" not in self.users[user_id]:
            print(f"{user_id} has no contact.")
            return []
        ids = []
        for id in self.users[user_id]["contact"]:
            if id < self.get_user_num():
                ids.append(id)
        return ids

    def get_relationships(self, user_id):
        """
        Get all relationships of a user.
        """
        if "contact" not in self.users[user_id]:
            print(f"{user_id} has no contact.")
            return dict()
        return self.users[user_id]["contact"]

    def get_relationship_names(self, user_id):
        """
        Get all relationship IDs of a user.
        """
        if "contact" not in self.users[user_id]:
            print(f"{user_id} has no contact.")
            return dict()
        relatiobnships = dict()
        for id in self.users[user_id]["contact"]:
            relatiobnships[self.users[id]["name"]] = self.users[user_id]["contact"][id]
        return relatiobnships

    def get_network_density(self):
        """
        Get the network density of the social network.
        """
        self.network_density = round(
            self.tot_relationship_num
            * 2
            / (self.get_user_num() * (self.get_user_num() - 1)),
            2,
        )
        return self.network_density

    def get_item_description_by_id(self, item_ids):
        """
        Get description of items by item id.
        """
        return [self.items[item_id]["description"] for item_id in item_ids]

    def get_item_description_by_name(self, item_names):
        """
        Get description of items by item name.
        """
        item_descriptions = []
        for item in item_names:
            found = False
            for item_id, item_info in self.items.items():
                if item_info["name"] == item.strip(" <>"):
                    item_descriptions.append(item_info["description"])
                    found = True
                    break
            if not found:
                item_descriptions.append("")
        return item_descriptions

    def get_genres_by_id(self, item_ids):
        """
        Get genre of items by item id.
        """
        # return [self.items[item_id]["genre"] for item_id in item_ids]
        return [
            genre
            for item_id in item_ids
            for genre in self.items[item_id]["genre"].split('|')
        ]
      
