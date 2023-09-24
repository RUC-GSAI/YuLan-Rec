import csv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


class Data:
    """
    Data class for loading data from local files.
    """
    def __init__(self, config):
        self.config = config
        self.items = {}
        self.users = {}
        self.db = None
        self.load_items(config["item_path"])
        self.load_users(config["user_path"])
        self.load_relationship(config["relationship_path"])
        self.load_faiss_db(config["index_name"])

    def load_faiss_db(self, index_name):
        """
        Load faiss db from local if exists, otherwise create a new one.
        
        """
        embeddings = OpenAIEmbeddings()
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
                user_1, user_2, relationship = row
                user_ids = self.get_user_ids([user_1, user_2])
                if user_ids[0] not in self.users or user_ids[1] not in self.users:
                    continue
                if "contact" not in self.users[user_ids[0]]:
                    self.users[user_ids[0]]["contact"] = []
                self.users[user_ids[0]]["contact"].append(user_2)
                if "contact" not in self.users[user_ids[1]]:
                    self.users[user_ids[1]]["contact"] = []
                self.users[user_ids[1]]["contact"].append(user_1)

    def load_items(self, file_path):
        """
        Load items from local file.
        """
        with open(file_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header line
            for row in reader:
                item_id, title, genre,description = row
                self.items[int(item_id)] = {"name": title, "genre": genre,"description":description}

    def load_users(self, file_path):
        """
        Load users from local file.
        """
        with open(file_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header line
            for row in reader:
                user_id, name, age, traits, status, observations = row
                self.users[int(user_id)] = {
                    "name": name,
                    "age": int(age),
                    "traits": traits,
                    "status": status,
                    "observations": observations,
                }

    def get_full_items(self):
        return list(self.items.keys())

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

    def get_full_users(self):
        return list(self.users.keys())

    def get_user_names(self, user_ids):
        return [self.users[user_id]["name"] for user_id in user_ids]

    def get_user_ids(self, user_names):
        user_ids = []
        for user in user_names:
            for user_id, user_info in self.users.items():
                if user_info["name"] ==user:
                    user_ids.append(user_id)
                    break
        return user_ids

    def search_items(self, item, k=5):
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
        return self.users[user_id]["contact"]
    
    def get_item_descriptions(self,item_names):
        """
        Get description of items.
        """
        item_descriptions = []
        for item in item_names:
            for item_id, item_info in self.items.items():
                if item_info["name"] in item:
                    item_descriptions.append(item_info["description"])
                    break
        return item_descriptions
