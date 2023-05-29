import csv
import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.nn as nn

import torch.nn.functional as F

from nltk.corpus import stopwords 

from collections import Counter

import re

import nltk

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

nltk.download('stopwords')

from torch.utils.data import Dataset

random.seed(42)

def generate_toy_fruit_data(num_pos_attr=5):
    fruits = ["Acai",
            "Ackee",
            "Apple",
            "Apricot",
            "Avocado",
            "Babaco",
            "Banana",
            "Bilberry",
            "Blackberry",
            "Blackcurrant",
            "Blood Orange",
            "Blueberry",
            "Boysenberry",
            "Breadfruit",
            "Brush Cherry",
            "Canary Melon",
            "Cantaloupe",
            "Carambola",
            "Casaba Melon",
            "Cherimoya",
            "Cherry",
            "Clementine",
            "Cloudberry",
            "Coconut",
            "Cranberry",
            "Crenshaw Melon",
            "Cucumber",
            "Currant",
            "Curry Berry",
            "Custard Apple",
            "Damson Plum",
            "Date",
            "Dragonfruit",
            "Durian",
            "Eggplant",
            "Elderberry",
            "Feijoa",
            "Finger Lime",
            "Fig",
            "Gooseberry",
            "Grapes",
            "Grapefruit",
            "Guava",
            "Honeydew Melon",
            "Huckleberry",
            "Italian Prune Plum",
            "Jackfruit",
            "Java Plum",
            "Jujube",
            "Kaffir Lime",
            "Kiwi",
            "Kumquat",
            "Lemon",
            "Lime",
            "Loganberry",
            "Longan",
            "Loquat",
            "Lychee",
            "Mammee",
            "Mandarin",
            "Mango",
            "Mangosteen",
            "Mulberry",
            "Nance",
            "Nectarine",
            "Noni",
            "Olive",
            "Orange",
            "Papaya",
            "Passion fruit",
            "Pawpaw",
            "Peach",
            "Pear",
            "Persimmon",
            "Pineapple",
            "Plantain",
            "Plum",
            "Pomegranate",
            "Pomelo",
            "Prickly Pear",
            "Pulasan",
            "Quine",
            "Rambutan",
            "Raspberries",
            "Rhubarb",
            "Rose Apple",
            "Sapodilla",
            "Satsuma",
            "Soursop",
            "Star Apple",
            "Star Fruit",
            "Strawberry",
            "Sugar Apple",
            "Tamarillo",
            "Tamarind",
            "Tangelo",
            "Tangerine",
            "Ugli",
            "Velvet Apple",
            "Watermelon"]
    attributes = ["adorable",
              "adventurous",
              "aggressive",
                "agreeable",
              "alert",
              "alive",
                "amused",
              "angry",
              "annoyed",
                "annoying",
              "anxious",
              "arrogant",
                "ashamed",
              "attractive",
              "average",
                "awful",
              "bad",
              "beautiful",
                "better",
              "bewildered",
              "black",
                "bloody",
              "blue",
              "blue-eyed",
                "blushing",
              "bored",
              "brainy",
                "brave",
              "breakable",
              "bright",
                "busy",
              "calm",
              "careful",
                "cautious",
              "charming",
              "cheerful",
                "clean",
              "clear",
              "clever",
                "cloudy",
              "clumsy",
              "colorful",
                "combative",
              "comfortable",
              "concerned",
                "condemned",
              "confused",
              "cooperative",
                "courageous",
              "crazy",
              "creepy",
                "crowded",
              "cruel",
              "curiouscute",
              "dangerous",
              "darkdead",
              "defeated",
              "defiant",
                "delightful",
              "depressed",
              "determined",
                "different",
              "difficult",
              "disgusted",
                "distinct",
              "disturbed",
              "dizzy",
                "doubtful",
              "drab",
              "dull",
                "eager",
              "easy",
              "elated",
                "elegant",
              "embarrassed",
              "enchanting",
                "encouraging",
              "energetic",
              "enthusiastic",
                "envious",
              "evil",
              "excited",
                "expensive",
              "exuberant",
              "fair",
                "faithful",
              "famous",
              "fancy",
                "fantastic",
              "fierce",
              "filthy",
                "fine",
              "foolish",
              "fragile",
                "frail",
              "frantic",
              "friendly",
                "frightened",
              "funny",
              "gentle",
                "gifted",
              "glamorous",
              "gleaming",
                "glorious",
              "good",
              "gorgeous",
                "graceful",
              "grieving",
              "grotesque",
                "grumpy",
              "handsome",
              "happy",
                "healthy",
              "helpful",
              "helpless",
                "hilarious",
              "homeless",
              "homely",
                "horrible",
              "hungry",
              "hurtill",
              "important",
              "impossible",
                "inexpensive",
              "innocent",
              "inquisitive",
                "itchy",
              "jealous",
              "jittery",
                "jolly",
              "joyous",
              "kind",
                "lazy",
              "light",
              "lively",
                "lonely",
              "long",
              "lovely",
                "lucky",
              "magnificent",
              "misty",
                "modern",
              "motionless",
              "muddy",
                "mushy",
              "mysterious",
              "nasty",
                "naughty",
              "nervous",
              "nice",
                "nutty",
              "obedient",
              "obnoxiousodd",
              "old-fashioned",
              "open",
                "outrageous",
              "outstanding",
              "panicky",
                "perfect",
              "plain",
              "pleasant",
                "poised",
              "poor",
              "powerful",
                "precious",
              "prickly",
              "proud",
                "putrid",
              "puzzled",
              "quaint",
                "real",
              "relieved",
              "repulsive",
                "rich",
              "scary",
              "selfish",
                "shiny",
              "shy",
              "silly",
                "sleepy",
              "smiling",
              "smoggy",
                "sore",
              "sparkling",
              "splendid",
                "spotless",
              "stormy",
              "strange",
                "stupid",
              "successful"]

    train_dataset = []
    for fruit in fruits:
        positive = random.choices(attributes, k=num_pos_attr)
        for attr in attributes:
            if attr in positive:
                train_dataset.append({"data": f"The {fruit} has attribute {attr}", "label": 1})
            else:
                train_dataset.append({"data": f"The {fruit} has attribute {attr}", "label": 0})

    keys = train_dataset[0].keys()
    with open('train_toy_fruit.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(train_dataset)

    return train_dataset

def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tokenize(x_train,y_train):
    word_list = []

    stop_words = set(stopwords.words('english')) 
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    # tokenize
    final_list_train, final_list_test = [],[]
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = [1 if label == 1 else 0 for label in y_train]  
    return np.array(final_list_train), np.array(encoded_train), onehot_dict

class CustomTextDataset(Dataset):
    def __init__(self, data):
        self.data = data
            
    def __getitem__(self, idx):
        data = self.data[idx]
        sample = {"data": data["data"], "label": data["label"]}
        return sample

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    x = CustomTextDataset(generate_toy_fruit_data)
    x[0]