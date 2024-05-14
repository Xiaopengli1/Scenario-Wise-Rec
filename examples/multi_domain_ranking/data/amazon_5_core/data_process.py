import json
import pandas as pd

# Amazon - Beauty
user = []
item = []
label = []
with open("reviews_Beauty_5.json", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        d = json.loads(line.strip())
        user.append(d["reviewerID"])
        item.append(d["asin"])
        label.append(int(d["overall"]))
dict = {'user': user, 'item': item, 'label': label}
df_beauty = pd.DataFrame(dict)
df_beauty["domain_indicator"] = 0

# Amazon - Clothing
user = []
item = []
label = []
with open("reviews_Clothing_Shoes_and_Jewelry_5.json", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        d = json.loads(line.strip())
        user.append(d["reviewerID"])
        item.append(d["asin"])
        label.append(int(d["overall"]))
dict = {'user': user, 'item': item, 'label': label}
df_cloth = pd.DataFrame(dict)
df_cloth["domain_indicator"] = 1

# Amazon - Health
user = []
item = []
label = []
with open("reviews_Health_and_Personal_Care_5.json", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        d = json.loads(line.strip())
        user.append(d["reviewerID"])
        item.append(d["asin"])
        label.append(int(d["overall"]))
dict = {'user': user, 'item': item, 'label': label}
df_Health = pd.DataFrame(dict)
df_Health["domain_indicator"] = 2

# Concat three domains
data = pd.concat([df_beauty,df_cloth,df_Health])
data.reset_index(drop=True,inplace=True)

# label mapping
data.label = data.label.apply(lambda x: 1 if x>3 else 0)
data.to_csv('amazon_5_core.csv', index=False)