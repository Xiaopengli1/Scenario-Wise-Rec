import pandas as pd

file_dir_dev = "/content/drive/MyDrive/rec_dataset/mind/dev"
file_dir_train = "/content/drive/MyDrive/rec_dataset/mind/train"

hebaviours = pd.read_table(file_dir_train+"/behaviors.tsv",sep='\t',header=None)[[1,4]]
hebaviours.columns = ["User_ID","Impressions"]
u_id_l = []
news_id_l = []
label_l = []

for i in range(len(hebaviours)):
    u_id = hebaviours['User_ID'][i]
    hebaviours_list = hebaviours['Impressions'][i].strip().split(" ")
    for item in hebaviours_list:
        news_id, label = item.split("-")[0], item.split("-")[1]
        u_id_l.append(u_id)
        news_id_l.append(news_id)
        label_l.append(label)

c={"User_ID" : u_id_l,
   "News_ID" : news_id_l,
   "label" : label_l}#将列表a，b转换成字典
data=pd.DataFrame(c)#将字典转换成为数据框

news = pd.read_table(file_dir_train+"/news.tsv",sep='\t',header=None)[[0,1,2]]
news.columns = ["News_ID","Category","SubCategory"]
data_new = pd.merge(data,news, on='News_ID',how='left')

train = data_new

hebaviours = pd.read_table(file_dir_dev+"/behaviors.tsv",sep='\t',header=None)[[1,4]]
hebaviours.columns = ["User_ID","Impressions"]
u_id_l = []
news_id_l = []
label_l = []

for i in range(len(hebaviours)):
    u_id = hebaviours['User_ID'][i]
    hebaviours_list = hebaviours['Impressions'][i].strip().split(" ")
    for item in hebaviours_list:
        news_id, label = item.split("-")[0], item.split("-")[1]
        u_id_l.append(u_id)
        news_id_l.append(news_id)
        label_l.append(label)

c={"User_ID" : u_id_l,
   "News_ID" : news_id_l,
   "label" : label_l}#将列表a，b转换成字典
data=pd.DataFrame(c)#将字典转换成为数据框

news = pd.read_table(file_dir_dev+"/news.tsv",sep='\t',header=None)[[0,1,2]]
news.columns = ["News_ID","Category","SubCategory"]
data_new = pd.merge(data,news, on='News_ID',how='left')

dev = data_new

data = pd.concat([train,dev])

Category_selected = set(["news","lifestyle","sports","finance"])
data_selected = data[data['Category'].apply(lambda x: x in Category_selected)]
data_selected.reset_index(drop=True,inplace=True)
data_selected.to_csv('mind_all.csv', index=False)

data_selected_sample = data_selected.sample(n=8000, random_state=1)
data_selected_sample.to_csv('mind_sample.csv', index=False)