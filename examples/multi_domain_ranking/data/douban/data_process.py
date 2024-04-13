import pandas as pd

user = pd.read_table("/Douban/douban_dataset(text information)/users_cleaned.txt",sep='\t',header=0)
user  = user[["living_place","join_time","UID"]]
user = user.rename(columns={'UID': 'user_id'})

Book = pd.read_table('/Douban/douban_dataset(text information)/bookreviews_cleaned.txt',sep='\t',header=0)
Book = Book[["user_id","book_id","rating"]]
Book["domain_id"] = 0
Book = Book.rename(columns={'book_id': 'item_id'})

Movie = pd.read_table('/Douban/douban_dataset(text information)/moviereviews_cleaned.txt',sep='\t',header=0)
Movie = Movie[["user_id","movie_id","rating"]]
Movie["domain_id"] = 1
Movie = Movie.rename(columns={'movie_id': 'item_id'})

Music = pd.read_table('/Douban/douban_dataset(text information)/musicreviews_cleaned.txt',sep='\t',header=0)
Music = Music[["user_id","music_id","rating"]]
Music["domain_id"] = 2
Music = Music.rename(columns={'music_id': 'item_id'})

Movie.item_id += Book.item_id.max()
Music.item_id += Movie.item_id.max()

data = pd.concat([Book,Movie,Music])
data.reset_index(drop=True,inplace=True)

data_new = pd.merge(data,user, on='user_id',how='left')

# rating mapping
data_new.rating = data_new.rating.apply(lambda x: 1 if x>3 else 0)

# join_time feature binning
data_new["join_time"] = pd.to_datetime(data_new["join_time"])
index5 = pd.date_range('2005-03','2018-04',freq='M')
val_l = list(range(len(index5)-1))
data_new["time_bin"] = pd.cut(data_new['join_time'], index5, labels=val_l)

data_new.to_csv('douban.csv', index=False)