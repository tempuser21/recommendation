from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json

path = 'data/'  
  
df_event = pd.read_csv(path + "df_event.csv")
df_meta = pd.read_csv(path + "df_meta.csv")
df_apriori = pd.read_csv(path + "df_apriori.csv")
df_product_detail = pd.read_csv(path + "df_product_detail.csv")


def join_frames(df_01, df_02, join_colum_list, join_type, report=True):

  join_res = pd.merge(df_01, df_02, how=join_type, on=join_colum_list)

  if report:
    print("frame 01 row size : " + str(df_01.shape[0]))
    print("frame 02 row size : " + str(df_02.shape[0]))
    print("join row size     : " + str(join_res.shape[0]))

  return join_res


def concat_df_columns(df_01, df_02):
  return pd.concat([df_01, df_02], axis=1)


def union_frames(df_01, df_02):
  return pd.concat([df_01, df_02])


def reco_list_to_names(productid, product_similarity_list, source_name):

  id_list = []

  similarity_list = []
  
  for similarity_pair in product_similarity_list:
    similarity_pair_list = similarity_pair.split(" - ")
    id_list.append(similarity_pair_list[0])
    similarity_list.append(similarity_pair_list[1])


  df = pd.DataFrame(
      {'productid': id_list,
      'similarity': similarity_list,
       'source': [source_name] * len(similarity_list)
      })

  return list(df_meta[df_meta['productid'] == productid]['name'])[0], join_frames(df, df_meta, ['productid'], 'inner', False)[['productid', 'name', 'category', 'subcategory', 'similarity', 'source']]


def apriori_to_reco(productid):
  reco_01_list = df_apriori[df_apriori['productid'] == productid]['suggestion'].tolist()
  if not reco_01_list:
    return []
  
  return reco_01_list[0].replace('[', '').replace(']', '').replace("'", "").split(", ")[:20]


def fill_recommendation_list(df_recommendation, number_of_random_products, product_id):

  ignore_list = df_recommendation['productid'].tolist()
  ignore_list.append(product_id)

  product_detail = df_product_detail[df_product_detail['productid'] == product_id]

  price = product_detail['price'].tolist()[0]
  brand = product_detail['brand'].tolist()[0]
  category = product_detail['category'].tolist()[0]
  subcategory = product_detail['subcategory'].tolist()[0]

  target_price_up = price * 3.0
  target_price_down = price * 0.5

  df_filtered_product = df_product_detail[(df_product_detail['subcategory'] == subcategory) & (df_product_detail['price'] >= target_price_down) & (df_product_detail['price'] <= target_price_up)]
  df_filtered_product = df_filtered_product[~df_filtered_product['productid'].isin(ignore_list)]

  df_filtered_product['price_similarity'] = (df_filtered_product['price'] - price).abs()
  df_filtered_product['price_similarity_normalized'] = 1 - (df_filtered_product['price_similarity']-df_filtered_product['price_similarity'].min())/(df_filtered_product['price_similarity'].max()-df_filtered_product['price_similarity'].min()) 

  df_filtered_product['similarity'] = df_filtered_product['price_similarity_normalized'] / 10
  df_filtered_product['source'] = 'random'
  df_filtered_product = df_filtered_product[['productid', 'name', 'category', 'subcategory', 'similarity', 'source']]

  df_similarity_03 = df_filtered_product.sample(number_of_random_products)
  df_similarity_03.sort_values(by=['similarity'], ascending=False, inplace=True)
  df_recommendation = union_frames(df_recommendation, df_similarity_03)

  return df_recommendation


def recommend_product(product_id):

  reco_list_02 = apriori_to_reco(product_id)
  product, df_similarity_02 = reco_list_to_names(product_id, reco_list_02, 'apriori')

  print("*** " + product + " ***\n")

  df_recommendation = df_similarity_02
  df_recommendation['similarity'] = df_recommendation['similarity'].astype(float)

  # embedding & apriori'den aynıları gelebilir diye productid'ye göre gruplama yapıyoruz
  df_recommendation = df_recommendation.groupby('productid').agg({'name': 'min', 'category': 'min', 'subcategory': 'min', 'similarity': 'max', 'source': 'sum'})
  df_recommendation.reset_index(inplace=True)

  df_recommendation.sort_values(by=['similarity'], ascending=False, inplace=True)

  number_of_target_recommendations = 10
  number_of_recommendations = df_recommendation.shape[0]

  if number_of_recommendations < number_of_target_recommendations:
      df_recommendation = fill_recommendation_list(df_recommendation, number_of_target_recommendations - number_of_recommendations, product_id)

  return df_recommendation#.head(number_of_target_recommendations)



def get_final_recommendation(product_id_list):

  all_recos = pd.DataFrame(columns=['productid', 'name', 'category', 'subcategory', 'similarity', 'source', 'product_number'])

  count = 1
  for product_id in product_id_list:

    # sırayla ürünler için tavsiyler geliyor
    rc = recommend_product(product_id)

    rc['product_number'] = str(count)
    rc['order'] = np.arange(rc.shape[0]) + 1

    # gelen tavsiyler bir araya toplanıyor
    all_recos = union_frames(all_recos, rc)

    all_recos = all_recos[~all_recos['productid'].isin(product_id_list)]

    count += 1

  all_recos.head(50)


  ### farlı ürünler için, aynı ürün tavsiye ediliyorsa, onlara öncelik verilecek, bunun sayısını öğrenmek için
  ### gruplama yapılıyor. product_number kolonunda, öoklu olan ifadeler concat'lanıyor

  res = all_recos.groupby('productid').agg({'name': 'min', 'category': 'min', 'subcategory': 'min', 'similarity': 'max', 'source': 'sum', 'product_number': 'sum', 'order': 'sum'})
  res.reset_index(inplace=True)

  # bu satırdaki ürün kaç farklı ürün için tavsiye edilmiş
  res['same_candidate_number'] = res['product_number'].str.len()

  # bunlar öncelikli olacağından üst sıraya alınıyor
  res.sort_values(by=['same_candidate_number'], ascending=False, inplace=True)

# ------------------------------------------------------------------------------------

  # birden fazla ürün için tavsiye edilmiş ürünler
  mutual_reco = res[res['same_candidate_number'] > 1]
    
  # bir kere tavsiye edilmiş ürünler
  product_reco = res[res['same_candidate_number'] <= 1]
    
  # bir kere tavsiye edilenler ürün'ün kendi içindeki tavsiye sırası ve ürün numarasına (product_number) göre küçükten büyüğe sıralanıyor
  product_reco.sort_values(by=['order', 'product_number'], ascending=True, inplace=True)
    
  # birden fazla tavsiye edilenler en üstte olacak 
  final_reco = union_frames(mutual_reco, product_reco)
    
  final_reco = final_reco.head(10)
    
  # birden fazla tavsiye edilen ürünler yukarıda olacak şekilde, tavsiye edilen ürünler benzerlik puanlarına göre sıralanıyor
  final_reco.sort_values(by=['same_candidate_number', 'similarity'], ascending=False, inplace=True)

  final_reco.head(10)

# ------------------------------------------------------------------------------------
  #### input olarak diyelim ki 7 giriş oldu ama farklı ürünler için aynıları tavsiye edildiğinde öncelikli olduğundan onlar üste çıkınca
  #### diğerlerine sıra gelmeyebiliyor. Bunun için, adına tavsiye yapılmamış ürünler tespit ediliyor (1)
  #### dahil edilmeyen ürün kadar satır sondan siliniyor (2)
  #### diğer ürünlerin ilk sıradaki tavsiyesi, sırayla sonra ekleniyor

  number_of_input = len(product_id_list)
  product_index_list = final_reco['product_number'].unique()

  product_index_str = '-'.join(product_index_list)

  not_included_list = []

  # (1)
  for i in range(1, number_of_input + 1):
    index = str(i)
    if index not in product_index_str:
      not_included_list.append(index)  


  if len(not_included_list) > 0:

    # (2)
    final_reco = final_reco.iloc[:-len(not_included_list)]

    # (3)
    for product_index in not_included_list:
      final_reco = union_frames(final_reco, product_reco[(product_reco['product_number'] == product_index) & (product_reco['order'] == 1)]) 

  final_reco.head(10)
  
  return final_reco

# ------------------------------------------------------------------------------------

product_id_list = ['HBV00000NE0SY', 'HBV00000QU3Z9', 'HBV00000NE25T', 'HBV00000NG8T3', 'AILEELITDIS5288B', 'HBV00000U27LJ']
product_id_list = ['HBV00000NE0SY']


