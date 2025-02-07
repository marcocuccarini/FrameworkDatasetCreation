# -*- coding: utf-8 -*-
"""DatasetCreationBase.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DO31yFZPvttsokb5dshODu5YMwpO2mZ3
"""


import pandas as pd
import numpy as np
from ast import literal_eval
from sentence_transformers import SentenceTransformer
from Model.BERTModel import BERT_Model



def dataset_add_postive_sample(df_train, dataset_question, dataset_answer, dataset_label):

  for i in range(len(df_train['question'])):

    dataset_question.append(df_train['question'][i])
    dataset_answer.append(df_train['answer'][i])
    dataset_label.append(1)

  return dataset_question, dataset_answer, dataset_label



def dataset_add_negative_sample(df_train, dataset_question, dataset_answer, dataset_label):
  for i in range(len(questions_enc)):

    cont=0
    sim_vect = similarity_extraction(questions_enc[i], answers_enc)
    list_score, list_candidate = zip(*sorted(zip(sim_vect, df_train['answer'].tolist()), reverse=True))

    for j in list_candidate:


      if not(j==df_train['answer'][i]):

        cont+=1
        dataset_question.append(df_train['question'].tolist()[i])
        dataset_answer.append(j)
        dataset_label.append(0)

        if cont==2:
          break

  return dataset_question, dataset_answer, dataset_label

def similarity_extraction(questions_enc, answers_enc):

    KB_sim=[]

    for i in range(len(answers_enc)):
      temp=[]
      temp.append(np.dot(questions_enc,answers_enc[i]))
      KB_sim.append(temp)

    return KB_sim

for z in range(5):
  df_train=pd.read_csv("Dataset/Dataset_Base/DataSetIndex"+str(z)+".csv", sep=",")


  


  BERT = BERT_Model("multi-qa-mpnet-base-dot-v1", SentenceTransformer)




  answers_enc=BERT.text_embedding(df_train['answer'].tolist())
  questions_enc=BERT.text_embedding(df_train['question'].tolist())

  dataset_question=[]
  dataset_answer=[]
  dataset_label=[]



  dataset_question, dataset_answer, dataset_label= dataset_add_postive_sample(df_train, dataset_question, dataset_answer, dataset_label)
  dataset_question, dataset_answer, dataset_label= dataset_add_negative_sample(df_train, dataset_question, dataset_answer, dataset_label)

  df=pd.DataFrame(list(zip(dataset_question, dataset_answer, dataset_label)), columns=['question', 'answer', 'label'])

  df.to_csv("/Index"+str(z)+"BaseTrain.csv", sep=";")

for i in range(5):
  if (i==0):

   df=pd.read_csv("/Index"+str(z)+"BaseTrain.csv", sep=";")
   df_train=df

  else:

    df=pd.read_csv("/Index"+str(z)+"BaseTrain.csv", sep=";")
    frame=[df_train,df]
    df_train=pd.concat(frame)
    del df_train['Unnamed: 0']

df_train.to_csv("/TrainArgu.csv", sep=";")