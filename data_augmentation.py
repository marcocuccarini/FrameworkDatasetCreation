from flask import Flask, render_template,jsonify,request
from flask_cors import CORS
import requests,os
from dotenv.main import load_dotenv
from langchain.memory import ConversationSummaryBufferMemory
import torch


from Script.Structure.Document import Document
from Script.Model.BERTModel import BERT_Model
from Script.Pipeline.PipelineChat import Pipeline_Chatbot
from Script.Varius import util
from sentence_transformers import SentenceTransformer
from Script.Model.LLama_Local_Augmentation import Local_LLama
from Script.Structure.Dataset_QA import Dataset_QA

#!pip install transformers==4.43.1

import transformers

import json


#classe che gestisce il modello LLama


model_LLama=Local_LLama()

index=20

r=Dataset_QA('Dataset/Dataset_QA/DataSetIndex'+str(index)+'.csv', model_LLama, index )

r.load_dataset()

df=r.period_augmentation(10)

df=r.question_augmentation(10)

df.to_csv("Dataset/Dataset_Augmented/Index"+str(index)+"Augmented.csv")