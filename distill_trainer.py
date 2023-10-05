from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.LoggingHandler import LoggingHandler
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.losses import MSELoss

import torch
from torch.utils.data import DataLoader
from sentence_transformers.datasets import ParallelSentencesDataset
from datetime import datetime
from datasets import load_dataset

import logging
import csv
import numpy as np



logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available else "cpu"



dataset = load_dataset("opus100","en-es")


train_data = dataset["train"]["translation"]

with open('train_data.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    writer.writerow(['English', 'Spanish'])

    for item in train_data:
        writer.writerow([item['en'], item['es']])


teacher_model_name = 'BAAI/bge-small-en'   #Our monolingual teacher model, we want to convert to multiple languages
#student_model_name = 'xlm-roberta-large' 
student_model_name = "thenlper/gte-small"

max_seq_length = 128                #Student model max. lengths for inputs (number of word pieces)
train_batch_size = 64               #Batch size for training
inference_batch_size = 64           #Batch size at inference
max_sentences_per_language = 500000 #Maximum number of  parallel sentences for training
train_max_sentence_length = 250     #Maximum length (characters) for parallel training sentences

num_epochs = 5                       #Train for x epochs
num_warmup_steps = 10000             #Warumup steps

num_evaluation_steps = 1000          #Evaluate performance after every xxxx steps
dev_sentences = 1000                 #Number of parallel sentences to be used for development


train_corpus = "train_data.csv"

logger.info("Load teacher model")

teacher_model = SentenceTransformer(teacher_model_name)

#teacher_model = SentenceTransformer(teacher_model_name).to(device) if device == "cuda" else SentenceTransformer(teacher_model_name)

logger.info("Create student model from scratch")
word_embedding_model = Transformer(student_model_name, max_seq_length=max_seq_length)
pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
#student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(device) if device == "cuda" else SentenceTransformer(modules=[word_embedding_model,pooling_model])

###### Read Parallel Sentences Dataset ######
train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model, batch_size=inference_batch_size, use_embedding_cache=True)
#for train_file in train_files:
train_data.load_data(train_corpus, max_sentences=max_sentences_per_language, max_sentence_length=train_max_sentence_length)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = MSELoss(model=student_model)

# Train the model
output_path = "model"
student_model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          output_path=output_path,
          optimizer_params= {'lr': 2e-5, 'eps': 1e-6}
          )

