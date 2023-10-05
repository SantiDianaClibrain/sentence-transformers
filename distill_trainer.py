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
import argparse
from huggingface_hub import HfApi





def main(args):
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


    teacher_model_name = args.teacher_model  #Our monolingual teacher model, we want to convert to multiple languages
    #student_model_name = 'xlm-roberta-large' 
    student_model_name = args.student_model

    max_seq_length = 128                #Student model max. lengths for inputs (number of word pieces)
    train_batch_size = 64               #Batch size for training
    inference_batch_size = 64           #Batch size at inference
    max_sentences_per_language = 500000 #Maximum number of  parallel sentences for training
    train_max_sentence_length = 250     #Maximum length (characters) for parallel training sentences

    num_epochs = 5                       #Train for x epochs


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
    configuration ={
        'lr': 2e-5,
        'eps': 1e-6,
        'Epochs':args.num_epochs,
        'Max sequence length': max_seq_length,
        'Batch size':train_batch_size,
        'Teacher model': teacher_model_name,
        'Student model': student_model_name,
        'Runtime name': args.runtime_name
    }
    student_model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            output_path=output_path,
            optimizer_params= {'lr': 2e-5, 'eps': 1e-6},
            configuration=configuration
            )
    
    api = HfApi()
    repo = args.repo
    api.upload_folder(
        folder_path=output_path,
        repo_id=repo,
        repo_type="model",
        token = args.token
    ) 
    
if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type= str, required=True, help = "HF name of teacher model")
    parser.add_argument("--student_model", type = str, required=True, help = "HF name of student model.")
    parser.add_argument("--num_epochs", type = int, default=5, help = "Number of epochs")
    parser.add_argument("--runtime_name", type=str,  required=True, help = "Runtime name for weights and biases")
    parser.add_argument("--repo", type=str, required=True, help = "Repository to save the trained model.")
    parser.add_argument("--token", type=str, required=True, help = "HF token")
    args = parser.parse_args()
    main(args)

