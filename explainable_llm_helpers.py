from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.corpus import brown
import shap
import os
import numpy as np
import random
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain_community.llms import HuggingFaceEndpoint
#from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
import langchain
from langchain_community.embeddings import HuggingFaceEmbeddings
import pdb
########################
import nltk
import torch
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from typing import AsyncIterator, Iterator 	

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from docx import Document as ddoc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from numpy import dot
import pandas as pd


nltk.download('brown')
brown_vocab = brown.words()
unique_brown_vocab = list(set(list(brown_vocab)))
model_encode = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


prompt = f"""Provide a summary of the below text and provide a list of words that led to the summary.

Text: {text_to_summarize}

Summary:

"""

def replace_with_random(word_pos,
                       word_list,
                       vocab,
                       samps = 5):

    new_random = []
    samp_words = random.sample(vocab,samps)
    for rand in samp_words:
        new_sentence = ' '.join(word_list[0:word_pos]+[rand]+word_list[word_pos+1:])
        new_random.append(new_sentence)

    return new_random


def replace_with_random_v2(word_pos,
                       word_list,
                       vocab,
                       length = 3,
                       trials = 5):

    new_random = []
    to_be_changed = ' '.join(word_list[word_pos:word_pos+length])
    for t in range(trials):
        samp_words = random.sample(vocab,length)
        new_sentence = ' '.join(word_list[0:word_pos]+samp_words+word_list[word_pos+length:])
        new_random.append(new_sentence)

    return new_random, to_be_changed


def get_local_model_pred(tokenizer
                         ,model
                         ,this_string
                        ):
    inputs = tokenizer(this_string, return_tensors="pt").input_ids
    outputs = model.generate(inputs)
    pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return pred[0]

def generate_llm_output(og_random_dict,
                       llm_remote,
                       prompt=None):
    for sentence in og_random_dict:
        #print(sentence)
        if prompt is not None:
            this_old_input = prompt(sentence)
        else:
            this_old_input = sentence
        og_output =   llm_remote.invoke(this_old_input)
        for loc_list in og_random_dict[sentence]:
            new_out = []
            these_samples = og_random_dict[sentence][loc_list]['new_sample']
            #print('*'*10,these_samples)
            for sent in these_samples:

                #wrap new sentence in a prompt before sending out
                if prompt is not None:
                    this_new_input = prompt(sent)
                else:
                    this_new_input = sent
                sent_output = llm_remote.invoke(this_new_input)
                new_out.append(sent_output)
                
            og_random_dict[sentence][loc_list]['new_output'] = new_out 
        og_random_dict[sentence]['og_output'] = og_output  
    return og_random_dict

def generate_change_metrics(results_dictionary
                    ,model_encode):
    result = {'og_input':[],
                      'og_output':[],
                      'changed':[],
                      'new_in':[],
                      'new_out':[],
                      'cos_sim':[],
                      'l2':[]}

    for question in results_dictionary:
       old_o =results_dictionary[question]['og_output']
       for pos_tup in results_dictionary[question]:
            
            if pos_tup!='og_output':
                new_samps = results_dictionary[question][pos_tup]['new_sample']
                for idx, new_o in enumerate(results_dictionary[question][pos_tup]['new_output']):
                    emb_new = model_encode.encode(new_o)
                    emb_old = model_encode.encode(old_o)
                    cos_sim = dot(emb_new, emb_old)/(norm(emb_new)*norm(emb_old))
                    l2_dist = np.sqrt(norm(emb_new-emb_old))
                    result['og_input'].append(question)
                    result['og_output'].append(old_o)
                    result['changed'].append(pos_tup[1])
                    result['new_out'].append(new_o)
                    result['new_in'].append(new_samps[idx])
                    result['cos_sim'].append(cos_sim)
                    result['l2'].append(l2_dist)
                    #print('new:',new_o)
                    #print('old:',old_o)
    result_df = pd.DataFrame(result)
    result_df[['mean_cos','mean_l2']] = result_df.groupby(['og_input','changed'])[['cos_sim','l2']].transform('mean')
    return result_df