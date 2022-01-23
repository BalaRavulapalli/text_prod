# Python standard libraries
# import gc
# import torch
from user import User
from db import init_db_command
# from transformers.file_utils import TF_CAUSAL_LM_SAMPLE
# from BalaQGFile import BalaQG
import pickle
# import gensim
# from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
# import tensorflow as tf
# from nltk.tree import Tree
# from nltk import tokenize
# from allennlp.predictors.predictor import Predictor
# import spacy
from typing import final
import random
# from flashtext import KeywordProcessor
# from nltk.tokenize import sent_tokenize
import traceback
# from nltk.corpus import wordnet
# from nltk.corpus import stopwords
# import pke
import itertools
import re
import string
import copy
from collections import OrderedDict
# from Questgen import main
import json
import os
import sqlite3
import openai
openai.api_key = "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"
# Third-party libraries
from flask import Flask, render_template, url_for, request, redirect, session, abort
from flask_wtf import FlaskForm, RecaptchaField
from flask_wtf.file import FileAllowed
from wtforms import FileField, IntegerField, TextAreaField, SelectMultipleField, SubmitField, PasswordField, StringField, widgets
from wtforms.validators import InputRequired, DataRequired, Length, Email, EqualTo
from werkzeug.utils import escape, unescape, secure_filename
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from oauthlib.oauth2 import WebApplicationClient
import requests
import datetime
from secrets import token_hex
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
from concurrent.futures import ThreadPoolExecutor as Executor

import logging
logging.basicConfig(filename = 'example.log', level  = logging.ERROR)
# startmodel load
# print('loading')
# nlp = spacy.load("en_core_web_sm")
# predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
# GPT2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# GPT2model = TFGPT2LMHeadModel.from_pretrained("gpt2",pad_token_id=GPT2tokenizer.eos_token_id)
# print('loaded')

# from sentence_transformers import SentenceTransformer, util
# import scipy
# BERT_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
qe = 'fds'
# qe= main.BoolQGen()

# end modle load

# qg = BalaQG()
# qg = main.QGen()
qg = 'fds'
app = Flask(__name__)


def copyer(qg, qe):
    qg2 = copy.copy(qg)
    qe2 = copy.copy(qe)
    return qg2, qe2


def mcq(qg, results, payload,  selected_specific, coref_sents, executor, uniqueUserId):
    MCQ_URL = "https://dockermcqv2gcp-sn6tlr3gzq-uc.a.run.app/getquestion"

    fixed_selected_specific = [[group[0], 1, group[2]] for group in selected_specific]
    def query(func_input):
        fixed_selected_specific, coref_sents, uniqueUserId = func_input
        data = json.dumps({"selected_specific": fixed_selected_specific, "coref_sents": coref_sents,"uniqueUserId": uniqueUserId})
        # print('before')
        response = requests.request("POST", MCQ_URL, data=data)
        # print('after')
        return json.loads(response.content.decode("utf-8"))
    mcq_request = []
    mcq_request.append(executor.submit(query, [fixed_selected_specific, coref_sents, uniqueUserId]))
    # try:
    # output = qg.predict_mcq(
    #     {'input_text': payload['input_text'], 'max_questions': payload['max_questions']['Multiple Choice']},  selected_specific, coref_sents, executor, uniqueUserId)
    # print(output)
    # except Exception as e:
    # with open('error.txt', mode = 'w') as myFile:
    #     print(e)
    #     myFile.write(e)
    return mcq_request
    
    # except:
        # pass


# def boolq(qe, results, payload):
#     results['Yes/No']['questions'] = []
#     results['Yes/No']['answers'] = []
#     output = qe.predict_boolq(
#         {'input_text': payload['input_text'], 'max_questions': payload['max_questions']['Yes/No']})
#     # print('count', payload['max_questions']['Yes/No'])
#     filtered = []
#     used = []
#     for i, q in enumerate(output['Boolean Questions']):
#         if q not in used:
#             filtered.append(i)
#         used.append(q)
#     filtered_dict = {'questions': [output['Boolean Questions'][i]
#                                    for i in filtered], 'answers': [output['Answer'] for i in filtered]}
#     if payload['max_questions']['Yes/No'] < len(filtered_dict['questions']):
#         sampled = [(filtered_dict['questions'][i], filtered_dict['answers'][i]) for i in sorted(
#             random.sample(range(len(filtered_dict['questions'])), payload['max_questions']['Yes/No']))]
#     else:
#         sampled = zip(filtered_dict['questions'], filtered_dict['answers'])
#     sampled = list(sampled)
#     sampled_dict = {'questions': [sample[0] for sample in sampled], 'answers': [
#         sample[1] for sample in sampled]}
#     # try:
#     results['Yes/No']['questions'] = sampled_dict['questions']
#     results['Yes/No']['answers'] = sampled_dict['answers']
    # except:
    # pass


class FB:
    def __init__(self, text, necessary_coref):
        self.text = text
        self.necessary_coref= necessary_coref

    def fb(self, sent_num=10, pos={'VERB', 'ADJ', 'NOUN', 'NUM', 'PROPN'}):
        # https://universaldependencies.org/u/pos/
        # def tokenize_sentences(text):
        #     sentences = sent_tokenize(text)
        #     sentences = [sentence.strip()
        #                  for sentence in sentences if len(sentence) > 20]
        #     return sentences

        # sentences = tokenize_sentences(self.text)

        def get_keywords(text, pos):
            out = []
            # try:
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=text)
            pos = set(pos)
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            extractor.candidate_selection(pos=pos, stoplist=stoplist)
            extractor.candidate_weighting(alpha=1.1,
                                          threshold=0.75,
                                          method='average')
            keyphrases = extractor.get_n_best(n=sent_num)

            for val in keyphrases:
                out.append(val[0])
            # except:
            #     out = []
            #     traceback.print_exc()

            return out

        keywords_list = get_keywords(text=self.text, pos=pos)
        logging.error('keywords len ' + str(len(keywords_list)))
        def get_sentences_for_keyword(keywords, sentences):
            keyword_processor = KeywordProcessor()
            keyword_sentences = OrderedDict({})
            for word in keywords:
                keyword_sentences[word] = []
                keyword_processor.add_keyword(word)
            for sentence in sentences:
                keywords_found = keyword_processor.extract_keywords(sentence)
                for key in keywords_found:
                    keyword_sentences[key].append(sentence)

            for key in keyword_sentences.keys():
                values = keyword_sentences[key]
                values = sorted(values, key=len, reverse=True)
                keyword_sentences[key] = values
            return keyword_sentences

        keyword_sentence_mapping = get_sentences_for_keyword(
            keywords_list, self.necessary_coref)
        # logging.error('keyword sentence mapping' + str(keyword_sentence_mapping))
        def get_fill_in_the_blanks(sentence_mapping):
            out = OrderedDict({})
            blank_sentences = []
            processed = []
            keys = []
            for key in sentence_mapping:
                if len(sentence_mapping[key]) > 0:
                    sents = sentence_mapping[key]
                    for sent in sents:
                        if sent not in processed:
                            insensitive_sent = re.compile(
                                re.escape(key), re.IGNORECASE)
                            no_of_replacements = len(re.findall(
                                re.escape(key), sent, re.IGNORECASE))
                            line = insensitive_sent.sub(' _________ ', sent)
                            if no_of_replacements < 2:
                                blank_sentences.append(line)
                                processed.append(sent)
                                keys.append(key)
                                break
            out["sentences"] = blank_sentences
            out["keys"] = keys
            # print('reached')
            return out
        fill_in_the_blanks = get_fill_in_the_blanks(keyword_sentence_mapping)
        return fill_in_the_blanks


def fitb(results, payload,  selected_specific, coref_sents, executor):
    FB_URL = "https://dockerfbgcp-sn6tlr3gzq-uc.a.run.app/create_fb"

    fixed_selected_specific = [[group[0], 1, group[2]] for group in selected_specific]
    def query(func_input):
        fixed_selected_specific, coref_sents = func_input
        data = json.dumps({"selected_specific": fixed_selected_specific, "coref_sents": coref_sents,"max_questions": payload['max_questions']['Fill in the Blanks']})
        # print('before')
        response = requests.request("POST", FB_URL, data=data)
        # print('after')
        return json.loads(response.content.decode("utf-8"))
    fb_request = []
    fb_request.append(executor.submit(query, [fixed_selected_specific, coref_sents]))
    return fb_request
    # results['Fill in the Blanks']['questions'] = []
    # results['Fill in the Blanks']['answers']
    # necessary_coref = []
    # for item in selected_specific:
    #     necessary_coref.append(coref_sents[item[2]])
    # selected_specific_sents = [item[0] for item in selected_specific]
    # logging.error('num sents fb ' + str(len(selected_specific_sents)))
    # fb = FB(' '.join(selected_specific_sents), necessary_coref)
    # output = fb.fb(int((payload['max_questions']['Fill in the Blanks']*2)), pos = {'ADJ', 'NOUN', 'NUM', 'PROPN'})
    # # print(output)
    # if payload['max_questions']['Fill in the Blanks'] < len(output['sentences']):
    #     sampled = [(output['sentences'][i], output['keys'][i]) for i in sorted(random.sample(
    #         range(len(output['sentences'])), payload['max_questions']['Fill in the Blanks']))]
    # else:
    #     sampled = zip(output['sentences'], output['keys'])
    # sampled = list(sampled)
    # sampled_dict = {'sentences': [sample[0] for sample in sampled], 'keys': [
    #     sample[1] for sample in sampled]}
    # # print(sampled_dict)
    # # try:
    # logging.error('output sents fb ' + str(len(sampled_dict['sentences'])))
    # results['Fill in the Blanks']['questions'] = sampled_dict['sentences']
    # results['Fill in the Blanks']['answers'] = sampled_dict['keys']
    # # except ValueError:
    # # pass


class TF:

    def __init__(self, payload):
        self.text = payload['input_text']
        self.num_sents = payload['max_questions']['True/False']

    # def tf_sum(self):
    #     sents = gensim.summarization.textcleaner.split_sentences(self.text)
    #     # print(sents)
    #     if self.num_sents >= len(sents):
    #         self.text = sents
    #     else:
    #         ratio = self.num_sents/len(sents)
    #         # print(ratio)
    #         # print(gensim.summarization.INPUT_MIN_LENGTH)
    #         text = gensim.summarization.summarize(
    #             self.text, ratio=ratio, split=True)
    #         # print(self.text)
    #         if not text:
    #             text = [sents[i] for i in sorted(
    #                 random.sample(range(len(sents)), self.num_sents))]
    #             print('random')
    #         self.text = text
    #     # print('summarized')

    def tf(self, selected_specific, coref_sents, gpt2_completions, executor):
        final_sents = []
        for selected_item in selected_specific:
            logging.error(str(type(selected_item[2])))
            logging.error(str(selected_item))
            logging.error(str(type(coref_sents[0])))
            my_item = coref_sents[selected_item[2]]
            item = my_item.rstrip('?:!.,;"')
            parser_output = predictor.predict(sentence=item)
            tree_string = parser_output['trees']
            tree = Tree.fromstring(tree_string)

            def get_flattened(t):
                sent_str_final = None
                if t is not None:
                    sent_str = [" ".join(x.leaves()) for x in list(t)]
                    sent_str_final = [" ".join(sent_str)]
                    sent_str_final = sent_str_final[0]
                return sent_str_final

            def get_right_most_P(parse_tree, last_P=None):
                if len(parse_tree.leaves()) <= 1:
                    return last_P
                last_subtree = parse_tree[-1]
                if last_subtree.label() in ["NP", "VP", "PP", "S"]:
                    if len(get_flattened(last_subtree)) < int(.3 * len(item)):
                        return last_P
                    else:
                        last_P = last_subtree

                return get_right_most_P(last_subtree, last_P)

            last_P = get_right_most_P(tree)
            if last_P:
                last_P_flattened = get_flattened(last_P)
            else:
                # return []
                continue

            def get_termination_portion(main_string, sub_string):

                # orig = tokenize.word_tokenize(main_string)
                # mini = tokenize.word_tokenize(sub_string)
                # orig.reverse()
                # mini.reverse()
                # rorig = iter(orig)
                # rmini = iter(mini)
                # brackets = []
                # for i, x in enumerate(rmini):
                #     y = next(rorig)
                #     if y in ['(', ')']:
                #         brackets.append(((i + len(brackets)), y))
                #         y = next(rorig)

                # for i, x in brackets:
                #     mini.insert(i, x)
                # orig.reverse()
                # mini.reverse()

                combined_sub_string = sub_string.replace(" ", "")
                # print(combined_sub_string)
                main_string_list = main_string.split()
                last_index = len(main_string_list)
                for i in range(last_index):
                    check_string_list = main_string_list[i:]
                    check_string = "".join(check_string_list)
                    check_string = check_string.replace(" ", "")
                    if check_string == combined_sub_string:
                        return " ".join(main_string_list[:i])

                return None

            last_P_flattened = re.sub(r"-LRB- ", "(", last_P_flattened)
            last_P_flattened = re.sub(r" -RRB-", ")", last_P_flattened)
            split_sentence = get_termination_portion(item, last_P_flattened)
            print('generating')
            # input_ids = GPT2tokenizer.encode(
            #     split_sentence, return_tensors='tf')
            # maximum_length = len(split_sentence.split())+40

            # sample_outputs = GPT2model.generate(
            #     input_ids,
            #     do_sample=True,
            #     max_length=maximum_length,
            #     top_p=0.80,  # 0.85
            #     top_k=30,  # 30
            #     repetition_penalty=10.0,
            #     num_return_sequences=10
            # )
            headers = {"Authorization": f"Bearer wJuTMiDhARWIeLvVxMBQjurqDblxYgTuFXqqsUmhtsfHLHDNdPtUpWJOadpUtckHbWsEVJHkIeYpfLISthsoHbqiQNvyjkuCgQYpiizklwwjkzimZCYDGVmZXeWZpiPn"}
            API_URL = "https://api-inference.huggingface.co/models/gpt2"
            if split_sentence:
                def query(text):
                    data = json.dumps({"inputs": text, "parameters":{"top_k": 30, "repitition_penalty":10.0, "top_p":.80, "num_return_sequences":10,  "max_length":int(len(text)*.7)},"options": {"wait_for_model": True, "use_cache": False}})
                    # print('before')
                    response = requests.request("POST", API_URL, headers=headers, data=data)
                    # print('after')
                    return json.loads(response.content.decode("utf-8"))
                gpt2_completions.append(executor.submit(query, split_sentence))
                final_sents.append(coref_sents[selected_item[2]])
        logging.error(str(final_sents))
        return final_sents


def rank_dissimilarity(gpt2_sentences):
    print(gpt2_sentences)
    dissimilar_final = []
    for question in gpt2_sentences:
        false_sentences_embeddings = BERT_model.encode(question[1])
        original_sentence_embedding = BERT_model.encode([question[0]])
        distances = scipy.spatial.distance.cdist(
            original_sentence_embedding, false_sentences_embeddings, "cosine")[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        dissimilar_sentences = []
        for idx, distance in results:
            dissimilar_sentences.append(question[1][idx])

        false_sentences_list_final = list(reversed(dissimilar_sentences))
        dissimilar_final.append((question[0], false_sentences_list_final[2:7]))
    return dissimilar_final


def tfq(results, payload, selected_specific, coref_sents, gpt2_completions, executor):
    # results['True/False']['correct'] = []
    # results['True/False']['incorrect'] = []
    # my_tf = TF(payload)
    # # try:
    # # my_tf.tf_sum()
    # # except ValueError:
    # # print('Skipped True/False')
    # # else:
    # final_sents = my_tf.tf(selected_specific, coref_sents, gpt2_completions, executor)
    # return final_sents
    # # if output:
    TF_URL = "https://dockertfv2gcp-sn6tlr3gzq-uc.a.run.app/create_tf"

    fixed_selected_specific = [[group[0], 1, group[2]] for group in selected_specific]
    def query(func_input):
        fixed_selected_specific, coref_sents = func_input
        data = json.dumps({"selected_specific": fixed_selected_specific, "coref_sents": coref_sents,
        "use_gpu": True
        })
        # print('before')
        response = requests.request("POST", TF_URL, data=data)
        # print('after')
        if response.status_code ==200:
            return json.loads(response.content.decode("utf-8"))
        else:
            return False
    tf_request = []
    tf_request.append(executor.submit(query, [fixed_selected_specific, coref_sents]))
    # try:
    # output = qg.predict_mcq(
    #     {'input_text': payload['input_text'], 'max_questions': payload['max_questions']['Multiple Choice']},  selected_specific, coref_sents, executor, uniqueUserId)
    # print(output)
    # except Exception as e:
    # with open('error.txt', mode = 'w') as myFile:
    #     print(e)
    #     myFile.write(e)
    return tf_request
def rank_tfq(results, used_sents, gpt2completions):
    final_sents = []
    # logging.error(str(used_sents))
    # logging.error(str(gpt2completions))
    for gpt2completion, used_sent in zip(gpt2completions, used_sents):
        generated_sentences = []
        for sample_output in gpt2completion:
            generated_sentences.append(nltk.sent_tokenize(sample_output['generated_text'])[0])
        generated_sentences = [x.strip() for x in generated_sentences]
        generated_sentences = list(set(generated_sentences))
        while used_sent in generated_sentences:
            generated_sentences.remove(used_sent)
        final_sents.append((used_sent, generated_sentences))
    ranked = rank_dissimilarity(final_sents)
    results['True/False']['correct'] = [rank[0] for rank in ranked]
    results['True/False']['incorrect'] = [rank[1] for rank in ranked]


# Internal imports

with open('client_creds.json', mode='r') as jsoncreds:
    creds = json.load(jsoncreds)

GOOGLE_CLIENT_ID = creds['web']['client_id']
GOOGLE_CLIENT_SECRET = creds['web']['client_secret']
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

app.config["SECRET_KEY"] = 'TVOC2UFO7W8PJ95AZW42W0U6QV1R49ZQ878CIHFITU8FFNBUGMAPRNHKUO7XFOPBRVWENPMESWE3Y1VA8CAD2Y5QZ1GJJDPV'
# app.config["RECAPTCHA_PUBLIC_KEY"] = "6Lcqw3MbAAAAAPGvlUS_xCDSTw6mB7yLn8wcwj8U"
# app.config["RECAPTCHA_PRIVATE_KEY"] = "6Lcqw3MbAAAAABO2ekkU6KtCQj7O94thjJoOTpKr"
app.config['ALLOWED_FILE_EXTENSIONS'] = ["pdf", "docx", "txt"]
app.config['MAX_CONTENT_LENGTH'] = 100*1024*1024
# app.config["TESTING"] = True

# class MultiCheckboxField(SelectMultipleField):
#     widget = widgets.ListWidget(prefix_label=False)
#     option_widget = widgets.CheckboxInput()


class QuestionsForm(FlaskForm):
    content = TextAreaField("Content",
                            validators=[InputRequired("Input is required."), DataRequired("Data is required."),
                                        Length(min=11, message="Input must be longer than 10 characters and shorter than 5000 characters", max = 5000)]
                            )
    # checkbox = MultiCheckboxField("Question Types", choices = [('Multiple Choice', 'Multiple Choice'), ('True/False', 'True/False'), ('Fill in the Blanks', 'Fill in the Blanks'), ('Yes/No', 'Yes/No'), ('Match Definitions', 'Match Definitions')], validators =
    #     [ InputRequired("You must select  at least one option."), DataRequired("Data is required.")])
    # upload = FileField('File Upload', validators =[FileAllowed(app.config['ALLOWED_FILE_EXTENSIONS'], 'Only PDF, Docx, and Text files are compatible.')])
    count0 = IntegerField('Multiple Choice')
    count1 = IntegerField('True/False')
    count2 = IntegerField('Fill in the Blanks')
    # count3 = IntegerField('Yes/No')
    # count4 = IntegerField('Match Definitions')
    # recaptcha = RecaptchaField()
    submit = SubmitField("Submit")


def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()


# User session management setup
# https://flask-login.readthedocs.io/en/latest
login_manager = LoginManager()
login_manager.init_app(app)

# Naive database setup
try:
    init_db_command()
except sqlite3.OperationalError:
    # Assume it's already been created
    pass

# # OAuth 2 client setup
client = WebApplicationClient(GOOGLE_CLIENT_ID)

# # Flask-Login helper to retrieve a user from our db


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


@app.route('/')
def home():
    # with Executor() as executor:
    #     # if 'True/False' in payload['question_types']:
    #     headers = {"Authorization": f"Bearer wJuTMiDhARWIeLvVxMBQjurqDblxYgTuFXqqsUmhtsfHLHDNdPtUpWJOadpUtckHbWsEVJHkIeYpfLISthsoHbqiQNvyjkuCgQYpiizklwwjkzimZCYDGVmZXeWZpiPn"}
    #     API_URL = "https://api-inference.huggingface.co/models/gpt2"

    #     def query(text):
    #         data = json.dumps({"inputs": text, "parameters":{"num_return_sequences":1,  "max_length":1},"options": {"wait_for_model": False, "use_cache": False, "use_gpu":True}})
    #         # print('before')
    #         response = requests.request("POST", API_URL, headers=headers, data=data)
    #         # print('after')
    #         return json.loads(response.content.decode("utf-8"))
    #     begin1 = datetime.datetime.now()
    #     executor.submit(query, 'a')
    #     logging.error('1' + str(datetime.datetime.now()-begin1))
    #         # logging.error('sent gpt2 request')
    #     # if 'Multiple Choice' in payload['question_types']:
    #     def fakequeryopenAI():
    #         output = openai.Completion.create(
    #         model="babbage:ft-natlang-ai-2021-12-01-02-21-24",
    #         max_tokens = 1,
    #         stop = ["####"],
    #         temperature = .8, 
    #         prompt=f"a", 
    #         user = random.randrange(1, 10)*"a")
    #         return output['choices'][0]['text']
    #     begin2 = datetime.datetime.now()
    #     executor.submit(fakequeryopenAI, 'a')
    #     logging.error('2' + str(datetime.datetime.now()-begin2))
    #     def cold_start(url):
    #         response = requests.request("GET", url)
    #         return json.loads(response.content.decode("utf-8"))
    #     begin4 = datetime.datetime.now()
    #     for url in ['https://dockerfbgcp-sn6tlr3gzq-uc.a.run.app', 'https://dockermcqgcp-sn6tlr3gzq-uc.a.run.app', 'https://dockerprepv2gcp-sn6tlr3gzq-uc.a.run.app', 'https://dockertfv2gcp-sn6tlr3gzq-uc.a.run.app']:
    #         begin3 = datetime.datetime.now()
    #         executor.submit(cold_start, url)
    #         logging.error('3' + str(datetime.datetime.now()-begin3))
    #     logging.error('4' + str(datetime.datetime.now()-begin4))
    if current_user.is_authenticated:
        # logging.error('4.1' + str(datetime.datetime.now()-begin4))
        return render_template('index.html', pic=current_user.profile_pic,
                               info=[current_user.name])
    else:
        # logging.error('4.1' + str(datetime.datetime.now()-begin4))
        return render_template('index.html',
                               # name = False,
                               info=[])


@app.route("/login")
def login():
    # Find out what URL to hit for Google login
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    # Use library to construct the request for Google login and provide
    # scopes that let you retrieve user's profile from Google
    request_uri = client.prepare_request_uri(authorization_endpoint, redirect_uri="https://natlangai.com/login/callback",
                                             scope=["openid", "email", "profile"])
    # print(request.base_url)
    return redirect(request_uri)


@app.route("/login/callback", methods=['GET', 'POST'])
def callback():
    # with open('error.txt', mode = "w") as myFile:
    # myFile.write("part1")
    # Get authorization code Google sent back to you
    code = request.args.get("code")
    # Find out what URL to hit to get tokens that allow you to ask for
    # things on behalf of a user
    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]
    # Prepare and send a request to get tokens! Yay tokens!
    token_url, headers, body = client.prepare_token_request(token_endpoint, authorization_response=(request.url.replace("http://localhost", "https://natlangai.com")),
                                                            redirect_url="https://natlangai.com/login/callback", code=code)
    token_response = requests.post(token_url, headers=headers, data=body,
                                   auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET))
    print(token_response.text)
    # myFile.write("part2")
    # Parse the tokens!
    client.parse_request_body_response(json.dumps(token_response.json()))
    # Now that you have tokens (yay) let's find and hit the URL
    # from Google that gives you the user's profile information,
    # including their Google profile image and email
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)
    # myFile.write("part3")
    # You want to make sure their email is verified.
    # The user authenticated with Google, authorized your
    # app, and now you've verified their email through Google!
    if userinfo_response.json().get("email_verified"):
        unique_id = userinfo_response.json()["sub"]
        users_email = userinfo_response.json()["email"]
        picture = userinfo_response.json()["picture"]
        users_name = userinfo_response.json()["given_name"]
    else:
        return "User email not available or not verified by Google.", 400
    # myFile.write("part4")
    # Create a user in your db with the information provided
    # by Google
    user = User(
        id_=unique_id, name=users_name, email=users_email, profile_pic=picture
    )

    # Doesn't exist? Add it to the database.
    if not User.get(unique_id):
        User.create(unique_id, users_name, users_email, picture)
    # myFile.write("part5")
    # myFile.write(users_name)
    # Begin user session by logging the user in
    login_user(user)
    # myFile.write("part6")

    # Send user back to homepage
    return redirect(url_for("home"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

@app.route('/coldstart', methods=['GET'])
def coldstart():
    with Executor() as executor:
        executed = []
        headers = {"Authorization": f"Bearer QRPtXHhrYcHElEtmHaXxlmfXvHRNFOOaJXIUhtbYFvmdpSMSLmUZDqXopJEnzSQWFTmMeLaYhUSIcggyugWzNwgMnatpHDdmIvvDMUMdzzMHjETXmJPDgNzdPWqoBeDj"}
        API_URL = "https://api-inference.huggingface.co/models/gpt2"

        def query(text):
            data = json.dumps({"inputs": text, "parameters":{"num_return_sequences":1,  "max_length":1},"options": {"wait_for_model": False, "use_cache": False, "use_gpu":True}})
            # print('before')
            response = requests.request("POST", API_URL, headers=headers, data=data)
            # print('after')
            return json.loads(response.content.decode("utf-8"))
        executed.append(executor.submit(query, 'a'))
            # logging.error('sent gpt2 request')
        # if 'Multiple Choice' in payload['question_types']:
        def fakequeryopenAI(text):
            output = openai.Completion.create(
            model="babbage:ft-natlang-ai-2021-12-01-02-21-24",
            max_tokens = 1,
            stop = ["####"],
            temperature = .8, 
            prompt=text, 
            user = random.randrange(1, 10)*"a")
            return output['choices'][0]['text']
        executed.append(executor.submit(fakequeryopenAI, 'a'))
        def cold_start(url):
            response = requests.request("GET", url)
            return json.loads(response.content.decode("utf-8"))
        for url in ['https://dockerfbgcp-sn6tlr3gzq-uc.a.run.app', 'https://dockermcqv2gcp-sn6tlr3gzq-uc.a.run.app', 'https://dockerprepv2gcp-sn6tlr3gzq-uc.a.run.app', 'https://dockertfv2gcp-sn6tlr3gzq-uc.a.run.app']:
            executed.append(executor.submit(cold_start, url))
        output = [call.result() for call in executed]
    return json.dumps(output)  
@app.route('/query/new', methods=['GET', 'POST'])
@login_required
def new():
    
        # if 'True/False' in payload['question_types']:
        # headers = {"Authorization": f"Bearer wJuTMiDhARWIeLvVxMBQjurqDblxYgTuFXqqsUmhtsfHLHDNdPtUpWJOadpUtckHbWsEVJHkIeYpfLISthsoHbqiQNvyjkuCgQYpiizklwwjkzimZCYDGVmZXeWZpiPn"}
        # API_URL = "https://api-inference.huggingface.co/models/gpt2"

        # def query(text):
        #     data = json.dumps({"inputs": text, "parameters":{"num_return_sequences":1,  "max_length":1},"options": {"wait_for_model": False, "use_cache": False, "use_gpu":True}})
        #     # print('before')
        #     response = requests.request("POST", API_URL, headers=headers, data=data, timeout = .1)
        #     # print('after')
        #     return json.loads(response.content.decode("utf-8"))
        # executor.submit(query, 'a')
        #     # logging.error('sent gpt2 request')
        # # if 'Multiple Choice' in payload['question_types']:
        # def fakequeryopenAI():
        #     output = openai.Completion.create(
        #     model="babbage:ft-natlang-ai-2021-12-01-02-21-24",
        #     max_tokens = 1,
        #     stop = ["####"],
        #     temperature = .8, 
        #     prompt=f"a", 
        #     user = random.randrange(1, 10)*"a")
        #     return output['choices'][0]['text']
        # executor.submit(fakequeryopenAI, 'a')
        # def cold_start(url):
        #     response = requests.request("GET", url)
        #     return json.loads(response.content.decode("utf-8"))
        # for url in ['https://dockerfbgcp-sn6tlr3gzq-uc.a.run.app', 'https://dockermcqgcp-sn6tlr3gzq-uc.a.run.app', 'https://dockerprepv2gcp-sn6tlr3gzq-uc.a.run.app', 'https://dockertfv2gcp-sn6tlr3gzq-uc.a.run.app']:
        #     executor.submit(cold_start, url)
        # logging.info('entered new 558')
        # global form
        # logging.info('before qform')
        form = QuestionsForm()
        # logging.info('after qform')
        if request.method == 'POST':
            if form.validate_on_submit():
                with Executor() as executor:
                    newStart = datetime.datetime.now()
                    payload = {'question_types': [], 'input_text': '', 'max_questions': {
                        'Multiple Choice': 0,
                        'True/False': 0, 
                        'Fill in the Blanks': 0,
                        #  'Yes/No': 0, 'Match Definitions': 0, 
                        }}
                    results = OrderedDict({'Multiple Choice': OrderedDict({'context': [], 'questions': [], 'answers': [], 'options': []}), 
                    # 'Yes/No': OrderedDict({'questions': [], 'answers': []}), 
                        'Fill in the Blanks': OrderedDict({'questions': [], 'answers': []}), 'True/False': OrderedDict({'correct': [], 'incorrect': []})})
                    # print(payload['input_text'])
                    if form.count0.data > 0:
                        payload['question_types'].append('Multiple Choice')
                        payload['max_questions']['Multiple Choice'] = form.count0.data
                    if form.count1.data > 0:
                        payload['question_types'].append('True/False')
                        payload['max_questions']['True/False'] = form.count1.data
                    if form.count2.data > 0:
                        payload['question_types'].append('Fill in the Blanks')
                        payload['max_questions']['Fill in the Blanks'] = form.count2.data
                    # if form.count3.data > 0:
                    #     payload['question_types'].append('Yes/No')
                    #     payload['max_questions']['Yes/No'] = form.count3.data
                    # if form.count4.data > 0:
                    #     payload['question_types'].append('Match Definitions')
                    #     payload['max_questions']['Match Definitions'] = form.count4.data
                    payload['input_text'] = form.content.data
                    # print(payload['input_text'])
                    # print(results)
                    # print(payload)
                    # startCopy = datetime.datetime.now()
                    myqg, myqe = copyer(qg, qe)
                    # logging.error('copy timer' + str(datetime.datetime.now()-startCopy))
                    # print('before')
                    # myqe= main.BoolQGen()
                    # myqg = main.QGen()
                    # print('after')
                    if len(payload['input_text']) > 10:
                        uniqueUserId = ''
                        for character in current_user.email:
                            uniqueUserId += str(ord(character))

                        # startExecute = datetime.datetime.now()
                        # with Executor() as executor:
                        # logging.error('startExecute' + str(datetime.datetime.now()-startExecute))
                        # if 'True/False' in payload['question_types']:
                        # headers = {"Authorization": f"Bearer wJuTMiDhARWIeLvVxMBQjurqDblxYgTuFXqqsUmhtsfHLHDNdPtUpWJOadpUtckHbWsEVJHkIeYpfLISthsoHbqiQNvyjkuCgQYpiizklwwjkzimZCYDGVmZXeWZpiPn"}
                        # API_URL = "https://api-inference.huggingface.co/models/gpt2"

                        # def query(text):
                        #     data = json.dumps({"inputs": text, "parameters":{"num_return_sequences":1,  "max_length":1},"options": {"wait_for_model": True, "use_cache": False, "use_gpu":True}})
                        #     # print('before')
                        #     response = requests.request("POST", API_URL, headers=headers, data=data)
                        #     # print('after')
                        #     return json.loads(response.content.decode("utf-8"))
                        # executor.submit(query, 'a')
                        #     # logging.error('sent gpt2 request')
                        # # if 'Multiple Choice' in payload['question_types']:
                        # def fakequeryopenAI():
                        #     output = openai.Completion.create(
                        #     model="babbage:ft-natlang-ai-2021-12-01-02-21-24",
                        #     max_tokens = 1,
                        #     stop = ["####"],
                        #     temperature = .8, 
                        #     prompt=f"a", 
                        #     user = uniqueUserId)
                        #     return output['choices'][0]['text']
                        # executor.submit(fakequeryopenAI, 'a')
                        # def cold_start(url):
                        #     response = requests.request("GET", url)
                        #     return json.loads(response.content.decode("utf-8"))
                        # for url in ['https://dockerfbgcp-sn6tlr3gzq-uc.a.run.app', 'https://dockermcqgcp-sn6tlr3gzq-uc.a.run.app', 'https://dockerprepv2gcp-sn6tlr3gzq-uc.a.run.app', 'https://dockertfv2gcp-sn6tlr3gzq-uc.a.run.app']:
                        #     executor.submit(cold_start, url)
                        # selected_specific, coref_sents = qg.filter_coref({'input_text':payload['input_text']})
                        filter_coref_response = requests.request("POST", 'https://dockerprepv2gcp-sn6tlr3gzq-uc.a.run.app/filter_coref', data = json.dumps({'input_text': payload['input_text'].replace("\r", "").strip()}))
                        logging.error('payload: ' + str(payload['input_text'].replace("\r", "").strip().__repr__()))
                        logging.error('orig input: ' + str(payload['input_text'].__repr__()))
                        logging.error('orig selected_specific len' + str(filter_coref_response.content.decode("utf-8")))
                        list_filter_coref = filter_coref_response.json()
                        selected_specific = list_filter_coref[0]
                        coref_sents = list_filter_coref[1]
                        for item in selected_specific:
                            assert item[2]<len(coref_sents)
                        # logging.error(json.dumps(selected_specific))
                        logging.error('orig selected_specific len' + str(len(selected_specific)))
                        selected_specific  = selected_specific[:payload['max_questions']['Multiple Choice'] + payload['max_questions']['True/False'] + payload['max_questions']['Fill in the Blanks']]
                        # random.shuffle(selected_specific)
                        offset = 0
                        offset_mapping = {}
                        if 'Multiple Choice' in payload['question_types']:
                            offset_mapping["Multiple Choice"]= offset
                            offset += payload['max_questions']['Multiple Choice']
                        if 'True/False' in payload['question_types']:
                            logging.error('offsetting TF')
                            offset_mapping["True/False"]= offset
                            offset += payload['max_questions']['True/False']
                        # if 'Yes/No' in payload['question_types']:
                        #     offset_mapping["Yes/No"]= offset
                        #     offset += payload['max_questions']['Yes/No']
                        if 'Fill in the Blanks' in payload['question_types']:
                            offset_mapping["Fill in the Blanks"]= offset
                            offset += payload['max_questions']['Fill in the Blanks']
                        # if 'Yes/No' in payload['question_types']:
                        #     boolq(myqe, results, payload)
                        if 'True/False' in payload['question_types']:
                            # logging.error(str(offset))
                            # logging.error(str(offset+payload['max_questions']['True/False']))
                            # logging.error(str(len(selected_specific[offset:offset+payload['max_questions']['True/False']])))
                            beginTF = datetime.datetime.now()
                            start = offset_mapping['True/False']
                            if selected_specific[start:start+payload['max_questions']['True/False']]:
                                logging.error('doing tf' + str(results))
                                gpt2_completions = []
                                tf_request = tfq(results, payload, selected_specific[start:start+payload['max_questions']['True/False']], coref_sents, gpt2_completions, executor)
                            logging.error(str(datetime.datetime.now()-beginTF))
                        if 'Multiple Choice' in payload['question_types']:
                            logging.info('before mcq')
                            start = offset_mapping['Multiple Choice']
                            if selected_specific[start:start+payload['max_questions']['Multiple Choice']]:
                                logging.error('doing mcq' + str(results))
                                mcq_request = mcq(myqg, results, payload, selected_specific[start:start+payload['max_questions']['Multiple Choice']], coref_sents, executor, uniqueUserId)
                            logging.info('after mcq', results)
                        if 'Fill in the Blanks' in payload['question_types']:
                            logging.error('total before ' + str(datetime.datetime.now()-newStart))
                            beginFB = datetime.datetime.now()
                            start = offset_mapping['Fill in the Blanks']
                            if selected_specific[start:start+payload['max_questions']['Fill in the Blanks']]:
                                logging.error('doing fb' + str(results))
                                fb_request = fitb(results, payload, selected_specific[start:start+payload['max_questions']['Fill in the Blanks']], coref_sents, executor)
                            # offset += payload['max_questions']['Fill in the Blanks']
                            logging.error('fitb timer'+str(datetime.datetime.now()-beginFB))
                        if 'Fill in the Blanks' in payload['question_types']:
                            start = offset_mapping['Fill in the Blanks']
                            if selected_specific[start:start+payload['max_questions']['Fill in the Blanks']]:
                                fb_output = [call.result() for call in fb_request]
                                results['Fill in the Blanks']['questions'] = []
                                results['Fill in the Blanks']['answers'] = []
                                # print(output)
                                # if payload['max_questions']['Fill in the Blanks'] < len(output['sentences']):
                                #     sampled = [(output['sentences'][i], output['keys'][i]) for i in sorted(random.sample(
                                #         range(len(output['sentences'])), payload['max_questions']['Fill in the Blanks']))]
                                # else:
                                #     sampled = zip(output['sentences'], output['keys'])
                                # sampled = list(sampled)
                                # sampled_dict = {'sentences': [sample[0] for sample in sampled], 'keys': [
                                #     sample[1] for sample in sampled]}
                                # print(sampled_dict)
                                # try:
                                logging.error('output sents fb ' + str(len(fb_output[0]['questions'])))
                                results['Fill in the Blanks']['questions'] = fb_output[0]['questions']
                                results['Fill in the Blanks']['answers'] = fb_output[0]['answers']
                                # except ValueError:
                                # pass
                        if 'Multiple Choice' in payload['question_types']:
                            start = offset_mapping['Multiple Choice']
                            if selected_specific[start:start+payload['max_questions']['Multiple Choice']]:
                                mcq_output = [call.result() for call in mcq_request]
                                results['Multiple Choice']['context'] = []
                                results['Multiple Choice']['questions'] = []
                                results['Multiple Choice']['answers'] = []
                                results['Multiple Choice']['options'] = []
                                # try:
                                for x in mcq_output[0]['Multiple Choice']['questions']:
                                    results['Multiple Choice']['context'].append(x['context'])
                                    results['Multiple Choice']['questions'].append(x['question_statement'])
                                    results['Multiple Choice']['answers'].append(x['answer'])
                                    results['Multiple Choice']['options'].append(x['options'])
                        if 'True/False' in payload['question_types']:
                            start = offset_mapping['True/False']
                            if selected_specific[start:start+payload['max_questions']['True/False']]:
                                logging.error('doing tf2' + str(results))
                                # beginRank = datetime.datetime.now()
                                # gpt2_outputs = [call.result() for call in gpt2_completions]
                                # rank_tfq(results, used_sents,gpt2_outputs)
                                tf_output = [call.result() for call in tf_request]
                                if tf_output[0] != False:
                                    correct = tf_output[0]['correct']
                                    incorrect = tf_output[0]['incorrect']
                                    results['True/False']['correct'] = correct
                                    results['True/False']['incorrect'] = incorrect
                                # logging.error(str(datetime.datetime.now()-beginRank))
                # try:
                data_list = {}
                # print(payload)
                logging.error(str(results))
                for qtype in results.keys():
                    # print(qtype)
                    qtype_list = []
                    for portion in results[qtype].keys():
                        portion_list = []
                        for text in results[qtype][portion]:
                            portion_list.append({portion: text})
                        qtype_list.append(portion_list)
                    zipped = list(map(list, zip(*qtype_list)))
                    data_list[qtype] = zipped
                if payload['input_text']:
                    payload['input_text'] = unescape(payload['input_text'])
                    # print(form.content.data)
                    payload['input_text'] = escape(payload['input_text'])
                # if form.content.data:
                #     pass
                # print('rendering')
                # print(data_list)
                empty = True
                for qtype in data_list:
                    if data_list[qtype]:
                        empty = False
                if empty:
                    data_list = {}
                # print('DataList')
                # print(data_list)
                # print(data_list)
                # print(payload['input_text'])
                
                url = "https://script.google.com/macros/s/AKfycbzducEgq-8BPARjTx6djntN8bOenVaq9X8Ku_5z51krxhbQ4OJ26lam7k3sNQKJuqXD/exec"
                formjson = {'data_list': data_list, 'email': current_user.email}
                # print(current_user.email)
                logging.info('here1')
                # print('form_json', formjson)
                # try:
                logging.info('here2')
                beginRequest = datetime.datetime.now()
                gform = requests.post(url=url, json=formjson)
                # logging.error('lenselected specific' + str(len(selected_specific)))
                logging.error('Request ' + str(datetime.datetime.now()-beginRequest))
                logging.info('2.1')
                gform.raise_for_status()
                logging.info('here3')
            # except Exception as e:
                logging.info('3.1')
                # with open('error.txt', mode = 'w') as myFile:
                #     myFile.write('closer' + repr(e))
                logging.info('3.2')
                # with open('error.txt', mode = 'w') as myFile:
                #     myFile.write(str(gform.status_code))
                #     myFile.write(str(gform.text))
                logging.info(gform.text)
                logging.info('here4')
                form = QuestionsForm()
                logging.info(gform.status_code)
                gform = gform.json()
                copyurl = re.sub(r"edit$", "copy", gform['link'])
                # gc.collect()
                # torch.cuda.empty_cache()
                text = requests.get(gform['html']).text
                logging.info(text)
                return render_template('questions.html', input_text=payload['input_text'],
                                    html=text,
                                    copy=copyurl,
                                    form=form, pic=current_user.profile_pic,
                                    info=[current_user.name])
                # except Exception as e:
                # with open('error.txt', mode = 'w') as myFile:
                #     myFile.write(repr(e))
                # form = QuestionsForm()
                return render_template('questions.html', input_text='',
                                    all_qtypes={},
                                    form=form, pic=current_user.profile_pic,
                                    info=[current_user.name])
                # print(results)
                # session['payload']=payload
                # session['results'] = results
                # print(session['payload']['input_text'])
                # print(session['results'])
                # print('next')
                # return redirect(url_for('questions', payload = json.dumps(payload), results = json.dumps(results)), code = 307)
                # if form.upload.data.filename or form.content.data:
                #     print('all is correct')
                #     return redirect(url_for('questions'))
                # else:
                #     form.content.errors.append('Either the Content box or the File Upload must have data.')
            else:
                print('all is not correct')
        return render_template('query.html', form=form,  pic=current_user.profile_pic,
                            info=[current_user.name])

# @app.route('/query/questions', methods = ['GET', 'POST'])
# @login_required
# def questions():
#     # global form
#     # try:
#     #     if form.upload.data.filename:
#     #         format = "%Y%m%d%T%H%M%S"
#     #         now = datetime.datetime.utcnow().strftime(format)
#     #         random_string = token_hex(2)
#     #         filename = random_string + "_" + now + "_" + form.upload.data.filename
#     #         filename = secure_filename(filename)
#     #         print(filename)
#     # except:
#     #     print('no upload')

#     try:
#         payload = session['payload']
#         received = session['results']
#         results = OrderedDict()
#         results['Multiple Choice'] = received['Multiple Choice']
#         results['Yes/No'] = received['Yes/No']
#         results['Fill in the Blanks'] = received['Fill in the Blanks']
#         results['True/False'] = received['True/False']
#         data_list = {}
#         # print(payload)
#         for qtype in results.keys():
#             # print(qtype)
#             qtype_list = []
#             for portion in results[qtype].keys():
#                 portion_list = []
#                 for text in results[qtype][portion]:
#                     portion_list.append({portion: text})
#                 qtype_list.append(portion_list)
#             zipped = list(map(list, zip(*qtype_list)))
#             data_list[qtype] = zipped
#         if payload['input_text']:
#             payload['input_text'] = unescape(payload['input_text'])
#             # print(form.content.data)
#             payload['input_text'] = escape(payload['input_text'])
#         # if form.content.data:
#         #     pass
#         # print('rendering')
#         form = QuestionsForm()
#         return render_template('questions.html', input_text = payload['input_text'],
#                             all_qtypes = data_list,
#                             form = form)
#     except:
#         form = QuestionsForm()
#     return render_template('questions.html', input_text = '',
#                             all_qtypes = {},
#                             form = form)
#         # pic = current_user.profile_pic,
#         #     info = [current_user.name])


@app.errorhandler(401)
def unauthorized(error):
    return render_template('unauthorized.html')


@app.errorhandler(413)
def largefile(error):
    return render_template('unauthorized.html')

@app.errorhandler(500)
def server(error):
    return render_template('balaerror.html')


@app.errorhandler(502)
def server(error):
    return render_template('size.html')


@app.errorhandler(413)
def server(error):
    return render_template('size.html')


print('reading to run')
if __name__ == '__main__':
    pass
    app.run(
        ssl_context="adhoc"
    )
