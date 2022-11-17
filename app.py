from br_login import br_login_callback_func, br_login_func
from br_login import flask_run_args
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from concurrent.futures import ThreadPoolExecutor as Executor
from secrets import token_hex
import datetime
import requests
from oauthlib.oauth2 import WebApplicationClient
from werkzeug.utils import escape, unescape, secure_filename
from wtforms.validators import InputRequired, DataRequired, Length, Email, EqualTo
from wtforms import FileField, IntegerField, TextAreaField, SelectMultipleField, SubmitField, PasswordField, StringField, widgets, StringField, ValidationError
from flask_wtf.file import FileAllowed
from flask_wtf import FlaskForm, RecaptchaField
from flask import Flask, render_template, url_for, request, redirect, session, abort, Response, send_file
import logging
from user import User
from db import init_db_command
from io import BytesIO
import base64
import pandas as pd
from googletrans import Translator
from youtube_transcript_api import YouTubeTranscriptApi
import pickle

from typing import final
import random

import traceback

import itertools
import re
import string
import copy
from collections import OrderedDict

import json
import os
import sqlite3
import openai
openai.api_key = "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"

# import logging
logging.basicConfig(filename='example.log', level=logging.ERROR)


app = Flask(__name__)
FB_URL = "https://dockerfbv2gcp-sn6tlr3gzq-uc.a.run.app/create_fb"
MCQ_URL = "https://dockermcqv8gcp-sn6tlr3gzq-uc.a.run.app/getquestion"


def mcq(url, selected_specific, coref_sents, executor, uniqueUserId):

    fixed_selected_specific = [[group[0], 1, group[2]]
                               for group in selected_specific]

    def query(func_input):
        fixed_selected_specific, coref_sents, uniqueUserId = func_input
        data = json.dumps({"selected_specific": fixed_selected_specific,
                          "coref_sents": coref_sents, "uniqueUserId": uniqueUserId})
        # print('before')
        response = requests.request("POST", url, data=data)
        # print(response.content)
        # print('after')
        return json.loads(response.content.decode("utf-8"))
    mcq_request = []
    mcq_request.append(executor.submit(
        query, [fixed_selected_specific, coref_sents, uniqueUserId]))
    return mcq_request









# def fitb(results, payload,  selected_specific, coref_sents, executor):
#     FB_URL = "https://dockerfbv2gcp-sn6tlr3gzq-uc.a.run.app/create_fb"

#     fixed_selected_specific = [[group[0], 1, group[2]]
#                                for group in selected_specific]

#     def query(func_input):
#         fixed_selected_specific, coref_sents = func_input
#         data = json.dumps({"selected_specific": fixed_selected_specific, "coref_sents": coref_sents,
#                           "max_questions": payload['max_questions']['Fill in the Blanks']})
#         # print('before')
#         response = requests.request("POST", FB_URL, data=data)
#         # print('after')
#         return json.loads(response.content.decode("utf-8"))
#     fb_request = []
#     fb_request.append(executor.submit(
#         query, [fixed_selected_specific, coref_sents]))
#     return fb_request



def tfq(results, payload, selected_specific, coref_sents, gpt2_completions, executor):

    TF_URL = "https://dockertfv2gcp-sn6tlr3gzq-uc.a.run.app/create_tf"

    fixed_selected_specific = [[group[0], 1, group[2]]
                               for group in selected_specific]

    def query(func_input):
        fixed_selected_specific, coref_sents = func_input
        data = json.dumps({"selected_specific": fixed_selected_specific, "coref_sents": coref_sents,
                           "use_gpu": False
                           })
        # print('before')
        response = requests.request("POST", TF_URL, data=data)
        # print('after')
        if response.status_code == 200:
            return json.loads(response.content.decode("utf-8"))
        else:
            return False
    tf_request = []
    tf_request.append(executor.submit(
        query, [fixed_selected_specific, coref_sents]))

    return tf_request





# Internal imports

with open('client_creds.json', mode='r') as jsoncreds:
    creds = json.load(jsoncreds)

GOOGLE_CLIENT_ID = creds['web']['client_id']
GOOGLE_CLIENT_SECRET = creds['web']['client_secret']
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

app.config["SECRET_KEY"] = 'TVOC2UFO7W8PJ95AZW42W0U6QV1R49ZQ878CIHFITU8FFNBUGMAPRNHKUO7XFOPBRVWENPMESWE3Y1VA8CAD2Y5QZ1GJJDPV'
app.config['ALLOWED_FILE_EXTENSIONS'] = ["pdf", "docx", "txt"]
app.config['MAX_CONTENT_LENGTH'] = 100*1024*1024



class QuestionsForm(FlaskForm):
    content = TextAreaField("Content",
                            # validators=[InputRequired("Input is required."), DataRequired("Data is required."),
                            #             Length(min=11, message="Input must be longer than 10 characters and shorter than 5000 characters", max=5000)]
                            )
    url = StringField('YouTube URL')

    def validate_url(form, field):
        if (form.content.data != "") and (form.url.data != ""):
            raise ValidationError(
                'Sumbit either a text input or a YouTube URL')
        if ((len(form.content.data) <= 10) or (len(form.content.data) >= 200000)) and (form.url.data == ""):
            raise ValidationError(
                "Input must be longer than 10 characters and shorter than 200,000 characters")
        if ((form.content.data == "") and (("youtube" not in form.url.data) and ("youtu.be" not in form.url.data))):
            raise ValidationError("Enter a proper YouTube URL")
        if (form.content.data == ""):
            if "youtube" in form.url.data:
                id = form.url.data.split("?v=")[1]
            else:
                id = form.url.data.split("youtu.be/")[1]
            transcript_list = YouTubeTranscriptApi.list_transcripts(id)
            valid_transcript = False
            for transcript in transcript_list:
                if transcript.is_generated == False:
                    valid_transcript = True
            if not valid_transcript:
                raise ValidationError(
                    "Selected YouTube video does not have a human generated transcript and is not compatible")

    count0 = IntegerField('Multiple Choice')
    count1 = IntegerField('True/False')
    count2 = IntegerField('Fill in the Blanks')

    def validate_count2(form, field):
        if ([form.count0.data, form.count1.data, form.count2.data] == [0, 0, 0]):
            raise ValidationError('Select a question type')
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
    if current_user.is_authenticated:
        # logging.error('4.1' + str(datetime.datetime.now()-begin4))
        return render_template('index.html',
                               pic=current_user.profile_pic,
                               info=[current_user.name]
                               )
    else:
        return render_template('index.html',
                               # pic=current_user.profile_pic,
                               #   info=[current_user.name]
                               )


@app.route("/login")
def login():
    # Find out what URL to hit for Google login
    # google_provider_cfg = get_google_provider_cfg()
    # authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    # # Use library to construct the request for Google login and provide
    # # scopes that let you retrieve user's profile from Google
    # request_uri = client.prepare_request_uri(authorization_endpoint, redirect_uri="https://balapoc.southcentralus.cloudapp.azure.com:5000/login/callback",
    #                                          scope=["openid", "email", "profile"])
    # # print(request.base_url)
    # return redirect(request_uri)
    print(1)
    return br_login_func(client)


@app.route("/login/callback", methods=['GET', 'POST'])
def callback():
    print(2)
    return br_login_callback_func(client)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


@app.route('/coldstart', methods=['GET'])
def coldstart():
    with Executor() as executor:
        # tester = {}
        executed = []
        headers = {"Authorization": f"Bearer QRPtXHhrYcHElEtmHaXxlmfXvHRNFOOaJXIUhtbYFvmdpSMSLmUZDqXopJEnzSQWFTmMeLaYhUSIcggyugWzNwgMnatpHDdmIvvDMUMdzzMHjETXmJPDgNzdPWqoBeDj"}
        API_URL = "https://api-inference.huggingface.co/models/gpt2"

        def query(text):
            # tester['hf'] = False
            data = json.dumps({"inputs": text, "parameters": {"num_return_sequences": 1,  "max_length": 1}, "options": {
                              "wait_for_model": True, "use_cache": False, "use_gpu": False}})
            # print('before')
            response = requests.request(
                "POST", API_URL, headers=headers, data=data)
            # print('after')
            # tester['hf'] = True
            return json.loads(response.content.decode("utf-8"))
        executed.append(executor.submit(query, 'a'))
        # logging.error('sent gpt2 request')
        # if 'Multiple Choice' in payload['question_types']:

        def fakequeryopenAI(text):
            # tester['openai'] = False
            output = openai.Completion.create(
                model="curie:ft-natlang-ai-2022-03-06-02-31-35",
                max_tokens=1,
                stop=["END"],
                temperature=.8,
                prompt=text,
                user=random.randrange(1, 10)*"a")
            # tester['openai'] = True
            return output['choices'][0]['text']
        executed.append(executor.submit(fakequeryopenAI, 'a'))

        def cold_start(url):
            # tester[url] = False
            response = requests.request("GET", url)
            # tester[url] = True
            return json.loads(response.content.decode("utf-8"))
        for url in ['https://dockerfbv2gcp-sn6tlr3gzq-uc.a.run.app', 'https://dockermcqv8gcp-sn6tlr3gzq-uc.a.run.app', 'https://dockerprepv3gcp-sn6tlr3gzq-uc.a.run.app/', 'https://dockertfv2gcp-sn6tlr3gzq-uc.a.run.app']:
            executed.append(executor.submit(cold_start, url))
        output = [call.result() for call in executed]
    return json.dumps(output)


@app.route('/query/new', methods=['GET', 'POST'])
@login_required
def new():

    form = QuestionsForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            api_check_begin = datetime.datetime.now()
            api_check = {'fb': False, 'tf': False, 'mcq': False, 'prep': False}
            url_mapping = {'fb': 'https://dockerfbv2gcp-sn6tlr3gzq-uc.a.run.app', 'mcq': 'https://dockermcqv8gcp-sn6tlr3gzq-uc.a.run.app',
                           'prep': 'https://dockerprepv3gcp-sn6tlr3gzq-uc.a.run.app/', 'tf': 'https://dockertfv2gcp-sn6tlr3gzq-uc.a.run.app'}
            while not all(api_check.values()):
                for key, value in url_mapping.items():
                    if api_check[key] == False:
                        try:
                            r = requests.get(value, timeout=(1, 1.5))
                            content = r.content.decode('utf-8')
                            # print(content)
                            if ('hi' in content) or ('hello' in content):
                                api_check[key] = True
                        except:
                            # print(key)
                            pass
            logging.error(
                "api_check " + str(datetime.datetime.now()-api_check_begin))
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
                if form.content.data != "":
                    payload['input_text'] = form.content.data
<<<<<<< HEAD
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
                        filter_coref_response = requests.request("POST", 'https://dockerprepv4gcp-sn6tlr3gzq-uc.a.run.app/filter_coref', data = json.dumps({'input_text': payload['input_text'].replace("\r", "").strip()}))
                        if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
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
                                if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
                                    logging.error('doing tf' + str(results))
                                gpt2_completions = []
                                tf_request = tfq(results, payload, selected_specific[start:start+payload['max_questions']['True/False']], coref_sents, gpt2_completions, executor)
                            logging.error(str(datetime.datetime.now()-beginTF))
                        if 'Multiple Choice' in payload['question_types']:
                            logging.info('before mcq')
                            start = offset_mapping['Multiple Choice']
                            if selected_specific[start:start+payload['max_questions']['Multiple Choice']]:
                                if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
                                    logging.error('doing mcq' + str(results))
                                mcq_request = mcq(myqg, results, payload, selected_specific[start:start+payload['max_questions']['Multiple Choice']], coref_sents, executor, uniqueUserId)
                            logging.info('after mcq', results)
                        if 'Fill in the Blanks' in payload['question_types']:
                            logging.error('total before ' + str(datetime.datetime.now()-newStart))
                            beginFB = datetime.datetime.now()
                            start = offset_mapping['Fill in the Blanks']
                            if selected_specific[start:start+payload['max_questions']['Fill in the Blanks']]:
                                if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
                                    logging.error('doing fb' + str(results))
                                fb_request = fitb(results, payload, selected_specific[start:start+payload['max_questions']['Fill in the Blanks']], coref_sents, executor, uniqueUserId)
                            # offset += payload['max_questions']['Fill in the Blanks']
                            logging.error('fitb timer'+str(datetime.datetime.now()-beginFB))
                        
                        if 'Multiple Choice' in payload['question_types']:
                            start = offset_mapping['Multiple Choice']
                            if selected_specific[start:start+payload['max_questions']['Multiple Choice']]:
                                mcq_output = [call.result() for call in mcq_request]
                                # results['Multiple Choice']['context'] = []
                                # results['Multiple Choice']['questions'] = []
                                # results['Multiple Choice']['answers'] = []
                                # results['Multiple Choice']['options'] = []
                                # try:
                                for x in mcq_output[0]['Multiple Choice']['questions']:
                                    results['Multiple Choice']['context'].append(x['context'])
                                    results['Multiple Choice']['questions'].append(x['question_statement'])
                                    results['Multiple Choice']['answers'].append(x['answer'])
                                    results['Multiple Choice']['options'].append(x['options'])

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
                                if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
                                    logging.error('output sents fb ' + str(fb_output))
                                # results['Multiple Choice']['context'] = []
                                # results['Fill in the Blanks']['questions'] = []
                                # results['Fill in the Blanks']['answers'] = []
                                # results['Fill in the Blanks']['options'] = []

                                for x in fb_output[0]['Multiple Choice']['questions']:
                                    results['Multiple Choice']['context'].append(x['context'])
                                    results['Multiple Choice']['questions'].append(x['question_statement'])
                                    results['Multiple Choice']['answers'].append(x['answer'])
                                    results['Multiple Choice']['options'].append(x['options'])
                                # results['Fill in the Blanks']['questions'] = fb_output[0]['questions']
                                # results['Fill in the Blanks']['answers'] = fb_output[0]['answers']
                                # except ValueError:
                                # pass

                        if 'True/False' in payload['question_types']:
                            start = offset_mapping['True/False']
                            if selected_specific[start:start+payload['max_questions']['True/False']]:
                                if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
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
                if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
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
=======
                else:
                    if "youtube" in form.url.data:
                        id = form.url.data.split("?v=")[1]
                    else:
                        id = form.url.data.split("youtu.be/")[1]
                    transcript_list = YouTubeTranscriptApi.list_transcripts(id)
                    valid_transcript = ''
                    for transcript in transcript_list:
                        if (transcript.is_generated == False) and ('en' in transcript.language_code):
                            valid_transcript = transcript
                    if valid_transcript == '':
                        for transcript in transcript_list:
                            if (transcript.is_generated == False):
                                valid_transcript = transcript
                    temp = ''
                    for item in valid_transcript.fetch():
                        temp += ' ' + item['text'].replace('\n', ' ')
                    payload['input_text'] = temp
                    # print(temp)
                # print(payload['input_text'])
                # print(results)
                # print(payload)
                # startCopy = datetime.datetime.now()
                # myqg, myqe = copyer(qg, qe)
                # logging.error('copy timer' + str(datetime.datetime.now()-startCopy))
                # print('before')
                # myqe= main.BoolQGen()
                # myqg = main.QGen()
                # print('after')
                if len(payload['input_text']) > 10:
                    translator = Translator()
                    lang = 'en'
                    try:
                        lang = translator.detect(
                            payload['input_text'][:150]).lang
                        if lang != 'en':
                            print(lang)
                            payload['input_text'] = translator.translate(
                                payload['input_text'], dest='en', src=lang).text
                    except:
                        pass
                    uniqueUserId = ''
                    for character in current_user.email:
                        uniqueUserId += str(ord(character))

                    beginPrep = datetime.datetime.now()
                    filter_coref_response = requests.request("POST", 'https://dockerprepv3gcp-sn6tlr3gzq-uc.a.run.app/filter_coref', data=json.dumps({
                                                             'input_text': payload['input_text'].replace("\r", "").strip()}))
                    logging.error("prepTime: " +str(datetime.datetime.now()-beginPrep))
                    if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
                        logging.error(
                            'payload: ' + str(payload['input_text'].replace("\r", "").strip().__repr__()))
                        logging.error('orig input: ' +
                                    str(payload['input_text'].__repr__()))
                        logging.error('orig selected_specific len' +
                                    str(filter_coref_response.content.decode("utf-8")))
                    
                    list_filter_coref = filter_coref_response.json()
                    print(list_filter_coref)
                    selected_specific = list_filter_coref[0]
                    coref_sents = list_filter_coref[1]
                    for item in selected_specific:
                        assert item[2] < len(coref_sents)
                    # logging.error(json.dumps(selected_specific))
                    logging.error('orig selected_specific len' +
                                  str(len(selected_specific)))
                    selected_specific = selected_specific[:payload['max_questions']['Multiple Choice'] +
                                                          payload['max_questions']['True/False'] + payload['max_questions']['Fill in the Blanks']]
                    # random.shuffle(selected_specific)
                    offset = 0
                    offset_mapping = {}
                    if 'Multiple Choice' in payload['question_types']:
                        offset_mapping["Multiple Choice"] = offset
                        offset += payload['max_questions']['Multiple Choice']
                    if 'True/False' in payload['question_types']:
                        logging.error('offsetting TF')
                        offset_mapping["True/False"] = offset
                        offset += payload['max_questions']['True/False']
                    # if 'Yes/No' in payload['question_types']:
                    #     offset_mapping["Yes/No"]= offset
                    #     offset += payload['max_questions']['Yes/No']
                    if 'Fill in the Blanks' in payload['question_types']:
                        offset_mapping["Fill in the Blanks"] = offset
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
                            if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
                                logging.error('doing tf' + str(results))
                            gpt2_completions = []
                            tf_request = tfq(
                                results, payload, selected_specific[start:start+payload['max_questions']['True/False']], coref_sents, gpt2_completions, executor)
                        logging.error(str(datetime.datetime.now()-beginTF))
                    if 'Multiple Choice' in payload['question_types']:
                        logging.info('before mcq')
                        start = offset_mapping['Multiple Choice']
                        if selected_specific[start:start+payload['max_questions']['Multiple Choice']]:
                            if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
                                logging.error('doing mcq' + str(results))
                            mcq_request = mcq(
                                MCQ_URL, selected_specific[start:start+payload['max_questions']['Multiple Choice']], coref_sents, executor, uniqueUserId)
                        logging.info('after mcq', results)
                    if 'Fill in the Blanks' in payload['question_types']:
                        logging.error('total before ' +
                                      str(datetime.datetime.now()-newStart))
                        beginFB = datetime.datetime.now()
                        start = offset_mapping['Fill in the Blanks']
                        if selected_specific[start:start+payload['max_questions']['Fill in the Blanks']]:
                            if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
                                logging.error('doing fb' + str(results))
                            fb_request = mcq(
                                FB_URL,  selected_specific[start:start+payload['max_questions']['Fill in the Blanks']], coref_sents, executor, uniqueUserId)
                        # offset += payload['max_questions']['Fill in the Blanks']
                        logging.error(
                            'fitb timer'+str(datetime.datetime.now()-beginFB))
                    results['Multiple Choice']['context'] = []
                    results['Multiple Choice']['questions'] = []
                    results['Multiple Choice']['answers'] = []
                    results['Multiple Choice']['options'] = []
                    if 'Multiple Choice' in payload['question_types']:
                        start = offset_mapping['Multiple Choice']
                        if selected_specific[start:start+payload['max_questions']['Multiple Choice']]:
                            mcq_output = [call.result()
                                          for call in mcq_request]
                            
                            # try:
                            for x in mcq_output[0]['Multiple Choice']['questions']:
                                results['Multiple Choice']['context'].append(
                                    x['context'])
                                results['Multiple Choice']['questions'].append(
                                    x['question_statement'])
                                results['Multiple Choice']['answers'].append(
                                    x['answer'])
                                results['Multiple Choice']['options'].append(
                                    x['options'])
                
                    if 'Fill in the Blanks' in payload['question_types']:
                        start = offset_mapping['Fill in the Blanks']
                        if selected_specific[start:start+payload['max_questions']['Fill in the Blanks']]:
                            fb_output = [call.result() for call in fb_request]
                            for x in fb_output[0]['Multiple Choice']['questions']:
                                results['Multiple Choice']['context'].append(
                                    x['context'])
                                results['Multiple Choice']['questions'].append(
                                    x['question_statement'])
                                results['Multiple Choice']['answers'].append(
                                    x['answer'])
                                results['Multiple Choice']['options'].append(
                                    x['options'])
                    if 'True/False' in payload['question_types']:
                        start = offset_mapping['True/False']
                        if selected_specific[start:start+payload['max_questions']['True/False']]:
                            if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
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
            if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
                logging.error(str(results))
            if lang != 'en':
                payload['input_text'] = translator.translate(
                    payload['input_text'], dest=lang, src='en').text
                for cat in results.keys():
                    for section in results[cat].keys():
                        # for group in results[cat][section]:
                        for i in range(len(results[cat][section])):
                            # print(results)
                            # print(results[cat][section][i])
                            if type(results[cat][section][i]) != list:
                                results[cat][section][i] = translator.translate(
                                    results[cat][section][i], dest=lang, src='en').text
                            else:
                                for innerI in range(len(results[cat][section][i])):
                                    results[cat][section][i][innerI] = translator.translate(
                                        results[cat][section][i][innerI], dest=lang, src='en').text

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
            # logging.info('here1')
            # print('form_json', formjson)
            # try:
            # logging.info('here2')
            beginRequest = datetime.datetime.now()
            text = ''
            copyurl = ''
            if ('torzi' not in current_user.email) and ('nibha' not in current_user.email) and ('vemurim09' not in current_user.email):
>>>>>>> 499ad77fc988b96295a3cfe71e3633eb1f614bec
                gform = requests.post(url=url, json=formjson)
            # logging.error('lenselected specific' + str(len(selected_specific)))
                logging.error(
                    'Request ' + str(datetime.datetime.now()-beginRequest))
                # logging.info('2.1')
                gform.raise_for_status()
<<<<<<< HEAD
                logging.info('here3')
            # except Exception as e:
                logging.info('3.1')
                # with open('error.txt', mode = 'w') as myFile:
                #     myFile.write('closer' + repr(e))
                logging.info('3.2')
                # with open('error.txt', mode = 'w') as myFile:
                #     myFile.write(str(gform.status_code))
                #     myFile.write(str(gform.text))
                if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
                    logging.info(gform.text)
                logging.info('here4')
=======
                # logging.info('here3')

>>>>>>> 499ad77fc988b96295a3cfe71e3633eb1f614bec
                form = QuestionsForm()
                # logging.info(gform.status_code)
                gform = gform.json()
                copyurl = re.sub(r"edit$", "copy", gform['link'])
                # gc.collect()
                # torch.cuda.empty_cache()
                text = requests.get(gform['html']).text
<<<<<<< HEAD
                if ('torzi' not in current_user.email) and ('nibha' not in current_user.email):
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
=======
                # logging.info(text)
                
            return render_template('questions.html', input_text=payload['input_text'],
                                   html=text,
                                   json_data=json.dumps(data_list).replace(r'\"', r'\\\"').replace(
                                       r"\'", r"\\\'").replace(r"\n", "").replace(r"\t", r""),
                                   copy=copyurl,
                                   form=form,
                                   pic=current_user.profile_pic,
                                   info=[current_user.name]
                                   )
        else:
            print('all is not correct')
    return render_template('query.html', form=form,
                           pic=current_user.profile_pic,
                           info=[current_user.name]
                           )




@app.route("/Kahootxlsx", methods=['POST'])
def Kahootxlsx():
    data = request.get_json(force=True)
    # print(data)
    columns = ["Question - max 120 characters", "Answer 1 - max 75 characters", "Answer 2 - max 75 characters", "Answer 3 - max 75 characters",
               "Answer 4 - max 75 characters", "Time limit (sec) – 5, 10, 20, 30, 60, 90, 120, or 240 secs", "Correct answer(s) - choose at least one"]
    df = pd.DataFrame(columns=columns)
    for group in data['Multiple Choice']:
        temp = ['', '', '', '', '', '60', '']
        temp[0] = group[1]['questions']
        ans = []
        ans.append((group[2]['answers'], 1))
        for item in group[3]['options']:
            ans.append((item, 0))
        random.shuffle(ans)
        for i, item in enumerate(ans):
            temp[i+1] = item[0]
            if item[1] == 1:
                temp[-1] = str(i+1)
        df.loc[len(df.index)] = temp

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    # taken from the original question
    df.to_excel(writer, startrow=0, index=False,
                merge_cells=False, sheet_name="Sheet_1")
    # the writer has done its job
    writer.close()

    # go back to the beginning of the stream
    output.seek(0)
    # output.getbuffer()

    # finally return the file
    return send_file(BytesIO(base64.b64encode(output.getbuffer())), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/Quizizzxlsx", methods=['POST'])
def Quizizzxlsx():
    data = request.get_json(force=True)
    # print(data)
    columns = ["Question Text", "Question Type", "Option 1", "Option 2", "Option 3", "Option 4", "Option 5",
               "Correct Answer", "Time in seconds", "Image Link"]
    df = pd.DataFrame(columns=columns)
    for group in data['Multiple Choice']:
        temp = ['', '', '', '', '', '', '', '', '', '']
        temp[0] = group[1]['questions']
        temp[1] = "Multiple Choice"
        ans = []
        ans.append((group[2]['answers'], 1))
        for item in group[3]['options']:
            ans.append((item, 0))
        random.shuffle(ans)
        for i, item in enumerate(ans):
            temp[i+2] = item[0]
            if item[1] == 1:
                temp[7] = str(i+1)
        df.loc[len(df.index)] = temp
    for group in data['Fill in the Blanks']:
        temp = ['', '', '', 'incorrect', '', '', '', '1', '', '']
        temp[0] = group[0]['questions']
        temp[1] = "Fill-in-the-Blank"
        temp[2] = group[1]['answers']
        df.loc[len(df.index)] = temp
    for group in data['True/False']:
        temp = ['', '', 'True', 'False', '', '', '', '', '', '']
        val = random.choice([0, 1])
        if (val == 0):
            temp[0] = group[0]['correct']
        if (val == 1):
            temp[0] = group[1]['incorrect'][-1]
        temp[1] = "Multiple Choice"
        temp[-3] = str(val + 1)
        df.loc[len(df.index)] = temp
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    # taken from the original question
    df.to_excel(writer, startrow=0, index=False,
                merge_cells=False, sheet_name="Sheet_1")
    # the writer has done its job
    writer.close()

    # go back to the beginning of the stream
    output.seek(0)
    # output.getbuffer()

    # finally return the file
    return send_file(BytesIO(base64.b64encode(output.getbuffer())), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/Gimkitcsv", methods=['POST'])
def Gimkitcsv():
    data = request.get_json(force=True)
    # print(data)
    columns = ["Gimkit Spreadsheet Import Template", "", "", "", ""]
    df = pd.DataFrame(columns=columns)
    df.loc[len(df.index)] = ['Question', 'Correct Answer',
                             'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']
    for group in data['Multiple Choice']:
        temp = ['', '', '', '', '']
        temp[0] = group[1]['questions']
        temp[1] = group[2]['answers']
        for i, item in enumerate(group[3]['options']):
            temp[i+2] = item
        df.loc[len(df.index)] = temp
    for group in data['True/False']:
        temp = ['', '', '', '', '']
        temp[0] = 'Which of the following is true?'
        temp[1] = group[0]['correct']
        for i, item in enumerate(group[1]['incorrect']):
            temp[i+2] = item
        df.loc[len(df.index)] = temp
    output = BytesIO()
    # writer = pd.ExcelWriter(output, engine='xlsxwriter')

    # taken from the original question
    df.to_csv(output, index=False)
    # the writer has done its job
    # output.close()

    # go back to the beginning of the stream
    output.seek(0)
    # output.close()
    # output.getbuffer()

    # finally return the file
    return send_file(BytesIO(base64.b64encode(output.getbuffer())), mimetype="text/csv")
>>>>>>> 499ad77fc988b96295a3cfe71e3633eb1f614bec


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


# print('reading to run')
if __name__ == '__main__':
    print('here')
    app.run(
        **flask_run_args
    )
# print('after run')
