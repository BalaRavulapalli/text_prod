from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification, ElectraForQuestionAnswering, ElectraTokenizerFast
import nltk
import torch
nltk.download('punkt')
from typing import List
from spacy.tokens import Doc, Span
from coreference_resolution.utils import load_models, print_clusters, print_comparison
import random
import numpy as np
from fastT5 import get_onnx_model,get_onnx_runtime_sessions,OnnxT5
from pathlib import Path
import os
import datetime
import logging
import openai
openai.api_key = "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"

def core_logic_part(document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
    final_token = document[coref[1]]
    if final_token.tag_ in ["PRP$", "POS"]:
        resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
    else:
        resolved[coref[0]] = mention_span.text + final_token.whitespace_
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""
    return resolved

def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
    spans = [doc[span[0]:span[1]+1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
        if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
    return span_noun_indices


def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end+1]
    return head_span, [head_start, head_end]


def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])


def improved_replace_corefs(document, clusters):
    resolved = list(tok.text_with_ws for tok in document)
    all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)

            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)

class BalaQG:
    def __init__(self):
        trained_model_path = './onnxt5/t5_squad_v1/models/'
        encoder_path = os.path.join(trained_model_path,f"t5_squad_v1-encoder-quantized.onnx")
        decoder_path = os.path.join(trained_model_path,f"t5_squad_v1-decoder-quantized.onnx")
        init_decoder_path = os.path.join(trained_model_path,f"t5_squad_v1-init-decoder-quantized.onnx")
        model_paths = encoder_path, decoder_path, init_decoder_path
        model_sessions = get_onnx_runtime_sessions(model_paths)
        c_q_model = OnnxT5(trained_model_path, model_sessions)

        c_q_tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
        predictor, nlp = load_models()
        self.c_a_model = ElectraForQuestionAnswering.from_pretrained("./electraAextraction18381/checkpoint-18381/")
        self.c_a_toker = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")
        self.c_q_model = c_q_model
        self.c_q_model.config.max_length = 512
        self.c_q_toker = c_q_tokenizer
        # self.a_o_model = GPT2LMHeadModel.from_pretrained("t5smallqano/checkpoint-10544/")
        # self.a_o_model.config.max_length = 128
        # self.a_o_toker = GPT2Tokenizer.from_pretrained("./t5smallqano/checkpoint-10544/", eos_token='<|endoftext|>')
        self.filter_model = AutoModelForSequenceClassification.from_pretrained("./distilbertfilter/checkpoint-650/")
        self.filter_toker = AutoTokenizer.from_pretrained("./distilbertfilter/checkpoint-650/")
        self.predictor = predictor
        self.nlp = nlp
    def filter_coref(self, input_dict):
        context = input_dict['input_text']
        with torch.no_grad():
            beginFilter = datetime.datetime.now()
            softer = torch.nn.Softmax(dim = 0)
            sents = nltk.sent_tokenize(context)
            specific = []
            for i, sent in enumerate(sents):
                sample = self.filter_toker(sent, return_tensors = "pt", add_special_tokens=True, truncation=True,  max_length=512)
                logits= self.filter_model(**sample).logits.squeeze()
                softed = softer(logits)
                max = np.argmax(softed)
                if (max == 1) and ('?' not in sent):
                    specific.append([sent, softed[1], i])
            specific.sort(reverse = True, key = lambda x: x[1])
            selected_specific = specific
            selected_specific.sort(key = lambda x: x [2])
            # sents = selected_specific
            beginCoref = datetime.datetime.now()
            coref_text = improved_replace_corefs(self.nlp(context), self.predictor.predict(context)['clusters'])
            coref_sents = nltk.sent_tokenize(coref_text)
            logging.error('filtering ' + str(datetime.datetime.now()-beginFilter))
        return [selected_specific, coref_sents]
    def predict_mcq(self, input_dict, selected_specific, coref_sents, executor, uniqueUserId):
        logging.basicConfig(filename = 'example.log', level  = logging.ERROR)
        context = input_dict['input_text']
        logging.error("selected_specific: " + str(selected_specific))
        logging.error("coref_sents: " + str(coref_sents))
        # max_length = input_dict['max_questions']
        with torch.no_grad():
            # beginFilter = datetime.datetime.now()
            softer = torch.nn.Softmax(dim = 0)
            # sents = nltk.sent_tokenize(context)
            # specific = []
            # for i, sent in enumerate(sents):
            #     sample = self.filter_toker(sent, return_tensors = "pt", add_special_tokens=True, truncation=True,  max_length=512)
            #     logits= self.filter_model(**sample).logits.squeeze()
            #     softed = softer(logits)
            #     max = np.argmax(softed)
            #     if (max == 1) and ('?' not in sent):
            #         specific.append([sent, softed[1], i])
            # specific.sort(reverse = True, key = lambda x: x[1])
            # selected_specific = specific[:max_length]
            # selected_specific.sort(key = lambda x: x [2])
            sents = selected_specific
            # logging.error('filtering ' + str(datetime.datetime.now()-beginFilter))
            # beginCoref = datetime.datetime.now()
            # coref_text = improved_replace_corefs(self.nlp(context), self.predictor.predict(context)['clusters'])
            # coref_sents = nltk.sent_tokenize(coref_text)
            # logging.error('coref ' + str(datetime.datetime.now()-beginCoref))
            answer_grouping = []
            beginAextraction = datetime.datetime.now()
            for group in sents:
                text = group[0]
                tokered = self.c_a_toker(text, return_tensors="pt", add_special_tokens=True, truncation=True,  max_length=512, return_token_type_ids=True)
                output = self.c_a_model(**tokered)
                starts = softer(output.start_logits.cpu()[0]).tolist()
                ends = softer(output.end_logits.cpu()[0]).tolist()
                scores = []
                for outerI, start in enumerate(starts):
                    for innerI, end in enumerate(ends):
                        if innerI > outerI:
                            scores.append([[outerI, innerI], start + end])
                scores.sort(key = lambda x: x[1], reverse = True)
                starttok = scores[0][0][0]
                endtok = scores[0][0][1]
                
                offsets = self.c_a_toker(text, return_offsets_mapping=True)['offset_mapping']
                startOffset = offsets[starttok][0]
                endOffset = offsets[endtok][0]
                answer = text[startOffset:endOffset]
                answer_grouping.append([coref_sents[group[2]], answer])
                if [starttok, endtok] != [np.argmax(starts), np.argmax(ends)]:
                    logging.error('ans do not match' + str(answer))
            logging.error('aExtraction ' + str(datetime.datetime.now()-beginAextraction))
            beginQG = datetime.datetime.now()
            question_grouping = []
            def openAIQuery(queryInput):
                result = openai.Completion.create(
                    model="babbage:ft-natlang-ai-2021-12-01-02-21-24",
                    max_tokens = 200,
                    stop = ["####"],
                    temperature = .75, 
                    prompt=f"question: {queryInput['question']}\nanswer: {queryInput['answer']}\n\n##\n\n",
                    user = uniqueUserId)['choices'][0]['text']
                return result
            def openAIRating(text):
                rating = openai.Completion.create(
                        engine="content-filter-alpha",
                        prompt = "<|endoftext|>"+text+"\n--\nLabel:",
                        temperature=0,
                        max_tokens=1,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        logprobs=10,
                        user = uniqueUserId
                        )["choices"][0]["logprobs"]["top_logprobs"][0]
                return rating
            openAIcompletions = []
            for sentence, answer in answer_grouping:
                # if len(question_grouping) < max_length:
                
                # c_q_input = {k:v.unsqueeze(0) for k, v in c_q_input.items()}
                result = self.c_q_toker.decode(self.c_q_model.generate(**self.c_q_toker(f'context: {sentence} answer: {answer} ', return_tensors = "pt"), num_beams = 1)[0])
                question = result.split('question: ')[1].split('</s>')[0]
                if (answer.lower() not in question.lower()):
                    openAIcompletions.append(executor.submit(openAIQuery, {'question': question, 'answer': answer}))
                    question_grouping.append([question, answer, sentence])
            logging.error('QG ' + str(datetime.datetime.now()-beginQG))
            beginDistractor = datetime.datetime.now()
            final_grouping = []
            openAICompleted = [call.result() for call in openAIcompletions]
            for item in openAICompleted:
                executor.submit(openAIRating, item)
            for group, completion in zip(question_grouping, openAICompleted):
                question, answer, sentence = group
                # input_txt = f"question: {question} answer: {answer}"
                # encoded = self.a_o_toker.encode(input_txt, return_tensors="pt")
                # output = self.a_o_toker.decode(self.a_o_model.generate(input_ids = encoded, num_beams = 1)[0], clean_up_tokenization_spaces=True)
                # logging.error(str(completion))
                output = completion
                print(output)
                options = []
                try:
                    split1 = output.split('option1:')[1]
                    split2 = split1.split('option2:')
                    options.append(split2[0].strip())
                    split2 = split2[1]
                    split3 = split2.split('option3:')
                    options.append(split3[0].strip())
                    options.append(split3[1].strip())
                except:
                    continue
                # split4 = split3[1].split(" <|endoftext|>")
                # options.append(split4[0])
                final_grouping.append([question, answer, options, sentence])
            logging.error('Distractor ' + str(datetime.datetime.now()-beginDistractor))
            func_output = {'Multiple Choice': {'questions': []}}   
            for item in final_grouping:
                answer = item[1].strip()
                options = [x.strip() for x in item[2]]
                while answer in options:
                    options.remove(answer)
                func_output['Multiple Choice']['questions'].append(
                    {'question_statement': item[0],
                    'answer': answer,
                    'options': list(set(options)),
                    'context': item[3]}
                )
        return func_output