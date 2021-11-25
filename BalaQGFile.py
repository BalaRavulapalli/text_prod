from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification, ElectraForQuestionAnswering, ElectraTokenizerFast
import nltk
import torch
nltk.download('punkt')
from typing import List
from spacy.tokens import Doc, Span
from coreference_resolution.utils import load_models, print_clusters, print_comparison
import random
import numpy as np


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
        predictor, nlp = load_models()
        self.c_a_model = ElectraForQuestionAnswering.from_pretrained("./electraAextraction18381/checkpoint-18381/")
        self.c_a_toker = ElectraTokenizerFast.from_pretrained("./electraAextraction18381/checkpoint-18381/")
        self.c_q_model = T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_squad_v1")
        self.c_q_model.config.max_length = 512
        self.c_q_toker = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_squad_v1")
        self.a_o_model = GPT2LMHeadModel.from_pretrained("t5smallqano/checkpoint-10544/")
        self.a_o_model.config.max_length = 128
        self.a_o_toker = GPT2Tokenizer.from_pretrained("./t5smallqano/checkpoint-10544/", eos_token='<|endoftext|>')
        self.filter_model = AutoModelForSequenceClassification.from_pretrained("./distilbertfilter/checkpoint-650/")
        self.filter_toker = AutoTokenizer.from_pretrained("./distilbertfilter/checkpoint-650/")
        self.predictor = predictor
        self.nlp = nlp

    def predict_mcq(self, input_dict):
        context = input_dict['input_text']
        max_length = input_dict['max_questions']
        with torch.no_grad():
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
            selected_specific = specific[:max_length]
            selected_specific.sort(key = lambda x: x [2])
            sents = selected_specific
            # i = 0
            # sents2 = []
            
            # # while i <= (len(sents) - 1):
            # #     if i ==(len(sents) - 1):
            # #         sents2.append(sents[i])
            # #     else:
            # #         sents2.append(sents[i] + ' '+ sents[i +1])
            # #     i +=2
                        
            # while i <= (len(sents) - 1):
            #     location = sents[i][2]
            #     temp_group = sents[i:i+2]
            #     sents_temp = [x[0] for x in temp_group]
            #     sents2.append([' '.join(sents_temp), location])
            #     i +=2
            # # text_mapping = {}
            # # sent2_count = []
            # # previous = 0
            # # tempLen = 0
            # # for sentGroup in sents2:
            # #     sent2_count.append(previous)
            # #     tempLen = len(nltk.sent_tokenize(sentGroup))
            # #     previous += tempLen
            coref_text = improved_replace_corefs(self.nlp(context), self.predictor.predict(context)['clusters'])
            coref_sents = nltk.sent_tokenize(coref_text)
            # assert len(sents2) == len(sent2_count)
            answer_grouping = []
            # softer = torch.nn.Softmax(dim=0)
            for group in sents:
                text = group[0]
                # if len(answer_grouping) < max_length:
                with torch.no_grad():
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
                offsets = self.c_a_toker(group, return_offsets_mapping=True)['offset_mapping']
                startOffset = offsets[starttok][0]
                endOffset = offsets[endtok][0]
                answer = text[startOffset:endOffset]
                answer_grouping.append([coref_sents[group[2]], answer])
                # output1 = output1.split('answers:')[1]
                # output1 = output1.split('</s>')[0]
                # output1 = output1.split('[A]')[:-1]
                # output1 = [x.strip() for x in output1]
                # output1= list(set(output1))
                # keeping = []
                # for output in output1:
                #     if output in output1:
                #         keeping.append(output)
                # textSents = nltk.sent_tokenize(text)
                # for kept in output1:
                #     for i, sent in enumerate(textSents):
                #         if kept in sent:
                #             # print('coref', len(coref_sents))
                #             # print('sent2_count', len(sent2_count))
                #             # print('outerI', outerI)
                #             # print('i', i)
                #             # print('sent2_count[outerI]', sent2_count[outerI])
                #             answer_grouping.append((coref_sents[location + i], kept))
                #             break
            # while len(answer_grouping) > max_length:
            #     answer_grouping.pop(random.choice(range(len(answer_grouping))))

            question_grouping = []
            for sentence, answer in answer_grouping:
                if len(question_grouping) < max_length:
                    result = self.c_q_toker.decode(self.c_q_model.generate(**self.c_q_toker(f'context: {sentence} answer: {answer} ', return_tensors = "pt"))[0])
                    question = result.split('question: ')[1].split('</s>')[0]
                    if (answer.lower() not in question.lower()):
                        question_grouping.append([question, answer, sentence])
            final_grouping = []
            for question, answer, sentence in question_grouping:
                input_txt = f"question: {question} answer: {answer}"
                encoded = self.a_o_toker.encode(input_txt, return_tensors="pt")
                output = self.a_o_toker.decode(self.a_o_model.generate(input_ids = encoded)[0], clean_up_tokenization_spaces=True)
                print(output)
                options = []
                split1 = output.split('option1: ')[1]
                split2 = split1.split('option2: ')
                options.append(split2[0])
                split2 = split2[1]
                split3 = split2.split('option3: ')
                options.append(split3[0])
                split4 = split3[1].split(" <|endoftext|>")
                options.append(split4[0])
                final_grouping.append([question, answer, options, sentence])
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
                


            

                
            

