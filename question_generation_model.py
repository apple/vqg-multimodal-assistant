#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

"""
This file contains the Question Generation Model class

"""
import logging
import random
import sys
import re

sys.path.append("./")
sys.path.append("../")

import boto3
import os

import requests
sys_random = random.SystemRandom()

from cluster import cluster_texts
from bert_layer import BertLayer, preprocess_bert_input
from elmo_layer import ELMoEmbedding#, ElmoEmbeddingLayer

from botocore.client import Config

from PIL import Image
from io import BytesIO
from numpy import array
import numpy as np
import tensorflow

import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Lambda, Layer, Input, Dense, Embedding, Dropout, LSTM, Concatenate, Add, Bidirectional
from tensorflow.keras.models import Model, load_model

from tensorflow.keras import backend as K
# If running ELMO, the below need to be imported
# otherwise import the two lines above
# from keras.optimizers import Adam
# from keras.models import Model, load_model
# from keras.layers import Lambda, Layer, Input, Dense, Embedding, Dropout, LSTM, Concatenate, Add, Bidirectional

try:
    import turibolt as bolt
except:
    print('Running on local')

mcqueen_url = "http://store-test.blobstore.apple.com"
mcqueen_region_name = "store-test"
mcqueen_access_key = "MKIAXRQON2I79WCA4Q19"
mcqueen_secret_key = "C9ED2E4ABA42DD317E498C391A9AD989099C6E9018DD68508A232738719A915B"


class QuestionGenerationModel:

    def __init__(self, datasets, logger, hidden_units=256, dropout=0.5):

        logging.info("Initialize model")
        self.logger = logger
        self.datasets = datasets
        self.embedding_dim = 200
        self.vocab_size = len(datasets.vocabulary)
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.word_to_idx = datasets.word_to_idx
        if 'glov' in self.datasets.embedding_file:
            self.embedding_matrix = self.load_embeddings()
        else:
            self.embedding_matrix = None
        self.no_of_samples = 0
        self.model = None
        self.tokenizer = None
        self.input_shape = 0
        self.s3 = boto3.resource(service_name='s3',
                                 endpoint_url=mcqueen_url,
                                 region_name=mcqueen_region_name,
                                 aws_access_key_id=mcqueen_access_key,
                                 aws_secret_access_key=mcqueen_secret_key,
                                 config=Config(s3={'addressing_style': 'path', 'signature_version': 's3'}))

    def load_embeddings(self):
        """
        Load Glove vectors
        :return:
        """
        glove_dir = os.getcwd()  # '/Volumes/Data/data/glove.6B'
        embeddings_index = {}  # empty dictionary
        f = open(os.path.join(glove_dir, self.datasets.embedding_file), encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        self.logger.info("Embedding_index: %s" % str(len(embeddings_index)))

        # Get 200-dim dense vector for each of the words in our vocabulary
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        for word, i in self.word_to_idx.items():
            # if i < max_words:
            embedding_vector = embeddings_index.get(word)

            if word == '<START>' or word == '<END>':
                embedding_vector = np.random.rand(self.embedding_dim)

            if embedding_vector is not None:

                embedding_matrix[i] = embedding_vector
            else:
                # Words not found in the embedding index will be all zeros
                embedding_matrix[i] = np.zeros(self.embedding_dim)

        return embedding_matrix

    def build_glove_model(self):
        """
        Build model definition file using GloVe embeddings and image input
        :return:
        """

        # image feature
        inputs1 = layers.Input(shape=(self.input_shape,))
        fe1 = Dropout(self.dropout)(inputs1)
        fe2 = layers.Dense(self.hidden_units, activation='relu')(fe1)

        # partial question sequence model
        inputs2 = Input(shape=(self.datasets.max_question_len,))
        se1 = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(self.dropout)(se1)
        # Bi directionasl is harder to train???
        question_seq_model = LSTM(self.hidden_units)(se2)
        # question_seq_model = Bidirectional(LSTM(self.hidden_units))(se2)
        # question_seq_model = layers.Dense(self.hidden_units, activation='relu')(question_seq_model)
        # se4 = Dropout(self.dropout)(se3)

        # decoder (feed forward) model
        decoder1 = Add()([fe2, question_seq_model])
        # keras.layers.Add()([x1, x2])

        decoder2 = layers.Dense(self.hidden_units, activation='relu')(decoder1)
        outputs = layers.Dense(self.vocab_size, activation='softmax')(decoder2)

        # merge the two input models
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        # make embedding layer weights fixed
        model.layers[2].set_weights([self.embedding_matrix])
        model.layers[2].trainable = False

        model.summary()
        optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # optimizer = SGD()

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def build_keyword_model(self):
        """
        Build model definition using GloVe embeddings and image, keyword input
        :return:
        """

        self.logger.debug('In keyword model')

        # image feature
        inputs1 = layers.Input(shape=(self.input_shape,))
        fe1 = Dropout(self.dropout)(inputs1)
        fe2 = layers.Dense(self.hidden_units, activation='relu')(fe1)

        # partial question sequence model
        inputs2 = Input(shape=(self.datasets.max_question_len,))
        se1 = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(self.dropout)(se1)
        question_seq_model = LSTM(self.hidden_units)(se2)
        # question_seq_model = Bidirectional(LSTM(self.hidden_units))(se2)
        # question_seq_model = layers.Dense(self.hidden_units, activation='relu')(question_seq_model)
        # se4 = Dropout(self.dropout)(se3)

        # keyword sequence model
        inputs3 = Input(shape=(self.datasets.max_keyword_len,))
        k1 = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(inputs3)
        k2 = Dropout(self.dropout)(k1)
        keyword_seq_model = LSTM(self.hidden_units)(k2)

        # decoder (feed forward) model
        decoder1 = Add()([fe2, question_seq_model, keyword_seq_model])
        # keras.layers.Add()([x1, x2])

        decoder2 = layers.Dense(self.hidden_units, activation='relu')(decoder1)
        outputs = layers.Dense(self.vocab_size, activation='softmax')(decoder2)

        # merge the two input models
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        # make embedding layer weights fixed
        model.layers[3].set_weights([self.embedding_matrix])
        model.layers[3].trainable = False

        model.layers[4].set_weights([self.embedding_matrix])
        model.layers[4].trainable = False

        model.summary()
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # optimizer = SGD()

        #TODO: try different optimizer?
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def build_elmo_model(self):
        """
        Build model definition using ELMO embeddings and image input
        TODO: This model is throwing out garbage words. Maybe because Lamba layer in untrainable.
        :return:
        """

        self.logger.info('Building elmo model')
        # image feature
        inputs1 = Input(shape=(self.input_shape,))
        fe1 = Dropout(self.dropout)(inputs1)
        fe2 = layers.Dense(self.hidden_units, activation='relu')(fe1)

        # partial question sequence model
        inputs2 = Input(shape=(1,), dtype="string")
        self.logger.info('Building elmo model 2')
        se1 = Lambda(ELMoEmbedding)(inputs2)
        # se1 = ElmoEmbeddingLayer(pooling='first', trainable=False)(inputs2)
        self.logger.info('Building elmo model 3')
        se2 = Dropout(self.dropout)(se1)
        question_seq_model = LSTM(self.hidden_units)(se2)
        self.logger.info('Building elmo model 3')

        # decoder (feed forward) model
        decoder1 = Add()([fe2, question_seq_model])
        self.logger.info('Building elmo model 4')
        # keras.layers.Add()([x1, x2])

        decoder2 = layers.Dense(self.hidden_units, activation='relu')(decoder1)
        ge1 = Dropout(self.dropout)(decoder2)
        outputs = layers.Dense(self.vocab_size, activation='softmax')(ge1)
        print('Output:', outputs)
        self.logger.info('Building elmo model 5')
        # merge the two input models
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.summary()
        optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #TODO: try different optimizer?
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.logger.info('Elmo model succesfully built')

        return model

    def build_bert_model(self):
        """
        Build model definition using BERT embeddings and image, keyword input
        :return:
        """

        self.logger.info('Constructing bert model...')
        # image feature
        inputs1 = Input(shape=(self.input_shape,))
        fe1 = Dropout(self.dropout)(inputs1)
        fe2 = layers.Dense(self.hidden_units, activation='relu')(fe1)

        # partial question sequence model
        in_id = Input(shape=(self.datasets.max_question_len,), name="input_ids")
        in_mask = Input(shape=(self.datasets.max_question_len,), name="input_masks")
        in_segment = Input(shape=(self.datasets.max_question_len,), name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]

        bert_output = BertLayer(n_fine_tune_layers=10, trainable=True)(bert_inputs)
        se2 = Dropout(self.dropout)(bert_output)
        question_seq_model = LSTM(self.hidden_units)(se2)
        # question_seq_model = Dense(self.hidden_units, activation='relu')(se2)

        # decoder (feed forward) model
        decoder1 = Add()([fe2, question_seq_model])
        # keras.layers.Add()([x1, x2])

        decoder2 = layers.Dense(self.hidden_units, activation='relu')(decoder1)
        outputs = layers.Dense(self.vocab_size, activation='softmax')(decoder2)

        # merge the two input models
        model = Model(inputs=[inputs1, in_id, in_mask, in_segment], outputs=outputs)

        model.summary()
        optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cleanText(self, text):
        """
        Standardizes an input string
        :param text: Input string
        :return: Standardized clean string according to the rules below
        """
        text = text.strip().replace("\n", " ").replace("\r", " ")
        # text = replace_contraction(text)
        # text = replace_links(text, "link")
        # text = remove_numbers(text)
        text = re.sub(r'[,!@#$%^&*)(|/><";:.?\'\\}{]', "", text)
        text = text.lower()
        return text

    def greedy_search(self, image_input, model, keyword=None):
        """
        Decoding strategy of choosing highest scoring token at each step
        :param image_input: image input features
        :param model: model definition file
        :param keyword: Keyword input features
        :return:
        """

        if 'glove' in self.datasets.embedding_file:
            max_seq_len = model.inputs[1].shape[1].value
        elif 'elmo' in self.datasets.embedding_file:
            max_seq_len = self.datasets.max_question_len
            image_input = np.repeat(image_input, axis=0, repeats=2)
        elif 'bert' in self.datasets.embedding_file:
            max_seq_len = self.datasets.max_question_len
            image_input = np.repeat(image_input, axis=0, repeats=2)
        self.logger.info('Max len %s' % max_seq_len)
        in_text = '<START>'
        prob = 0
        for i in range(max_seq_len):
            sequence = [self.word_to_idx[w] for w in in_text.split(' ') if w in self.word_to_idx]
            if self.datasets.use_keyword:
                sequence = pad_sequences([sequence], maxlen=max_seq_len, padding='post')
                yhat = model.predict([image_input, sequence, keyword])[0]
            elif 'glove' in self.datasets.embedding_file:
                sequence = pad_sequences([sequence], maxlen=max_seq_len, padding='post')
                yhat = model.predict([image_input, sequence])[0]
            elif 'elmo' in self.datasets.embedding_file:
                sequence = ' '.join([w for w in in_text.split(' ')])
                sequence = self.cleanText(sequence)
                sequence = np.array([[sequence], [sequence]])
                self.logger.debug('Sequence %s shape %s image %s' % (sequence, sequence.shape, image_input.shape))

                yhat = model.predict([image_input, sequence])[0]
            elif 'bert' in self.datasets.embedding_file:
                sequence = ' '.join([w for w in in_text.split(' ')[1:]])
                sequence = self.cleanText(sequence)
                sequence = [[sequence], [sequence]]
                # Preprocess bert
                input_ids, input_masks, segment_ids, _ = preprocess_bert_input(sequence, [None] * len(sequence),
                                                                               self.datasets.max_question_len, self.tokenizer, self.vocab_size)
                yhat = model.predict([image_input, input_ids, input_masks, segment_ids])[0]
            else:
                self.logger.error('Embedding strategy not supported')
                exit(-1)

            yhat_max = np.argmax(yhat)
            prob += yhat[yhat_max]
            word = self.datasets.idx_to_word[yhat_max]
            in_text += ' ' + word
            if word == '<END>':
                break
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)

        if final not in self.datasets.unique_train_questions:
            self.logger.info(
             'Unique generated questions not seen in training data: %s' % final)
        self.datasets.unique_generated_questions.add(final)
        self.datasets.generated_questions.append([final])

        self.logger.info('Final greedy candidate: %s' % final)
        return {final: prob}

    def beam_search(self, image_input, model, beam_size, keyword=None):
        """
        This function performs simple beam search
        :param image_input: Image encoded feature
        :param model: Model definition file
        :param beam_size: Beam size to be used for decoding
        :param keyword: Keyword input feature to be used with some model architectures
        :return:
        """
        start = [self.word_to_idx['<START>']]

        if 'glov' in self.datasets.embedding_file:
            max_seq_len = model.inputs[1].shape[1].value
        else:
            max_seq_len = self.datasets.max_question_len
            image_input = np.repeat(image_input, axis=0, repeats=2)
        self.logger.info('max len %s' % max_seq_len)
        start_word = [[start, 0.0]]

        EOS_utterances= []

        while len(start_word[0][0]) < max_seq_len:

            temp = []
            for s in start_word:
                if self.datasets.use_keyword:
                    sequence = pad_sequences([s[0]], maxlen=max_seq_len, padding='post')
                    preds = model.predict([image_input, sequence, keyword])
                elif 'glove' in self.datasets.embedding_file:
                    sequence = pad_sequences([s[0]], maxlen=max_seq_len, padding='post')
                    preds = model.predict([image_input, sequence])
                elif 'elmo' in self.datasets.embedding_file:
                    sequence = ' '.join([self.datasets.idx_to_word[idx] for idx in s[0]])
                    sequence = self.cleanText(sequence)
                    sequence = np.array([sequence, sequence])
                    preds = model.predict([image_input, sequence])
                elif 'bert' in self.datasets.embedding_file:
                    sequence = ' '.join([self.datasets.idx_to_word[idx] for idx in s[0][1:]])
                    sequence = self.cleanText(sequence)
                    sequence = [[sequence], [sequence]]
                    # sequence = self.cleanText(sequence)
                    input_ids, input_masks, segment_ids, _ = preprocess_bert_input(sequence, [None] * len(sequence),
                                                                                   self.datasets.max_question_len,
                                                                                   self.tokenizer, self.vocab_size)
                    preds = model.predict([image_input, input_ids, input_masks, segment_ids])
                else:
                    exit(-1)

                word_preds = np.argsort(preds[0])[-2 * beam_size:]

                # Getting the top <beam_size>(n) predictions and creating a
                # new list so as to put them via the model again

                for w in word_preds:
                    next_quest, prob = s[0][:], s[1]
                    next_quest.append(w)
                    if w == 2:
                        EOS_utterances.append([next_quest, prob])
                    prob += preds[0][w]
                    temp.append([next_quest, prob])

            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_size:]

        candidates = dict()
        max_prob = 0
        final_candidate = ''
        unique_questions_not_seen_training_data = set()
        thresh = 2.0
        for st_wd in start_word + EOS_utterances:
            prob = st_wd[1]
            st_wd = st_wd[0]

            intermediate_question = [self.datasets.idx_to_word[i] for i in st_wd]

            final_question = []

            for i in intermediate_question:
                if i != '<END>':
                    final_question.append(i)
                else:
                    break

            final_question = ' '.join(final_question[1:])

            if prob > max_prob:
                max_prob = prob
                final_candidate = final_question
            if prob > thresh:
                if final_question in candidates:
                    if prob > candidates[final_question]:
                        candidates[final_question] = prob
                else:
                    candidates[final_question] = prob

            if final_question not in self.datasets.unique_train_questions:
                unique_questions_not_seen_training_data.add(final_question)

        self.logger.info('Unique generated questions not seen in training data: %s' % unique_questions_not_seen_training_data)
        self.datasets.unique_generated_questions.update(candidates.keys())
        self.logger.info('Final Simple BS candidates: %s' % candidates)

        self.datasets.generated_questions += list(candidates.keys())

        # Inventiness:
        # Number of unique questions not seen in training data / Total number of generated questions for that image
        self.logger.info('Inventiveness: %s' % str(len(unique_questions_not_seen_training_data)/len(candidates)))
        return [final_candidate], candidates

    def dissimilarity_grouping(self, temp, num_groups, beam_size):
        """
        This function is responsible for clustering hypothesis at each step to produce diverse solutions
        :param temp: Candidate solutions
        :param num_groups: Number of clusters
        :param beam_size: Beam size to be used for decoding
        :return:
        """
        all_sentences = []
        for s, prob in temp:
            intermediate_question = [self.datasets.idx_to_word[i] for i in s]
            all_sentences.append(' '.join(intermediate_question[1:]))
        self.logger.debug('Sentences being clustered %s' % all_sentences)
        dict_map_sentence_to_idx = dict()
        for i, s in enumerate(all_sentences):
            if s not in dict_map_sentence_to_idx:
                dict_map_sentence_to_idx[s] = i

        uniq_sentences = list(dict_map_sentence_to_idx.keys())

        try:
            retained = dict(cluster_texts(uniq_sentences, num_groups))
        except:
            # logging.error('Error with clustering in diversity beam search')
            return temp

        # Add best 2 out of the box in retained_sentences
        picked = [len(all_sentences)-2, len(all_sentences)-1]
        retained_sentences = [temp[idx] for idx in picked]
        while(1):
            if len(retained_sentences) >= beam_size:
                break
            for key, values in retained.items():
                # Pick one from each cluster
                idx = random.choice(values)
                if idx not in picked:
                    picked.append(idx)
                    if len(retained_sentences) >= beam_size:
                        break
                    retained_sentences.append(temp[idx])
        self.logger.debug('Filtered sentences')
        for s in retained_sentences:
            iq = ' '.join([self.datasets.idx_to_word[i] for i in s[0]])
            self.logger.debug(iq)

        return retained_sentences

    def diverse_beam_search(self, image_input, model, beam_size, num_groups, keyword=None):
        """
        This function performs diverse beam search borrowing idea from https://arxiv.org/abs/1610.02424
        :param image_input: Image encoded feature
        :param model: Model definition file
        :param beam_size: Beam size to be used for decoding
        :param num_groups: Number of clusters
        :param keyword: Keyword input feature to be used with some model architectures
        :return:
        """
        # Ignore PAD, START, END tokens
        start = [self.word_to_idx["<START>"]]
        if 'glov' in self.datasets.embedding_file:
            max_seq_len = model.inputs[1].shape[1].value
        else:
            max_seq_len = self.datasets.max_question_len
            image_input = np.repeat(image_input, axis=0, repeats=2)
        self.logger.info('max len %s', max_seq_len)

        start_word = [[start, 0.0]]
        EOS_utterances = []
        map_EOS_utterances = dict()

        while len(start_word[0][0]) < max_seq_len:
            temp = []
            self.logger.debug('\n\n\nCurrent it: %s Max seq len: %s' %(len(start_word[0][0]), max_seq_len))

            for s in start_word:
                # self.logger.info('Start word Tuple %s' % s)
                if self.datasets.use_keyword:
                    sequence = pad_sequences([s[0]], maxlen=max_seq_len, padding='post')
                    preds = model.predict([image_input, sequence, keyword])
                elif 'glove' in self.datasets.embedding_file:
                    sequence = pad_sequences([s[0]], maxlen=max_seq_len, padding='post')
                    preds = model.predict([image_input, sequence])
                elif 'elmo' in self.datasets.embedding_file:
                    sequence = ' '.join([self.datasets.idx_to_word[idx] for idx in s[0]])
                    sequence = self.cleanText(sequence)
                    sequence = np.array([sequence, sequence])
                    preds = model.predict([image_input, sequence])
                elif 'bert' in self.datasets.embedding_file:
                    sequence = ' '.join([self.datasets.idx_to_word[idx] for idx in s[0][1:]])
                    sequence = self.cleanText(sequence)
                    sequence = [[sequence], [sequence]]
                    input_ids, input_masks, segment_ids, _ = preprocess_bert_input(sequence, [None] * len(sequence),
                                                                                   self.datasets.max_question_len,
                                                                                   self.tokenizer, self.vocab_size)
                    preds = model.predict([image_input, input_ids, input_masks, segment_ids])
                else:
                    exit(-1)

                word_preds = np.argsort(preds[0])[- 2*beam_size:]

                # Getting the top <beam_size>(n) predictions and creating a
                # new list so as to put them via the model again
                for w in word_preds:
                    next_quest, prob = s[0][:], s[1]
                    next_quest.append(w)
                    # If END token is found then keep the utterance
                    if w == 2:
                        intermediate_question = ' '.join([self.datasets.idx_to_word[i] for i in next_quest])
                        if intermediate_question not in map_EOS_utterances:
                            map_EOS_utterances[intermediate_question] = prob
                            EOS_utterances.append([next_quest, prob])
                    prob += preds[0][w]
                    temp.append([next_quest, prob])

            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            if len(start_word[0][0]) > 2:
                start_word = start_word[-2 * beam_size:]
                start_word = self.dissimilarity_grouping(start_word, num_groups, beam_size)
            else:
                start_word = start_word[-beam_size:]

        candidates = dict()
        max_prob = 0
        final_candidate = ''
        start_word += EOS_utterances
        unique_questions_not_seen_training_data = set()
        thresh = 2.0
        for st_wd in start_word:
            # import pdb;pdb.set_trace()
            prob = st_wd[1]
            st_wd = st_wd[0]

            intermediate_question = [self.datasets.idx_to_word[i] for i in st_wd]

            final_question = []

            for i in intermediate_question:
                if i != '<END>':
                    final_question.append(i)
                else:
                    break

            final_question = ' '.join(final_question[1:])

            if prob > max_prob:
                max_prob = prob
                final_candidate = final_question
            if prob > thresh:
                if final_question in candidates:
                    if prob > candidates[final_question]:
                        candidates[final_question] = prob
                else:
                    candidates[final_question] = prob

            if final_question not in self.datasets.unique_train_questions:
                    unique_questions_not_seen_training_data.add(final_question)

        self.logger.info(
            'Unique generated questions not seen in training data: %s' % unique_questions_not_seen_training_data)
        self.datasets.unique_generated_questions.update(candidates.keys())
        self.logger.info('Final DBS candidates: %s' % candidates)

        self.datasets.generated_questions += list(candidates.keys())

        # Inventiness:
        # Number of unique questions not seen in training data / Total number of generated questions for that image
        self.logger.info('Inventiveness: %s' % str(len(unique_questions_not_seen_training_data) / len(candidates)))
        return [final_candidate], candidates

    def generate_batch(self, batch_size, graph, id_question_dict, id_imagefeat_dict, id_keyword_dict=None, shuffle=True):
        """
        Generator function resposible for generating batches during training

        :param batch_size: Batch size to be used
        :param graph: Tensorflow graph
        :param id_question_dict: Dict with image id as key and question list as value
        :param id_imagefeat_dict: Dict with image id as key and image features as value
        :param id_keyword_dict: Dict with image id as key and keyword as value
        :param shuffle: If shuffle is true, randomly shuffle the image ids
        :return:
        """
        with graph.as_default():

            while True:
                image_ids = list(id_question_dict.keys())
                image_ids = [id for id in image_ids if id in id_imagefeat_dict]
                num_samples = len(image_ids)

                if shuffle:
                    random.shuffle(image_ids)

                # Get index to start each batch:
                # [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
                for offset in range(0, num_samples, batch_size):
                    X1 = list()
                    X2 = list()
                    X3 = list()
                    Y = list()
                    bert_label = list()
                    # Get the samples you'll use in this batch
                    batch_samples = image_ids[offset:offset + batch_size]
                    for image_id in batch_samples:
                        try:
                            image_feature = id_imagefeat_dict[image_id]
                            if image_feature is None:
                                self.logger.debug('Image has no feature %s' % image_id)
                                continue
                        except:
                            self.logger.error('Image %s not found' % image_id)
                            continue

                        try:
                            keyword = id_keyword_dict[image_id]
                        except:
                            keyword = " "

                        x1 = image_feature
                        image_questions = id_question_dict[image_id]

                        for image_question in image_questions:
                            token_seq = [self.datasets.word_to_idx[word] for word in image_question.split(' ') if
                                         word in self.datasets.word_to_idx]

                            for i in range(1, len(token_seq)):
                                in_seq, out_seq = token_seq[:i], token_seq[i]

                                bert_label.append(out_seq)
                                y = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                                Y.append(y)

                                X1.append(x1)
                                if self.datasets.use_keyword:
                                    x2_glove = pad_sequences([in_seq], maxlen=self.datasets.max_question_len, padding='post')[0]
                                    X2.append(x2_glove)

                                    keyword_token_seq = [self.datasets.word_to_idx[word] for word in keyword.split(' ') if
                                                        word in self.datasets.word_to_idx]
                                    keyword_tokens = pad_sequences([keyword_token_seq], maxlen=self.datasets.max_keyword_len, padding='post')[0]
                                    X3.append(keyword_tokens)
                                elif 'glove' in self.datasets.embedding_file:
                                    x2_glove = pad_sequences([in_seq], maxlen=self.datasets.max_question_len, padding='post')[0]
                                    X2.append(x2_glove)
                                # Input format for ELMO and Bert
                                elif 'elmo' in self.datasets.embedding_file:
                                    x2_elmo = ' '.join([self.datasets.idx_to_word[idx] for idx in in_seq[1:]])
                                    x2_elmo = self.cleanText(x2_elmo)
                                    X2.append([x2_elmo])
                                # Input format for ELMO and Bert
                                elif 'bert' in self.datasets.embedding_file:
                                    x2_bert = ' '.join([self.datasets.idx_to_word[idx] for idx in in_seq[1:]])
                                    x2_bert = self.cleanText(x2_bert)
                                    X2.append([x2_bert])

                    if self.datasets.use_keyword:
                        yield [[array(X1), array(X2), array(X3)], array(Y)]
                    # Bert input is slightly different from the rest
                    elif 'bert' in self.datasets.embedding_file:
                        input_ids, input_masks, segment_ids, labels = preprocess_bert_input(X2,
                                                                                            bert_label,
                                                                                            self.datasets.max_question_len,
                                                                                            self.tokenizer,
                                                                                            self.vocab_size)
                        yield [[array(X1), array(input_ids), array(input_masks), array(segment_ids)], array(labels)]
                    else:
                        yield [[array(X1), array(X2)], array(Y)]

    def train_model(self, model, graph, model_dir, epoch, batch_size, decoder_algorithm, beam_size, sess):
        """
        Training function
        :param model: model definition
        :param graph: Tensorflow graph
        :param model_dir: Directory where model is saved
        :param epoch: Number of epochs for training
        :param batch_size: Batch size
        :param decoder_algorithm: Decoder algorithm to be used with model: greedy, simple beam search or diverse beam search
        :param beam_size: Beam size to be used for decoding
        :param sess: Tensorflow session
        :return:
        """

        report_metric_on_epoch_end = LambdaCallback(
            on_epoch_end=lambda epoch, logs: bolt.send_metrics({
                                                                'train_accuracy': logs['acc'],
                                                                'train_loss': logs['loss']}, iteration=epoch)
        )

        #
        # Some functions are commented out to increase training speed
        #

        model_bucket_name = 'experimental_models'
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2000)
        filepath = os.path.join(model_dir, "model_{epoch:02d}.h5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        steps = int(len(self.datasets.train_image_id_questions_dict) // batch_size)
        val_steps = int(len(self.datasets.dev_image_id_questions_dict) // batch_size)
        self.logger.info('Train steps %s' % steps)
        self.logger.info('Validation steps %s' % val_steps)
        generator = self.generate_batch(batch_size,
                                        graph,
                                        self.datasets.train_image_id_questions_dict,
                                        self.datasets.train_image_id_imagefeat_dict,
                                        self.datasets.train_image_id_keyword_dict)
        # val_generator = self.generate_batch(batch_size,
        #                                     graph,
        #                                     self.datasets.dev_image_id_questions_dict,
        #                                     self.datasets.dev_image_id_imagefeat_dict,
        #                                     self.datasets.dev_image_id_keyword_dict,
        #                                     shuffle=False)
        # test_generator = self.generate_batch(1,
        #                                      graph,
        #                                      self.datasets.test_image_id_questions_dict,
        #                                      self.datasets.test_image_id_imagefeat_dict,
        #                                      shuffle=False,
        #
        #                                      test=False)
        for no_epoch in range(epoch):
            self.logger.info('\n'*5)
            self.logger.info(no_epoch)

            history = model.fit_generator(generator,
                                          steps_per_epoch=steps,
                                          epochs=1,
                                          verbose=2,
                                          callbacks=[report_metric_on_epoch_end])
            # history = model.fit_generator(generator, validation_generator=val_generator,
            #                               steps_per_epoch=steps,
            #                               val_steps = val_steps
            #                               epochs=1,
            #                               verbose=2,
            #                               callbacks=[report_metric_on_epoch_end, checkpoint. es])

            model_file_name = os.path.join(model_dir, 'model_' + str(no_epoch) + '.h5')
            self.logger.info('Model saved at %s' % model_file_name)
            model.save(model_file_name)

            # Test model on ids sampled from test set
            test_epoch_condition = 0
            test_img_count_condition = 2
            if no_epoch < test_epoch_condition:
                test_img_count = 0

                while test_img_count < test_img_count_condition:

                    sess.run(tensorflow.local_variables_initializer())
                    sess.run(tensorflow.global_variables_initializer())
                    sess.run(tensorflow.tables_initializer())
                    K.set_session(sess)

                    id = sys_random.choice(list(self.datasets.test_image_id_url_dict.keys()))
                    test_image_url = self.datasets.test_image_id_url_dict[id]
                    self.logger.info('\n\n\n\n\nImage url: %s' % test_image_url)
                    try:
                        output_questions = self.test_model(test_image_url, model, decoder_algorithm, beam_size)
                        gt_questions = self.datasets.test_image_id_questions_dict[id]
                        self.logger.info('GT  ---->%s' % gt_questions)
                    except:
                        self.logger.error('Error with inference code')
                        pass

                    test_img_count += 1

        # upload model to McQueen
        # self.datasets.store_data_to_mcqueen(model_bucket_name, filepath)
        return epoch

    def test_model(self, test_image_url, model, decoder_algorithm, beam_size, keyword=None):
        """
        Function to perform inference on model
        :param test_image_url: Test image url
        :param model: Model definition file
        :param decoder_algorithm: Decoder algorithm: Greedy, simple beam search or diverse beam search
        :param beam_size: Beam size to be used for decoding
        :param keyword: Keyword to be used for some model architecture
        :return:
        """

        response = requests.get(test_image_url)
        test_image_content = Image.open(BytesIO(response.content))
        test_image_content = test_image_content.resize((224, 224))
        self.logger.debug('Extracting image feature')
        try:
            test_image_feature = self.datasets.extract_features_from_image(test_image_content)
        except:
            self.logger.error('Error in extracting image feature')
            test_image_content = test_image_content.convert('RGB')
            test_image_feature = self.datasets.extract_features_from_image(test_image_content)

        self.logger.debug('Succesfully extracted image feature')
        test_image_feature = test_image_feature.reshape((1, self.input_shape))

        if keyword is not None:
            keyword_token_seq = [self.datasets.word_to_idx[word] for word in keyword.split(' ') if
                                 word in self.datasets.word_to_idx]
            keyword = pad_sequences([keyword_token_seq], maxlen=self.datasets.max_keyword_len, padding='post')[0]
            keyword = array([keyword])

        output = None

        if decoder_algorithm == 'sbs':
            self.logger.info('Simple beam search ---->')
            output, candidates = self.beam_search(test_image_feature, model, beam_size)
        elif decoder_algorithm == 'dbs':
            try:
                self.logger.info('Diverse beam search ---->')
                output, candidates = self.diverse_beam_search(test_image_feature, model, beam_size, num_groups=2)
            except:
                self.logger.error('Failed to perform diverse beam search')
        elif decoder_algorithm == 'greedy':
            self.logger.info('Greedy search ---->')
            output = self.greedy_search(test_image_feature, model, keyword)
            #
            # Perform all 3 below
            #
            # self.logger.info('Simple beam search ---->')
            # output, candidates = self.beam_search(test_image_feature, model, beam_size, keyword)
            # self.logger.info('Diverse beam search ---->')
            # output, candidates = self.diverse_beam_search(test_image_feature, model, beam_size, num_groups=2, keyword=keyword)

        self.logger.info('Final Output:  ---->%s', output)
        self.logger.info('\n')
        return output

    def test_model_demo(self, test_image_url, model, beam_size):

        response = requests.get(test_image_url)
        test_image_content = Image.open(BytesIO(response.content))
        test_image_content = test_image_content.resize((224, 224))
        self.logger.debug('Extracting image feature')
        try:
            test_image_feature = self.datasets.extract_features_from_image(test_image_content)
        except:
            self.logger.error('Error in extracting image feature')
            test_image_content = test_image_content.convert('RGB')
            test_image_feature = self.datasets.extract_features_from_image(test_image_content)

        self.logger.debug('Succesfully extracted image feature')
        test_image_feature = test_image_feature.reshape((1, self.input_shape))

        self.logger.info('Greedy search ---->')
        candidates1 = self.greedy_search(test_image_feature, model)

        self.logger.info('Simple beam search ---->')
        output2, candidates2 = self.beam_search(test_image_feature, model, beam_size)


        try:
            self.logger.info('Diverse beam search ---->')
            output3, candidates3 = self.diverse_beam_search(test_image_feature, model, beam_size, num_groups=2)
        except:
            self.logger.error('Failed to perform diverse beam search')
            candidates3 = None
            output3 = None

        self.logger.info('Final Output:  ---->%s', candidates1)
        self.logger.info('\n')
        return candidates1, candidates2, candidates3


if __name__ == "__main__":
    pass



