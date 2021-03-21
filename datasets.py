"""
This file contains the Dataset Class
"""
import sys
import os
import numpy as np
from PIL import Image
import pickle
import requests
from io import BytesIO

import logging

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet201

from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_net_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_net_preprocess
# from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_net_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as dense_net_preprocess


import string

import warnings
warnings.filterwarnings(action='once')

import boto3

from botocore.client import Config

mcqueen_url = "http://store-test.blobstore.apple.com"
mcqueen_region_name = "store-test"
mcqueen_access_key = "MKIAXRQON2I79WCA4Q19"
mcqueen_secret_key = "C9ED2E4ABA42DD317E498C391A9AD989099C6E9018DD68508A232738719A915B"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Datasets:
    """
    Datasets class.
    Contains all attributes important to manage dataset
    """

    def __init__(self, train_file_name, validation_file_name, test_file_name, embedding_file_name, max_train,
                 image_encoding_algo, use_keyword=False, build_vocab_dev=True):

        self.s3 = boto3.resource(service_name='s3',
                                 endpoint_url=mcqueen_url,
                                 region_name=mcqueen_region_name,
                                 aws_access_key_id=mcqueen_access_key,
                                 aws_secret_access_key=mcqueen_secret_key,
                                 config=Config(s3={'addressing_style': 'path', 'signature_version': 's3'}))

        bucket_name = 'vqg-data'
        self.image_encoding_algo = image_encoding_algo
        self.build_vocab_dev = build_vocab_dev

        if not os.path.exists(train_file_name):
            logger.info("Downloading training data to %s" % train_file_name)
            self.store_data_from_mcqueen(bucket_name, train_file_name.split('/')[-1], train_file_name)

        if not os.path.exists(validation_file_name):
            logger.info("Downloading validation data to %s" % validation_file_name)
            self.store_data_from_mcqueen(bucket_name, validation_file_name.split('/')[-1], validation_file_name)

        if not os.path.exists(test_file_name):
            logger.info("Downloading testing data to %s" % test_file_name)
            self.store_data_from_mcqueen(bucket_name, test_file_name.split('/')[-1], test_file_name)

        if 'glove' in embedding_file_name and not os.path.exists(embedding_file_name):
            logger.info("Downloading embedding file from %s" % embedding_file_name)
            self.store_data_from_mcqueen('glov_data', embedding_file_name.split('/')[-1], embedding_file_name)

        self.train_file = train_file_name
        self.validation_file = validation_file_name
        self.test_file = test_file_name
        self.embedding_file = embedding_file_name

        self.max_samples = max_train
        if self.image_encoding_algo == 'VGG19':
            self.image_encoding_model = VGG19(weights='imagenet', include_top=False)
        elif self.image_encoding_algo == 'MobileNet':
            self.image_encoding_model = MobileNetV2(weights='imagenet', include_top=False)
        elif image_encoding_algo == 'ResNet':
            self.image_encoding_model = ResNet50(weights='imagenet', include_top=False)
        # elif self.image_encoding_algo=='Inception':
        #     self.image_encoding_model = InceptionV3(weights='imagenet', include_top=False)
        elif self.image_encoding_algo == 'DenseNet':
            self.image_encoding_model = DenseNet201(weights='imagenet', include_top=False)

        logger.info('%s model loaded' % image_encoding_algo)

        self.vocabulary = dict()
        self.idx_to_word = dict()
        self.word_to_idx = dict()
        self.train_image_id_questions_dict = dict()
        self.train_image_id_imagefeat_dict = dict()
        self.train_image_id_keyword_dict = dict()

        self.dev_image_id_questions_dict = dict()
        self.dev_image_id_imagefeat_dict = dict()
        self.dev_image_id_keyword_dict = dict()

        self.test_image_id_questions_dict = dict()
        self.test_image_id_imagefeat_dict = dict()
        self.test_image_id_url_dict = dict()
        self.test_image_id_keyword_dict = dict()

        self.unique_train_questions = set()
        self.unique_generated_questions = set()
        self.generated_questions = []

        self.max_question_len = -1
        self.max_keyword_len = 10
        self.no_of_samples = 0
        if use_keyword == 'YES':
            self.use_keyword = True
        else:
            self.use_keyword = False

        # self.load_data(self.train_file)

        self.build_vocabulary()

    def store_data_from_mcqueen(self, bucket_name, key_name, file_name):
        """
        Downloads data from mcqueen
        :param bucket_name: Name of bucket on mcqueen
        :param key_name: Key name on mcqueen
        :param file_name: Name of file to be stored
        :return:
        """

        s3object = self.s3.Object(bucket_name=bucket_name, key=key_name).get()
        data = s3object['Body'].read().decode('utf-8')

        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(data)

    def store_binary_data_from_mcqueen(self, bucket_name, key_name, file_name):
        """
        Downloads binary from mcqueen. We use this when the function above fails in case of binary encoded files.
        :param bucket_name: Name of bucket on mcqueen
        :param key_name: Key name on mcqueen
        :param file_name: Name of file to be stored
        :return:
        """
        try:
            self.s3.Bucket(bucket_name).download_file(key_name, file_name)

        except:
            logger.error('Error trying another style')
            with open(file_name, 'wb') as f:
                self.s3.download_fileobj(Filename=f, Key=key_name, Bucket=bucket_name)

    def store_data_to_mcqueen(self, model_bucket_name, key_name, file_name):
        """
        Uploads data to mcqueen
        :param bucket_name: Name of bucket on mcqueen
        :param key_name: Key name on mcqueen
        :param file_name: Name of file to be uploaded
        :return:
        """

        if self.s3.Bucket(model_bucket_name) not in self.s3.buckets.all():
            bucket = self.s3.create_bucket(Bucket=model_bucket_name)

        self.s3.Bucket(model_bucket_name).upload_file(file_name, key_name)

    def get_processed_fields(self, line, file_name, build_image_feature = False):
        """
        This function splits a record into id and questions. It also extracts image features from a given url
        :param line: String containing image id, image url and questions
        :param file_name: File name containing dataset name
        :param:build_image_feature: Flag if image features need to be built
        :return:
        """
        line = line.strip()
        records = line.split(",")
        image_id = records[0]
        image_url = records[1]
        if 'bing' in file_name:
            image_questions = records[3].split("---")
        else:
            image_questions = records[2].split("---")

        if self.use_keyword:
            keyword = records[3]

        else:
            keyword = None

        cleaned_image_questions = list()
        for question in image_questions:
            question = self.preprocess_text(question)
            if question == 'none':
                continue
            cleaned_image_questions.append(question)

        image_feature = None
        if build_image_feature:
            # If image feature dict exists ignore
            if 'train' in file_name and image_id in self.train_image_id_imagefeat_dict:
                image_feature = self.train_image_id_imagefeat_dict[image_id]
            elif 'test' in file_name and image_id in self.test_image_id_imagefeat_dict:
                image_feature = self.test_image_id_imagefeat_dict[image_id]
            elif 'dev' in file_name and image_id in self.dev_image_id_imagefeat_dict:
                image_feature = self.train_image_id_imagefeat_dict[image_id]
            else:
                # Image feature doesnt exist in dictionary so build it
                try:
                    logger.debug('Building feature for image: %s' % image_url)
                    image_feature = self.get_processed_image_features(image_url)
                except:
                    logger.error('Image url has error %s' % image_url)

        return image_id, image_feature, cleaned_image_questions, image_url, keyword

    def load_test_data(self):
        """
        Function loads test data from test file name into datasets class
        :return:
        """
        for id, train_questions in self.train_image_id_questions_dict.items():
            for question in train_questions:
                question = question.split()[1:-1]
                question = ' '.join(question)
                if question in ['None', 'none']:
                    continue
                self.unique_train_questions.add(question)
        logger.info('Done creating unique training question set. Size: %s' % str(len(self.unique_train_questions)))

        with open(self.test_file, 'r', encoding='utf-8') as file:
            count = 0
            header = file.readline()
            for line in file:
                count += 1
                if count % 100 == 0:
                    logger.info('Processing image id # : %s' % count)

                image_id, \
                    image_feature, \
                    cleaned_image_questions, \
                    image_url, keyword = self.get_processed_fields(line,
                                                                   self.test_file,
                                                                   build_image_feature=True)
                self.test_image_id_questions_dict[image_id] = cleaned_image_questions
                self.test_image_id_url_dict[image_id] = image_url
                self.test_image_id_imagefeat_dict[image_id] = image_feature
                self.test_image_id_keyword_dict[image_id] = keyword

                if count >= self.max_samples:
                    break
        logger.info("Test data loaded")

    def load_data(self, file_name):
        """
        Load training data from training file name into datasets class
        :param file_name:
        :return:
        """
        count = 0

        with open(file_name, 'r', encoding='utf-8') as file:
            header = file.readline()
            for line in file:
                count += 1
                try:
                    if count % 100 == 0:
                        logger.info('Processing image id # : %s' % count)

                    image_id, \
                        image_feature, \
                        cleaned_image_questions, \
                        image_url, keyword = self.get_processed_fields(line, file_name, build_image_feature=True)
                    self.train_image_id_imagefeat_dict[image_id] = image_feature
                    self.train_image_id_questions_dict[image_id] = cleaned_image_questions
                    self.train_image_id_keyword_dict[image_id] = keyword

                    if count >= self.max_samples:
                        break
                except:
                    logger.error('Image url has an issue: %s' % image_url)

            logger.info("Training data loaded")

    def load_dev_data(self, file_name):
        """
        Load dev set data from dev file name into datasets class
        :param file_name:
        :return:
        """
        count = 0

        with open(file_name, 'r', encoding='utf-8') as file:
            header = file.readline()
            for line in file:
                count += 1
                try:
                    if count % 100 == 0:
                        logger.info('Processing image id # : %s' % count)
                    image_id, \
                        image_feature, \
                        cleaned_image_questions, \
                        image_url, keyword = self.get_processed_fields(line, file_name, build_image_feature=True)
                    self.dev_image_id_imagefeat_dict[image_id] = image_feature
                    self.dev_image_id_questions_dict[image_id] = cleaned_image_questions
                    self.dev_image_id_keyword_dict[image_id] = keyword

                    if count >= self.max_samples:
                        break
                except:
                    pass
                    # logger.error('Image url has an issue')

            logger.info("Validation data loaded")

    def get_processed_image_features(self, image_url):
        """
        This function opens an image and extracts encoded features from it.
        :param image_url: Image URL
        :return: Extracted features after running through an image encoding algorithm
        """
        response = requests.get(image_url, verify=False)
        image_content = Image.open(BytesIO(response.content))
        image_content = image_content.resize((224, 224))

        try:
            image_feature = self.extract_features_from_image(image_content)
        except:
            image_content = image_content.convert('RGB')
            image_feature = self.extract_features_from_image(image_content)
        return image_feature

    def extract_features_from_image(self, image_content):
        """
        This function performs image encoding based on selected algorithm
        """
        # self.image_encoding_model.summary()
        # print(type(image_content))

        img_data = image.img_to_array(image_content)
        img_data = np.expand_dims(img_data, axis=0)

        if self.image_encoding_algo == 'MobileNet':
            img_data = mobile_net_preprocess(img_data)
        elif self.image_encoding_algo == 'VGG19':
            img_data = vgg_preprocess(img_data)
        elif self.image_encoding_algo == 'DenseNet':
            img_data = dense_net_preprocess(img_data)
        # elif self.image_encoding_algo == 'Inception':
        #     img_data = inception_net_preprocess(img_data)
        elif self.image_encoding_algo == 'ResNet':
            img_data = res_net_preprocess(img_data)

        image_encoding_feature = self.image_encoding_model.predict(img_data)
        image_encoding_feature_np = np.array(image_encoding_feature)
        image_encoding_feature_vector = image_encoding_feature_np.flatten()

        return image_encoding_feature_vector

    def preprocess_text(self, text):
        """
        String pre processing function that standardizes an input string
        :param text: Input string
        :return:
        """

        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))

        tokens = text.split(" ")

        result = list()
        result.append('<START>')

        for token in tokens:
            token = token.strip()

            result.append(token)

        result.append('<END>')

        self.max_question_len = max(self.max_question_len, len(result))
        self.no_of_samples += len(result)

        return ' '.join(result)

    def build_vocabulary(self):
        """
        Function that builds vocabulary
        :return:
        """

        self.vocabulary['<PAD>'] = 0
        self.word_to_idx['<PAD>'] = 0
        self.idx_to_word[0] = '<PAD>'

        self.vocabulary['<START>'] = 1
        self.word_to_idx['<START>'] = 1
        self.idx_to_word[1] = '<START>'

        self.vocabulary['<END>'] = 2
        self.word_to_idx['<END>'] = 2
        self.idx_to_word[2] = '<END>'

        keyword_list = []

        i = 0
        with open(self.train_file, 'r') as file:
            # skip the file line of the file as header
            file.readline()
            for line in file:
                image_id, \
                    _, \
                    cleaned_image_questions, \
                    _, keyword = self.get_processed_fields(line, self.train_file, build_image_feature=False)
                if self.use_keyword:
                    keyword_list.append(keyword.split())

                self.train_image_id_questions_dict[image_id] = cleaned_image_questions
                i += 1

        if self.build_vocab_dev:
            i = 0
            with open(self.validation_file, 'r') as file:
                # skip the file line of the file as header
                file.readline()
                for line in file:
                    image_id, \
                        _, \
                        cleaned_image_questions, \
                        _, keyword = self.get_processed_fields(line, self.validation_file, build_image_feature=False)
                    if self.use_keyword:
                        keyword_list.append(keyword.split())
                    self.dev_image_id_questions_dict[image_id] = cleaned_image_questions
                    i += 1

        # Start the word index from 3 after PAD, START, END tokens
        count = 3
        for image_id in self.train_image_id_questions_dict:
            for question in self.train_image_id_questions_dict[image_id]:
                tokens = question.split(" ")
                for token in tokens:
                    if token not in self.vocabulary:
                        self.vocabulary[token] = count
                        self.word_to_idx[token] = count
                        self.idx_to_word[count] = token
                        count += 1

        if self.build_vocab_dev:
            for image_id in self.dev_image_id_questions_dict:
                for question in self.dev_image_id_questions_dict[image_id]:
                    tokens = question.split(" ")
                    for token in tokens:
                        if token not in self.vocabulary:
                            self.vocabulary[token] = count
                            self.word_to_idx[token] = count
                            self.idx_to_word[count] = token
                            count += 1

        # Add keywords to the vocabulary
        if self.use_keyword:
            for keywords in keyword_list:
                for token in keywords:
                    if token not in self.vocabulary:
                        self.vocabulary[token] = count
                        self.word_to_idx[token] = count
                        self.idx_to_word[count] = token
                        count += 1


if __name__ == "__main__":
    train_file_name = "coco_train_all.csv"
    valid_file_name = "coco_dev_all.csv"
    test_file_name = "coco_test_all.csv"
    embedding_file_name = 'glove.6B.200d.txt'

    datasets = Datasets(train_file_name, valid_file_name, test_file_name, embedding_file_name, 10)

    print("Vocabulary: " + str(len(datasets.vocabulary)))

    print("No. of samples: " + str(datasets.no_of_samples))

