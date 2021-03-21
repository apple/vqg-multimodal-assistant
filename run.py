"""
Entry point code for modeling_pkg
"""


from datasets import Datasets
from question_generation_model import QuestionGenerationModel

from commons_pkg.commons_utils import load_config
from evaluation_pkg.evaluate import run_evaluation
from pprint import pprint
from tensorflow.keras import backend as K
import argparse
import csv
import logging
import os
import pickle
import tensorflow
import subprocess
import sys

os.environ['TFHUB_CACHE_DIR'] = os.path.join(os.getcwd(), 'misc')
from bert_layer import create_tokenizer_from_hub_module, initialize_vars
sys.path.append("./")
sys.path.append("../")

# # Initialize session
sess = tensorflow.Session()


def parse_arguments():
    """
    Defines parser arguments
    :return: parser arguments
    """
    parser = argparse.ArgumentParser(
        description='Run modeling tasks on visual question geenration task')
    parser.add_argument('-model_dir', type=str, default='model',
                        help='Directory to store saved model files')
    parser.add_argument('-c', type=str, default='bolt/config.yaml',
                        help='Config file path')

    args = parser.parse_args()
    return args


def get_logger(log_level):
    """
    Provides logging functionality
    :param log_level: describes log level of logger functionality
    :return: logger
    """
    file_handler = logging.FileHandler(filename='run.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler, file_handler]
    if log_level == 'd':
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers)

    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger


def load_model(question_generator):
    """
    Load model
    :param question_generator: Class containing all question generator modules. Defined in question_generator_model.py
    :return: model defition file
    """
    # Build the model
    if question_generator.datasets.use_keyword:
        model = question_generator.build_keyword_model()
    elif 'glove' in question_generator.datasets.embedding_file:
        model = question_generator.build_glove_model()
    elif 'elmo' in question_generator.datasets.embedding_file:
        model = question_generator.build_elmo_model()
    elif 'bert' in question_generator.datasets.embedding_file:
        bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        # Instantiate tokenizer
        question_generator.tokenizer = create_tokenizer_from_hub_module(bert_path, sess)
        model = question_generator.build_bert_model()
    else:
        logging.error('Embedding model not found')
        exit(-1)

    return model


def save_obj(obj, name):
    """
    This module saves obj into a pkl file

    :param obj: pickle object to be saved
    :param name: Name of file
    :return: None
    """
    print('Saving', name)
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    This module loads the objects defined under name

    :param name: Name of pickle object to be loaded
    :return: Pickle object
    """
    print('Loading', name)
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":

    args = parse_arguments()
    config_path = args.c
    config = load_config(config_path)['parameters']
    pprint(config)
    train_file = config['datasets']['train_file']
    validation_file = config['datasets']['validation_file']
    embedding_file = config['datasets']['embedding_file']
    test_file = config['datasets']['test_file']
    max_train_size = config['datasets']['max_train_size']
    is_training = config['is_training']
    use_keyword = config['datasets']['keyword']

    model_file_name = config['model_parameters']['inference']['model_file']
    batch_size = config['model_parameters']['training']['batch_size']
    epoch = config['model_parameters']['training']['epoch']
    decoder_algorithm = config['model_parameters']['decoder']['algorithm']
    beam_size = config['model_parameters']['decoder']['beam_size']
    user_input = config['model_parameters']['inference']['user_input']
    dataset = config['datasets']['name']
    image_encoding_algo = config['model_parameters']['image_encoder']['algorithm']
    image_embedding_dim = config['model_parameters']['image_encoder']['image_embedding_dim']
    hidden_units = config['model_parameters']['text_encoder']['hidden_dim']
    dropout = config['model_parameters']['text_encoder']['dropout']
    logging_level = config['logging_level']

    logger = get_logger(logging_level)

    # Remove existing inference results
    if os.path.exists('result_' + test_file):
        os.remove('result_' + test_file)

    if os.path.exists('gt_' + test_file):
        os.remove('gt_' + test_file)

    path = os.path.join(args.model_dir, model_file_name)

    datasets = Datasets(train_file, validation_file, test_file, embedding_file, max_train_size,
                        image_encoding_algo, use_keyword=use_keyword)
    logger.info("Max question len: %s" % datasets.max_question_len)
    logger.info("Max training samples: %s" % datasets.max_samples)
    logger.info("Vocabulary: %s" % str(len(datasets.vocabulary)))
    question_generator = QuestionGenerationModel(datasets, logger, hidden_units, dropout)
    question_generator.input_shape = image_embedding_dim

    # Calculate image features and store them if save is True
    obj_dir = os.path.join('data', dataset, 'obj')
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)

    train_imagefeat_dict_name = os.path.join(obj_dir, 'train_imagefeat_dict.pkl')
    test_imagefeat_dict_name = os.path.join(obj_dir, 'test_imagefeat_dict.pkl')
    dev_imagefeat_dict_name = os.path.join(obj_dir, 'dev_imagefeat_dict.pkl')

    if os.path.exists(test_imagefeat_dict_name):
        datasets.test_image_id_imagefeat_dict = load_obj(test_imagefeat_dict_name)
    datasets.load_test_data()

    # Set this variable to True if you want to save the image features
    save = False

    if save:
        save_obj(datasets.test_image_id_imagefeat_dict, test_imagefeat_dict_name)
    # K.set_session(sess)
    if is_training == 'YES':

        if os.path.exists(train_imagefeat_dict_name):
            datasets.train_image_id_imagefeat_dict = load_obj(train_imagefeat_dict_name)

        # Validation part of pipeline is commented out to speedup training process
        # if os.path.exists(dev_imagefeat_dict_name):
        #     datasets.dev_image_id_imagefeat_dict = load_obj(dev_imagefeat_dict_name)

        datasets.load_data(train_file)
        # datasets.load_dev_data(validation_file)

        model = load_model(question_generator)

        if save:
            save_obj(datasets.dev_image_id_imagefeat_dict, dev_imagefeat_dict_name)
            save_obj(datasets.train_image_id_imagefeat_dict, train_imagefeat_dict_name)

        count = 0
        vocab_size = len(datasets.vocabulary)

        graph = tensorflow.get_default_graph()

        logger.info("Training size: %s" % str(len(datasets.train_image_id_questions_dict)))
        logger.info("Validation size: %s" % str(len(datasets.dev_image_id_questions_dict)))

        sess.run(tensorflow.local_variables_initializer())
        sess.run(tensorflow.global_variables_initializer())
        sess.run(tensorflow.tables_initializer())
        K.set_session(sess)
        # Train the model
        last_epoch = question_generator.train_model(model,
                                       graph,
                                       args.model_dir,
                                       epoch=epoch,
                                       batch_size=batch_size,
                                       decoder_algorithm=decoder_algorithm,
                                       beam_size=beam_size,
                                       sess=sess)
        model_file_name = 'model_' + str(last_epoch - 1) + '.h5'

    # Perform inference
    path = os.path.join(args.model_dir, model_file_name)
    if not os.path.exists(path):
        logger.info('Downloading {}  file'.format(model_file_name))
        datasets.store_binary_data_from_mcqueen('experimental_models', model_file_name, path)
    else:
        logger.info('Model {} exists'.format(model_file_name))

    if 'glov' in datasets.embedding_file:
        new_model = tensorflow.keras.models.load_model(path)
        new_model.load_weights(path)
        initialize_vars(sess)

    elif 'elmo' in datasets.embedding_file:
        K.set_session(sess)
        new_model = tensorflow.keras.models.load_model(path)
        new_model.load_weights(path)
        initialize_vars(sess)


    else:
        new_model = load_model(question_generator)
        new_model.load_weights(path)
        initialize_vars(sess)

    logger.info('Model {} loaded'.format(model_file_name))
    new_model.summary()

    test_file_name = test_file.split('/')[-1]

    with open(args.model_dir + '/result_' + test_file_name, mode='w') as f:
        with open(args.model_dir + '/gt_' + test_file_name, mode='w') as gt_file:

            file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['image_id', 'image_url', 'questions'])

            gt_file_writer = csv.writer(gt_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            gt_file_writer.writerow(['image_id', 'questions'])

            i = 0
            for id, test_image_url in datasets.test_image_id_url_dict.items():
                i = i + 1

                try:
                    if user_input == 'YES':
                        test_image_url = input("Provide the url of the image: ")
                        # test_image_url =
                        # 'https://vision.ece.vt.edu/data/mscoco/images/train2014/./COCO_train2014_000000314392.jpg'
                        output_questions = question_generator.test_model(test_image_url, new_model,
                                                                         decoder_algorithm, beam_size)
                    else:
                        logger.info('\n\n\nImage url: %s' % test_image_url)
                        if id in datasets.test_image_id_keyword_dict:
                            keyword = datasets.test_image_id_keyword_dict[id]
                        else:
                            keyword = None
                        output_questions = question_generator.test_model(test_image_url, new_model,
                                                                         decoder_algorithm, beam_size, keyword)
                        file_writer.writerow([i, test_image_url, '---'.join(output_questions)])

                        gt_questions = datasets.test_image_id_questions_dict[id]
                        ground_truth = []
                        for question in gt_questions:
                            gt = question.split()[1:-1]
                            gt = ' '.join(gt)
                            if gt in ['None', 'none']:
                                continue
                            ground_truth.append(gt)
                        logger.info('GT  ---->%s' % ground_truth)
                        gt_file_writer.writerow([i, '---'.join(ground_truth)])
                except:
                    logger.error('%s has inference issues. Most likely doesnt exist' % test_image_url)
                    continue

                # Generative strength:
                # No of unique questions averaged over number of images
                logger.info('Generative strength: %s' % str(len(datasets.unique_generated_questions)/float(i)))

                f.flush()
                gt_file.flush()

                # Inventiveness:
                unique_questions_not_seen_training_data = set()
                print('Unique generated:', len(datasets.unique_generated_questions))
                print('All generated', len(datasets.generated_questions))
                for q in datasets.unique_generated_questions:
                    if q not in datasets.unique_train_questions:
                        unique_questions_not_seen_training_data.add(q)

                logger.info(
                    'Unique generated questions not seen in training data: %s' % unique_questions_not_seen_training_data)

                logger.info('I %s Overall Inventiveness %s\n\n' % (str(i), str(len(unique_questions_not_seen_training_data)/len(datasets.generated_questions))))

    prediction_file_path = "{}/result_{}_test_all.csv".format(args.model_dir, dataset)
    ground_truth_file_path = "{}/gt_{}_test_all.csv".format(args.model_dir, dataset)
    logger.info("Running evaluation script on prediction %s and gt %s file" % (prediction_file_path, ground_truth_file_path))
    run_evaluation(prediction_file_path, ground_truth_file_path)








