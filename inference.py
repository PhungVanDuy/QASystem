#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import bottle
from bottle import route, run
import threading
import json
import numpy as np
from model import Model
import os
from prepro import convert_to_features, word_tokenize
from time import sleep
import torch
flags = tf.flags

home = os.path.expanduser("~")
train_file = os.path.join(home, "data", "squad", "train-v1.1.json")
dev_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
test_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
glove_word_file = os.path.join(home, "data", "glove", "glove.840B.300d.txt")

train_dir = "train"
model_name = "FRC"
dir_name = os.path.join(train_dir, model_name)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(os.path.join(os.getcwd(),dir_name)):
    os.mkdir(os.path.join(os.getcwd(),dir_name))
target_dir = "data"
log_dir = os.path.join(dir_name, "event")
save_dir = os.path.join(dir_name, "model")
answer_dir = os.path.join(dir_name, "answer")
train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
word_dictionary = os.path.join(target_dir, "word_dictionary.json")
char_dictionary = os.path.join(target_dir, "char_dictionary.json")
answer_file = os.path.join(answer_dir, "answer.json")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

flags.DEFINE_string("mode", "train", "Running mode train/debug/test")


flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("glove_word_file", glove_word_file, "Glove word embedding source file")

flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer")
flags.DEFINE_string("word_dictionary", word_dictionary, "Word dictionary")
flags.DEFINE_string("char_dictionary", char_dictionary, "Character dictionary")


flags.DEFINE_integer("glove_char_size", 94, "Corpus size for Glove")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")

flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("ans_limit", 30, "Limit length for answers")
flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
flags.DEFINE_integer("char_limit", 16, "Limit length for character")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_integer("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("num_steps", 60000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.1, "Dropout prob across the layers")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
flags.DEFINE_integer("hidden", 128, "Hidden size")
flags.DEFINE_integer("num_heads", 1, "Number of heads in self attention")
flags.DEFINE_boolean("q2c", True, "Whether to use query to context attention or not")
flags.DEFINE_integer("early_stop", 3, "Checkpoints for early stop")

# Extensions (Uncomment corresponding code in download.sh to download the required data)
glove_char_file = os.path.join(home, "data", "glove", "glove.840B.300d-char.txt")
flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding source file")
flags.DEFINE_boolean("pretrained_char", False, "Whether to use pretrained character embedding")

fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding source file")
flags.DEFINE_boolean("fasttext", False, "Whether to use fasttext")


class Inference(object):

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.query = []

    def request(self, passage, question):
        self.query = (passage, question)
        return self.backend()

    def backend(self):

        with open(self.config.word_dictionary, "r") as fh:
            word_dictionary = json.load(fh)
        with open(self.config.char_dictionary, "r") as fh:
            char_dictionary = json.load(fh)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with self.model.graph.as_default():

            with tf.Session(config=sess_config) as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(self.config.save_dir))
                if self.config.decay < 1.0:
                    sess.run(self.model.assign_vars)
                if self.query:
                    context = word_tokenize(self.query[0].replace("''", '" ').replace("``", '" '))
                    c,ch,q,qh = convert_to_features(self.config, self.query, word_dictionary, char_dictionary)
                    fd = {'context:0': [c],
                        'question:0': [q],
                        'context_char:0': [ch],
                        'question_char:0': [qh]}
                    yp1,yp2,probs = sess.run([self.model.yp1, self.model.yp2, self.model.prob_score], feed_dict = fd)
                    print(probs)
                    yp2[0] += 1
                    response = " ".join(context[yp1[0]:yp2[0]])
                    print(probs[0].reshape(-1))
                    print(probs[0].reshape(-1).shape[0])
                    scores = torch.ger(torch.Tensor(probs[0].reshape(-1)), torch.Tensor(probs[1].reshape(-1)))
                    scores.triu_().tril_(probs[0].reshape(-1).shape[0] - 1)
                    scores = scores.numpy()
                    scores_flat = scores.flatten()
                    sc = max(scores_flat)
                    return (response, sc)

def main():
    config = flags.FLAGS
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)
    model = Model(config, None, word_mat, char_mat, trainable=False, demo = True)
    inference = Inference(model, config)
    passage = u'Ronaldo Luis Nazario de Lima born 18 September 1976, commonly known as Ronaldo, is a retired Brazilian professional footballer who played as a striker.\
    Popularly dubbed "O Fenomeno" (The Phenomenon), he is widely considered to be one of the greatest football players of all time.\
    In his prime, he was known for his dribbling at speed, feints, and clinical finishing. \
    At his best in the 1990s, Ronaldo starred at club level for Cruzeiro, PSV, Barcelona, and Internazionale. \
    His moves to Spain and Italy made him only the second player, after Diego Maradona, to break the world transfer record twice, all before his 21st birthday.\
    At age 23, he had scored over 200 goals for club and country. \
    After almost three years of inactivity due to serious knee injuries and recuperation, Ronaldo joined Real Madrid in 2002, which was followed by spells at Milan and Corinthians.'
    passage2 = u'Ronaldo won the FIFA World Player of the Year three times, in 1996, 1997 and 2002, and the Ballon d\'Or twice, in 1997 and 2002, as well as the UEFA Club Footballer of the Year in 1998.\
    He was La Liga Best Foreign Player in 1997, when he also won the European Golden Boot after scoring 34 goals in La Liga, and he was named Serie A Footballer of the Year in 1998. \
    One of the most marketable sportsmen in the world, the first Nike Mercurial boots–R9–were commissioned for Ronaldo in 1998. \
    He was named in the FIFA 100, a list of the greatest living players compiled in 2004 by Pelé, and was inducted into the Brazilian Football Museum Hall of Fame and the Italian Football Hall of Fame.'

    print(inference.request(passage, u'What is the nickname of Ronaldo?'))
    passage3 = u'Population of Vietnam as standing at approximately 90.7 million people. The population had grown significantly from the 1979 census, which showed the total population of reunified Vietnam to be 52.7 million. In 2012, the country population was estimated at approximately 90.3 million. Currently, the total fertility rate of Vietnam is 1.8 (births per woman), which is largely due to the government family planning policy, the two-child policy.'
    print(inference.request(passage3, u'How many population of Vietnam?'))


if __name__=='__main__':
    main()
