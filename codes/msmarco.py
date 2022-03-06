"""
Author: Dylan JH Ju 
E-mail: jhjoo@citi.sinica.edu.tw
"""
import tensorflow as tf
import functools

# Task 1: [MONO-RANK] msmrco passage ranking
def msmarco_passage_pointwise_ranking_prep(ds):
    def normalize_text(text):
        text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
        return text
    def to_inputs_and_targets(ex):
        return {
            "inputs": normalize_text(ex["qp-pair"]),
            "targets": normalize_text(ex["relevance"])
        }
    return ds.map(to_inputs_and_targets, 
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
def msmarco_passage_pointwise_ranking_ds(split, shuffle_files):
    '''
    Input: [Query: <q> Document: <d> Relevant:]
    Output: [true] or [false]
    '''
    dataset = tf.data.TextLineDataset("gs://t541r/data/triples.train.small.monot5.tsv")
    dataset = dataset.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(lambda *ex: dict(zip(["qp-pair", "relevance"], ex)))
    
    return dataset

# Task2 : [D2Q] msmrco passage to query
def msmarco_passage_to_query_prep(ds):
    def normalize_text(text):
        text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
        return text
    prefix = ""
    postfix = " Translate Document to Query:"
    def to_inputs_and_targets(ex):
        return {
            "inputs": prefix + normalize_text("Document: " + ex["p-relevant"]) + postfix,
            "targets": normalize_text(ex["q-relevant"])
        }
    return ds.map(to_inputs_and_targets,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

def msmarco_passage_to_query_ds(split, shuffle_files):
    '''
    Input: [Document: <d> Translate Document to Query:]
    Output: [<q>]
    '''
    dataset = tf.data.TextLineDataset("gs://t541r/data/rel_pai.train.p2q.tsv")
    dataset = dataset.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(lambda *ex: dict(zip(["p-relevant", "q-relevant"], ex)))
    
    return dataset

# Task 3: [DUO-RANK] msmrco passage pairwise-ranking
def msmarco_passage_pairwise_ranking_prep(ds):
    def normalize_text(text):
        text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
        return text
    def to_inputs_and_targets(ex):
        return {
            "inputs": normalize_text(ex["qqp-pair"]),
            "targets": normalize_text(ex["relevance"])
        }
    return ds.map(to_inputs_and_targets, 
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
def msmarco_passage_pairwise_ranking_ds(split, shuffle_files):
    '''
    Input: [Query: <q> Document0: <d0> Document1: <d1> Relevant:]
    Output: [true] or [false]
    '''
    dataset = tf.data.TextLineDataset("gs://t541r/data/triples.train.small.duot5.tsv")
    dataset = dataset.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(lambda *ex: dict(zip(["qqp-pair", "relevance"], ex)))
    
    return dataset
