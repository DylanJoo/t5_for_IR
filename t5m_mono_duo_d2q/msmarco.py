"""
Author: Dylan JH Ju 
E-mail: jhjoo@citi.sinica.edu.tw
"""
import tensorflow as tf
import functools

# Task 1: [MONO-RANK] msmrco passage ranking
def msmarco_passage_ranking_prep(ds):
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
def msmarco_passage_ranking_ds(split, shuffle_files):
    '''
    Input: [Query: <q> Document: <d> Relevant:]
    Output: [true] or [false]
    '''
    if split == "full":
        dataset = tf.data.TextLineDataset("gs://castorini/monot5/data/query_doc_pairs.train.tsv")
        # [TODO] upload the full triplet if needed.
    else:
        dataset = tf.data.TextLineDataset("gs://castorini/monot5/data/query_doc_pairs.train.tsv")
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
    dataset = tf.data.TextLineDataset("gs://conv-ir/msmarco/doubles.train.qrels.tsv")
    dataset = dataset.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(lambda *ex: dict(zip(["p-relevant", "q-relevant"], ex)))
    
    return dataset

