# t5_for_IR
Leverage T5 pretrained with mulitview learning framework to improve IR problems.

> [2021/12/20] Updates: Codes of multi-view learning with "pointwise reranking" and "doc2query".
> [2021/03/19] Updates: Try d2q with duo multi-view learning.

<hr/>

## Types:
0. Standard BM25
1. MonoT5
2. MonoT5 + D2qT5
3. MonoT5 + D2qT5 + DuoT5


## Results:
Evaluating bm25_top1k.trec...
MRR @10: 0.18736452221767383
Evaluating t5_mono_rerank_top1k.trec...
MRR @10: 0.39302224041479095
Evaluating t5m_mono_d2q_duo_rerank_top1k.trec...
MRR @10: 0.39709936553417985
Evaluating t5m_mono_d2q_rerank_top1k.trec...
MRR @10: 0.3995522354118344
Evaluating t5m_mono_duo_modified_rerank_top1k.trec...
MRR @10: 0.3887470437076461
Evaluating t5m_mono_duo_rerank_top1k.trec...
MRR @10: 0.3927071678719255
