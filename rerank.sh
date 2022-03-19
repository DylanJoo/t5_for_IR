for folder in ~/git/t5_for_IR/t5m_mono_d2q_duo;do
    echo $folder
    file=${folder##*/}
    echo $file
    k=1

    # converting
    python3 rerank_by_logits.py \
        -flogits ${folder}/qp_pairs.dev.small.flogits \
        -tlogits ${folder}/qp_pairs.dev.small.tlogits \
        -score ${folder}/qp_relevance.dev.small.scores \
        -runs ./data/dev/query_doc_id_pairs.dev.small.tsv \
        -rerank_runs ./results/${file}_rerank_top${k}k.trec \
        -topk ${k}000 \
        --resoftmax \
        --trec
done
