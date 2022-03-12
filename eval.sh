for file in ~/git/t5_for_IR/results/t5_mono_*.trec;do
    k=1
    ANSERINI='/home/jhju/treccast/2021/retrieval/anserini'
    echo "Evaluating ${file##*/}..."

    # (1) trec_eval
    # ${ANSERINI}/tools/eval/trec_eval.9.0.4/trec_eval \
    #     -c -m map \
    #     -c -m recip_rank.10 \
    #     -c -m recall.1000 \
    #     ${ANSERINI}/src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt \
    #     ${file}

    # (2) Anserini's
    ## convert trec to runs
    python3 ${ANSERINI}/tools/scripts/msmarco/convert_trec_to_msmarco_run.py \
        --input ${file} --output temp.runs

    ## evaluate the runs
    python3 ${ANSERINI}/tools/scripts/msmarco/msmarco_passage_eval.py \
        ./data/dev/qrels.dev.small.tsv temp.runs
    rm temp.runs
done
