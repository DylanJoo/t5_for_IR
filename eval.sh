for file in ~/git/t5_for_IR/results/*.trec;do
    k=1
    ANSERINI='/home/jhju/treccast/2021/retrieval/anserini'
    echo "==================="
    echo "Evaluating ${file}..."

    ${ANSERINI}/tools/eval/trec_eval.9.0.4/trec_eval \
        -c -m map \
        -c -m recip_rank.10 \
        -c -m recall.1000 \
        ${ANSERINI}/src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt \
        ${file}
    echo "==================="
done
