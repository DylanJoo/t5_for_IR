# T5_for_IR
Leverage T5 pretrained with mulitview learning framework to improve IR problems.
## T5 multiview learning for passage reranking
> paper: Text-to-text Multi-view Learning for Passage Re-ranking.

### Requirements 
- GCP with TPU

This codes will need the Google Cloud Platform' VM and TPU. \
So make sure the GCP account is ready or check the detail in GCP's information.
- Environment setup 
```
export PATH=$PATH:/home/${GCP_ID}/.local/bin
```
- Bucket accessibility

You may need to manually modify accessibility of Goole's bucket, and allowed the VM and TPU API to access.

### Package requirements
```
pip3 install tensorflow==2.3.0
pip3 install tensorflow-text==2.3.0 
pip3 install mesh-tensorflow==0.1.17
pip3 install t5[gcp]==0.7.1
```
> So far, only test on the latest version of these packages.


### Instructions
Modifiy the T5 prediction into logit-form, instead of the text output.

1. Append the content of 'layers_add.py' 
> ${PYTHON_PACKAGES}/mesh_tensorflow/utils.py \

2. Append the content of 'uilts_add.py'  
> ${PYTHON_PACKAGES}/mesh_tensorflow/transformers/utils.py

3. Add the multiview learning tasks files:
```
cp msmarco.py ${PYTHON_PACKAGES}/t5/data/
```

4. Append the msmarco tasks and mixtures
- tasks_add.py
- mixtures_add.py
> ${PYTHON_PACKAGES}/t5/data/


### Finetunning
Add the MSMARCO pasage ranking task of (monoT5 + D2Q)
> ${PYTHON_PACKAGES}/mesh_tensorflow/transformers/utils.py

Run the mulit-view learning framework by T5's API

### Inferencing (prediction)
```
for ITER in {00..20}; do
  echo "Running iter: $ITER" >> process-t5m.out
  nohup t5_mesh_transformer \
    --tpu="${YOUR TPU}" \
    --gcp_project="${PRJ_NAME}" \
    --tpu_zone="europe-west4-a" \
    --model_dir="gs://${MODEL_DIR}" \
    --gin_file="gs://t5-data/pretrained_models/${MODEL_SIZE}/operative_config.gin" \
    --gin_file="infer.gin" \
    --gin_file="score_from_file.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
    --gin_param="infer_checkpoint_step = ${CKPT}" \
    --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
    --gin_param="Bitransformer.decode.max_decode_length = 2" \
    --gin_param="inputs_filename = 'gs://${INPUT_FILE}'" \
    --gin_param="targets_filename = 'gs://${TARGET_FILE}'" \
    --gin_param="scores_filename = 'gs://${OUTPUT_FILE}'" \
    --gin_param="Bitransformer.decode.beam_size = 1" \
    --gin_param="Bitransformer.decode.temperature = 0.0" \
    --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" 
```

### Refereces:
- Text-to-text Transfer Transformers (T5)
- T5 document ranking 
- doc2TTTTTquery
