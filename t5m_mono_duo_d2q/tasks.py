# Copyright 2020 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add Tasks to registry."""
import functools

from t5.data import postprocessors
from t5.data import preprocessors
from t5.data.dataset_providers import Feature
from t5.data.dataset_providers import TaskRegistry
from t5.data.dataset_providers import TfdsTask
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
from t5.data.glue_utils import get_super_glue_metric
from t5.data.utils import get_default_vocabulary
from t5.data.utils import set_global_cache_dirs
from t5.evaluation import metrics
import tensorflow_datasets as tfds
from t5.data.msmarco import msmarco_passage_ranking_prep, msmarco_passage_to_query_prep
from t5.data.msmarco import msmarco_passage_ranking_ds, msmarco_passage_to_query_ds

# ==================================== MSMARCO start ======================================
TaskRegistry.remove('msmarco_passage_ranking')
TaskRegistry.remove('msmarco_passage_to_query')

# [MONO-RANK] Passage Ranking task 
TaskRegistry.add(
    "msmarco_passage_ranking_pair_wise",
    dataset_fn=msmarco_passage_ranking_ds,
    text_preprocessor=[msmarco_passage_ranking_prep],
    metric_fns=[],
    splits=['train'])

# [D2Q] Passage to query task
TaskRegistry.add(
    "msmarco_passage_to_query",
    dataset_fn=msmarco_passage_to_query_ds,
    text_preprocessor=[msmarco_passage_to_query_prep],
    metric_fns=[], 
    splits=['train'])

# ==================================== MSMARCO end ======================================
