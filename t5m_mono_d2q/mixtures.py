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

"""Add Mixtures to the registry.

This module contains different mixtures for training T5 models.
"""
from t5.data.dataset_providers import MixtureRegistry
from t5.data.glue_utils import get_glue_weight_mapping
from t5.data.glue_utils import get_super_glue_weight_mapping
import t5.data.tasks  # pylint: disable=unused-import
from t5.data.utils import rate_num_examples
from t5.data.utils import rate_unsupervised

# ============================= msmarco  ================================
MixtureRegistry.remove("msmarco_mono-0.85+d2q-0.15")

MixtureRegistry.add(
    "msmarco_mono-0.85+d2q-0.15",
    [("msmarco_passage_ranking_pairwise", 0.85), ("msmarco_passage_to_query", 0.15)]
)
