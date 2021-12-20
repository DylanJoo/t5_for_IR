# coding=utf-8
# Copyright 2020 The Mesh TensorFlow Authors.
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

"""Layers implemented in Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gin

from mesh_tensorflow import ops_with_redefined_builtins as mtf

import tensorflow.compat.v1 as tf

def modified_softmax_cross_entropy_with_logits(logits, targets, vocab_dim, z_loss=0.0):
  """Per-example softmax loss.

  `logits` is a Tensor with floating-point dtype, containing the predicted
  relative log probabilities of the classes.

  Either hard targets or soft targets are supported.

  In the case of hard targets, `targets` is a Tensor with integer dtype whose
  values are in the range [0, vocab_dim.size).  `targets` should have the same
  set of dimensions as `logits`, but without `vocab_dim`.

  In the case of soft targets, `targets` is a Tensor with floating point dtype
  and the same dimensions as `logits.  Reducing `targets` along `vocab_dim`
  should result in all ones.

  if z_loss is nonzero, we add a loss equal to z_loss*log(z)^2, where z is the
  partition function.  Example value: z_loss=1e-4.  Two uses of z_loss are:
  - To keep the logits from drifting too far from zero, which can cause
     unacceptable roundoff errors in bfloat16.
  - To encourage the logits to be normalized log-probabilities.

  Args:
    logits: a mtf.Tensor whose shape contains vocab_dim
    targets: a mtf.Tensor representing hard or soft targets (see comments)
    vocab_dim: a mtf.Dimension
    z_loss: a float

  Returns:
    a mtf.Tensor whose shape is equal to logits.shape - vocab_dim

  Raises:
    ValueError: if the shapes do not match.
  """
  if targets.dtype.is_integer:
    # hard targets
    if (set(targets.shape.dims)
        != set(logits.shape.dims).difference([vocab_dim])):
      raise ValueError(
          "softmax_cross_entropy_with_logits with hard targets "
          "dims in targets=%s should be dims in logits=%s other than "
          "vocab_dim=%s" % (targets, logits, vocab_dim))
    targets = mtf.one_hot(targets, vocab_dim, dtype=logits.dtype)
  elif set(targets.shape.dims) != set(logits.shape.dims):
    raise ValueError(
        "softmax_cross_entropy_with_logits with soft targets "
        "dims in targets=%s should be dims in logits=%s"% (targets, logits))
  if vocab_dim not in logits.shape.dims:
    raise ValueError("vocab_dim must be in logits.shape.dims")
  log_z = mtf.reduce_logsumexp(logits, vocab_dim)
  log_softmax = logits - log_z
  # The target is the one-hot mask
  loss = mtf.negative(
      mtf.reduce_sum(logits * targets, reduced_dim=vocab_dim))
  
  return loss # negative logit of target token
  #return mtf.gather(loss, 0, loss.shape.dims[1])
  
  #return loss
