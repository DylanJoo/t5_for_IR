# This modified code is made for text-to-text transfer transformers (T5)
#
# Substitiute the following function in '${pytho_libary_path}/mesh_tensorflow/utils.py'
#
# (1) Extract the true/false logit in the prediction mode.
# (2) Append the true/false logit to the predictions.

@gin.configurable
def tpu_estimator_model_fn(model_type,
                           transformer_model,
                           vocabulary,
                           model_dir,
                           use_tpu,
                           mesh_shape,
                           layout_rules,
                           batch_size,
                           sequence_length,
                           autostack,
                           keep_checkpoint_max,
                           save_checkpoints_steps,
                           learning_rate_schedule=None,
                           optimizer=None,
                           outer_batch_size=1,
                           tpu_summaries=False,
                           predict_fn=None,
                           score_in_predict_mode=False,
                           variable_filter=None,
                           init_checkpoint=None,
                           ensemble_inputs=None,
                           mesh_devices=None,
                           model_info_file=None,
                           hierarchical_tiling_spec=None):
  """Create a TPUEstimator model function.

  Args:
    model_type: a string. One of "bitransformer", "lm", "delimited_lm",
      "aligned", or "bi_teacher_student"
    transformer_model: a transformer.Unitransformer or transformer.Bitransformer
    vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
      targets_vocabulary) tuple. Used for decoding in predict mode.
    model_dir: a string, directory to save the model to.
    use_tpu: a boolean
    mesh_shape: a mtf.Shape
    layout_rules: a mtf.LayoutRules
    batch_size: an integer
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    autostack: a boolean
    keep_checkpoint_max: an integer, maximum number of checkpoints to keep
    save_checkpoints_steps: an integer, save a checkpoint every this number of
      steps
    learning_rate_schedule: a constant or a function from step to learning rate
    optimizer: a class extending optimize.Optimizer, required for training
    outer_batch_size: outer batch dimension that could be used to enable the mix
      of data-parallel and model-parallel training of Mixture of Experts (MoE)
      models
    tpu_summaries: a boolean, use rewrites to make summaries work on TPU.  This
      may be slow, since it uses a host call hack.
    predict_fn: an optional function, see docs for `run` for more information.
    score_in_predict_mode: compute log-likelihood scores instead of predictions
    variable_filter: controls which variables are trained.
      If None (default), train all trainable variables.
      If a string regex, train all variables that match this regex.
      If a function (mtf.Variable -> boolean), then train variables for which
        the function returns True.
    init_checkpoint: a string, if not None then read in variables from this
      checkpoint path when initializing variables. Will only initialize
      variables that appear both in the current graph and the checkpoint.
    ensemble_inputs: an optional integer - pass the size of the ensemble to
      train an ensemble where each model gets different inputs.
      You also need to configure Unitransformer.ensemble  to the right size.
      If None, then all models are trained on the same inputs.
    mesh_devices: a list of strings, the device names to use for each mesh
      slice. Only required for GPU.
    model_info_file: an optional string, information about variables and
      operations will be logged to this file during the TRAIN mode.
    hierarchical_tiling_spec: an optional list that can be passed as the
      spec argument to simd_mesh_impl.HierarchicalTiling
  Returns:
    a function to be passed to TPUEstimator
  """
  mesh_devices = mesh_devices or [""] * mesh_shape.size

  def my_model_fn(features, labels, mode, params=None, config=None):
    """Estimator model function.

    Args:
      features: dictionary where keys are strings like "inputs" and "targets"
        and the values are the actual values of "inputs". See TPUEstimator's
        docs for more information
      labels: ignored argument
      mode: a tf.estimator.ModeKeys
      params: dictionary containing the key "context"
      config: ignored argument

    Returns:
      a TPUEstimatorSpec
    """
    del labels, config
    if mode == tf.estimator.ModeKeys.PREDICT and score_in_predict_mode:
      mode = "score"
    global_step = tf.train.get_global_step()
    if use_tpu and "context" in params:
      ctx = params["context"]
      num_hosts = ctx.num_hosts
      host_placement_fn = ctx.tpu_host_placement_function
      device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
      # TODO(ylc): Better estimation of replica cache size?
      replica_cache_size = 300 * 1000000  # 300M per replica
      # Worker 0 caches all the TPU binaries.
      worker0_mem = replica_cache_size * ctx.num_replicas
      devices_memeory_usage = [worker0_mem] + [0] * (num_hosts - 1)
      var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                    devices_memeory_usage)
      physical_shape = [int(i) for i in
                        params["context"].device_assignment.topology.mesh_shape]
      if len(physical_shape) == 4:
        physical_shape = (
            mtf.simd_mesh_impl.physical_shape_3d_from_topology_proto_4d(
                physical_shape))
      if hierarchical_tiling_spec is not None:
        logical_to_physical = mtf.simd_mesh_impl.HierarchicalTiling(
            hierarchical_tiling_spec,
            physical_shape).logical_to_physical
      else:
        logical_to_physical = mtf.simd_mesh_impl.auto_logical_to_physical_tpu(
            mesh_shape.to_integer_list, physical_shape)
      mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
          mesh_shape, layout_rules, mesh_devices, ctx.device_assignment,
          logical_to_physical=logical_to_physical)
    else:
      var_placer = None
      mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
          mesh_shape, layout_rules, mesh_devices)

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh", var_placer)

    if (outer_batch_size and
        mode not in [tf.estimator.ModeKeys.PREDICT, "score"]):
      outer_batch_dim = mtf.Dimension("outer_batch", outer_batch_size)
      batch_dim = mtf.Dimension("batch", batch_size // outer_batch_size)
      batch_dims = [outer_batch_dim, batch_dim]
    else:
      batch_dim = mtf.Dimension("batch", batch_size)
      batch_dims = [batch_dim]
    ensemble_dims = ([mtf.Dimension("ensemble", ensemble_inputs)]
                     if ensemble_inputs else [])

    mtf_features = {}
    for key, x in features.items():
      # Some auxiliary features may have been generated in packing.
      # The names of these new features are of the form
      #   "<original_feature_name>_<suffix>", e.g. "inputs_segmentation".
      #   We look up the lengths based on the original feature name, without
      #   the "_<suffix>".
      feature_length = sequence_length[key.split("_")[0]]
      length_dim = mtf.Dimension("length", feature_length)
      feature_shape = mtf.Shape(
          ensemble_dims + batch_dims + [length_dim])
      x = tf.cast(features[key], tf.int32)
      x = tf.reshape(x, feature_shape.to_integer_list)
      if not use_tpu:
        tf.logging.info("feature %s : %s" % (key, x))
      mtf_features[key] = mtf.import_fully_replicated(
          mesh, x, feature_shape, name=key)

    def _verify_feature_exists(feature_name, should_exist):
      if should_exist != (feature_name in mtf_features):
        message = (
            "mode=%s model_type=%s should%s have feature %s" %
            (mode, model_type, "" if should_exist else " not", feature_name))
        if "lm" in model_type:
          message += (
              "\nA common mistake is that model_type=\"lm\" should be used "
              "with tasks that produce inputs and targets, while "
              "model_type=\"delimited_lm\" should be used with tasks that "
              "produce targets only.")
        raise ValueError(message)

    # Verify that the right features exist, and transform them if necessary
    if mode == tf.estimator.ModeKeys.PREDICT:
      _verify_feature_exists("inputs", True)
      # "targets" may or may not exist depending on whether we are doing
      # evaluation or open-ended inference.
    elif model_type in ("lm", "delimited_lm") and mode == "score":
      # in scoring mode the inputs and targets may already be combined.
      if "inputs" in mtf_features:
        if model_type == "lm":
          tf.logging.warning(
              "Scoring of lm models will include loss from the 'inputs'.")
        mtf_features = _dynamic_text2self(mtf_features)
    else:
      _verify_feature_exists("targets", True)
      _verify_feature_exists("inputs", model_type != "lm")
      if model_type == "delimited_lm":
        mtf_features = _dynamic_text2self(mtf_features)

    # Detokenize in the graph if supported by vocabulary and accelerator.
    def _maybe_detokenize(ids, vocab):
      if not use_tpu and hasattr(vocab, "decode_tf"):
        return vocab.decode_tf(ids)
      return ids
    if mode == "score":
      # compute log-likelihoods per sequence
      if predict_fn:
        # predict_fn contains a custom scoring function
        # this code-path has not been tested
        scores = predict_fn(
            model=transformer_model,
            features=mtf_features,
            variable_dtype=get_variable_dtype())
      targets = mtf_features["targets"]
      if isinstance(transformer_model, transformer.Unitransformer):
        length_dim = targets.shape.dims[-1]
        inputs = transformer.autoregressive_inputs(
            mtf_features["targets"])
      elif isinstance(transformer_model,
                      (transformer.Bitransformer,
                       transformer.StudentTeacher)):
        inputs = mtf_features["inputs"]
      else:
        raise ValueError("unrecognized class")
      logits, _ = transformer_model.call_simple(
          inputs=inputs,
          targets=targets,
          compute_loss=False,
          mode=mode,
          variable_dtype=get_variable_dtype())
      logits = mtf.cast(logits, tf.float32)
      _, length_dim, vocab_dim = logits.shape.dims

      # < Modified start >
      cross_entropy = mtf.layers.modified_softmax_cross_entropy_with_logits( #Modified
          logits, mtf_features["targets"], vocab_dim)
      # < Modified end >

      # 0=padding and negative targets are a hack to indicate no loss
      cross_entropy *= mtf.cast(
          mtf.greater(targets, 0), cross_entropy.dtype)
      if model_type == "delimited_lm":
        cross_entropy *= mtf.cast(mtf.logical_not(
            transformer.delimited_lm_inputs_mask(targets)), cross_entropy.dtype)
      # Log-likelihood of the sentences 
      scores = -mtf.reduce_sum(cross_entropy, reduced_dim=length_dim)

      # likelihood of setences Modified
      scores = mtf.exp(scores)
      
      scores = mtf.anonymize(scores)

      # < Modified start >
      true_logit = mtf.gather(logits, 1176, vocab_dim)
      true_logit = mtf.gather(true_logit, 0, true_logit.shape.dims[-1])
      true_logit = mtf.anonymize(true_logit)

      false_logit = mtf.gather(logits, 6136, vocab_dim) 
      false_logit = mtf.gather(false_logit, 0, false_logit.shape.dims[-1])
      false_logit = mtf.anonymize(false_logit)
      # < Modified end >
      
      targets = mtf.anonymize(targets)
      lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)
      targets = clean_decodes(lowering.export_to_tf_tensor(targets))
      targets = _maybe_detokenize(targets, targets_vocabulary(vocabulary))
      
      # < Modified start >
      predictions = {
          "targets": targets,
          "scores": lowering.export_to_tf_tensor(scores), 
          "true_logit": lowering.export_to_tf_tensor(true_logit), 
          "false_logit": lowering.export_to_tf_tensor(false_logit), 
      }
      # < Modified end >

    elif mode == tf.estimator.ModeKeys.PREDICT:
      inputs = mtf_features["inputs"]
      if predict_fn:
        mtf_samples = predict_fn(
            model=transformer_model,
            features=mtf_features,
            variable_dtype=get_variable_dtype())
      elif isinstance(transformer_model, transformer.Unitransformer):
        # pad so that there is enough room for the targets
        inputs = mtf.pad(
            inputs, [0, sequence_length["targets"]], length_dim.name)
        mtf_samples = transformer_model.sample_autoregressive(
            inputs, variable_dtype=get_variable_dtype(),
            remove_partial_sequences=True)
      elif isinstance(
          transformer_model,
          (transformer.Bitransformer, transformer.StudentTeacher)):
        mtf_samples = transformer_model.decode(
            inputs, variable_dtype=get_variable_dtype())
      else:
        raise ValueError("unrecognized class")
      mtf_samples = mtf.anonymize(mtf_samples)
      inputs = mtf.anonymize(inputs)
      lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)
      inputs = clean_decodes(lowering.export_to_tf_tensor(inputs))
      outputs = clean_decodes(lowering.export_to_tf_tensor(mtf_samples))

      inputs = _maybe_detokenize(inputs, inputs_vocabulary(vocabulary))
      outputs = _maybe_detokenize(outputs, targets_vocabulary(vocabulary))

      predictions = {
          "inputs": inputs,
          "outputs": outputs}

    if mode in ["score", tf.estimator.ModeKeys.PREDICT]:
      # When exporting a model, we need to communicate to TF-Serving that
      # master variables need to be copied to their slave slice variables.
      # Estimator uses a Scaffold's "local_init_op" for this purpose, so we
      # augment the default "local_init_op" here.
      #
      # The "ready_op" is also constructed here to ensure the variables
      # initialized by "local_init_op" are the same ones checked by "ready_op".
      #
      # WARNING: Any variables created outside of this model_fn()
      # (e.g. tpu_estimator/iterations_per_loop) will NOT be initialized nor
      # checked by these ops.
      def scaffold_fn():
        return tf.train.Scaffold(
            local_init_op=tf.group(
                tf.train.Scaffold.default_local_init_op(),
                lowering.copy_masters_to_slices(),
                name="mtf_local_init_op"),
            ready_op=tf.concat(
                [tf.report_uninitialized_variables(),
                 resources.report_uninitialized_resources()],
                axis=0,
                name="mtf_ready_op"))

      return tpu_estimator.TPUEstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          scaffold_fn=scaffold_fn,
          prediction_hooks=[mtf.MtfRestoreHook(lowering)])

    assert (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL)

    def logits_and_loss(mtf_features, num_microbatches=1):
      """Compute logits and loss.

      Args:
        mtf_features: a dictionary
        num_microbatches: integer
      Returns:
        logits: a mtf.Tensor
        loss: a mtf.Tensor
      """
      if model_type in ["lm", "delimited_lm"]:
        inputs = transformer.autoregressive_inputs(
            mtf_features["targets"],
            sequence_id=mtf_features.get("targets_segmentation", None))
      else:
        inputs = mtf_features["inputs"]

      if isinstance(transformer_model, transformer.Unitransformer):
        position_kwargs = dict(
            sequence_id=mtf_features.get("targets_segmentation", None),
            position=mtf_features.get("targets_position", None),
        )
      elif isinstance(
          transformer_model,
          transformer.Bitransformer) or model_type == "bi_student_teacher":
        position_kwargs = dict(
            encoder_sequence_id=mtf_features.get("inputs_segmentation", None),
            decoder_sequence_id=mtf_features.get("targets_segmentation",
                                                 None),
            decoder_subsequence_id=mtf_features.get("targets_subsegmentation",
                                                    None),
            encoder_position=mtf_features.get("inputs_position", None),
            decoder_position=mtf_features.get("targets_position", None),
        )
      else:
        raise ValueError("unrecognized class")

      return transformer_model.call_simple(
          inputs=inputs,
          targets=mtf_features["targets"],
          compute_loss=True,
          mode=mode,
          variable_dtype=get_variable_dtype(),
          num_microbatches=num_microbatches,
          **position_kwargs)

    if mode == tf.estimator.ModeKeys.TRAIN:
      num_microbatches = serialize_num_microbatches(batch_dim,
                                                    sequence_length,
                                                    mesh_shape,
                                                    layout_rules)
      if num_microbatches > 1:
        def serialized_fn(mtf_features):
          return {"loss": logits_and_loss(mtf_features, num_microbatches)[1]}
        var_grads, loss_dict = mtf.serialize_training_step(
            mtf_features, serialized_fn, batch_dim, num_microbatches)
        loss = loss_dict["loss"]
      else:
        loss = logits_and_loss(mtf_features)[1]
        var_grads = mtf.gradients(
            [loss], [v.outputs[0] for v in graph.trainable_variables])

      if tpu_summaries:
        mtf.scalar_summary("loss", loss)
        for g in var_grads:
          grad_norm = mtf.sqrt(mtf.reduce_sum(mtf.square(g)))
          mtf.scalar_summary("grads/norm" + g.name[:-2], grad_norm)

      if callable(learning_rate_schedule):
        # the following happens on CPU since TPU can't handle summaries.
        with mtf.utils.outside_all_rewrites():
          learning_rate = learning_rate_schedule(
              step=tf.train.get_global_step())
          tf.summary.scalar("learning_rate", learning_rate)
      else:
        learning_rate = learning_rate_schedule

      if isinstance(variable_filter, str):
        pattern = re.compile(variable_filter)
        variable_filter_fn = lambda v: pattern.search(v.name)
      elif variable_filter is None:
        variable_filter_fn = lambda v: True
      elif callable(variable_filter):
        variable_filter_fn = variable_filter
      else:
        raise ValueError(
            "variable_filter must be None, a string, or a callable function")
      trainable_vars = [
          v for v in graph.trainable_variables if variable_filter_fn(v)]
      trainable_var_grads = [
          g for g, v in zip(var_grads, graph.trainable_variables)
          if variable_filter_fn(v)]
      if len(trainable_vars) != len(graph.trainable_variables):
        tf.logging.info("Variables being trained:")
        tf.logging.info([v.name for v in trainable_vars])
        tf.logging.info("Variables not being trained:")
        tf.logging.info([v.name for v in graph.trainable_variables
                         if not variable_filter_fn(v)])

      update_ops = optimizer(learning_rate=learning_rate).apply_grads(
          trainable_var_grads, trainable_vars
      )

      lowering = mtf.Lowering(
          graph, {mesh: mesh_impl},
          autostack=autostack,
          log_file=model_info_file)

      tf_loss = lowering.export_to_tf_tensor(loss)
      tf_loss = tf.cast(tf_loss, tf.float32)
      if not use_tpu:
        tf_loss = tf.Print(tf_loss, [tf_loss, tf.train.get_global_step()],
                           "step, tf_loss")

      tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
      tf_update_ops.append(tf.assign_add(global_step, 1))
      train_op = tf.group(tf_update_ops)

      if hasattr(transformer_model, "initialize"):
        with mtf.utils.outside_all_rewrites():
          transformer_model.initialize()

      if tpu_summaries:
        # has to be outside of
        # with mtf.utils.outside_all_rewrites()
        host_call = mtf.utils.create_host_call(model_dir)
        mtf.utils.remove_summaries()
      else:
        host_call = None

      with mtf.utils.outside_all_rewrites():

        if init_checkpoint:
          ckpt_vars = {v for v, _ in tf.train.list_variables(init_checkpoint)}
          global_vars = {v.op.name for v in tf.global_variables()}
          restore_vars = ckpt_vars.intersection(global_vars)
          tf.logging.info("Initializing variables from %s:", init_checkpoint)
          tf.logging.debug("\n".join(sorted(restore_vars)))
          tf.logging.info("Variables in %s but not in graph:", init_checkpoint)
          tf.logging.info("\n".join(sorted(ckpt_vars - global_vars)))
          tf.logging.info("Variables in graph but not in %s:", init_checkpoint)
          tf.logging.info("\n".join(sorted(global_vars - ckpt_vars)))
          tf.train.init_from_checkpoint(
              init_checkpoint, {v: v for v in restore_vars}
          )

        # Copy master variables to slices. Must be called first.
        restore_hook = mtf.MtfRestoreHook(lowering)
        saver = tf.train.Saver(
            tf.global_variables(),
            sharded=True,
            max_to_keep=keep_checkpoint_max,
            keep_checkpoint_every_n_hours=2,
            defer_build=False,
            save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        saver_hook = tf.train.CheckpointSaverHook(
            model_dir,
            save_steps=save_checkpoints_steps,
            saver=saver,
            listeners=[saver_listener])
        gin_config_saver_hook = gin.tf.GinConfigSaverHook(
            model_dir, summarize_config=True, include_step_in_filename=False)

        if use_tpu:
          return tpu_estimator.TPUEstimatorSpec(
              mode=tf.estimator.ModeKeys.TRAIN,
              loss=tf_loss,
              train_op=train_op,
              host_call=host_call,
              training_hooks=[
                  restore_hook,
                  saver_hook,
                  gin_config_saver_hook,
              ])
        else:
          return tf.estimator.EstimatorSpec(
              tf.estimator.ModeKeys.TRAIN,
              loss=tf_loss,
              train_op=train_op,
              training_chief_hooks=[
                  restore_hook,
                  saver_hook,
                  gin_config_saver_hook,
              ])
    elif mode == tf.estimator.ModeKeys.EVAL:
      # perplexity eval
      logits, loss = logits_and_loss(mtf_features)
      # compute cross-entropy while still on TPU to avoid having to outfeed the
      # logits, which might be big.
      logits = mtf.cast(logits, tf.float32)
      vocab_dim = logits.shape.dims[-1]
      targets = mtf_features["targets"]
      cross_entropy = mtf.layers.softmax_cross_entropy_with_logits(
          logits, targets, vocab_dim)
      anon_cross_entropy = mtf.anonymize(cross_entropy)
      predictions = mtf.cast(mtf.argmax(logits, vocab_dim), targets.dtype)
      anon_predictions = mtf.anonymize(predictions)
      anon_targets = mtf.anonymize(targets)
      # 0=padding and negative targets are a hack to indicate no loss
      anon_weights = mtf.cast(mtf.greater(anon_targets, 0), tf.float32)
      if model_type == "delimited_lm":
        anon_weights *= mtf.cast(
            mtf.logical_not(transformer.delimited_lm_inputs_mask(anon_targets)),
            dtype=tf.float32)

      lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)
      tf_loss = tf.cast(lowering.export_to_tf_tensor(loss), tf.float32)
      tf_loss = tf.cast(tf_loss, tf.float32)
      tf_predictions = lowering.export_to_tf_tensor(anon_predictions)
      tf_cross_entropy = lowering.export_to_tf_tensor(anon_cross_entropy)

      def simple_metrics(xent, predictions, labels, weights):
        """Simple metrics for teacher-forced eval."""
        token_correct = tf.cast(
            tf.equal(predictions, labels), tf.float32) * weights
        sequence_correct = tf.cast(
            tf.equal(tf.reduce_sum(token_correct, -1),
                     tf.reduce_sum(weights, -1)),
            tf.float32)
        sequence_weights = tf.cast(
            tf.not_equal(tf.reduce_sum(weights, -1), 0),
            tf.float32)
        # the purpose of "mean_label" is as a checksum to ensure that
        # models were evaluated on the same data.
        return {"neg_log_perplexity": tf.metrics.mean(-xent, weights),
                "token_accuracy": tf.metrics.mean(token_correct, weights),
                "sequence_accuracy": tf.metrics.mean(
                    sequence_correct, sequence_weights),
                "mean_label": tf.metrics.mean(
                    tf.cast(labels, tf.float32), weights),
                "num_eval_tokens": metric_sum(weights, name="num_eval_tokens"),
                "max_targets_length": metric_max(tf.reduce_sum(
                    weights, axis=-1), name="max_targets_length"),
               }

      labels = lowering.export_to_tf_tensor(anon_targets)
      weights = lowering.export_to_tf_tensor(anon_weights)
      eval_metrics = (simple_metrics, [
          tf_cross_entropy, tf_predictions, labels, weights])
      with mtf.utils.outside_all_rewrites():
        restore_hook = mtf.MtfRestoreHook(lowering)
      return tpu_estimator.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          evaluation_hooks=[restore_hook],
          loss=tf_loss,
          eval_metrics=eval_metrics)

  return my_model_fn
