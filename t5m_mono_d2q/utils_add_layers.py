# This modified code is made for text-to-text transfer transformers (T5)
#
# Substitiute the following function in '${pytho_libary_path}/mesh_tensorflow/utils.py'
#
# (1) Save the true and false logit
# (2) change the logit floating type to .16f

@gin.configurable
def save_scores(results, vocabulary,
                scores_filename=None, save_example_text=False):
  """Processes results from scoring examples and maybe saves them to disk.

  Args:
    results: list of dictionaries containing the results for each scored
        example.
    vocabulary: a function that that returns a tf.data.Dataset with examples
      containing the string field 'targets' and optionally the field 'inputs'
    scores_filename: a string (path of file to write scores to). If None, scores
        are returned but not written to disk.
    save_example_text: a boolean - If True, then the text for each example is
        also saved/returned.

  Returns:
    List of float scores, one score per example. If save_example_text is True,
    the text of the inputs/targets for each example are also returned.
  """
  if not results:
    raise ValueError("No examples were scored.")

  scores = [r["scores"] for r in results]
  tlogits = [r["true_logit"] for r in results]
  flogits = [r["false_logit"] for r in results]

  if scores_filename is not None:
    write_lines_to_file(["%.16f" % f for f in scores], scores_filename+".scores")
    write_lines_to_file(["%.16f" % f for f in flogits], scores_filename+".flogits")
    write_lines_to_file(["%.16f" % f for f in tlogits], scores_filename+".tlogits")

  if save_example_text:
    # Targets will always exist.
    targets = [r.get("targets_plaintext", r["targets"]) for r in results]
    targets = _maybe_decode_python(targets, targets_vocabulary(vocabulary))
    #if scores_filename is not None:
      #write_lines_to_file(targets, scores_filename+".targets")

    # Inputs may only exist for some tasks.
    if "inputs" in results[0]:
      inputs = [r.get("inputs_plaintext", r["inputs"]) for r in results]
      inputs = _maybe_decode_python(inputs, inputs_vocabulary(vocabulary))
      if scores_filename is not None:
        write_lines_to_file(inputs, scores_filename+".inputs")
      return scores, inputs, targets
    else:
      return scores, targets

  return scores
