# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
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

"""TED talk multilingual data set."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow_datasets.core import api_utils
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
Massively multilingual (60 language) data set derived from TED Talk transcripts.
Each record consists of parallel arrays of language and text. Missing
translations and missing portions of translations will be indicated with the
string "__NULL__".
"""

_CITATION = """\
@InProceedings{qi-EtAl:2018:N18-2,
  author    = {Qi, Ye  and  Sachan, Devendra  and  Felix, Matthieu  and  Padmanabhan, Sarguna  and  Neubig, Graham},
  title     = {When and Why Are Pre-Trained Word Embeddings Useful for Neural Machine Translation?},
  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
  month     = {June},
  year      = {2018},
  address   = {New Orleans, Louisiana},
  publisher = {Association for Computational Linguistics},
  pages     = {529--535},
  abstract  = {The performance of Neural Machine Translation (NMT) systems often suffers in low-resource scenarios where sufficiently large-scale parallel corpora cannot be obtained. Pre-trained word embeddings have proven to be invaluable for improving performance in natural language analysis tasks, which often suffer from paucity of data. However, their utility for NMT has not been extensively explored. In this work, we perform five sets of experiments that analyze when we can expect pre-trained word embeddings to help in NMT tasks. We show that such embeddings can be surprisingly effective in some cases -- providing gains of up to 20 BLEU points in the most favorable setting.},
  url       = {http://www.aclweb.org/anthology/N18-2084}
}
"""

_DATA_URL = "http://phontron.com/data/ted_talks.tar.gz"


class TedTranslateConfig(tfds.core.BuilderConfig):
  """BuilderConfig for TedTranslate."""

  @api_utils.disallow_positional_args
  def __init__(self, text_encoder_config=None, **kwargs):
    """BuilderConfig for TedTranslate.

    Args:
      text_encoder_config: `tfds.features.text.TextEncoderConfig`, configuration
        for the `tfds.features.text.TextEncoder` used for the `"text"` feature.
      **kwargs: keyword arguments forwarded to super.
    """
    super(TedTranslateConfig, self).__init__(**kwargs)
    self.text_encoder_config = (
        text_encoder_config or tfds.features.text.TextEncoderConfig())


class TedTranslate(tfds.core.GeneratorBasedBuilder):
  """TED talk multilingual data set."""

  BUILDER_CONFIGS = [
      TedTranslateConfig(
          name="plain_text",
          version="0.0.1",
          description="Plain text",
      )
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        # Parallel arrays of (language, text). If no translation exists for a
        # language, it will be excluded from the parallel arrays.
        features=tfds.features.SequenceDict({
            "language": tfds.features.Text(),
            "text": tfds.features.Text(),
        }),
        urls=["https://github.com/neulab/word-embeddings-for-nmt"],
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    dl_dir = dl_manager.download_and_extract(_DATA_URL)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=1,
            gen_kwargs={
                "data_file": os.path.join(dl_dir, "all_talks_train.tsv")
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            num_shards=1,
            gen_kwargs={"data_file": os.path.join(dl_dir,
                                                  "all_talks_dev.tsv")}),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            num_shards=1,
            gen_kwargs={
                "data_file": os.path.join(dl_dir, "all_talks_test.tsv")
            }),
    ]

  def _generate_examples(self, data_file):
    """This function returns the examples in the raw (text) form."""
    with tf.io.gfile.GFile(data_file) as f:
      languages = _parse_line(f.readline())
      for row in f:
        texts = _parse_line(row)

        # Gracefully handle empty lines.
        if not texts:
          continue

        assert len(languages) == len(
            texts), "Sizes do not match: %d languages vs %d translations" % (
                len(languages), len(texts))
        yield {
            "language": languages,
            "text": texts,
        }


def _parse_line(line):
  # The line will have a trailing new line. Ignore the first column as it is the
  # talk_name. We assume that the input does not contain escaped tabs.
  return line.rstrip("\n").split("\t")[1:]
