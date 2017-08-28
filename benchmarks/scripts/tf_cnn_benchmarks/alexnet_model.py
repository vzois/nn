# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Alexnet model configuration.

References:
  Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton
  ImageNet Classification with Deep Convolutional Neural Networks
  Advances in Neural Information Processing Systems. 2012
"""

import model


class AlexnetModel(model.Model):
  """Alexnet cnn model."""

  def __init__(self):
    super(AlexnetModel, self).__init__('alexnet', 224 + 3, 512, 0.005)

  def add_inference(self, cnn):
    # Note: VALID requires padding the images by 3 in width and height
    print cnn.conv(64, 11, 11, 4, 4, 'VALID')#55*55*9
    print cnn.mpool(3, 3, 2, 2)
    print cnn.conv(192, 5, 5)
    print cnn.mpool(3, 3, 2, 2)
    print cnn.conv(384, 3, 3)
    print cnn.conv(384, 3, 3)
    print cnn.conv(256, 3, 3)
    print cnn.mpool(3, 3, 2, 2)
    print cnn.reshape([-1, 256 * 6 * 6])
    print cnn.affine(4096)
    print cnn.dropout()
    print cnn.affine(4096)
    print cnn.dropout()
