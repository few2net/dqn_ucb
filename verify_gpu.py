from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
