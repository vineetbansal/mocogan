"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import tensorflow as tf
import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):
    def __init__(self, log_dir, suffix=None):
        #self.writer = tf.summary.FileWriter(log_dir, filename_suffix=suffix)
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value.item(), step)
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        #self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        with self.writer.as_default():
            tf.summary.image(name='%s_images' % (tag,), data=images, step=step)

    def video_summary(self, tag, videos, step):

        sh = list(videos.shape)
        sh[-1] = 1

        separator = np.zeros(sh, dtype=videos.dtype)
        videos = np.concatenate([videos, separator], axis=-1)

        with self.writer.as_default():
            for i, vid in enumerate(videos):

                v = vid.transpose(1, 2, 3, 0)
                v = [np.squeeze(f) for f in np.split(v, v.shape[0], axis=0)]

                # create one looong image with everything stacked horizontally
                # img = np.concatenate(v, axis=1)[:, :-1, :]
                img = np.stack(v)

                tf.summary.image(name='%s/%d' % (tag, i), data=img, step=step)
