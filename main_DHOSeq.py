import os
import scipy.misc
import numpy as np
import tensorflow as tf


from Deep_HOSeq import Deep_HOSeq
from utils import pp, show_all_variables


flags = tf.app.flags
flags.DEFINE_integer("epoch", 300, "Epoch to train [25]")
flags.DEFINE_integer("hidden_a", 5, "Dimensions in audio")
flags.DEFINE_integer("hidden_v", 5, "Dimensions in video")
flags.DEFINE_integer("hidden_t", 5, "Dimensions in text")
flags.DEFINE_integer("Sequence_Count", 20, "Sequence Count")
flags.DEFINE_integer("LSTM_hid_v", 5, "Dimensions in visual")
flags.DEFINE_integer("LSTM_hid_a", 10, "Dimensions in acoustic")
flags.DEFINE_integer("LSTM_hid_t", 128, "Dimensions in text")
flags.DEFINE_integer("text_out", 64, "Dimensions in text_out")
flags.DEFINE_integer("Conv_filt", 1, "Dimensions in text_out")
flags.DEFINE_float("learning_rate", 0.006, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_float("momentum", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 256, "The size of batch images [64]")
flags.DEFINE_string("data_dir", "XXXXXXX","directory of the data")
flags.DEFINE_string("Optimizer", "Adam","Adam, Grad, or Momentum")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:

        deep_hoseq = Deep_HOSeq(
            sess,
            data_dir    = FLAGS.data_dir,
            batch_size  = FLAGS.batch_size,

            hidden_v    = FLAGS.hidden_v,
            hidden_a    = FLAGS.hidden_a,
            hidden_t    = FLAGS.hidden_t,
            LSTM_hid_t  = FLAGS.LSTM_hid_t,
            text_out    = FLAGS.text_out,
            LSTM_hid_v  = FLAGS.LSTM_hid_v,
            LSTM_hid_a  = FLAGS.LSTM_hid_a,
            Seq_count   = FLAGS.Sequence_Count,
            Conv_filt   = FLAGS.Conv_filt

        )

        show_all_variables()

        deep_hoseq.train(FLAGS)



if __name__ == '__main__':
    tf.app.run()
