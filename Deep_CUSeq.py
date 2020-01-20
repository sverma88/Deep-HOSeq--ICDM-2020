import numpy as np
import scipy.io as sio
import tensorflow as tf
import os
import h5py

from ops import *
from utils import *


class Deep_CUSeq():

    def __init__(self, sess, data_dir, batch_size, hidden_v, hidden_a, hidden_t, LSTM_hid_t, text_out,
                 LSTM_hid_v, LSTM_hid_a, Seq_count, Conv_filt):
        """

        Args:
          sess: TensorFlow Session
          batch_size: The size of batch. Should be specified before training.
          data_dir: path to the director of the dataset
        """
        self.sess       = sess
        self.y_dim      = 1

        self.data_dir   = data_dir
        self.batch_size = batch_size

        self.hv 	= hidden_v
        self.ha 	= hidden_a
        self.ht 	= hidden_t
        self.t_out 	= text_out

        self.LSTM_hid_v = LSTM_hid_v
        self.LSTM_hid_a = LSTM_hid_a
        self.LSTM_hid_t = LSTM_hid_t
        self.Conv_filt  = Conv_filt

        self.sequence_count = Seq_count

        # batch normalization for Common Parts
        self.v_bn = batch_norm(name='video_subnet')
        self.a_bn = batch_norm(name='audio_subnet')
        self.t_bn = batch_norm(name='text_subnet')
        self.t1_bn = batch_norm(name='text_subnet1')


        self.Cbn_u1 = batch_norm(name="unique_conv1")
        self.Cbn_u2 = batch_norm(name="unique_conv1")
        self.Cbn_c1 = batch_norm(name="common_conv1")
        self.Cbn_c2 = batch_norm(name="common_conv2")

        self.build_model()

    def build_model(self):

        audio_data, video_data, text_data, _, _, _, _, _, _, _, _, _ = self.load_data()

        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        video_dim = video_data.shape
        audio_dim = audio_data.shape
        text_dim = text_data.shape

        self.video_inputs   = tf.placeholder(tf.float32, [None, video_dim[1], video_dim[2]], name='video_data')
        self.learning_rate  = tf.placeholder(tf.float32, [], name='learning_rate')
        self.audio_inputs   = tf.placeholder(tf.float32, [None, audio_dim[1], audio_dim[2]], name='audio_data')
        self.text_inputs    = tf.placeholder(tf.float32, [None, text_dim[1], text_dim[2]], name='text_data')
        self.drop_ratio     = tf.placeholder(tf.float32, [], name='drop_ratio')

        #### fc layers from subnetworks
        self.Unique_fc  = self.Unique_train(self.video_inputs, self.audio_inputs, self.text_inputs,
                                               self.hv, self.ha, self.ht, self.Conv_filt,
                                            dropl=self.drop_ratio, reuse=False)

        self.Unique_fc_ = self.Unique_test(self.video_inputs, self.audio_inputs, self.text_inputs,
                                               self.hv, self.ha, self.ht, self.Conv_filt)

        self.Common_fc  = self.Common_train(self.video_inputs, self.audio_inputs, self.text_inputs, self.hv, self.ha,
                                          self.ht, self.LSTM_hid_t, self.t_out, self.LSTM_hid_v, self.LSTM_hid_a,
                                          self.Conv_filt, dropl = self.drop_ratio, reuse=False)

        self.Common_fc_ = self.Common_test(self.video_inputs, self.audio_inputs, self.text_inputs,self.hv, self.ha,
                                           self.ht, self.LSTM_hid_t, self.t_out, self.LSTM_hid_v, self.LSTM_hid_a, self.Conv_filt)


        #### Perform the fusion
        self.logits     = self.Fusion_train(self.Unique_fc, self.Common_fc, reuse=False)
        self.logits_    = self.Fusion_test(self.Unique_fc_, self.Common_fc_)

        ## trainining function
        self.cnn_loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=self.y, predictions=self.logits))
        self.Accuracy = tf.reduce_sum(tf.abs(tf.subtract(self.y, self.logits_)))
        self.Pred = self.logits_
        self.diff = tf.abs(tf.subtract(self.y, self.logits_))

        self.saver = tf.train.Saver()

    #### Train the Network

    def train(self, config):

        if (config.Optimizer == "Adam"):
            cnn_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=config.beta1) \
                .minimize(self.cnn_loss)
        elif (config.Optimizer == "RMS"):
            cnn_optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cnn_loss)
        else:
            cnn_optim = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cnn_loss)

        tf.global_variables_initializer().run()

        # Loading data
        Au_trdat, Vi_trdat, Tx_trdat, Au_trlab, Au_tsdat, Vi_tsdat, \
        Tx_tsdat, Au_tslab, Au_vdat, Vi_vdat, Tx_vdat, Au_vlab = self.load_data()

        train_batches = Au_trdat.shape[0] // self.batch_size
        test_batches = Au_tsdat.shape[0] // self.batch_size
        val_batches = Au_vdat.shape[0] // self.batch_size

        left_index_test = Au_tsdat.shape[0] - (test_batches * config.batch_size)
        left_index_train = Au_trdat.shape[0] - (train_batches * config.batch_size)
        left_index_val = Au_vdat.shape[0] - (val_batches * config.batch_size)

        dropout_list = np.arange(0.0, 0.8, 0.05)

        for drop1 in dropout_list:

            tf.global_variables_initializer().run()
            seed = 20

            print("dropout ratio --->", drop1)

            #### Start training the model
            lr = config.learning_rate

            for epoch in range(config.epoch):
                seed += 1
                Val_Loss    = 0.0
                Test_loss   = 0.0

                if np.mod(epoch + 1, 5) == 0:
                    lr = lr - lr * 0.1

                random_index = np.random.RandomState(seed=seed).permutation(Au_trdat.shape[0])
                train_data_au = Au_trdat[random_index]
                train_data_vi = Vi_trdat[random_index]
                train_data_tx = Tx_trdat[random_index]
                train_lab_au = Au_trlab[random_index]

                for idx in range(train_batches):
                    batch_au = train_data_au[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_vi = train_data_vi[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_tx = train_data_tx[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = train_lab_au[idx * config.batch_size:(idx + 1) * config.batch_size]

                    _ = self.sess.run([cnn_optim],
                                      feed_dict={
                                          self.audio_inputs: batch_au,
                                          self.video_inputs: batch_vi,
                                          self.text_inputs: batch_tx,
                                          self.y: batch_labels,
                                          self.learning_rate: lr,
                                          self.drop_ratio: drop1
                                      })

                ##### Printing Loss on each epoch to monitor convergence
                ##### Apply Early stoping procedure to report results

                print(" Epoch:--->", epoch)


                random_index = np.random.permutation(Au_vdat.shape[0])
                VAL_data_au = Au_vdat[random_index]
                VAL_data_vi = Vi_vdat[random_index]
                VAL_data_tx = Tx_vdat[random_index]
                VAL_lab_au = Au_vlab[random_index]

                for idx in range(val_batches):
                    batch_au = VAL_data_au[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_vi = VAL_data_vi[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_tx = VAL_data_tx[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = VAL_lab_au[idx * config.batch_size:(idx + 1) * config.batch_size]

                    Val_Loss += self.Accuracy.eval({
                        self.audio_inputs: batch_au,
                        self.video_inputs: batch_vi,
                        self.text_inputs: batch_tx,
                        self.y: batch_labels
                    })

                batch_au = train_data_au[-left_index_val:]
                batch_vi = train_data_vi[-left_index_val:]
                batch_tx = train_data_tx[-left_index_val:]
                batch_labels = train_lab_au[-left_index_val:]

                Val_Loss += self.Accuracy.eval({
                    self.audio_inputs: batch_au,
                    self.video_inputs: batch_vi,
                    self.text_inputs: batch_tx,
                    self.y: batch_labels
                })

                Val_MAE = Val_Loss / (Au_vdat.shape[0])

                ##### Calculate Test Loss

                for idx in range(test_batches):
                    batch_au = Au_tsdat[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_vi = Vi_tsdat[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_tx = Tx_tsdat[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = Au_tslab[idx * config.batch_size:(idx + 1) * config.batch_size]

                    Test_loss += self.Accuracy.eval({
                        self.audio_inputs: batch_au,
                        self.video_inputs: batch_vi,
                        self.text_inputs: batch_tx,
                        self.y: batch_labels

                    })


                ### Do it for the left exampels which does not account in batches
                batch_au = Au_tsdat[-left_index_test:]
                batch_vi = Vi_tsdat[-left_index_test:]
                batch_tx = Tx_tsdat[-left_index_test:]
                batch_labels = Au_tslab[-left_index_test:]

                Test_loss += self.Accuracy.eval({
                    self.audio_inputs: batch_au,
                    self.video_inputs: batch_vi,
                    self.text_inputs: batch_tx,
                    self.y: batch_labels
                })

                Test_MAE = Test_loss / Au_tsdat.shape[0]

                print("******* Results ************ ")

                print("Validation MAE ---->", Val_MAE)
                print("Test MAE ---->", Test_MAE)

    def Unique_train(self, data_v, data_a, data_t, unq_hid_t, unq_hid_a, unq_hid_v, conv_filt, dropl, reuse=False):
        with tf.variable_scope("Unique_network") as scope:
            if reuse:
                scope.reuse_variables()

            # processing text feature with linear layers
            ht_0 = tf.layers.dense(data_t, 150, tf.nn.elu, name='fc1_text')
            ht_1 = tf.layers.dropout(ht_0, dropl)
            ht_2 = tf.layers.dense(ht_1, unq_hid_t, tf.nn.elu, name='fc3_text')

            # processing audio feature with linear layers
            ha_0 = tf.layers.dense(data_a, unq_hid_a, tf.nn.elu, name='fc1_audio')

            # processing visual feature with linear layers
            hv_0 = tf.layers.dense(data_v, unq_hid_v, tf.nn.elu, name='fc1_visual')

            out_ta = tf.einsum('ijk,ijl->ijkl', ht_2, ha_0)
            out_tav = tf.einsum('ijkl,ijm->ijklm', out_ta, hv_0)

            # adding channels
            out_tav = tf.expand_dims(out_tav, [-1])

            ##### Convolution Layer 1
            Conv_L1_list = []
            for i in range(self.sequence_count):
                # process tensor of each sequence independently
                tensor_Seq_i = out_tav[:, i, :, :, :, :]
                conv_out_i = conv3d(tensor_Seq_i, tensor_Seq_i.shape[-1], conv_filt, name=('conv1_' + str(i)))
                pool_conv_i = tf.nn.max_pool3d(conv_out_i, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                                               padding='SAME')
                Conv_L1_list.append(tf.expand_dims(pool_conv_i, 1))

            #### Combine the list
            Conv_L1_Output = tf.concat(Conv_L1_list, 1)
            Conv_L1_Output = tf.nn.elu(self.Cbn_u1(Conv_L1_Output))

            ### Flatten the output of eacch Sequence
            Conv_L1_flatten = []
            for i in range(self.sequence_count):
                Seq_i = Conv_L1_Output[:, i, :, :, :, :]
                Conv_L1_flatten.append(tf.expand_dims(tf.layers.flatten(Seq_i), 1))

            Conv_L1_flatten = tf.concat(Conv_L1_flatten, 1)
            Conv_L1_flatten = tf.layers.dropout(Conv_L1_flatten, dropl)

            fc1 = tf.reduce_mean(tf.layers.dense(Conv_L1_flatten, 8, tf.nn.elu, name='fc1'), axis=1)
            # fc1 = tf.layers.dropout(fc1, dropl)
            # h0 = tf.nn.elu(linear(fc1, 30, 'h0_lin'))

            return fc1

    def Unique_test(self, data_v, data_a, data_t, unq_hid_t, unq_hid_a, unq_hid_v, conv_filt):
        with tf.variable_scope("Unique_network") as scope:
            scope.reuse_variables()

            # processing text feature with linear layers
            ht_0 = tf.layers.dense(data_t, 150, tf.nn.elu, name='fc1_text')
            ht_1 = tf.layers.dropout(ht_0, 0.0)
            ht_2 = tf.layers.dense(ht_1, unq_hid_t, tf.nn.elu, name='fc3_text')

            # processing audio feature with linear layers
            ha_0 = tf.layers.dense(data_a, unq_hid_a, tf.nn.elu, name='fc1_audio')

            # processing visual feature with linear layers
            hv_0 = tf.layers.dense(data_v, unq_hid_v, tf.nn.elu, name='fc1_visual')


            out_ta = tf.einsum('ijk,ijl->ijkl', ht_2, ha_0)
            out_tav = tf.einsum('ijkl,ijm->ijklm', out_ta, hv_0)

            # adding channels
            out_tav = tf.expand_dims(out_tav, [-1])


            ##### Convolution Layer 1
            Conv_L1_list = []
            for i in range(self.sequence_count):
                # process tensor of each sequence independently
                tensor_Seq_i = out_tav[:, i, :, :, :, :]
                conv_out_i = conv3d(tensor_Seq_i, tensor_Seq_i.shape[-1], conv_filt, name=('conv1_' + str(i)))
                pool_conv_i = tf.nn.max_pool3d(conv_out_i, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                                               padding='SAME')
                Conv_L1_list.append(tf.expand_dims(pool_conv_i, 1))

            Conv_L1_Output = tf.concat(Conv_L1_list, 1)
            Conv_L1_Output = tf.nn.elu(self.Cbn_u1(Conv_L1_Output, train=False))


            ### Flatten the output of eacch Sequence
            Conv_L1_flatten = []
            for i in range(self.sequence_count):
                Seq_i = Conv_L1_Output[:, i, :, :, :, :]
                Conv_L1_flatten.append(tf.expand_dims(tf.layers.flatten(Seq_i), 1))

            Conv_L1_flatten = tf.concat(Conv_L1_flatten, 1)

            fc1 = tf.reduce_mean(tf.layers.dense(Conv_L1_flatten, 8, tf.nn.elu, name='fc1'), axis=1)
            # h0 = tf.nn.elu(linear(fc1, 30, 'h0_lin'))

            return fc1

    def Common_train(self, data_v, data_a, data_t, hidden_v, hidden_a, hidden_t, LSTM_hid_t, text_out,
                     LSTM_hid_v, LSTM_hid_a, conv_filt, dropl, reuse=False):
        with tf.variable_scope("Common_network") as scope:
            if reuse:
                scope.reuse_variables()

            #### Text Subnet LSTM
            LSTM_cell_t = tf.contrib.rnn.BasicLSTMCell(LSTM_hid_t, name='text_lstm')
            LSTM_cell_t = tf.contrib.rnn.DropoutWrapper(LSTM_cell_t, output_keep_prob=1.0 - dropl)

            _, state_t = tf.nn.dynamic_rnn(LSTM_cell_t, data_t, dtype=tf.float32)

            h0t1 = tf.nn.elu(self.t_bn(linear(state_t.h, text_out, 'h0_t1')))
            h0t1 = tf.layers.dropout(h0t1, dropl)
            h0t2 = tf.nn.elu(self.t1_bn(linear(h0t1, hidden_t, 'h0_t2')))

            #### visual Subnet LSTM
            LSTM_cell_v = tf.contrib.rnn.BasicLSTMCell(LSTM_hid_v, name='visual_lstm')
            LSTM_cell_v = tf.contrib.rnn.DropoutWrapper(LSTM_cell_v, output_keep_prob=1.0 - dropl)

            _, state_v = tf.nn.dynamic_rnn(LSTM_cell_v, data_v, dtype=tf.float32)
            h0v = tf.nn.elu(self.v_bn(linear(state_v.h, hidden_v, 'h0_v')))


            #### audio Subnet LSTM
            LSTM_cell_a = tf.contrib.rnn.BasicLSTMCell(LSTM_hid_a, name='audio_lstm')
            LSTM_cell_a = tf.contrib.rnn.DropoutWrapper(LSTM_cell_a, output_keep_prob=1.0 - dropl)

            _, state_a = tf.nn.dynamic_rnn(LSTM_cell_a, data_a, dtype=tf.float32)
            h0a = tf.nn.elu(self.a_bn(linear(state_a.h, hidden_a, 'h0_a')))


            ### now combine them in a tensor
            TF_tv = tf.einsum('ij,ik->ijk', h0t2, h0v)
            TF_avt = tf.einsum('ijk,il->ijkl', TF_tv, h0a)

            TF_avt = tf.expand_dims(TF_avt, [-1])

            conv0 = self.Cbn_c1(conv3d(TF_avt, TF_avt.shape[-1], conv_filt, name='conv_0'))
            conv0 = tf.nn.max_pool3d(conv0,ksize=[1,3,3,3,1],strides=[1,2,2,2,1], padding='SAME')
            conv0 = tf.nn.elu(conv0)

            f_conv = tf.layers.flatten(conv0)

            h1 = tf.layers.dropout(f_conv, dropl)
            h2 = tf.nn.elu(linear(h1, 8, 'h2_lin'))
            # h2 = tf.layers.dropout(h2, dropl)
            # h3 = tf.nn.elu(linear(h2, 30, 'h3_lin'))

            return h2

    def Common_test(self, data_v, data_a, data_t, hidden_v, hidden_a, hidden_t, LSTM_hid_t, text_out,
                    LSTM_hid_v, LSTM_hid_a, conv_filt):
        with tf.variable_scope("Common_network") as scope:
            scope.reuse_variables()

            #### Text Subnet LSTM
            LSTM_cell_t = tf.contrib.rnn.BasicLSTMCell(LSTM_hid_t, name='text_lstm')
            LSTM_cell_t = tf.contrib.rnn.DropoutWrapper(LSTM_cell_t, output_keep_prob=1.0)

            _, state_t = tf.nn.dynamic_rnn(LSTM_cell_t, data_t, dtype=tf.float32)
            h0t1 = tf.nn.elu(self.t_bn(linear(state_t.h, text_out, 'h0_t1'),train=False))
            h0t1 = tf.layers.dropout(h0t1, 0)
            h0t2 = tf.nn.elu(self.t1_bn(linear(h0t1, hidden_t, 'h0_t2'),train=False))

            #### visual Subnet LSTM
            LSTM_cell_v = tf.contrib.rnn.BasicLSTMCell(LSTM_hid_v, name='visual_lstm')
            LSTM_cell_v = tf.contrib.rnn.DropoutWrapper(LSTM_cell_v, output_keep_prob=1.0)

            _, state_v = tf.nn.dynamic_rnn(LSTM_cell_v, data_v, dtype=tf.float32)
            h0v = tf.nn.elu(self.v_bn(linear(state_v.h, hidden_v, 'h0_v'),train=False))

            #### audio Subnet LSTM
            LSTM_cell_a = tf.contrib.rnn.BasicLSTMCell(LSTM_hid_a, name='audio_lstm')
            LSTM_cell_a = tf.contrib.rnn.DropoutWrapper(LSTM_cell_a, output_keep_prob=1.0)

            _, state_a = tf.nn.dynamic_rnn(LSTM_cell_a, data_a, dtype=tf.float32)
            h0a = tf.nn.elu(self.a_bn(linear(state_a.h, hidden_a, 'h0_a'),train=False))


            ### now combine them in a tensor

            TF_tv = tf.einsum('ij,ik->ijk', h0t2, h0v)
            TF_avt = tf.einsum('ijk,il->ijkl', TF_tv, h0a)
            TF_avt = tf.expand_dims(TF_avt, [-1])


            conv0 = self.Cbn_c1(conv3d(TF_avt, TF_avt.shape[-1], conv_filt, name='conv_0'), train=False)

            conv0 = tf.nn.max_pool3d(conv0,ksize=[1,3,3,3,1],strides=[1,2,2,2,1], padding='SAME')
            conv0 = tf.nn.elu(conv0)

            f_conv = tf.layers.flatten(conv0)
            h1 = tf.layers.dropout(f_conv, 0)

            h2 = tf.nn.elu(linear(h1, 8, 'h2_lin'))
            # h2 = tf.layers.dropout(h2, dropl)
            # h3 = tf.nn.elu(linear(h2, 30, 'h3_lin'))

            return h2

    def Fusion_train(self, unique_fc, common_fc, reuse=False):
        with tf.variable_scope("Fusion_network") as scope:
            if reuse:
                scope.reuse_variables()

            Fused_layer     =  tf.reduce_mean((unique_fc,common_fc), axis=0)
            h0_lin          =  tf.nn.relu(linear(Fused_layer, 5, 'h0_lin'))
            # h0_lin      = Fused_layer
            h1              =  linear(h0_lin, 1, 'h1_lin')

            return h1

    def Fusion_test(self, unique_fc, common_fc):
        with tf.variable_scope("Fusion_network") as scope:
            scope.reuse_variables()

            Fused_layer     = tf.reduce_mean((unique_fc, common_fc), axis=0)
            h0_lin          = tf.nn.relu(linear(Fused_layer, 5, 'h0_lin'))
            # h0_lin    = Fused_layer
            h1              = linear(h0_lin, 1, 'h1_lin')

            return h1


    def load_data(self):
        path_base = self.data_dir

        path_a_train = path_base + 'audio_train.h5'
        path_v_train = path_base + 'video_train.h5'
        path_t_train = path_base + 'text_train_emb.h5'
        path_y_train = path_base + 'y_train.h5'

        path_a_test = path_base + 'audio_test.h5'
        path_v_test = path_base + 'video_test.h5'
        path_t_test = path_base + 'text_test_emb.h5'
        path_y_test = path_base + 'y_test.h5'

        path_a_valid = path_base + 'audio_valid.h5'
        path_v_valid = path_base + 'video_valid.h5'
        path_t_valid = path_base + 'text_valid_emb.h5'
        path_y_valid = path_base + 'y_valid.h5'

        # load the H5 files
        # ****************   train   *****************
        with h5py.File(path_a_train, 'r') as h5:
            content = h5['d1']
            a_train = np.ones(content.shape)
            content.read_direct(a_train)
        with h5py.File(path_v_train, 'r') as h5:
            content = h5['d1']
            v_train = np.ones(content.shape)
            content.read_direct(v_train)
        with h5py.File(path_t_train, 'r') as h5:
            content = h5['d1']
            t_train = np.ones(content.shape)
            content.read_direct(t_train)
        # get the y_train
        with h5py.File(path_y_train, 'r') as h5:
            content = h5['d1']
            y_train = np.ones(content.shape)
            content.read_direct(y_train)

        with h5py.File(path_a_test, 'r') as h5:
            content = h5['d1']
            a_test = np.ones(content.shape)
            content.read_direct(a_test)
        with h5py.File(path_v_test, 'r') as h5:
            content = h5['d1']
            v_test = np.ones(content.shape)
            content.read_direct(v_test)
        with h5py.File(path_t_test, 'r') as h5:
            content = h5['d1']
            t_test = np.ones(content.shape)
            content.read_direct(t_test)
            # get the y_test
        with h5py.File(path_y_test, 'r') as h5:
            content = h5['d1']
            y_test = np.ones(content.shape)
            content.read_direct(y_test)

        with h5py.File(path_a_valid, 'r') as h5:
            content = h5['d1']
            a_valid = np.ones(content.shape)
            content.read_direct(a_valid)
        with h5py.File(path_v_valid, 'r') as h5:
            content = h5['d1']
            v_valid = np.ones(content.shape)
            content.read_direct(v_valid)
        with h5py.File(path_t_valid, 'r') as h5:
            content = h5['d1']
            t_valid = np.ones(content.shape)
            content.read_direct(t_valid)
        # get the y_valid
        with h5py.File(path_y_valid, 'r') as h5:
            content = h5['d1']
            y_valid = np.ones(content.shape)
            content.read_direct(y_valid)

        # rip of the NaN
        a_train[a_train != a_train] = 0
        v_train[v_train != v_train] = 0

        a_test[a_test != a_test] = 0
        v_test[v_test != v_test] = 0

        a_valid[a_valid != a_valid] = 0
        v_valid[v_valid != v_valid] = 0

        y_train = np.expand_dims(y_train, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
        y_valid = np.expand_dims(y_valid, axis=1)

        return a_train, v_train, t_train, y_train, a_test, v_test, t_test, y_test, a_valid, v_valid, t_valid, y_valid
