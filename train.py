import os
import tensorflow as tf
import pickle
import config
import numpy as np
from attention import attention_layer

if not os.path.exists(config.model_save_path): os.mkdir(config.model_save_path)

with open("train.pkl", "rb")as f:
    train_data = pickle.load(f)
for i in train_data["length"]:
    print(len(i))
train_data["text"] = np.array(train_data["text"])
train_data["label"] = np.array(train_data["label"])
train_data["length"] = np.array(train_data["length"])
print(np.array(train_data["length"]))

# data = {"a": np.array([1, 2, 3, 4]),
#         "b": np.array([[22, 2], [2, 2], [3, 6], [4, 8]]),
#         "c": np.array([[[2, 4], [5, 7]], [[7, 8], [9, 4]], [[5, 0], [5, 1]], [[4, 6], [5, 8]]])}
# print(data["c"])

train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_dataset = train_dataset.batch(3)
train_dataset = train_dataset.shuffle(buffer_size=2000)
iterator = train_dataset.make_one_shot_iterator()
get_batch = iterator.get_next()

with tf.name_scope("input_layer"):
    inputs = tf.placeholder(tf.int32, (None, config.max_s, config.max_w))  # (batch,max_sentences,max_sentence_len)
    length = tf.placeholder(tf.int32, (None, config.max_s))
    labels = tf.placeholder(tf.int32, (None,))  # (batch,num_clasess)

with tf.name_scope("embeeding_layer"):
    inputs_reshape = tf.reshape(inputs, (-1, config.max_w))  # (batch*max_sentences , max_sentence_len)
    embeeding = tf.Variable(tf.random_uniform((config.max_vocab, config.embeeding_size), maxval=1, minval=-1))
    embeeding_input = tf.nn.embedding_lookup(embeeding,
                                             inputs_reshape)  # (batch*max_sentences , max_sentence_len,embeeding_size)

with tf.name_scope("words_encoder_layer"):
    fw_cell = tf.nn.rnn_cell.LSTMCell(config.hidden_size)
    bw_cell = tf.nn.rnn_cell.LSTMCell(config.hidden_size)
    outputs_1, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=embeeding_input, dtype=tf.float32)
    outputs_2 = tf.concat(outputs_1, axis=2)  # (batch*max_sentences , max_sentence_len,hidden_size * 2)
    outputs_3 = attention_layer(outputs_2, config.attention_size,
                                level="word")  # (batch,max_sentences,attention_size) 3 3 256

with tf.name_scope("sentences_encoder_layer"):
    fw_cell_1 = tf.nn.rnn_cell.LSTMCell(config.hidden_size, name="a")
    bw_cell_1 = tf.nn.rnn_cell.LSTMCell(config.hidden_size, name="h")
    outputs_4, state = tf.nn.bidirectional_dynamic_rnn(fw_cell_1, bw_cell_1, inputs=outputs_3, dtype=tf.float32)
    outputs_5 = tf.concat(outputs_4, axis=2)
    outputs_6 = attention_layer(outputs_5, attention_size=config.attention_size, level="sentence")

with tf.name_scope("full_connected_layer"):
    w_1 = tf.Variable(tf.random_normal((config.attention_size, config.num_classes)))
    b_1 = tf.Variable(tf.random_normal((config.num_classes,)))
    logits = tf.add(tf.matmul(outputs_6, w_1), b_1)

loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
train_op = tf.train.AdamOptimizer(config.lr).minimize(loss=loss)
#
# saver = tf.train.Saver()
#
# init = tf.initialize_all_variables()
# step = 0
# print("=============================================================")
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
step = 0
with tf.Session() as sess:
    sess.run(init_op)
    while True:

        try:
            batch_data = sess.run(get_batch)
            sess.run(train_op, feed_dict={inputs: batch_data["text"], labels: batch_data["label"]})

            if step % config.display_interval == 0:
                print(sess.run(loss, feed_dict={inputs: batch_data["text"], labels: batch_data["label"]}))
            if step % config.save_interval == 0:
                saver.save(sess, config.model_save_path, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            print("train done!")
            break
