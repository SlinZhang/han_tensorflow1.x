import tensorflow as tf
import config


def attention_layer(input, attention_size, level="word"):
    if level == "word":
        hidden_size = input.shape[-1].value  # BI-LSTM output size
        print(hidden_size)
        w = tf.Variable(tf.random_normal((hidden_size, attention_size)))
        b = tf.Variable(tf.random_normal((attention_size,)))
        u = tf.Variable(tf.random_normal((attention_size,)))
        out = tf.tanh(tf.add(tf.tensordot(input, w, axes=1), b))  # (batch * max_sentences,max_word,attention_size)
        out = tf.reshape(out, (-1, config.max_s, config.max_w, attention_size))  # (batch,max_sentences,max_words,attention_size)
        alpha = tf.nn.softmax(tf.tensordot(out, u, axes=1)) # (batch,max_sentences,max_word)
        alpha = tf.expand_dims(alpha, -1)
        result = out * alpha
        result = tf.reduce_sum(result, axis=2)  # (batch,max_sentences,attention_size)
        return result
    elif level == "sentence":
        hidden_size = input.shape[-1].value
        w = tf.Variable(tf.random_normal((hidden_size, attention_size)))
        b = tf.Variable(tf.random_normal((attention_size,)))
        u = tf.Variable(tf.random_normal((attention_size,)))
        out = tf.tanh(tf.add(tf.tensordot(input, w, axes=1), b))
        alpha = tf.nn.softmax(tf.tensordot(out, u, axes=1))
        output = tf.reduce_sum(out * tf.expand_dims(alpha, -1), axis=1)
        return output
    else:
        print("you must select 'word' or 'sentence' ")
