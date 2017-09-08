"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests
from tensorflow.python.layers.pooling import MaxPooling2D

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)

view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

#print('Dataset Stats')
#print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
#print('Number of sentences: {}'.format(len(sentences)))
#print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

#print()
#print('English sentences {} to {}:'.format(*view_sentence_range))
#print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
#print()
#print('French sentences {} to {}:'.format(*view_sentence_range))
#print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function

    #source_id_text = [source_vocab_to_int.get(word) for word in source_text.split(' ')]
    #target_id_text = [target_vocab_to_int.get(word) for word in target_text.split(' ')]

    source_id_text = []
    target_id_text = []
    for sentence in source_text.split("\n"):
        out = [source_vocab_to_int[w] for w in sentence.split()]
        source_id_text.append(out)
    for sentence in target_text.split("\n"):
        out = [target_vocab_to_int[w] for w in sentence.split()]
        out.append(target_vocab_to_int['<EOS>'])
        target_id_text.append(out)
    return source_id_text, target_id_text

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper
import problem_unittests as tests

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
from tensorflow.python.layers.core import Dense

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    # TODO: Implement Function
    input = tf.placeholder(tf.int32, shape=(None,None), name='input')
    targets = tf.placeholder(tf.int32, shape=(None,None), name='target')
    learning_rate = tf.placeholder(tf.float32, shape=None, name='lr')
    keep_probability = tf.placeholder(tf.float32, shape=None, name='keep_prob')
    target_sequence_length = tf.placeholder(tf.float32, shape=None, name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.float32, shape=None, name='source_sequence_length')

    return input, targets, learning_rate, keep_probability, target_sequence_length,\
           max_target_sequence_length, source_sequence_length


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    return None

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_encoding_input(process_decoder_input)