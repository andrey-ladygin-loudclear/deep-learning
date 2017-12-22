import pickle
import numpy as np
import matplotlib.pyplot as plt

def normalize(features):
    return features/255.

def one_hot_encode(x):
    n = len(x)
    b = np.zeros((n, max(x)+1))
    b[np.arange(n), x] = 1
    return b

def preprocess_and_save_data(cifar10_dataset_folder_path):
    def _save_data(name, save_as=None, start_index=None, end_index=None):
        features, labels = load_cfar10_batch_by_name(cifar10_dataset_folder_path+'/'+name)
        data_len = len(features)

        if start_index and start_index < 1: start_index = int(data_len * start_index)
        if end_index and end_index < 1: end_index = int(data_len * end_index)

        if start_index and end_index:
            features = features[start_index: end_index]
            labels = labels[start_index: end_index]
        elif start_index:
            features = features[start_index:]
            labels = labels[start_index:]
        else:
            features = features[:end_index]
            labels = labels[:end_index]

        features = normalize(features)
        labels = one_hot_encode(labels)
        if not save_as: save_as = str(name) + '.p'
        print(cifar10_dataset_folder_path + '/' + save_as, len(features))
        pickle.dump((features, labels), open(cifar10_dataset_folder_path + '/' + save_as, 'wb'))

    _save_data('data_batch_1', 'preprocess_batch_1.p')
    _save_data('data_batch_2', 'preprocess_batch_2.p')
    _save_data('data_batch_3', 'preprocess_batch_3.p')
    _save_data('data_batch_4', 'preprocess_batch_4.p')
    _save_data('data_batch_5', 'preprocess_batch_5.p', end_index=0.5)
    _save_data('data_batch_5', 'preprocess_dev.p', start_index=0.5)
    _save_data('test_batch', 'preprocess_train.p')




def load_cfar10_batch_by_name(file):
    with open(file, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    """
    Load a batch of the dataset
    """
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    """
    Display Stats of the the dataset
    """
    batch_ids = list(range(1, 6))

    if batch_id not in batch_ids:
        print('Batch Id out of Range. Possible Batch Ids: {}'.format(batch_ids))
        return None

    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch {}:'.format(batch_id))
    print('Samples: {}'.format(len(features)))
    print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True)))))
    print('First 20 Labels: {}'.format(labels[:20]))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]
    label_names = _load_label_names()

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    plt.axis('off')
    plt.imshow(sample_image)
    plt.show()


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]



def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'cifar-10-batches-py/preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


def _load_label_names():
    """
    Load the label names from file
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']