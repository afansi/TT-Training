import tensorflow as tf
import pathlib
import csv
import random
import os
import time
import math

import data_augmentation as da

data_root = '/network/tmp1/bertranh/rainforest/'

data_seed = 1234567890  # seed for data split
compression_type = "GZIP"
im_size = 224
AUTOTUNE = tf.data.experimental.AUTOTUNE


def read_labels():
    """ Read the labels associated with each example in the dataset.
        Assign to each label a unique ID and convert the labels assocated
        to each example into a multi-label 0/1 representation.
        Args:

        Outputs:
            image_labels (dict): a dictionnary of (key_image, labels) items
                where `labels` represent the multi-label 0/1 representation
                view of the labels associated to the image represented
                by `key_image`.
            label_to_index (dict): a dictionnary of (label, index) items
                containing the numerical index associated to each label.
            index_to_label (list): An indexed representation of the labels.
                `index_to_label[i]` will return the label associated with
                the numerical index `i`.
    """
    label_file = data_root + 'train_v2.csv'

    all_label_set = set()
    image_labels = {}

    with open(label_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:  # header
                continue
            file_name = row[0]
            labels = row[1].split(' ')

            all_label_set.update(labels)

            image_labels[file_name] = labels

    all_label_set = list(all_label_set)
    all_label_set = sorted(all_label_set)

    label_to_index = dict(
        (label, i)
        for i, label in enumerate(all_label_set)
    )
    index_to_label = all_label_set

    keys = list(image_labels.keys())

    for key in keys:
        image_labels[key] = [
            1 if u in image_labels[key] else 0
            for u in index_to_label
        ]

    return image_labels, label_to_index, index_to_label


def read_image_path(seed=data_seed):
    """ Read the paths of the images that are part of the dataset
        and split them into training/valid/test subsets
        Args:
            seed (int, optional): the seed of the random generator
                responsible for making the split. Default: 1234567890

        Outputs:
            train_images (list): a list of string representing the paths
                of images part of the training dataset.
            eval_images (list): similar to`train_images` but for
                validation dataset.
            test_images (list): similar to`train_images` but for
                test dataset.
    """

    image_dir = data_root + 'train-jpg'

    image_root = pathlib.Path(image_dir)

    image_paths = list(image_root.glob('**/*.jpg'))
    image_paths = [str(path) for path in image_paths]

    random.seed(seed)

    random.shuffle(image_paths)

    num_images = len(image_paths)

    num_train = int(num_images * 0.6)
    num_eval = int(num_images * 0.2)

    train_images = image_paths[0:num_train]
    eval_images = image_paths[num_train:num_train + num_eval]
    test_images = image_paths[num_train + num_eval:num_images]

    return train_images, eval_images, test_images


def retrieve_labels(image_paths, all_images_labels):
    """ Retrieve the lables associated to a provided set of image paths.
        Args:
            image_paths (list): a list of string representing the paths
                of images whose labels need to be retrieved.
            all_images_labels (dict): a dictionnary of (key_image, labels)
                items where `labels` represent the multi-label 0/1
                representation view of the labels associated to the image
                represented by `key_image`.

        Outputs:
            image_labels (list): a list retrieved labels. The order of the
            labels matches the one of `image_paths`.
    """
    image_labels = []
    for p in image_paths:
        key = p.split(os.sep)[-1].split('.')[-2]
        label = all_images_labels[key]

        image_labels.append(label)

    return image_labels


def preprocess_image(image):
    """ Preprocess a given image. the method first resizes the image
        into 224 x 224 dimensions and, then normalize its values
        within the [0, 1] range.
        Args:
            image (tf.Tensor):a tensor representing the image

        Outputs:
            image (tf.Tensor): the pre-processed image.
    """
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [im_size, im_size])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    """ load an image from the specified path and preprocess it.
        Args:
            path (str):the path of the image to load

        Outputs:
            image (tf.Tensor): the pre-processed image.
    """
    image = tf.io.read_file(path)
    return preprocess_image(image)


def parse_image(x):
    """ Transforms a serialized proto representing an image into a Tensor.
        Args:
            x: A scalar tensor of type string containing a serialized proto.

        Outputs:
            result (tf.Tensor): the de-serialized tensor representing an image.
    """
    result = tf.io.parse_tensor(x, out_type=tf.float32)
    result = tf.reshape(result, [im_size, im_size, 3])
    return result


def parse_label(x):
    """ Transforms a serialized proto representing a label into a Tensor.
        Args:
            x: A scalar tensor of type string containing a serialized proto.

        Outputs:
            result (tf.Tensor): the de-serialized tensor representing a label.
    """
    result = tf.io.parse_tensor(x, out_type=tf.int64)
    return result


def prepare_datasets(output_dir='./', seed=data_seed):
    """ Prepare the training/validation/test datasets as TFRecord dataset files
        Args:
            output_dir (str): The directory where to store the TFRecord files
            seed (int, optional): the seed of the random generator
                responsible for making the split. Default: 1234567890
    """

    image_labels, label_to_index, index_to_label = read_labels()
    train_images, eval_images, test_images = read_image_path(seed)

    train_labels = retrieve_labels(train_images, image_labels)
    eval_labels = retrieve_labels(eval_images, image_labels)
    test_labels = retrieve_labels(test_images, image_labels)

    with open(output_dir + '/' + 'label_info.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['# num_classes', str(len(index_to_label))])
        writer.writerow(
            ['# num_examples_train', str(len(train_labels))]
        )
        writer.writerow(
            ['# num_examples_eval', str(len(eval_labels))]
        )
        writer.writerow(
            ['# num_examples_test', str(len(test_labels))]
        )
        for i, label in enumerate(index_to_label):
            writer.writerow([str(i), str(label)])

    train_image_ds = tf.data.Dataset.from_tensor_slices(
        train_images
    ).map(
        load_and_preprocess_image, num_parallel_calls=AUTOTUNE
    ).map(
        tf.io.serialize_tensor, num_parallel_calls=AUTOTUNE
    )
    train_label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(train_labels, tf.int64)
    ).map(
        tf.io.serialize_tensor, num_parallel_calls=AUTOTUNE
    )

    eval_image_ds = tf.data.Dataset.from_tensor_slices(
        eval_images
    ).map(
        load_and_preprocess_image, num_parallel_calls=AUTOTUNE
    ).map(
        tf.io.serialize_tensor, num_parallel_calls=AUTOTUNE
    )
    eval_label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(eval_labels, tf.int64)
    ).map(
        tf.io.serialize_tensor, num_parallel_calls=AUTOTUNE
    )

    test_image_ds = tf.data.Dataset.from_tensor_slices(
        test_images
    ).map(
        load_and_preprocess_image, num_parallel_calls=AUTOTUNE
    ).map(
        tf.io.serialize_tensor, num_parallel_calls=AUTOTUNE
    )
    test_label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(test_labels, tf.int64)
    ).map(
        tf.io.serialize_tensor, num_parallel_calls=AUTOTUNE
    )

    tfrec_train_images = tf.data.experimental.TFRecordWriter(
        output_dir + '/' + 'Train_Images.TFRecord',
        compression_type=compression_type
    )
    tfrec_train_images_op = tfrec_train_images.write(train_image_ds)
    tfrec_train_labels = tf.data.experimental.TFRecordWriter(
        output_dir + '/' + 'Train_Labels.TFRecord',
        compression_type=compression_type
    )
    tfrec_train_labels_op = tfrec_train_labels.write(train_label_ds)

    tfrec_eval_images = tf.data.experimental.TFRecordWriter(
        output_dir + '/' + 'Eval_Images.TFRecord',
        compression_type=compression_type
    )
    tfrec_eval_images_op = tfrec_eval_images.write(eval_image_ds)
    tfrec_eval_labels = tf.data.experimental.TFRecordWriter(
        output_dir + '/' + 'Eval_Labels.TFRecord',
        compression_type=compression_type
    )
    tfrec_eval_labels_op = tfrec_eval_labels.write(eval_label_ds)

    tfrec_test_images = tf.data.experimental.TFRecordWriter(
        output_dir + '/' + 'Test_Images.TFRecord',
        compression_type=compression_type
    )
    tfrec_test_images_op = tfrec_test_images.write(test_image_ds)
    tfrec_test_labels = tf.data.experimental.TFRecordWriter(
        output_dir + '/' + 'Test_Labels.TFRecord',
        compression_type=compression_type
    )
    tfrec_test_labels_op = tfrec_test_labels.write(test_label_ds)

    if not tf.executing_eagerly():
        with tf.Session() as sess:
            _ = sess.run([
                tfrec_train_images_op,
                tfrec_train_labels_op,
                tfrec_eval_images_op,
                tfrec_eval_labels_op,
                tfrec_test_images_op,
                tfrec_test_labels_op,

            ])


def input_dataset_fn(
        batch_size, image_file=None, label_file=None,
        repeat=False, shuffle=False, data_augmentation=False,
        drop_remainder=False):
    """ Define a tf.Dataset (dataloader) based on some provided characteristics
        Args:
            batch_size (int): The batch_size of the dataloader
            image_file (str, optional): path to a TFRecord file of the images
            label_file (str, optional): path to a TFRecord file of the labels
            repeat (bool, optional): flag to loop over the data. Default: False
            shuffle (bool, optional): flag to shuffle the data. Default: False
            data_augmentation (bool, optional): flag to augment the data.
                Default: False
            drop_remainder (bool, optional): flag to drop the last batch if it
                does not contain `batch_size` examples. Default: False

        Outputs:
            ds (tf.Dataset): the corresponding dataloader.
    """

    if (image_file is None) and (label_file is None):
        return None

    if image_file is None:
        repeat = False
        shuffle = False

    image_ds = None
    if not (image_file is None):
        image_ds = tf.data.TFRecordDataset(
            image_file, compression_type=compression_type
        )
        image_ds = image_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
        if data_augmentation:
            augmentations = [
                da.flip_image,
                da.color_augmentation,
                da.zoom_image,
                da.rotate_image
            ]
            for f in augmentations:
                # Apply the augmentation, run jobs in parallel.
                image_ds = image_ds.map(f, num_parallel_calls=AUTOTUNE)

            # Make sure that the values are still in [0, 1]
            image_ds = image_ds.map(
                lambda x: tf.clip_by_value(x, 0.0, 1.0),
                num_parallel_calls=AUTOTUNE
            )

    label_ds = None
    if not (label_file is None):
        label_ds = tf.data.TFRecordDataset(
            label_file, compression_type=compression_type
        )
        label_ds = label_ds.map(parse_label, num_parallel_calls=AUTOTUNE)

    if image_ds is None:
        ds = label_ds
    elif label_ds is None:
        ds = image_ds
    else:
        ds = tf.data.Dataset.zip((image_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(AUTOTUNE)

    return ds


def timeit(ds, steps=10):
    """ Mehod for timing the loading performance of a dataloader.
        Args:
            ds (tf.Dataset): dataloader to be evaluated
            steps (int, optional): number of steps to perform during evaluation
                Default: 10
    """
    overall_start = time.time()
    # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
    # before starting the timer
    it = iter(ds.take(steps + 1))
    next(it)

    start = time.time()
    for i, (images, labels) in enumerate(it):
        if i == 0:
            batch_size = tf.shape(images)[0]
        if i % 10 == 0:
            print('.', end='')
    print()
    end = time.time()

    duration = end - start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(
        tf.cast(batch_size * steps, tf.float32) / duration))
    print("Total time: {}s".format(end - overall_start))


if __name__ == '__main__':

    tf.enable_eager_execution()
    # prepare_datasets()

    # exit()

    base_dir = '/network/tmp1/fansitca/TT-Training/'  # './'

    ds = input_dataset_fn(
        batch_size=32,
        image_file=base_dir + 'Eval_Images.TFRecord',
        label_file=base_dir + 'Eval_Labels.TFRecord',
        repeat=True, shuffle=False,
        data_augmentation=True,
    )

    size = 8095  # 24287  #
    batch_size = 32

    steps_per_epoch = int(math.ceil(size / batch_size))

    timeit(ds, steps=2 * steps_per_epoch + 1)
