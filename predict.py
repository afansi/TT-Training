import tensorflow as tf
import argparse
import os
import csv
import numpy as np

import dataloader
import model


def manage_arguments():
    parser = argparse.ArgumentParser('TT-Training-Predict')

    parser.add_argument(
        '--chkpt_dir', type=str,
        help='directory where to load the model under evaluation'
    )
    parser.add_argument(
        '--ouput_file', type=str, default='./prediction.csv',
        help='path of the file where to output the prediction result'
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='/network/tmp1/fansitca/TT-Training/',
        help='Directory where to find the data'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='batch size for training'
    )
    parser.add_argument(
        '--kernel_size', type=int, default=3,
        help='the kernel for convolution layers'
    )
    parser.add_argument(
        '--strides', type=int, default=1,
        help='the strides for convolution layers'
    )
    parser.add_argument(
        '--pool_size', type=int, default=2,
        help='the pool size for max_pooling layers'
    )
    parser.add_argument(
        '--num_fc', type=int, default=2,
        help='the number of dense layers before output predictions'
    )
    parser.add_argument(
        '--fc_size', type=int, default=1000,
        help='the output size of the dense layers before output predictions'
    )
    parser.add_argument(
        '--dropout_rate', type=float, default=0.5,
        help='the dropout rate applied within the model'
    )
    args = parser.parse_args()
    return args


def main_process():

    tf.logging.set_verbosity(tf.logging.INFO)
    args = manage_arguments()

    # prepare the dataset if necessary
    if not os.path.isfile(args.data_dir + '/Test_Images.TFRecord'):
        tf.gfile.MakeDirs(args.data_dir)
        dataloader.prepare_datasets(args.data_dir)

    real_labels = {}
    with open(args.data_dir + '/label_info.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if row[0].strip().startswith('#'):  # header
                continue
            index = int(row[0])
            label = row[1]
            real_labels[index] = label

    tf.enable_eager_execution()

    # define the model
    conv_list = [(2, 64), (2, 128), (3, 256)]
    dense_list = [args.fc_size] * args.num_fc
    dense_list.append(17)
    network_model = model.create_model(
        input_shape=(224, 224, 3),
        conv_list=conv_list,
        dense_list=dense_list,
        kernel_size=args.kernel_size,
        strides=args.strides,
        pool_size=args.pool_size,
        dropout_rate=args.dropout_rate,
        output_activation='sigmoid',
        layer_activation='relu'
    )
    network_model.summary()

    ckpt = tf.train.Checkpoint(
        model=network_model
    )
    status = ckpt.restore(tf.train.latest_checkpoint(args.chkpt_dir))

    status.assert_existing_objects_matched()

    # define the train/valid dataloaders
    ds_test_x = dataloader.input_dataset_fn(
        batch_size=args.batch_size,
        image_file=args.data_dir + 'Test_Images.TFRecord',
        label_file=None,
        repeat=False, shuffle=False, drop_remainder=False,
        data_augmentation=False,
    )
    predictions = network_model.predict(
        ds_test_x,
    )
    num_elements = len(predictions)
    ds_test_y = dataloader.input_dataset_fn(
        batch_size=num_elements,
        image_file=None,
        label_file=args.data_dir + 'Test_Labels.TFRecord',
        repeat=False, shuffle=False, drop_remainder=False,
        data_augmentation=False,
    )
    ground_truth = next(iter(ds_test_y))

    # define the loss function, metrics, and optimizer
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]

    loss_test = loss_fn(ground_truth, predictions)
    for i in range(len(metrics)):
        metrics[i].update_state(ground_truth, predictions)
    metric_values = dict(
        ('test_' + m.name, m.result().numpy()) for m in metrics
    )

    metric_values['loss_test'] = loss_test.numpy()
    test_precision = metric_values.get('test_precision', 0.0)
    test_recall = metric_values.get('test_recall', 0.0)
    denom = test_precision + test_recall
    if denom <= 0:
        denom = 1e-5
    test_f1 = 2 * (test_precision * test_recall) / denom
    metric_values['test_f1'] = test_f1

    predicted_classes = np.argwhere(predictions > 0.5)
    item_classes = {}
    for i in range(len(predicted_classes)):
        index, a_class = predicted_classes[i]
        if not (index in item_classes):
            item_classes[index] = []
        item_classes[index].append(
            real_labels[a_class]
        )

    print("\n\n")
    print(metric_values)
    print("\n\n")


def main(_):
    main_process()


if __name__ == '__main__':
    tf.app.run(main=main)
