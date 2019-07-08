import tensorflow as tf
import argparse
import os
import math
import mlflow

import dataloader
import training
import model


def manage_arguments():
    parser = argparse.ArgumentParser('TT-Training')

    parser.add_argument(
        '--output_dir', type=str, default='./output/',
        help='Base directory where to output the checkpoints'
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='/network/tmp1/fansitca/TT-Training/',
        help='Directory where to find the data'
    )
    parser.add_argument(
        '--mlflow_experiment_name', type=str,
        default='TT-Training',
        help='Name of the experiment under MlFlow'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='batch size for training'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=10,
        help='number of training epochs'
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
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='the learning rate of the training process'
    )
    parser.add_argument(
        '--num_max_chkpts', type=int, default=5,
        help='maximum number of checkpoint to keep'
    )
    parser.add_argument(
        '--num_max_best_model', type=int, default=1,
        help='maximum number of best models checkpoint to keep'
    )
    args = parser.parse_args()
    return args


def main_process():

    tf.logging.set_verbosity(tf.logging.INFO)
    args = manage_arguments()

    output_dir = args.output_dir

    # define the output directory of the experiment
    output_dir += "/model-fc-{}-{}-{}-{}-{}-{}-{}".format(
        args.fc_size, args.num_fc, args.kernel_size,
        args.strides, args.pool_size,
        str(args.dropout_rate).replace('.', '_'),
        str(args.learning_rate).replace('.', '_')
    )
    tf.gfile.MakeDirs(output_dir)

    # prepare the dataset if necessary
    if not os.path.isfile(args.data_dir + '/Train_Images.TFRecord'):
        tf.gfile.MakeDirs(args.data_dir)
        dataloader.prepare_datasets(args.data_dir)

    tf.enable_eager_execution()

    # define the train/valid dataloaders
    ds_train = dataloader.input_dataset_fn(
        batch_size=args.batch_size,
        image_file=args.data_dir + 'Train_Images.TFRecord',
        label_file=args.data_dir + 'Train_Labels.TFRecord',
        repeat=True, shuffle=True, drop_remainder=False,
        data_augmentation=True,
    )
    steps_per_epoch = int(math.ceil(24287 / args.batch_size))
    ds_eval = dataloader.input_dataset_fn(
        batch_size=args.batch_size,
        image_file=args.data_dir + 'Eval_Images.TFRecord',
        label_file=args.data_dir + 'Eval_Labels.TFRecord',
        repeat=False, shuffle=False, drop_remainder=False,
        data_augmentation=False,
    )
    validation_steps = int(math.ceil(8095 / args.batch_size))

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

    # define the loss function, metrics, and optimizer
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
    best_metric_indicator_index = 0
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # set up the mlflow experiment name and eventually retrieve the mlflow
    # run_id meta data. Afterward, log the params used in mlflow and start
    # the training process. At the end, log the returned metrics in mlflow
    # and return the metrics recorded within this instance of mlflow run.
    out_data = None
    mlflow.set_experiment(args.mlflow_experiment_name)
    run_id = training.get_mlflow_run_id(output_dir)
    run_metrics = None
    with mlflow.start_run(run_id=run_id) as run:
        run_id = run.info.run_id
        training.save_mlflow_run_id(run_id, output_dir)
        mlflow.log_param("pool_size", args.pool_size)
        mlflow.log_param("strides", args.strides)
        mlflow.log_param("kernel_size", args.kernel_size)
        mlflow.log_param("fc_size", args.fc_size)
        mlflow.log_param("num_fc", args.num_fc)
        mlflow.log_param("dropout_rate", args.dropout_rate)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("num_epochs", args.num_epochs)
        out_data = training.training_loop(
            network_model, optimizer, loss_fn,
            ds_train, args.num_epochs, steps_per_epoch, metrics=metrics,
            ds_eval=ds_eval, validation_steps=validation_steps,
            validation_freq=1, step_log_freq=20, checkpoint_path=output_dir,
            chkpt_freq=1, best_metric_coef=1.0,
            best_metric_indicator_index=best_metric_indicator_index,
            num_max_chkpts=args.num_max_chkpts,
            num_max_best_model=args.num_max_best_model, verbose=True
        )
        if out_data is not None:
            for k in out_data.keys():
                mlflow.log_metric(k, out_data[k])

        run_metrics = dict(run.data.metrics)

    return run_metrics


def main(_):
    main_process()


if __name__ == '__main__':
    tf.app.run(main=main)
