import tensorflow as tf
import mlflow
import six
import time


def one_step_exec(batch, model, x, y, loss_fn, optimizer, metrics, training):
    """ Execute one batch of data within the model. In training mode, it
        optimizes the model by applying the resulting gradients. It also
        computed the different metrics that are monitored.
        Args:
            batch (int): the id of the batch
            model (tf.keras.Model): the model under execution
            x (tf.Tensor): the input batch data
            y (tf.Tensor): the target associated to the input data
            loss_fn (callable): the loss function associated to the model
            optimizer (tf.keras.optimizers.Optimizer): the model's optimizer
            metrics (list): list of metrics to monitor
            training (bool): flag indicating the training/eval mode.

        Outputs:
            loss_value (tf.Tensor): the loss value of `y` w.r.t. `model(x)`.
    """
    if training:
        with tf.GradientTape() as tape:
            output = model(x, training)
            loss_value = loss_fn(y, output)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, model.trainable_variables)
        )
    else:
        output = model(x, training)
        loss_value = loss_fn(y, output)

    # name = "" if training else "val_"
    # name = "batch_" + name + "loss"
    # mlflow.log_metric(name, loss_value.numpy(), step=batch)

    if metrics is not None:
        for i in range(len(metrics)):
            metrics[i].update_state(y, output)

    return loss_value


def reset_metric_state(metrics):
    """ Reset the metric values.
        Args:
            metrics (list): list of monitored metrics
    """
    if metrics is None:
        return
    for i in range(len(metrics)):
        metrics[i].reset_states()
    return


def log_mlflow_metrics(epoch, metrics, training, out_result):
    """ Log the metric values in mlflow.
        Args:
            epoch (int): the epoch for which the metrics are logged
            metrics (list): list of monitored metrics
            training (bool): flag indicating the training/eval mode.
                In eval mode, the prefix "val_" is added in front of
                the metric name.
            out_result (dict): a dictionnary that keeps track of the
                logged values with the corresponding name.
    """
    if metrics is None:
        return
    prefix = "" if training else "val_"
    for i in range(len(metrics)):
        value = metrics[i].result().numpy()
        name = prefix + metrics[i].name
        mlflow.log_metric(name, value, step=epoch)
        out_result[name] = value


def save_mlflow_run_id(run_id, checkpoint_path='./'):
    """ Save the mlflow run_id property. This is useful for resuming training.
        Args:
            run_id (str): the run_id of the mlflow run
            checkpoint_path (str): the directory for storing the data. run_id
                will be stored as a checkpoint in `checkpoint_path`/mlflow/.
    """
    checkpoint_dir = checkpoint_path + '/mlflow/'
    ckpt = tf.train.Checkpoint(
        run_id=tf.Variable(run_id)
    )
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=1
    )
    ckpt_save_path = ckpt_manager.save()
    tf.logging.info(
        'Save mlflow run_id meta info: {}'.format(ckpt_save_path)
    )


def get_mlflow_run_id(checkpoint_path='./'):
    """ Get the mlflow run_id property from a given directory if found.
        return None otherwise.
        Args:
            checkpoint_path (str): the directory for loading the data. run_id
                is assumed being stored as a checkpoint in
                `checkpoint_path`/mlflow/.
        Outputs:
            run_id (str): the retrieved run_id data or None if no data found.
    """
    checkpoint_dir = checkpoint_path + '/mlflow/'
    ckpt = tf.train.Checkpoint(
        run_id=tf.Variable("")
    )
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=1
    )
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        run_id = str(ckpt.run_id.numpy(), 'utf-8')
        tf.logging.info('Latest ml_flow run_id restored: {}.'.format(run_id))
        return run_id
    else:
        return None


def training_loop(
        model, optimizer, loss_fn,
        ds_train, num_epochs, steps_per_epoch, metrics=None,
        ds_eval=None, validation_steps=None, validation_freq=1,
        initial_epoch=0, step_log_freq=20, checkpoint_path='./',
        chkpt_freq=1, best_metric_indicator_index=-1, best_metric_coef=-1.0,
        num_max_chkpts=5, num_max_best_model=5, verbose=True):
    """ Main loop of the training process.
        Args:
            model (tf.keras.Model): the model to train
            optimizer (tf.keras.optimizers.Optimizer): the model's optimizer
            loss_fn (callable): the loss function associated to the model
            ds_train (tf.Dataset): train dataloder
            num_epochs (int): number of training epochs
            steps_per_epoch (int): number of training steps per epoch
            metrics (list): list of metrics to monitor
            ds_eval (tf.Dataset, optional): validation dataloder. Default: None
            validation_steps (int): number of steps per validation stage.
                Default: None
            validation_freq (int): frequency at which the validation stage
                is performed. Default: 1
            initial_epoch (int): epoch at which to start training (useful
                for resuming a previous training run). Default: 0
            step_log_freq (int): frequency on which to perform logging
                (in terms of batches). Default: 20
            checkpoint_path (str): the directory for storing the checkpoints.
                best_models are stored in `checkpoint_path`/best_models/.
                checkpoints are stored in `checkpoint_path`/trained_models/.
                mlflow metadata are stored in `checkpoint_path`/mlflow/.
                Default: './'
            chkpt_freq (int): frequency on which to perform checkpointing
                (in terms of epochs). Default: 1
            best_metric_indicator_index (int): the index of the metric to be
                used for determining the best model. if negative, use the loss
                value. only applicable if validation dataset is provided.
                Default: -1
            best_metric_coef (float): coeeficient to applied to the monitored
                metric for determining the best model. Here, the monitored
                metric is maximized. Thus, the coef should be -1 for loss based
                metric and +1 for accuracy based metric for example.
                Default: -1.0
            num_max_chkpts (int): maximum number of checkpoints to be kept.
                Default: 5
            num_max_best_model (int): maximum number of best_models to be kept.
                Default: 5
            verbose (bool): flag for enabling logging

        Outputs:
            out_result (dict): a dictionnary containing the final loss value
                and the monitored metrics for the training and eventually the
                validattion datasets.
    """

    out_result = {}
    epoch = initial_epoch
    loss_history = []
    train_loss_epoch_history = []
    eval_loss_history = None
    eval_loss_epoch_history = None
    best_val_score = None

    checkpoint_train_path = checkpoint_path + '/trained_models/'
    ckpt_train = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=tf.Variable(epoch)
    )
    ckpt_train_manager = tf.train.CheckpointManager(
        ckpt_train, checkpoint_train_path, max_to_keep=num_max_chkpts
    )

    # Begin step 1: **** Restore the latest best model found so far ******
    if not (ds_eval is None):
        # chkpt_freq should be at most equals to the validation frequency
        chkpt_freq = min(chkpt_freq, validation_freq)

        eval_loss_history = []
        eval_loss_epoch_history = []
        best_val_score = -100000.0

        checkpoint_best_path = checkpoint_path + '/best_models/'
        ckpt_best = tf.train.Checkpoint(
            best_val_score=tf.Variable(best_val_score),
            model=model,
        )
        ckpt_best_manager = tf.train.CheckpointManager(
            ckpt_best, checkpoint_best_path, max_to_keep=num_max_best_model
        )
        # if a best checkpoint exists,
        # restore the latest one at the condition
        # that the weight of the model will be replaced later
        # by the last checkpoint of the ckpt_train_manager.
        if ckpt_best_manager.latest_checkpoint:
            ckpt_best.restore(ckpt_best_manager.latest_checkpoint)
            best_val_score = ckpt_best.best_val_score.numpy()
            out_result['val_performance'] = best_val_score
            tf.logging.info('Latest best checkpoint restored!!')
    # End step 1: *** End of Restoring the latest best model found so far ****

    # Begin step 2: **** Restore the latest checkpoint saved so far ******
    if ckpt_train_manager.latest_checkpoint:
        ckpt_train.restore(ckpt_train_manager.latest_checkpoint)
        epoch = int(ckpt_train.epoch.numpy())
        tf.logging.info('Latest checkpoint restored!!')
    # End step 2: **** End of Restoring the latest checkpoint saved so far ***

    # Begin step 3: **** initialize the metrics to be monitored ******
    if metrics is not None:
        for i in range(len(metrics)):
            if isinstance(metrics[i], six.string_types):
                metrics[i] = tf.keras.metrics.get(metrics[i])
        reset_metric_state(metrics)

        str_msg = "indicator index must be less than the number of metrics"
        assert best_metric_indicator_index < len(metrics), str_msg
    # End step 3: **** End of initializing the metrics to be monitored ******

    # Begin step 4: **** compute the number of training steps ******
    max_steps = (num_epochs - epoch) * steps_per_epoch
    max_steps = max(max_steps, 0)
    # End step 4: **** End of computing the number of training steps ******

    if max_steps == 0:
        return

    # Begin step 5: **** begin of the training loop ******
    ds_tmp_train = ds_train.take(max_steps)
    start = time.time()
    for (batch, (images, labels)) in enumerate(ds_tmp_train):

        loss_value = one_step_exec(
            batch, model, images, labels, loss_fn, optimizer, metrics, True
        )
        loss_history.append(loss_value.numpy())

        if verbose and ((batch + 1) % step_log_freq == 0):
            tf.logging.info("Loss at step {:04d}: {:.5f}".format(
                batch + 1, loss_history[-1]
            ))

        # Test on the end of a training epoch
        if (batch + 1) % steps_per_epoch == 0:
            epoch = epoch + 1
            ckpt_train.epoch.assign_add(1)
            epoch_loss = sum(loss_history[-steps_per_epoch:]) / steps_per_epoch
            train_loss_epoch_history.append(epoch_loss)
            log_string = "Epoch {:03d}: loss: {:.5f}".format(
                epoch, epoch_loss)

            # log and reset the monitored metrics on training data
            name = "loss"
            mlflow.log_metric(name, epoch_loss, step=epoch)
            out_result[name] = epoch_loss
            log_mlflow_metrics(epoch, metrics, True, out_result)
            reset_metric_state(metrics)

            if not (ds_eval is None) and (epoch % validation_freq == 0):

                # begin of the evaluation stage
                if not (validation_steps is None):
                    ds_tmp_eval = ds_eval.take(validation_steps)
                else:
                    ds_tmp_eval = ds_eval
                num_steps = 0
                for (batch, (images, labels)) in enumerate(ds_tmp_eval):
                    num_steps += 1
                    loss_value = one_step_exec(
                        batch, model, images, labels, loss_fn, None,
                        metrics, False
                    )
                    eval_loss_history.append(loss_value.numpy())

                eval_epoch_loss = sum(
                    eval_loss_history[-num_steps:]) / num_steps
                eval_loss_epoch_history.append(eval_epoch_loss)
                log_string += " val_loss: {:.5f}".format(eval_epoch_loss)

                # log the validation loss
                name = "val_loss"
                mlflow.log_metric(
                    name, eval_epoch_loss, step=epoch // validation_freq
                )
                out_result[name] = eval_epoch_loss

                # check if it is the best performance so far and save the model
                if best_metric_indicator_index < 0:
                    performance_value = -eval_epoch_loss
                else:
                    performance_value = metrics[
                        best_metric_indicator_index
                    ].result().numpy() * best_metric_coef
                if performance_value > best_val_score:
                    best_val_score = performance_value
                    out_result['val_performance'] = best_val_score
                    ckpt_best.best_val_score.assign(best_val_score)
                    ckpt_save_path = ckpt_best_manager.save()
                    tf.logging.info(
                        'Save Best checkpt: epoch {}: perf {:.5f} - {}'.format(
                            epoch, best_val_score, ckpt_save_path
                        )
                    )

                # log and reset the monitored metrics on validation data
                log_mlflow_metrics(
                    epoch // validation_freq, metrics, False,
                    out_result
                )
                reset_metric_state(metrics)

            elapsed = time.time() - start
            for k in out_result.keys():
                if k not in ('loss', 'val_loss'):
                    log_string += " {}: {:.5f}".format(k, out_result[k])
            log_string += " time: {:.5f} sec".format(elapsed)

            if verbose:
                tf.logging.info(log_string)

            # perform checkpointing
            if (epoch % chkpt_freq == 0) or (epoch == num_epochs):
                ckpt_save_path = ckpt_train_manager.save()
                tf.logging.info(
                    'Save checkpt for epoch {}: {}'.format(
                        epoch, ckpt_save_path
                    )
                )
            start = time.time()

            if not (num_epochs is None):
                if epoch == num_epochs:
                    break
    # End step 5: **** end of the training loop ******

    # Begin step 6: **** final logging ******
    if verbose:
        tf.logging.info("Final Loss at step {:04d}: {:.5f}".format(
            len(loss_history), loss_history[-1]
        ))
        log_string = "Final Epoch: {:03d}:  Train Loss {:.5f}".format(
            epoch, train_loss_epoch_history[-1]
        )
        if not (ds_eval is None):
            log_string += " Eval Loss {:.5f}".format(
                eval_loss_epoch_history[-1])
        tf.logging.info(log_string)
    # End step 6: **** end of final logging ******

    return out_result
