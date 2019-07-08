#!/usr/bin/env python
import tensorflow as tf
from orion.client import report_results

from main import main_process


def main(_):
    """ This function aims at calling the main training process from which
        it retrieves the metrics and computes the objective value to be
        optimized by ORION. In this case, we optimize for the F1-score on
        the validation dataset.
    """

    run_metrics = main_process()
    default_value = 1000000.0
    orion_objective = default_value
    if not (run_metrics is None):
        # val_acc = run_metrics.get('val_binary_accuracy', 0.0)
        val_precision = run_metrics.get('val_precision', 0.0)
        val_recall = run_metrics.get('val_recall', 0.0)

        denom = val_precision + val_recall
        if denom <= 0:
            denom = 1e-5

        val_f1 = 2 * (val_precision * val_recall) / denom

        orion_objective = -val_f1  # -val_acc

    tf.logging.info("FOUND OBJECTIVE: {}".format(orion_objective))
    report_results(
        [dict(
            name='orion_objective',
            type='objective',
            value=orion_objective)
         ]
    )


if __name__ == '__main__':
    tf.app.run(main=main)
