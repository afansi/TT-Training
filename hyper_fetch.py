import pprint
import argparse
# import datetime
from orion.core.io.experiment_builder import ExperimentBuilder


def manage_arguments():
    parser = argparse.ArgumentParser(
        'ORION-HYPER-SEARCH-FETCH: Fetching the results of an ORION Experiment'
    )

    parser.add_argument(
        '--experiment_name', type=str, default='ORION-HYPER-SEARCH',
        help='Experiment name whose trials need to be fetched.'
    )
    parser.add_argument(
        '--only_completed', default=False, action='store_true',
        help='Fetch only completed trials'
    )
    args = parser.parse_args()
    return args


def main():
    """ This function aims at fetching the results of an ORION Experiment.
        An `experiment_name` needs to be provided. Also, it is possible to
        retrieve results from complete trials.
    """
    args = manage_arguments()

    # some_datetime = datetime.datetime.now() - datetime.timedelta(minutes=5)

    experiment = ExperimentBuilder().build_view_from(
        {"name": args.experiment_name}
    )

    pprint.pprint(experiment.stats)

    query = {}
    if args.only_completed:
        query['status'] = 'completed'

    # query['end_time'] = {'$gte': some_datetime}

    for trial in experiment.fetch_trials(query):
        print(trial.id)
        print(trial.status)
        print(trial.params)
        print(trial.results)
        print()
        pprint.pprint(trial.to_dict())


if __name__ == '__main__':
    main()
