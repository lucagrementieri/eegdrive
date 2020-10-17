#!/usr/bin/env python3.8

import argparse
import sys

from eegdrive.eegdrive import EEGDrive


class CLI:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Command line interface for EEG classification assessment',
            usage=(
                'python3 -m eegdrive <command> [<args>]\n'
                '\n'
                'ingest      Process EEG data\n'
                'train       Train classification model\n'
                #: TODO 'test        Test SpeaREC EER on test pairs\n'
            ),
        )
        parser.add_argument(
            'command', type=str, help='Sub-command to run', choices=('ingest', 'train'),
        )
        args = parser.parse_args(sys.argv[1:2])
        command = args.command.replace('-', '_')
        if not hasattr(self, command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        getattr(self, command)()

    @staticmethod
    def ingest() -> None:
        parser = argparse.ArgumentParser(
            description='Process EEG data',
            usage='python3 -m eegdrive ingest DATA-PATH OUTPUT-DIR',
        )
        parser.add_argument(
            'data_path', metavar='data-path', type=str, help='Session HDF5 file path'
        )
        parser.add_argument(
            'output_dir',
            metavar='output-dir',
            type=str,
            help='Output episodes directory',
        )
        args = parser.parse_args(sys.argv[2:])
        EEGDrive.ingest(args.data_path, args.output_dir)

    @staticmethod
    def train() -> None:
        parser = argparse.ArgumentParser(
            description='Train action classification model',
            usage='python3 -m eegdrive train DATASET-DIR '
                  '[--runs-dir RUNS-DIR --label-type LABEL-TYPE]',
        )
        parser.add_argument(
            'dataset_dir',
            metavar='dataset-dir',
            type=str,
            help='Path to directory of ingested data',
        )
        parser.add_argument(
            '--runs-dir', type=str, default='./runs', help='Output directory'
        )
        parser.add_argument(
            '--label-type', type=str, default='action', help='Label type'
        )

        args = parser.parse_args(sys.argv[2:])

        EEGDrive.train(args.dataset_dir, args.runs_dir, 'action')


"""
    @staticmethod
    def test() -> None:
        parser = argparse.ArgumentParser(
            description='Test SpeaREC EER on test pairs',
            usage='python3 -m torchspearec test CHECKPOINT SUMMARY-PATH PAIRS-PATH',
        )
        parser.add_argument('checkpoint', type=str, help='Model checkpoint')
        parser.add_argument(
            'summary_path',
            metavar='summary-path',
            type=str,
            help='Path to data summary',
        )
        parser.add_argument(
            'pairs_path', metavar='pairs-path', type=str, help='Path to test pairs file'
        )
        args = parser.parse_args(sys.argv[2:])

        print(f'Testing SpeaREC using data listed in from {args.summary_path}')
        scores = Spearec.test(args.checkpoint, args.summary_path, args.pairs_path)
        print(f'EERs using intermediate features', scores)
"""

if __name__ == '__main__':
    CLI()
