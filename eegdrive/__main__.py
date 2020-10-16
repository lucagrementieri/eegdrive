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
                #: TODO 'train       Train SpeaREC model\n'
                #: TODO 'test        Test SpeaREC EER on test pairs\n'
            ),
        )
        parser.add_argument(
            'command', type=str, help='Sub-command to run', choices=('ingest',),
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


"""
    @staticmethod
    def train() -> None:
        parser = argparse.ArgumentParser(
            description='Train SpeaREC model',
            usage='python3 -m torchspearec train SUMMARY-PATH [--runs-dir  RUNS-DIR '
                  '--batch-size BATCH-SIZE --epochs EPOCHS --lr LR '
                  '--test-summary-path TEST-SUMMARY-PATH --test-pairs-path TEST-PAIRS-PATH]',
        )
        parser.add_argument(
            'summary_path',
            metavar='summary-path',
            type=str,
            help='Path to data summary',
        )
        parser.add_argument(
            '--runs-dir', type=str, default='./runs', help='Output directory'
        )
        parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
        parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
        parser.add_argument(
            '--lr', type=float, default=0.001, help='Initial learning rate'
        )
        parser.add_argument(
            '--test-summary-path', type=str, help='Path to test data summary'
        )
        parser.add_argument(
            '--test-pairs-path', type=str, help='Path to test pairs file'
        )

        args = parser.parse_args(sys.argv[2:])

        print(f'Training SpeaREC using data listed in {args.summary_path}')
        Spearec.train(
            args.summary_path,
            args.runs_dir,
            args.batch_size,
            args.epochs,
            args.lr,
            args.test_summary_path,
            args.test_pairs_path,
        )

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
