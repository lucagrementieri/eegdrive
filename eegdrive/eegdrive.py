import json
import logging
from pathlib import Path

from .ingestion import ingest_session
from .utils import initialize_logger


class EEGDrive:
    @staticmethod
    def ingest(data_path: str, output_dir: str) -> None:
        initialize_logger()
        logging.info(f'Data ingestion from {data_path}')
        data_path = Path(data_path).expanduser()
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True)
        statistics = ingest_session(data_path, output_dir)
        with open(output_dir / f'{data_path.stem}_statistics.json') as f:
            json.dump(statistics, f, indent=4)


"""
    @staticmethod
    def train(
            summary_path: str,
            output_dir: str,
            batch_size: int,
            epochs: int,
            lr: float,
            test_summary_path: Optional[str] = None,
            test_pairs_path: Optional[str] = None,
    ) -> None:
        summary_path = Path(summary_path).expanduser()
        run_dir = Path(output_dir) / str(int(time.time()))
        run_dir.mkdir(parents=True)
        initialize_logger(run_dir)

        logging.info(f'Batch size: {batch_size}')
        logging.info(f'Learning rate: {lr}')

        dataset = FBankDataset(summary_path, max_frames)
        train_loader, val_loader = get_train_val_loaders(dataset, batch_size)
        architecture = CustomResnet34()
        model = Model(architecture)

        if test_summary_path is not None and test_pairs_path is not None:
            test_summary_path = Path(test_summary_path).expanduser()
            test_dataset = FBankDataset(test_summary_path)
            test_pairs = get_test_pairs(test_pairs_path)
            test_loader = get_test_loader(test_dataset, test_pairs)
        else:
            test_pairs = None
            test_loader = None

        model.fit(
            run_dir,
            train_loader,
            dataset.num_classes,
            epochs,
            lr,
            val_loader,
            test_loader,
            test_pairs,
        )

    @staticmethod
    def test(checkpoint: str, summary_path: str, pairs_path: str) -> Dict[str, float]:
        summary_path = Path(summary_path).expanduser()
        test_dataset = FBankDataset(summary_path)
        test_pairs = get_test_pairs(pairs_path)
        test_loader = get_test_loader(test_dataset, test_pairs)
        architecture = CustomResnet34()
        model = Model(architecture)
        model.module.load_state_dict(torch.load(checkpoint, map_location=model.device))
        scores = model.test(test_loader, test_pairs)
        return scores

    @staticmethod
    def embed(checkpoint: str, audio_path: str, feature_name: str) -> torch.Tensor:
        audio_path = Path(audio_path).expanduser()
        segment = ingest_audio(audio_path)
        architecture = CustomResnet34()
        model = Model(architecture)
        model.module.load_state_dict(torch.load(checkpoint, map_location=model.device))
        embedding = model.embed(segment, feature_name)
        return embedding

    @staticmethod
    def compile(checkpoint: str, output_path: str, gpu: bool = False) -> None:
        output_path = Path(output_path).expanduser()
        device = torch.device('cuda:0' if gpu else 'cpu')
        architecture = CustomResnet34(compile=True).to(device)
        architecture.load_state_dict(torch.load(checkpoint, map_location=device))
        architecture.eval()
        example = torch.rand(1, 100, feature_dim).to(device)
        with torch.jit.optimized_execution(True):
            traced_script_module = torch.jit.trace(architecture, (example,))
        traced_script_module.save(str(output_path))
"""
