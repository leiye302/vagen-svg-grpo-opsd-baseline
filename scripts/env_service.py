"""Loopback SVG reward service using only prepared local model weights."""
import argparse
from pathlib import Path
import sys

from baseline_runtime import add_source_paths, safe_environment


def install_local_reward_models(work_dir):
    """Change asset locations only; retain upstream render/reward mathematics."""
    work = Path(work_dir).resolve()
    from transformers import AutoImageProcessor, AutoModel
    from vagen.env.svg.dino import DINOScoreCalculator
    import vagen.env.svg.dreamsim as dreamsim_module

    def local_dino(self, model_size):
        if model_size != 'small':
            raise RuntimeError('This frozen reward contract requires DINOv2-small')
        path = str(work/'models/dinov2-small')
        return (AutoModel.from_pretrained(path, local_files_only=True),
                AutoImageProcessor.from_pretrained(path, local_files_only=True))

    DINOScoreCalculator.get_DINOv2_model = local_dino
    original_class = dreamsim_module.DreamSimScoreCalculator

    class LocalDreamSimScoreCalculator(original_class):
        def __init__(self, pretrained=True, cache_dir=None, device=None):
            super().__init__(pretrained=pretrained, cache_dir=str(work/'cache/dreamsim'), device=device)

    dreamsim_module.DreamSimScoreCalculator = LocalDreamSimScoreCalculator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--port', type=int, default=5019)
    parser.add_argument('--owner-token', required=True)
    args = parser.parse_args()
    import os
    os.environ.update(safe_environment(args.work_dir))
    def local_connections_only(event, arguments):
        if event == 'socket.connect':
            address = arguments[1]
            if isinstance(address, tuple) and address[0] not in {'127.0.0.1', '::1', 'localhost'}:
                raise RuntimeError('Reward service requires complete local assets; external connections are disabled')
    sys.addaudithook(local_connections_only)
    add_source_paths()
    install_local_reward_models(args.work_dir)
    from omegaconf import OmegaConf
    from vagen.server.server import BatchEnvServer
    from flask import jsonify
    config = OmegaConf.create({
        'use_state_reward': False,
        'server': {'host': '127.0.0.1', 'port': args.port, 'debug': False, 'use_state_reward': False},
        'svg': {'max_workers': 8, 'model_size': 'small', 'preload_models': True,
                'device': {'dino': 'cuda:0', 'dreamsim': 'cuda:0'}, 'use_state_reward': False}})
    server = BatchEnvServer(config)

    @server.app.route('/baseline_identity', methods=['GET'])
    def identity():
        return jsonify({'owner_token': args.owner_token, 'work_dir': str(Path(args.work_dir).resolve()),
                        'local_reward_weights': True, 'external_judge': False})

    server.start(background=False)


if __name__ == '__main__':
    main()
