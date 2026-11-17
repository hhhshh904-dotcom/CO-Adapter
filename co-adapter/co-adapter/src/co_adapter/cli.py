"""Command-line interface; all pretrained weights and datasets stay external."""
import argparse
import os
from pathlib import Path
import signal
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='CO-Adapter: single-device no-cache training')
    parser.add_argument('command', choices=('smoke', 'train', 'evaluate'))
    parser.add_argument('--upstream-root', type=Path, required=True,
                        help='Official MSE-ChatGLM3-6B source directory (obtained separately)')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset', choices=('simsv2', 'mosei', 'meld', 'cherma'), required=True)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--variant', choices=('co', 'mixed'), default='co')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke-only', action='store_true',
                        help='With evaluate: replay smoke validation only, never load test')
    cli = parser.parse_args(argv)
    if cli.resume and cli.command == 'evaluate':
        parser.error('--resume applies only to smoke/train')
    if cli.smoke_only and cli.command != 'evaluate':
        parser.error('--smoke-only applies only to evaluate')
    # Offline operation is deliberate; obtain assets separately before invoking.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.dont_write_bytecode = True
    import torch
    from . import engine
    from .upstream import OfficialRuntime, configure
    from .provenance import inventory

    device = torch.device(cli.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('Only a CPU or a single CUDA device is supported')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    for path in (cli.upstream_root, cli.model_path, cli.data_root):
        if not path.is_dir():
            parser.error(f'External directory does not exist: {path}')
    runtime = OfficialRuntime(cli.upstream_root)
    args = configure(cli.dataset, cli.seed, cli.model_path, cli.data_root, device)
    resources = {'upstream_root': str(cli.upstream_root.resolve()),
                 'model_path': str(cli.model_path.resolve()), 'data_root': str(cli.data_root.resolve()),
                 'inventory': inventory(cli.upstream_root)}
    signal.signal(signal.SIGINT, engine.request_stop)
    signal.signal(signal.SIGTERM, engine.request_stop)
    out = cli.output_dir.resolve()
    smoke = cli.command == 'smoke' or cli.smoke_only
    if smoke:
        out = out / 'smoke'
    try:
        if cli.command in ('train', 'smoke'):
            result = engine.train(args, cli.variant, out, runtime, resources, cli.resume, smoke)
        else:
            result = engine.replay_and_evaluate(args, cli.variant, out, runtime, resources, smoke)
    except InterruptedError as exc:
        print(str(exc), file=sys.stderr)
        return 130
    print(result)
    return 0
