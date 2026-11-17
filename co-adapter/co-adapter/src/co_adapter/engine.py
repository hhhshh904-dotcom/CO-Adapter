"""Single-device constant-LR training and fresh-validation test gate."""
import contextlib
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .state import (PROCESS, model_state, read_checkpoint, read_json, restore_model,
                    restore_rng, rng_state, save_checkpoint, save_json, seed_all)

STOP = False
PROTOCOL = 'co-four-tokens-nocache-constant-clip-v1'


def request_stop(signum, frame):
    global STOP
    STOP = True
    print('Stop requested; saving after the current completed training batch.', flush=True)


def amp(args):
    return torch.autocast('cuda', dtype=torch.float16) if args.device.type == 'cuda' else contextlib.nullcontext()


def loader(dataset, args):
    # Preserve official shuffled validation/test grouping as well as training.
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)


def move_batch(batch, args):
    text = (batch['text'].to(args.device), batch['text_lengths'].to(args.device))
    audio = (batch['audio'].to(args.device), batch['audio_lengths'].to(args.device))
    video = (batch['vision'].to(args.device), batch['vision_lengths'].to(args.device))
    labels = batch['labels']['M']
    if args.train_mode == 'regression':
        labels = labels.view(-1).to(args.device)
    return labels, text, audio, video


@contextlib.contextmanager
def no_cache(model):
    """Change only stream generation's use_cache, not EOS/masks/decoding/parser."""
    backbone = model.LLM.model
    original = backbone.stream_generate
    had_instance = 'stream_generate' in backbone.__dict__
    previous = backbone.__dict__.get('stream_generate')

    def uncached(*args, **kwargs):
        return original(*args, **{**kwargs, 'use_cache': False})

    backbone.stream_generate = uncached
    try:
        yield
    finally:
        if had_instance:
            backbone.stream_generate = previous
        else:
            del backbone.stream_generate


@contextlib.contextmanager
def record_responses(model):
    tokenizer = model.LLM.tokenizer
    original = tokenizer.batch_decode
    had_instance = 'batch_decode' in tokenizer.__dict__
    previous = tokenizer.__dict__.get('batch_decode')
    responses = []

    def decode(*args, **kwargs):
        result = original(*args, **kwargs)
        responses[:] = result
        return result

    tokenizer.batch_decode = decode
    try:
        yield responses
    finally:
        if had_instance:
            tokenizer.batch_decode = previous
        else:
            del tokenizer.batch_decode


def parses(raw, regression):
    if regression:
        raw = raw.replace('–', '-').replace('一', '-').replace('：', '').replace('/', '').replace('(', '').replace(':', '')
    try:
        return math.isfinite(float(raw))
    except ValueError:
        return False


def evaluate_predictions(model, dataset, args, runtime, path):
    model.eval()
    rows = []
    with torch.no_grad(), no_cache(model), record_responses(model) as responses:
        for batch in loader(dataset, args):
            labels, text, audio, video = move_batch(batch, args)
            responses.clear()
            with amp(args):
                predictions = model.generate(text, audio, video)
            if len(predictions) != len(labels) or len(responses) != len(labels):
                raise ValueError('Generation/response/sample count differs')
            for i, (pred, true, raw) in enumerate(zip(predictions, labels.cpu().tolist(), responses)):
                if not math.isfinite(pred):
                    raise ValueError(f'Official parser produced nonfinite prediction: {raw!r}')
                rows.append({'index': int(batch['index'][i]), 'prediction': pred,
                             'label': true, 'raw': raw,
                             'parsed': parses(raw, args.train_mode == 'regression')})
            if STOP:
                raise InterruptedError('Evaluation interrupted; restart from saved training state')
    if not rows or len(rows) != len(dataset) or len({r['index'] for r in rows}) != len(rows):
        raise ValueError('Evaluation omitted or duplicated samples')
    metrics = runtime.metrics(args, [r['prediction'] for r in rows], [r['label'] for r in rows])
    metrics = {k: float(v) if math.isfinite(v) else None for k, v in metrics.items()}
    metrics.update(samples=len(rows), parse_failures=sum(not r['parsed'] for r in rows))
    if args.train_mode == 'classification':
        metrics['out_of_range'] = sum(r['prediction'] not in range(7) for r in rows)
    result = {'metrics': metrics, 'predictions': sorted(rows, key=lambda r: r['index'])}
    save_json(path, result)
    return result


def improves(value, best, metric):
    if value is None or not math.isfinite(value):
        raise FloatingPointError('Nonfinite validation selection metric')
    return best is None or (value <= best - 1e-6 if metric == 'MAE' else value >= best + 1e-6)


def optimizer_update(loss, model, optimizer, scaler):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() < scale


def identity(args, variant, resources, smoke):
    return {'protocol': PROTOCOL, 'dataset': args.datasetName, 'seed': args.seed,
            'variant': variant, 'smoke': smoke, 'resources': resources,
            'torch': torch.__version__, 'device': str(args.device),
            'batch_size': args.batch_size, 'learning_rate': 5e-5,
            'max_grad_norm': 1.0, 'selection': args.KeyEval,
            'patience': args.early_stop, 'max_new_tokens': args.max_new_tokens}


def require_smoke(out, expected):
    smoke_identity = {**expected, 'smoke': True}
    completion = read_json(out / 'smoke' / 'completed.json')
    replay = read_json(out / 'smoke' / 'reloaded.json')
    if (completion['identity'] != smoke_identity or replay['identity'] != smoke_identity
            or completion.get('passed') is not True or completion.get('updates', 0) <= 0
            or replay.get('exact_validation_replay') is not True
            or replay.get('test_access') is not False):
        raise ValueError('Run smoke and a separate evaluate --smoke-only first')


def train(args, variant, out, runtime, resources, resume=False, smoke=False,
          stop_after_batches=None):
    """The optional batch stop is an internal CPU-test hook, not a protocol knob."""
    out = Path(out)
    expected = identity(args, variant, resources, smoke)
    if not smoke:
        require_smoke(out, expected)
    out.mkdir(parents=True, exist_ok=True)
    identity_path = out / 'run.json'
    if identity_path.exists():
        if read_json(identity_path) != expected:
            raise ValueError('Run configuration/resources changed')
        if not resume:
            raise FileExistsError('Existing run; pass --resume')
    else:
        if any(out.glob('*.pt')) or (out / 'completed.json').exists():
            raise ValueError('Checkpoints exist without run metadata')
        if resume:
            raise FileNotFoundError('Cannot resume without run metadata')
        save_json(identity_path, expected)
    seed_all(args.seed, args.device)
    train_set = runtime.load_split(args, 'train')
    valid_set = runtime.load_split(args, 'valid')
    if hasattr(train_set, 'ids') and set(map(str, train_set.ids)) & set(map(str, valid_set.ids)):
        raise ValueError('Train/validation ID overlap')
    args.seq_lens = train_set.get_seq_len()
    save_json(out / 'config.json', {k: str(v) if isinstance(v, torch.device) else v for k, v in dict(args).items()})
    if smoke:
        if len(train_set) < 64 or len(valid_set) < 8:
            raise ValueError('Smoke requires 64 training and 8 validation samples')
        train_set, valid_set = Subset(train_set, range(64)), Subset(valid_set, range(8))
    data = loader(train_set, args)
    model = runtime.make_model(args, variant)
    initial = model_state(model)['parameters'] if smoke else None
    if any(p.requires_grad for p in model.LLM.model.parameters()):
        raise ValueError('Backbone must be frozen')
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-4,
                                  betas=(.9, .999), weight_decay=.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.type == 'cuda')
    state = {'epoch': 1, 'batch': 0, 'loss_sum': 0., 'best_epoch': 0,
             'best_valid': None, 'phase': 'train', 'epoch_rng': None,
             'updates': 0, 'skipped_updates': 0}
    latest = out / 'latest.pt'
    if resume and latest.exists():
        saved = read_checkpoint(latest, expected)
        restore_model(model, saved)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        state = saved['state']
        restore_rng(saved['rng'])
        if state['phase'] == 'finished' and (out / 'completed.json').exists():
            completed = read_json(out / 'completed.json')
            if completed['identity'] != expected or completed['best_epoch'] != state['best_epoch']:
                raise ValueError('Completion conflicts with latest checkpoint')
            return completed
    elif resume and (out / 'best.pt').exists():
        raise FileNotFoundError('Best exists without complete latest checkpoint')
    # A resume process must not authorize its own reload later in that process.
    save_json(out / 'training.json', {'process': PROCESS, 'identity': expected})

    def checkpoint():
        save_checkpoint(latest, {**model_state(model), 'identity': expected,
            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
            'state': state, 'rng': rng_state()})

    while state['phase'] != 'finished':
        epoch = state['epoch']
        if state['phase'] == 'train':
            model.train()
            if state['epoch_rng'] is None:
                state['epoch_rng'] = rng_state()
            current_rng = rng_state()
            restore_rng(state['epoch_rng'])
            iterator = iter(data)
            for _ in range(state['batch']):
                next(iterator)
            if state['batch']:
                restore_rng(current_rng)
            for batch in iterator:
                optimizer.zero_grad()
                labels, text, audio, video = move_batch(batch, args)
                with amp(args):
                    loss = model(labels, text, audio, video)['Loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                skipped = optimizer_update(loss, model, optimizer, scaler)
                state['updates'] += int(not skipped)
                state['skipped_updates'] += int(skipped)
                state['batch'] += 1
                state['loss_sum'] += loss.item()
                stopped = STOP or (stop_after_batches is not None and state['batch'] >= stop_after_batches)
                if state['batch'] % 25 == 0 or state['batch'] == len(data) or stopped:
                    checkpoint()
                    print(f'Epoch {epoch}, batch {state["batch"]}/{len(data)}, loss {loss.item():.6f}', flush=True)
                if stopped:
                    raise InterruptedError('Completed batch and RNG/optimizer/loader state saved')
            state['phase'] = 'valid'
            checkpoint()
        evaluation_rng = rng_state()
        result = evaluate_predictions(model, valid_set, args, runtime, out / f'valid_{epoch:03d}.json')
        metrics = result['metrics']
        if improves(metrics[args.KeyEval], state['best_valid'], args.KeyEval):
            state['best_valid'], state['best_epoch'] = metrics[args.KeyEval], epoch
            save_checkpoint(out / 'best.pt', {**model_state(model), 'identity': expected,
                'epoch': epoch, 'metrics': metrics, 'evaluation_rng': evaluation_rng})
        save_json(out / f'epoch_{epoch:03d}.json', {'epoch': epoch, 'valid': metrics,
            'train_loss': state['loss_sum'] / len(data), 'best_epoch': state['best_epoch']})
        finished = smoke or epoch - state['best_epoch'] >= args.early_stop
        state.update(epoch=epoch + 1, batch=0, loss_sum=0., epoch_rng=None,
                     phase='finished' if finished else 'train')
        checkpoint()
    best = read_checkpoint(out / 'best.pt', expected)
    result = {'identity': expected, 'best_epoch': best['epoch'], 'metrics': best['metrics'],
              'updates': state['updates'], 'test_access': False}
    if smoke:
        changed = any(not torch.equal(initial[n], p) for n, p in model_state(model)['parameters'].items())
        frozen = all(p.grad is None and p._version == frozen_versions[n]
                     for n, p in model.named_parameters() if not p.requires_grad)
        if not changed or not frozen or state['updates'] <= 0:
            raise RuntimeError('Smoke requires effective updates and unchanged frozen parameters')
        result.update(passed=True, parameters_changed=changed, backbone_frozen=frozen)
    save_json(out / 'completed.json', result)
    return result


def replay_and_evaluate(args, variant, out, runtime, resources, smoke=False):
    """Always rebuild and replay best-valid exactly before any held-out access."""
    out = Path(out)
    expected = identity(args, variant, resources, smoke)
    if not smoke:
        require_smoke(out, expected)
    if read_json(out / 'run.json') != expected:
        raise ValueError('Run configuration/resources changed')
    if read_json(out / 'training.json')['process'] == PROCESS:
        raise RuntimeError('Reload requires a fresh process')
    if (out / 'test.json').exists():
        raise FileExistsError('Held-out test already exists; refusing to overwrite')
    completion = read_json(out / 'completed.json')
    latest = read_checkpoint(out / 'latest.pt', expected)
    best = read_checkpoint(out / 'best.pt', expected)
    if (completion['identity'] != expected or latest['state']['phase'] != 'finished'
            or completion['best_epoch'] != best['epoch']
            or latest['state']['best_epoch'] != best['epoch']
            or latest['state']['best_valid'] != best['metrics'][args.KeyEval]
            or completion['metrics'] != best['metrics']):
        raise ValueError('Training completion/selection mismatch')
    seed_all(args.seed, args.device)
    data = runtime.load_split(args, 'valid')
    args.seq_lens = data.get_seq_len()
    if smoke:
        if completion.get('passed') is not True:
            raise ValueError('Smoke did not pass')
        data = Subset(data, range(8))
    model = runtime.make_model(args, variant)
    restore_model(model, best)
    restore_rng(best['evaluation_rng'])
    result = evaluate_predictions(model, data, args, runtime, out / 'reload.json')
    if result != read_json(out / f'valid_{best["epoch"]:03d}.json') or result['metrics'] != best['metrics']:
        raise AssertionError('Exact full best-validation replay differs; held-out test forbidden')
    marker = {'identity': expected, 'best_epoch': best['epoch'],
              'exact_validation_replay': True, 'test_access': False}
    save_json(out / 'reloaded.json', marker)
    if not smoke:
        # No marker can bypass the replay: this is the only test-loading site.
        del data
        data = runtime.load_split(args, 'test')
        restore_rng(latest['rng'])
        result = evaluate_predictions(model, data, args, runtime, out / 'test.json')
        save_json(out / 'evaluated.json', {**marker, 'test_access': True, 'metrics': result['metrics']})
    return result
