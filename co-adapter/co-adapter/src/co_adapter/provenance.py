"""Small source/dependency inventory, never a historical execution manifest."""
import hashlib
import importlib.metadata
from pathlib import Path

DEPENDENCIES = ('torch', 'numpy', 'scipy', 'transformers', 'tokenizers', 'scikit-learn',
                'sentencepiece', 'modelscope', 'pandas')


def inventory(upstream_root):
    package = Path(__file__).resolve().parent
    release = package.parents[1]
    local = list(package.glob('*.py')) + [release / 'requirements.txt', release / 'pyproject.toml', release / 'train.py']
    root = Path(upstream_root).resolve()
    external = [p for folder in ('models', 'data', 'config', 'utils') for p in (root / folder).rglob('*.py')]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return {'release_sources': {str(p.relative_to(release)): digest(p) for p in sorted(local) if p.exists()},
            'upstream_sources': {str(p.relative_to(root)): digest(p) for p in sorted(external)},
            'dependencies': {name: importlib.metadata.version(name) for name in DEPENDENCIES}}
