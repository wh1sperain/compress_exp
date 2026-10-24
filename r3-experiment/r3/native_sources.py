"""Load unmodified algorithm functions from hash-checked, frozen author files.

AST extraction avoids importing five incompatible packages called `llava` into
one process. It is for selector validation, not a replacement for model loading.
"""
from __future__ import annotations
import ast
import logging
from pathlib import Path
from .common import Blocked, digest, read


def require_adaptive_agile():
    import os
    # The author branches on presence, so even an empty value must be rejected.
    if 'DIST_THRESHOLD' in os.environ:
        raise Blocked('AgilePruner DIST_THRESHOLD overrides adaptive thresholds')


class AuthorSources:
    def __init__(self, lock_path):
        self.lock = read(lock_path)

    def file(self, repo, relative):
        entry = self.lock[repo]
        path = Path(entry['path']) / relative
        expected = entry['files'].get(relative)
        if not expected or digest(path.read_bytes()) != expected:
            raise Blocked('Author file differs from source lock: '+relative)
        return path

    def functions(self, repo, relative, names, class_name=None, constants=()):
        import torch
        path = self.file(repo, relative)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        nodes = tree.body
        if class_name:
            nodes = next(n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name)
        chosen = []
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                chosen.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
                chosen.append(node)
        if {n.name for n in chosen if isinstance(n, ast.FunctionDef)} != set(names):
            raise Blocked('Required author functions missing: '+relative)
        # Deferred annotations preserve source annotations without importing the
        # incompatible enclosing model. Bodies and decorators are unchanged.
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + chosen, type_ignores=[])
        scope = {'torch': torch, 'logger': logging.getLogger('author_reference')}
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
        return scope

    def priortr(self, backbone):
        if backbone not in ('llava', 'qwen3'):
            raise ValueError(backbone)
        relative = ('image/LLaVA/llava/vtr/strategy/priortr.py' if backbone == 'llava'
                    else 'image/Qwen3-VL/visual_token_pruning/strategy/priortr.py')
        return self.functions('PriorTR', relative, ['compute_scores'], 'PriorTRStrategy')['compute_scores']

    def agile(self):
        require_adaptive_agile()
        return self.functions('AgilePruner', 'llava/model/llava_arch.py',
                              ['calculate_adaptive_tau', 'effective_rank', 'select_diverse_tokens_by_attention_and_distance'],
                              constants=('ERANK_AVG_REF', 'TAU_MAX'))

    def visionzip(self):
        return self.functions('VisionZip', 'visionzip/clip_encoder.py', ['forward'],
                              'CLIPVisionTower_VisionZip')['forward']


def visionzip_arguments(keep):
    if keep not in (144, 288):
        raise ValueError('Unregistered VisionZip budget')
    contextual = int((keep-1)*10/64+0.5)
    dominant_patches = keep-contextual-1
    # Public visionzip() subtracts one before assigning _info['dominant'].
    return {'dominant': dominant_patches+1, 'contextual': contextual}
