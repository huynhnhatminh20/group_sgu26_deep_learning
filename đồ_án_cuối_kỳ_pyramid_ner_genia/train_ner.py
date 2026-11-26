import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os

from utils import *
from data import *
from models import *


import argparse
import hashlib
import json
import random
from pathlib import Path

BIOWORDVEC_SOURCE = 'https://ndownloader.figshare.com/files/12551780'
BIOWORDVEC_SHA256 = '8ccc18c568dd9b754e622af2aaa116b8c200537da38d2119cbc89a15714bf050'

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def none_or_str(value):
    if value == 'None':
        return None
    return value

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

parser = argparse.ArgumentParser(description='Arguments for training.')

#### Train
parser.add_argument('--model_class',
                    default='SeqProto',
                    action='store',)

parser.add_argument('--model_read_ckpt',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--model_write_ckpt',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--pretrained_wv',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--dataset',
                    default='ACE05',
                    action='store',)
parser.add_argument('--genia_protocol', choices=('original', 'cleaned'))
parser.add_argument('--final_eval_only', action='store_true')
parser.add_argument('--seed', type=int)
parser.add_argument('--num_workers', default=1, type=int)

parser.add_argument('--max_depth',
                    default=None, type=none_or_int,
                    action='store',)

parser.add_argument('--batch_size',
                    default=32, type=int,
                    action='store',)

parser.add_argument('--evaluate_interval',
                    default=1000, type=int,
                    action='store',)

parser.add_argument('--max_steps',
                    default=int(1e9), type=int,
                    action='store')

parser.add_argument('--max_epoches',
                    default=100, type=int,
                    action='store')
parser.add_argument('--resume_from', default=None)

parser.add_argument('--early_stopping_min_epochs', default=0, type=int)
parser.add_argument('--early_stopping_patience', default=0, type=int)
parser.add_argument('--early_stopping_min_delta', default=0.0, type=float)

parser.add_argument('--decay_rate',
                    default=0.05, type=float,
                    action='store')

#### Model Config
parser.add_argument('--token_emb_dim',
                    default=None, type=int,
                    action='store',)

parser.add_argument('--char_encoder',
                    default='lstm',
                    action='store',)

parser.add_argument('--char_emb_dim',
                    default=0, type=int,
                    action='store',)

parser.add_argument('--cased',
                    default=None, type=int,
                    action='store',)

parser.add_argument('--hidden_dim',
                    default=200, type=int,
                    action='store',)

parser.add_argument('--loss_reduction',
                    default='sum',
                    action='store',)

parser.add_argument('--maxlen',
                    default=None, type=int,
                    action='store',)

parser.add_argument('--dropout',
                    default=0.3, type=float,
                    action='store',)

parser.add_argument('--optimizer',
                    default='sgd',
                    action='store',)

parser.add_argument('--lr',
                    default=0.01, type=float,
                    action='store',)

parser.add_argument('--vocab_size',
                    default=20000, type=int,
                    action='store',)

parser.add_argument('--vocab_file',
                    default=None, type=none_or_str,
                    action='store',)

parser.add_argument('--tag_vocab_size',
                    default=100, type=int,
                    action='store',)

parser.add_argument('--tag_form',
                    default='iob2',
                    action='store',)

parser.add_argument('--freeze_wv',
                    default=None, type=int,
                    action='store',)

parser.add_argument('--lm_emb_path',
                    default=None,
                    action='store',)

parser.add_argument('--lm_emb_dim',
                    default=0, type=int,
                    action='store',)
parser.add_argument('--ablation', default='none',
                    choices=('none', 'no_char', 'no_layernorm',
                             'independent_decoder', 'mean_pool', 'max_pool'))

parser.add_argument('--device',
                    default=None, type=none_or_str,
                    action='store',)


def main():
    args = parser.parse_args()
    if args.final_eval_only and not args.model_read_ckpt:
        parser.error('--final_eval_only requires --model_read_ckpt')

    if args.dataset == 'GENIA':
        if args.genia_protocol is None:
            parser.error('GENIA requires --genia_protocol original or cleaned')
        if args.model_class not in ('PyramidNestNER', 'BiPyramidNestNER'):
            parser.error('GENIA requires PyramidNestNER or BiPyramidNestNER')
        if args.ablation != 'none' and args.model_class != 'PyramidNestNER':
            parser.error('Ablation variants require PyramidNestNER Basic')
        if args.ablation == 'no_char' and args.char_emb_dim != 0:
            parser.error('no_char requires --char_emb_dim 0')
        if args.ablation == 'independent_decoder' and args.max_depth is None:
            parser.error('independent_decoder requires a fixed --max_depth')
        if args.seed is None:
            parser.error('GENIA requires an explicit --seed')
        if args.vocab_file is not None and not args.model_read_ckpt:
            parser.error('GENIA vocabulary must be fitted from TRAIN; omit --vocab_file')
        args.token_emb_dim = 200 if args.token_emb_dim is None else args.token_emb_dim
        args.cased = 1 if args.cased is None else args.cased
        args.freeze_wv = 1 if args.freeze_wv is None else args.freeze_wv
        if args.token_emb_dim != 200 or args.freeze_wv != 1:
            parser.error('GENIA BioWordVec requires --token_emb_dim 200 and frozen word weights')
        if not args.final_eval_only and args.pretrained_wv is None and args.model_read_ckpt is None:
            args.pretrained_wv = '../BioWordVec/bio_embedding_extrinsic'
    else:
        args.token_emb_dim = 100 if args.token_emb_dim is None else args.token_emb_dim
        args.cased = 0 if args.cased is None else args.cased
        args.freeze_wv = 0 if args.freeze_wv is None else args.freeze_wv

    if args.device is None and torch.cuda.is_available():
        gpu_idx, gpu_mem = set_max_available_gpu()
        args.device = f"cuda:{gpu_idx}"
    elif args.device is None:
        args.device = 'cpu'
    if str(args.device).startswith('cuda'):
        torch.cuda.set_device(args.device)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    folder = 'protocol_clean_v1' if args.dataset == 'GENIA' and args.genia_protocol == 'cleaned' else 'unified'
    paths = {split: Path(f'./datasets/{folder}/{split}.{args.dataset}.json')
             for split in ('train', 'valid', 'test')}
    dataset_hashes = {split: sha256(path) for split, path in paths.items()}
    pretrained_hash = None
    if args.dataset == 'GENIA' and not args.final_eval_only and args.model_read_ckpt is None:
        pretrained_hash = sha256(args.pretrained_wv)
        if pretrained_hash != BIOWORDVEC_SHA256:
            raise ValueError('BioWordVec SHA-256 differs from the verified PubMed+MeSH extrinsic file')

    # Model
    config = Config(**{**args.__dict__, 'pretrained_wv_sha256': pretrained_hash,
                       'pretrained_wv_path': args.pretrained_wv,
                       'pretrained_wv_source': BIOWORDVEC_SOURCE if args.dataset == 'GENIA' else None,
                       'dataset_hashes': dataset_hashes})
    ModelClass = globals().get(args.model_class)
    if not isinstance(ModelClass, type):
        parser.error(f'Unknown model class: {args.model_class}')
    model = ModelClass.load(args.model_read_ckpt) if args.model_read_ckpt else ModelClass(config)

    if args.model_read_ckpt and args.dataset == 'GENIA':
        if model.config.genia_protocol != args.genia_protocol or model.config.dataset_hashes != dataset_hashes:
            raise ValueError('Checkpoint protocol or dataset checksums differ from this run')
        args.max_depth = model.config.max_depth
        args.tag_form = model.config.tag_form

    Trainer = model.get_default_trainer_class()
    trainer_kwargs = dict(
        model=model,
        train_path=str(paths['train']),
        test_path=str(paths['test']),
        valid_path=str(paths['valid']),
        batch_size=int(args.batch_size),
        tag_form=args.tag_form, num_workers=args.num_workers,
        max_depth=args.max_depth,
    )
    if args.dataset == 'GENIA':
        trainer_kwargs['final_evaluation_mode'] = args.final_eval_only
    trainer = Trainer(**trainer_kwargs)

    if args.pretrained_wv and not args.model_read_ckpt and not args.final_eval_only:
        if args.dataset == 'GENIA':
            stats = model.token_embedding.load_biowordvec(args.pretrained_wv)
            print('BioWordVec TRAIN vocabulary coverage:', stats)
        else:
            model.token_embedding.load_pretrained(args.pretrained_wv, freeze=bool(args.freeze_wv))

    print(json.dumps({
        'dataset_protocol': args.genia_protocol if args.dataset == 'GENIA' else folder,
        'dataset_paths': {key: str(value.resolve()) for key, value in paths.items()},
        'dataset_sha256': dataset_hashes,
        'word_vocabulary_size': len(model.token_embedding.token_indexing.vocab),
        'biowordvec_source': model.config.pretrained_wv_source,
        'biowordvec_path': model.config.pretrained_wv_path,
        'biowordvec_sha256': model.config.pretrained_wv_sha256,
        'embedding_dim': model.config.token_emb_dim,
        'word_embeddings_frozen': not model.token_embedding.token_embedding.weight.requires_grad,
        'seed': args.seed, 'model_class': args.model_class,
        'hyperparameters': vars(args),
        'checkpoint_model_config': model.config.__dict__ if args.model_read_ckpt else None,
    }, indent=2))

    if args.final_eval_only:
        result = trainer.evaluate_model(test_type='test')
        print('FINAL TEST:', {key: result[key] for key in ('precision', 'recall', 'f1')})
    else:
        if args.model_write_ckpt:
            Path(args.model_write_ckpt).parent.mkdir(parents=True, exist_ok=True)
        print("=== start training ===")
        from runs.wandb_tracking import tracking_run
        experiment_id = os.environ.get('EXPERIMENT_ID', f'{args.model_class}_seed{args.seed}')
        tracking_config = {**vars(args), 'experiment_id': experiment_id,
                           'dataset_sha256': dataset_hashes,
                           'biowordvec_sha256': pretrained_hash,
                           'upstream_commit': 'd94ed0e63a6eb2fb79118834a0911628ddda38a3'}
        output_dir = Path(args.model_write_ckpt).parent if args.model_write_ckpt else Path('runs/results')
        with tracking_run(output_dir, experiment_id, tracking_config):
            trainer.train_model(args=args)






if __name__ == '__main__':
    main()
