"""Verified, frozen upstream observation encoders. No clinical labels or future data."""
from __future__ import annotations
import argparse
import ast
from functools import partial
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from .run_io import file_hash,read_json

SLEEP_COMMIT='2bcbae04c3592f61352addb7ac3d4193f0a3ca25'
LABRAM_COMMIT='c431221e6cfd23dbfa9950e0180682fb322b0548'


def restricted_checkpoint(path):
    unsafe=set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    known={'argparse.Namespace','numpy.dtype','numpy.core.multiarray.scalar'}
    if not unsafe<=known:raise ValueError('Unexpected objects in a foundation checkpoint')
    allowed=[argparse.Namespace,np.dtype,(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),
             np.dtypes.Float64DType,np.dtypes.Float32DType,np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):return torch.load(path,map_location='cpu',weights_only=True)


def checked_source(root,relative):
    receipt=read_json(root/'source-receipt.json')
    if receipt['state']!='SOURCE_FETCHED_NOT_EXECUTED':raise ValueError('Source retrieval receipt is incomplete')
    record=next((f for f in receipt['files'] if f['path']==relative),None)
    if record is None or file_hash(root/relative)!=record['sha256']:raise ValueError('Upstream encoder source changed: '+relative)
    return root/relative


def import_verified(root,relative,name):
    path=checked_source(root,relative)
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module


def checked_weight(project,model_name,path):
    audit=read_json(project/'reports/encoder-checkpoint-audit-20261025.json')
    item=next(m for m in audit['models'] if m['model']==model_name)
    if item['state']!='BYTES_AND_STATE_DICTIONARY_VERIFIED' or file_hash(path)!=item['checkpoint_sha256']:
        raise ValueError('Foundation weight differs from its upstream Git-blob audit')
    return item


def electrode_vocabulary(root):
    """Read the publisher's literal channel list without importing training utils."""
    tree=ast.parse(checked_source(root,'utils.py').read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='standard_1020' for n in node.targets):
            names=ast.literal_eval(node.value)
            if not all(isinstance(n,str) for n in names):break
            return [n.upper() for n in names]
    raise ValueError('Publisher channel vocabulary is unavailable')


def mapped_positions(position_table,montage,vocabulary):
    """Known bipolar tokens stay exact; new bipolar tokens use endpoint means.

    Endpoint averaging is an explicit spatial observation adapter. It does not
    reconstruct monopolar electrodes from two bipolar measurements.
    """
    positions=[position_table[:,0:1]];mapping=[]
    if montage is None:raise ValueError('Unknown montage cannot be assigned a fabricated electrode location')
    for channel in montage:
        name=channel.upper()
        if name in vocabulary:
            index=vocabulary.index(name)+1;positions.append(position_table[:,index:index+1]);mapping.append('publisher_token')
        else:
            parts=name.split('-')
            if len(parts)!=2 or any(p not in vocabulary for p in parts):raise ValueError('Unsupported electrode names: '+name)
            indices=[vocabulary.index(p)+1 for p in parts]
            positions.append(position_table[:,indices].mean(dim=1,keepdim=True));mapping.append('bipolar_endpoint_mean')
    return torch.cat(positions,dim=1),mapping


class FrozenObservationEncoders:
    def __init__(self,project,device='cpu'):
        self.project=Path(project);self.device=torch.device(device)
        sr=self.project/'data/vendor_sources/sleepfm-clinical'/SLEEP_COMMIT
        lr=self.project/'data/vendor_sources/LaBraM'/LABRAM_COMMIT
        sleep_mod=import_verified(sr,'sleepfm/models/models.py','virtual_upstream_sleepfm_'+SLEEP_COMMIT)
        finetune=import_verified(lr,'modeling_finetune.py','virtual_upstream_labram_blocks_'+LABRAM_COMMIT)
        previous=sys.modules.get('modeling_finetune');sys.modules['modeling_finetune']=finetune
        try:labram_mod=import_verified(lr,'modeling_pretrain.py','virtual_upstream_labram_pretrain_'+LABRAM_COMMIT)
        finally:
            if previous is None:sys.modules.pop('modeling_finetune',None)
            else:sys.modules['modeling_finetune']=previous
        config_path=checked_source(sr,'sleepfm/checkpoints/model_base/config.json')
        config=read_json(config_path)
        sleep_path=self.project/'data/raw/sleepfm_base'/SLEEP_COMMIT/'best.pt'
        labram_path=self.project/'data/raw/labram_base'/LABRAM_COMMIT/'labram-base.pth'
        self.source_audits={'sleepfm':checked_weight(self.project,'sleepfm',sleep_path),
                           'labram':checked_weight(self.project,'labram',labram_path)}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.sleep=sleep_mod.SetTransformer(in_channels=config['in_channels'],patch_size=config['patch_size'],
                    embed_dim=config['embed_dim'],num_heads=config['num_heads'],num_layers=config['num_layers'],
                    pooling_head=config['pooling_head'],dropout=config['dropout'])
            self.labram=labram_mod.NeuralTransformerForMaskedEEGModeling(patch_size=200,embed_dim=200,depth=12,
                    num_heads=10,qkv_bias=False,qk_norm=partial(nn.LayerNorm,eps=1e-6),
                    norm_layer=partial(nn.LayerNorm,eps=1e-6),init_values=.1,use_abs_pos_emb=True)
        sleep_state=restricted_checkpoint(sleep_path)['state_dict']
        self.sleep.load_state_dict({k.removeprefix('module.'):v for k,v in sleep_state.items()},strict=True)
        all_state=restricted_checkpoint(labram_path)['model']
        student={k.removeprefix('student.'):v for k,v in all_state.items() if k.startswith('student.')}
        self.labram.load_state_dict(student,strict=True)
        # The verified pretraining vocabulary head is not an observation feature.
        self.labram.lm_head=nn.Identity()
        self.vocabulary=electrode_vocabulary(lr)
        for model in (self.sleep,self.labram):model.requires_grad_(False);model.eval();model.to(self.device)

    @torch.inference_mode()
    def sleep_features(self,signals,channel_valid):
        """Five completed seconds, 128 Hz; publisher forward, frozen BN/dropout."""
        x=torch.as_tensor(signals,dtype=torch.float32,device=self.device)
        valid=torch.as_tensor(channel_valid,dtype=torch.bool,device=self.device)
        if x.ndim!=3 or x.shape[-1]!=640 or valid.shape!=x.shape[:2]:raise ValueError('Malformed SleepFM observation block')
        x=torch.where(valid[:,:,None],x,0.)
        if not torch.isfinite(x).all():raise ValueError('Nonfinite valid SleepFM samples')
        good=valid.any(dim=1);out=torch.zeros((len(x),128),device=self.device)
        if good.any():out[good]=self.sleep(x[good],~valid[good])[0]
        if not torch.isfinite(out).all():raise FloatingPointError('Nonfinite frozen SleepFM features')
        return out.cpu().numpy(),good.cpu().numpy()

    @torch.inference_mode()
    def labram_features(self,signals,channel_valid,montage):
        """Five completed seconds, 200 Hz/uV divided by100, two source bipolar channels."""
        x=torch.as_tensor(signals,dtype=torch.float32,device=self.device)
        valid=torch.as_tensor(channel_valid,dtype=torch.bool,device=self.device)
        if x.ndim!=4 or x.shape[1:]!=(2,5,200) or valid.shape!=x.shape[:2]:raise ValueError('Malformed LaBraM observation block')
        x=torch.where(valid[:,:,None,None],x,0.)
        if not torch.isfinite(x).all():raise ValueError('Nonfinite valid LaBraM samples')
        out=torch.zeros((len(x),200),device=self.device);good=valid.any(dim=1)
        if montage is None:return out.cpu().numpy(),np.zeros(len(x),bool),['unknown_source_montage']
        position,mapping=mapped_positions(self.labram.pos_embed,montage,self.vocabulary)
        if good.any():
            xx=x[good];mask=(~valid[good]).repeat_interleave(5,dim=1)
            patches=self.labram.patch_embed(xx)
            patches=torch.where(mask[:,:,None],self.labram.mask_token,patches)
            tokens=torch.cat([self.labram.cls_token.expand(len(xx),-1,-1),patches],dim=1)
            spatial=position[:,1:].unsqueeze(2).expand(len(xx),2,5,-1).flatten(1,2)
            # Match this pinned pretraining forward exactly: its CLS position is
            # the first expanded channel position (modeling_pretrain.py:143–144).
            tokens=tokens+torch.cat([spatial[:,:1],spatial],dim=1)
            tokens[:,1:]+=self.labram.time_embed[:,:5].unsqueeze(1).expand(len(xx),2,5,-1).flatten(1,2)
            tokens=self.labram.pos_drop(tokens)
            for block in self.labram.blocks:tokens=block(tokens,rel_pos_bias=None)
            tokens=self.labram.norm(tokens)[:,1:]
            usable=(~mask).to(tokens.dtype)
            out[good]=(tokens*usable[:,:,None]).sum(dim=1)/usable.sum(dim=1,keepdim=True).clamp_min(1)
        if not torch.isfinite(out).all():raise FloatingPointError('Nonfinite frozen LaBraM features')
        return out.cpu().numpy(),good.cpu().numpy(),mapping

    def inventory(self):
        return {'sleepfm_frozen_parameters':sum(p.numel() for p in self.sleep.parameters()),
                'labram_frozen_encoder_parameters':sum(p.numel() for p in self.labram.parameters()),
                'trainable_parameters':0,'checkpoint_audits':self.source_audits,
                'pretraining_overlap_status':'not_established_no_claim_of_external_independence',
                'labram_unused_pretraining_prediction_head_removed':True}
