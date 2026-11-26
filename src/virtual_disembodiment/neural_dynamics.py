"""Trainable causal dynamical models with explicit brain/body/shared routes.

The peripheral-to-neural route has a single addressable message interface.
Neither the shared driver nor the readout receives future body observations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def positional_encoding(length, dimension, offset, device, dtype):
    at=torch.arange(offset,offset+length,device=device,dtype=torch.float32)
    frequency=torch.exp(torch.arange(0,dimension,2,device=device,dtype=torch.float32)*(-math.log(10000)/dimension))
    out=torch.zeros(length,dimension,device=device,dtype=torch.float32)
    out[:,0::2]=torch.sin(at[:,None]*frequency)
    out[:,1::2]=torch.cos(at[:,None]*frequency[:out[:,1::2].shape[1]])
    return out.to(dtype)


class CausalAttentionBlock(nn.Module):
    def __init__(self,dimension,heads):
        super().__init__()
        if dimension%heads:raise ValueError('Attention dimension must divide by head count')
        self.dimension=dimension;self.heads=heads
        self.norm1=nn.LayerNorm(dimension);self.qkv=nn.Linear(dimension,dimension*3)
        self.proj=nn.Linear(dimension,dimension);self.norm2=nn.LayerNorm(dimension)
        self.ff=nn.Sequential(nn.Linear(dimension,dimension*4),nn.GELU(),nn.Linear(dimension*4,dimension))

    def forward(self,x,cache=None):
        b,l,d=x.shape
        qkv=self.qkv(self.norm1(x)).reshape(b,l,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        previous=0
        if cache is not None:
            previous=cache[0].shape[2]
            k=torch.cat([cache[0],k],dim=2);v=torch.cat([cache[1],v],dim=2)
        if not previous:
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        elif l==1:
            y=F.scaled_dot_product_attention(q,k,v,is_causal=False)
        else:
            visible=torch.arange(previous+l,device=x.device)[None,:] <= (previous+torch.arange(l,device=x.device)[:,None])
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=visible)
        y=y.transpose(1,2).reshape(b,l,d)
        x=x+self.proj(y)
        x=x+self.ff(self.norm2(x))
        return x,(k,v)


class CausalStack(nn.Module):
    def __init__(self,dimension,layers,heads=8):
        super().__init__();self.blocks=nn.ModuleList([CausalAttentionBlock(dimension,heads) for _ in range(layers)])
        self.final_norm=nn.LayerNorm(dimension);self.dimension=dimension

    def forward(self,x,cache=None,offset=0):
        x=x+positional_encoding(x.shape[1],self.dimension,offset,x.device,x.dtype)[None]
        result=[]
        for i,block in enumerate(self.blocks):
            x,c=block(x,None if cache is None else cache[i]);result.append(c)
        return self.final_norm(x),result


class RecurrentStack(nn.Module):
    def __init__(self,dimension,layers,heads=8):
        super().__init__();self.gru=nn.GRU(dimension,dimension,layers,batch_first=True)
        self.norm=nn.LayerNorm(dimension)

    def forward(self,x,cache=None,offset=0):
        y,c=self.gru(x,cache)
        return self.norm(y),c


@dataclass
class EncodedHistory:
    brain:torch.Tensor
    peripheral:torch.Tensor|None
    shared:torch.Tensor
    brain_cache:Any
    peripheral_cache:Any
    shared_cache:Any
    length:int
    reconstruction:torch.Tensor
    history_reconstruction_target:torch.Tensor
    history_reconstruction_mask:torch.Tensor


class CoupledDynamics(nn.Module):
    """Causal transformer primary backbone, gated recurrent replication.

    kind='brain_only' has no peripheral encoder, dynamics or auxiliary body
    target. 'memory_replacement' uses an internal module driven by neural
    history instead of measured body data. 'shared_driver' has no ascending
    route, including during history encoding.
    """
    def __init__(self,neural_dim=16,body_dim=14,forcing_dim=2,dimension=256,layers=8,
                 backbone='causal_transformer',kind='coupled'):
        super().__init__()
        if kind not in ('coupled','brain_only','shared_driver','memory_replacement'):
            raise ValueError('Unrecognized mechanism model')
        if backbone not in ('causal_transformer','gated_recurrent'):
            raise ValueError('Unrecognized backbone')
        self.kind=kind;self.dimension=dimension;self.neural_dim=neural_dim
        self.body_dim=body_dim;self.forcing_dim=forcing_dim;self.backbone=backbone
        self.has_peripheral=kind!='brain_only'
        stack=CausalStack if backbone=='causal_transformer' else RecurrentStack
        self.brain_input=nn.Linear(2*neural_dim+forcing_dim,dimension)
        self.shared_input=nn.Linear(2*neural_dim+forcing_dim,dimension)
        self.brain=stack(dimension,layers);self.shared=stack(dimension,layers)
        self.brain_step=nn.Linear(2*dimension+forcing_dim,dimension)
        self.shared_step=nn.Linear(2*dimension+forcing_dim,dimension)
        self.neural_head=nn.Linear(dimension,2*neural_dim)
        self.reconstruction_head=nn.Linear(dimension,neural_dim)
        self.neural_to_state=nn.Linear(neural_dim,dimension)
        if self.has_peripheral:
            input_dim=2*(neural_dim if kind=='memory_replacement' else body_dim)+forcing_dim
            self.peripheral_input=nn.Linear(input_dim,dimension)
            self.peripheral=stack(dimension,layers)
            self.peripheral_step=nn.Linear(2*dimension+forcing_dim,dimension)
            self.ascending=nn.Linear(dimension,dimension,bias=False)
            self.descending=nn.Linear(dimension,dimension,bias=False)
            self.body_head=nn.Linear(dimension,2*body_dim)
        self.log_state_noise=nn.Parameter(torch.full((3 if self.has_peripheral else 2,dimension),-5.0))

    def _check_history(self,neural,neural_mask,body,body_mask,forcing):
        if neural.ndim!=3 or neural.shape[-1]!=self.neural_dim or forcing.shape[:2]!=neural.shape[:2]:
            raise ValueError('Malformed history or forcing')
        if neural.shape!=neural_mask.shape:
            raise ValueError('Neural observation mask has wrong shape')
        if body.shape!=body_mask.shape or body.shape[:2]!=neural.shape[:2] or body.shape[-1]!=self.body_dim:
            raise ValueError('Peripheral observations and masks do not align')
        for x in (neural,body,forcing):
            if not torch.isfinite(x).all():raise ValueError('Inputs must be masked and finite before encoding')

    def encode_history(self,neural,neural_mask,body,body_mask,forcing,observation_dropout=0.,remove_body_measurements=False):
        self._check_history(neural,neural_mask,body,body_mask,forcing)
        nm=neural_mask.to(neural.dtype);pm=body_mask.to(body.dtype)
        dropped=torch.zeros_like(neural_mask,dtype=torch.bool)
        if observation_dropout:
            if not self.training or not 0<=observation_dropout<1:
                raise ValueError('Observation masking is a declared training objective only')
            dropped=(torch.rand_like(neural)<observation_dropout)&neural_mask.bool()
            nm=nm*(~dropped).to(nm.dtype)
        ninput=torch.cat([neural*nm,nm,forcing],dim=-1)
        ne=self.brain_input(ninput)
        ce=self.shared_input(ninput)  # Shared driver never has a direct body-input route.
        pe=None
        if self.has_peripheral:
            if self.kind=='memory_replacement':
                pinput=ninput
            else:
                if remove_body_measurements:pm=torch.zeros_like(pm)
                pinput=torch.cat([body*pm,pm,forcing],dim=-1)
            pe=self.peripheral_input(pinput)
            if self.kind!='shared_driver':ne=ne+self.ascending(pe)
            pe=pe+self.descending(self.brain_input(ninput))
        nb,ncache=self.brain(ne);cc,ccache=self.shared(ce)
        pp,pcache=(None,None) if pe is None else self.peripheral(pe)
        reconstruction=self.reconstruction_head(nb)
        return EncodedHistory(nb[:,-1],None if pp is None else pp[:,-1],cc[:,-1],ncache,pcache,ccache,
                              neural.shape[1],reconstruction,neural,dropped)

    def simulate(self,encoded,future_forcing,ascending_gain=1.,descending_gain=1.,
                 innovations=None,message_delay=0,reverse_history_messages=False,
                 peripheral_initial=None,damage_mask=None,controller=None,controller_source=None):
        if future_forcing.ndim!=3 or future_forcing.shape[-1]!=self.forcing_dim:
            raise ValueError('Explicit future forcing must have shape batch x steps x forcing_dim')
        if not (math.isfinite(ascending_gain) and math.isfinite(descending_gain) and
                0<=ascending_gain<=1 and 0<=descending_gain<=1):
            raise ValueError('Registered gate gains lie within [0,1]')
        if message_delay<0:raise ValueError('Negative delays would access ungenerated future messages')
        if controller is not None and controller_source not in ('closed_loop_synthetic','autonomous_peripheral'):
            raise ValueError('Controller source is not an allowed autonomous rescue')
        if controller_source=='actual_future_observation':
            raise ValueError('Actual future physiology cannot enter free rollout')
        b,t,_=future_forcing.shape;d=self.dimension
        components=3 if self.has_peripheral else 2
        if innovations is None:
            innovations=torch.randn(b,t,components,d,device=encoded.brain.device,dtype=encoded.brain.dtype)
        if innovations.shape!=(b,t,components,d):raise ValueError('Paired innovations have wrong shape')
        n,p,c=encoded.brain,encoded.peripheral,encoded.shared
        if peripheral_initial is not None:
            if p is None or peripheral_initial.shape!=p.shape:raise ValueError('Swapped peripheral state has wrong dimensions')
            p=peripheral_initial
        nc,pc,cc=encoded.brain_cache,encoded.peripheral_cache,encoded.shared_cache
        means=[];scales=[];body_means=[];body_scales=[];latent_penalties=[]
        messages=[]
        noise=F.softplus(self.log_state_noise).to(n.dtype)
        for step in range(t):
            u=future_forcing[:,step].to(n.dtype)
            ntoken=self.brain_step(torch.cat([n,c,u],dim=-1))
            ctoken=self.shared_step(torch.cat([c,n,u],dim=-1))
            ptoken=None
            if p is not None:
                message=p
                if controller is not None:
                    message=controller(n,p,c,u,step)
                    if message.shape!=p.shape:raise ValueError('Rescue controller changed interface dimensions')
                messages.append(message)
                if message_delay:
                    # Earlier generated messages only; initial replay retains scale.
                    message=messages[max(0,step-message_delay)]
                if reverse_history_messages:
                    raise ValueError('Use a frozen history surrogate provider, not future rollout reversal')
                gain=0. if self.kind=='shared_driver' else ascending_gain
                ntoken=ntoken+gain*self.ascending(message)
                ptoken=self.peripheral_step(torch.cat([p,c,u],dim=-1))+descending_gain*self.descending(n)
            if damage_mask is not None:
                if damage_mask.shape[-1]!=d:raise ValueError('Damage mask must name neural latent dimensions')
                ntoken=ntoken*damage_mask
            ntoken=ntoken+noise[0]*innovations[:,step,0]
            ctoken=ctoken+noise[1]*innovations[:,step,1]
            nn,nc=self.brain(ntoken[:,None],nc,encoded.length+step)
            cn,cc=self.shared(ctoken[:,None],cc,encoded.length+step)
            n,c=nn[:,0],cn[:,0]
            if ptoken is not None:
                ptoken=ptoken+noise[2]*innovations[:,step,2]
                pn,pc=self.peripheral(ptoken[:,None],pc,encoded.length+step);p=pn[:,0]
            mean,log_scale=self.neural_head(n).chunk(2,dim=-1)
            scale=F.softplus(log_scale)+1e-4
            means.append(mean);scales.append(scale)
            latent_penalties.append((self.neural_to_state(mean)-n).square().mean(dim=-1))
            if p is not None:
                bm,bs=self.body_head(p).chunk(2,dim=-1)
                body_means.append(bm);body_scales.append(F.softplus(bs)+1e-4)
        result={'neural_mean':torch.stack(means,dim=1),'neural_scale':torch.stack(scales,dim=1),
                'dynamics_consistency':torch.stack(latent_penalties,dim=1),
                'history_reconstruction':encoded.reconstruction,
                'history_reconstruction_target':encoded.history_reconstruction_target,
                'history_reconstruction_mask':encoded.history_reconstruction_mask}
        if body_means:
            result.update(body_mean=torch.stack(body_means,dim=1),body_scale=torch.stack(body_scales,dim=1))
        return result

    def intervene(self,encoded,future_forcing,**kwargs):
        return self.simulate(encoded,future_forcing,**kwargs)

    def restore(self,encoded,future_forcing,innovations):
        return self.simulate(encoded,future_forcing,innovations=innovations)

    def parameter_inventory(self):
        return {'total':sum(p.numel() for p in self.parameters()),
                'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'backbone':self.backbone,'kind':self.kind,'dimension':self.dimension}


def gaussian_crps(mean,scale,target):
    mean,scale,target=mean.float(),scale.float().clamp_min(1e-6),target.float()
    z=(target-mean)/scale
    return scale*(z*torch.erf(z/math.sqrt(2))+math.sqrt(2/math.pi)*torch.exp(-z*z/2)-1/math.sqrt(math.pi))


def probabilistic_training_loss(prediction,batch,weights=(.1,.1,.1)):
    y=batch['target_neural'].float();mask=batch['target_neural_mask'].float()
    mean=prediction['neural_mean'].float();scale=prediction['neural_scale'].float().clamp_min(1e-5)
    nll=.5*((y-mean)/scale).square()+torch.log(scale)
    neural=((nll*mask).sum(dim=(1,2))/mask.sum(dim=(1,2)).clamp_min(1)).mean()
    peripheral=neural.new_zeros(())
    if 'body_mean' in prediction:
        b=batch['loss_target_body'].float();bm=batch['loss_target_body_mask'].float()
        bs=prediction['body_scale'].float().clamp_min(1e-5)
        bnll=.5*((b-prediction['body_mean'].float())/bs).square()+torch.log(bs)
        peripheral=((bnll*bm).sum(dim=(1,2))/bm.sum(dim=(1,2)).clamp_min(1)).mean()
    rm=prediction['history_reconstruction_mask'].float()
    residual=(prediction['history_reconstruction'].float()-prediction['history_reconstruction_target'].float()).square()
    reconstruction=((residual*rm).sum(dim=(1,2))/rm.sum(dim=(1,2)).clamp_min(1)).mean()
    consistency=prediction['dynamics_consistency'].float().mean()
    loss=neural+weights[0]*peripheral+weights[1]*reconstruction+weights[2]*consistency
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
    return loss,{'neural_nll':float(neural.detach()),'peripheral_nll':float(peripheral.detach()),
                 'reconstruction':float(reconstruction.detach()),'consistency':float(consistency.detach())}
