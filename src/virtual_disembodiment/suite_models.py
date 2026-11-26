"""Observation-prior dynamics and nonrecursive/shortcut model alternatives."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .neural_dynamics import CoupledDynamics,EncodedHistory


class PriorDynamics(CoupledDynamics):
    def __init__(self,*args,observation_prior='scratch',offline_encoder_parameters=0,**kwargs):
        super().__init__(*args,**kwargs);self.observation_prior=observation_prior
        self.offline_encoder_parameters=int(offline_encoder_parameters)
        nd=128 if observation_prior=='sleepfm_frozen' else 200 if observation_prior=='labram_frozen' else 0
        if nd:
            self.neural_prior=nn.Linear(nd+1,self.dimension)
            self.shared_prior=nn.Linear(nd+1,self.dimension)
            if self.kind=='memory_replacement':self.memory_prior=nn.Linear(nd+1,self.dimension)
        if observation_prior=='sleepfm_frozen' and self.has_peripheral and self.kind!='memory_replacement':
            self.body_prior=nn.Linear(129,self.dimension)

    @staticmethod
    def prior_input(values,mask):return torch.cat([values*mask.to(values.dtype),mask.to(values.dtype)],dim=-1)

    def encode_history(self,neural,neural_mask,body,body_mask,forcing,observation_dropout=0.,remove_body_measurements=False,
                       prior_neural=None,prior_neural_mask=None,prior_body=None,prior_body_mask=None,**unused):
        if self.observation_prior=='scratch':
            return super().encode_history(neural,neural_mask,body,body_mask,forcing,observation_dropout,remove_body_measurements)
        self._check_history(neural,neural_mask,body,body_mask,forcing)
        if prior_neural is None or prior_neural_mask is None:raise ValueError('Frozen neural observations are required')
        nm=neural_mask.to(neural.dtype);pm=body_mask.to(body.dtype);dropped=torch.zeros_like(neural_mask,dtype=torch.bool)
        if observation_dropout:
            if not self.training or not 0<=observation_dropout<1:raise ValueError('Invalid declared training mask')
            dropped=(torch.rand_like(neural)<observation_dropout)&neural_mask.bool();nm=nm*(~dropped).to(nm.dtype)
        ninput=torch.cat([neural*nm,nm,forcing],dim=-1)
        prior=self.prior_input(prior_neural,prior_neural_mask)
        initial_neural=self.brain_input(ninput)+self.neural_prior(prior)
        ne=initial_neural;ce=self.shared_input(ninput)+self.shared_prior(prior);pe=None
        if self.has_peripheral:
            if self.kind=='memory_replacement':pe=self.peripheral_input(ninput)+self.memory_prior(prior)
            else:
                if remove_body_measurements:pm=torch.zeros_like(pm)
                pe=self.peripheral_input(torch.cat([body*pm,pm,forcing],dim=-1))
                if hasattr(self,'body_prior'):
                    if prior_body is None or prior_body_mask is None:raise ValueError('Frozen peripheral observations are required')
                    if remove_body_measurements:prior_body=torch.zeros_like(prior_body);prior_body_mask=torch.zeros_like(prior_body_mask)
                    pe=pe+self.body_prior(self.prior_input(prior_body,prior_body_mask))
            if self.kind!='shared_driver':ne=ne+self.ascending(pe)
            pe=pe+self.descending(initial_neural)
        nb,ncache=self.brain(ne);cc,ccache=self.shared(ce)
        pp,pcache=(None,None) if pe is None else self.peripheral(pe)
        return EncodedHistory(nb[:,-1],None if pp is None else pp[:,-1],cc[:,-1],ncache,pcache,ccache,
                    neural.shape[1],self.reconstruction_head(nb),neural,dropped)

    def parameter_inventory(self):
        value=super().parameter_inventory();value['offline_frozen_observation_parameters']=self.offline_encoder_parameters
        value['total']+=self.offline_encoder_parameters;value['observation_prior']=self.observation_prior
        value['reconstruction_scope']='masked handcrafted features; unmasked completed-block prior is an alternate past view'
        return value


class FeedForwardAlternative(nn.Module):
    """Forecast horizons are parallel readouts; no generated brain/body feedback."""
    KINDS=('feedforward_multimodal','drug_only','procedure_time_only','body_only','missingness_only','metadata_only')
    def __init__(self,kind='feedforward_multimodal',context=60,width=1920,depth=3,metadata_dim=35,
                 observation_prior='scratch',offline_encoder_parameters=0):
        super().__init__()
        if kind not in self.KINDS:raise ValueError('Unknown nonrecursive comparator')
        self.kind=kind;self.context=context;self.dimension=width;self.neural_dim=16;self.body_dim=14
        self.has_peripheral=kind in ('feedforward_multimodal','body_only')
        self.observation_prior=observation_prior;self.offline_encoder_parameters=int(offline_encoder_parameters)
        if observation_prior!='scratch' and kind!='feedforward_multimodal':
            raise ValueError('Shortcut controls cannot receive an extra neural or peripheral observation view')
        per_token={'feedforward_multimodal':62,'drug_only':2,'body_only':28,'missingness_only':30}
        size=context*per_token[kind] if kind in per_token else 4 if kind=='procedure_time_only' else metadata_dim
        if observation_prior!='scratch':
            nd=128 if observation_prior=='sleepfm_frozen' else 200 if observation_prior=='labram_frozen' else None
            if nd is None:raise ValueError('Unknown frozen observation prior')
            self.neural_prior=nn.Linear(nd+1,16);size+=context*17
            if observation_prior=='sleepfm_frozen':self.body_prior=nn.Linear(129,14);size+=context*15
        self.input=nn.Linear(size,width)
        self.hidden=nn.Sequential(*[nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU()) for _ in range(depth)])
        self.horizon=nn.Embedding(15,64);self.neural_head=nn.Linear(width+68,32)
        if self.has_peripheral:self.body_head=nn.Linear(width+68,28)
        self.reconstruction_head=nn.Linear(width,context*16)

    def encode_history(self,neural,neural_mask,body,body_mask,forcing,observation_dropout=0.,history_time_sec=None,
                       static_metadata=None,remove_body_measurements=False,prior_neural=None,prior_neural_mask=None,
                       prior_body=None,prior_body_mask=None,**unused):
        if neural.shape[1]!=self.context:raise ValueError('Feedforward context differs from its input projection')
        nm=neural_mask.to(neural.dtype);pm=body_mask.to(body.dtype);dropped=torch.zeros_like(neural_mask,dtype=torch.bool)
        if observation_dropout:
            if not self.training or not 0<=observation_dropout<1:raise ValueError('Invalid training mask')
            dropped=(torch.rand_like(neural)<observation_dropout)&neural_mask.bool();nm=nm*(~dropped).to(nm.dtype)
        if remove_body_measurements:pm=torch.zeros_like(pm)
        if self.kind=='feedforward_multimodal':
            fields=[neural*nm,nm,body*pm,pm,forcing]
            if hasattr(self,'neural_prior'):
                if prior_neural is None or prior_neural_mask is None:raise ValueError('Frozen neural observations are required')
                fields.extend([self.neural_prior(PriorDynamics.prior_input(prior_neural,prior_neural_mask)),prior_neural_mask.to(neural.dtype)])
            if hasattr(self,'body_prior'):
                if prior_body is None or prior_body_mask is None:raise ValueError('Frozen peripheral observations are required')
                if remove_body_measurements:prior_body=torch.zeros_like(prior_body);prior_body_mask=torch.zeros_like(prior_body_mask)
                fields.extend([self.body_prior(PriorDynamics.prior_input(prior_body,prior_body_mask)),prior_body_mask.to(neural.dtype)])
            x=torch.cat(fields,dim=-1).flatten(1)
        elif self.kind=='drug_only':x=forcing.flatten(1)
        elif self.kind=='body_only':x=torch.cat([body*pm,pm],dim=-1).flatten(1)
        elif self.kind=='missingness_only':x=torch.cat([neural_mask,body_mask],dim=-1).to(neural.dtype).flatten(1)
        elif self.kind=='procedure_time_only':
            if history_time_sec is None:raise ValueError('Procedure-time control requires the real elapsed clock')
            t=history_time_sec[:,-1]/3600.
            x=torch.cat([t,torch.log1p(t),torch.sin(t),torch.cos(t)],dim=-1)
        else:
            if static_metadata is None:raise ValueError('Metadata-only control requires baseline static fields')
            x=static_metadata
        z=self.hidden(F.gelu(self.input(x)))
        reconstruction=self.reconstruction_head(z).reshape(len(z),self.context,16)
        return EncodedHistory(z,None,z,None,None,None,self.context,reconstruction,neural,dropped)

    def simulate(self,encoded,future_forcing,**kwargs):
        if any(k!='innovations' for k in kwargs):
            raise ValueError('A nonrecursive predictor has no message gate; this intervention is not applicable')
        b,t,_=future_forcing.shape
        if t!=15:raise ValueError('The nonrecursive comparator has15 registered horizon readouts')
        if self.kind not in ('feedforward_multimodal','drug_only'):future_forcing=torch.zeros_like(future_forcing)
        # Only the forcing prefix through each readout time can enter that readout.
        horizon=self.horizon(torch.arange(t,device=future_forcing.device))[None].expand(b,-1,-1)
        x=torch.cat([encoded.brain[:,None].expand(-1,t,-1),horizon,future_forcing,future_forcing.cumsum(dim=1)],dim=-1)
        mean,raw=self.neural_head(x).chunk(2,dim=-1)
        output={'neural_mean':mean,'neural_scale':F.softplus(raw)+1e-4,
                'dynamics_consistency':mean.new_zeros((b,t)),'history_reconstruction':encoded.reconstruction,
                'history_reconstruction_target':encoded.history_reconstruction_target,
                'history_reconstruction_mask':encoded.history_reconstruction_mask}
        if self.has_peripheral:
            bm,bs=self.body_head(x).chunk(2,dim=-1);output.update(body_mean=bm,body_scale=F.softplus(bs)+1e-4)
        return output

    def parameter_inventory(self):
        total=sum(p.numel() for p in self.parameters())
        return {'total':total+self.offline_encoder_parameters,'trainable':total,'kind':self.kind,'backbone':'nonrecursive_feedforward',
                'dimension':self.dimension,'context_sec':self.context,
                'future_forcing_used':self.kind in ('feedforward_multimodal','drug_only'),
                'dynamics_consistency_auxiliary':'not_applicable_zero_weight_effect',
                'offline_frozen_observation_parameters':self.offline_encoder_parameters,'observation_prior':self.observation_prior}
