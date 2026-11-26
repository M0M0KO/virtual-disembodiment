"""Generated signals test actual dynamics routing and paired stochastic execution."""
import unittest
import numpy as np
import torch
from virtual_disembodiment.paired_interventions import (Condition,basic_conditions,noise_bank,paired_prediction,
                                                       HistoryReplay,masked_history,joint_energy_score)
from virtual_disembodiment.suite_models import PriorDynamics,FeedForwardAlternative

torch.set_num_threads(1)


def batch():
    torch.manual_seed(7)
    n=torch.randn(2,20,16);body=torch.randn(2,20,14)
    return {'history_neural':n,'history_neural_mask':torch.ones_like(n,dtype=torch.bool),
            'history_body':body,'history_body_mask':torch.ones_like(body,dtype=torch.bool),
            'history_forcing':torch.zeros(2,20,2),'future_forcing':torch.zeros(2,15,2)}


def rows():
    return [{'participant_id':'p'+str(i),'recording_id':'r'+str(i),'forecast_origin_sec':60} for i in range(2)]


class PairedInterventionTests(unittest.TestCase):
    def test_bank_is_window_keyed_and_not_batch_partition_or_order_dependent(self):
        model=PriorDynamics(dimension=16,layers=1)
        full=noise_bank(model,rows(),'fixture-model',11,samples=3)
        reverse=noise_bank(model,list(reversed(rows())),'fixture-model',11,samples=3)
        single=noise_bank(model,rows()[:1],'fixture-model',11,samples=3)
        for name in ('latent','emission'):
            np.testing.assert_array_equal(full[name],reverse[name][::-1]);np.testing.assert_array_equal(full[name][:1],single[name])

    def test_restore_is_exact_for_both_actual_backbone_implementations(self):
        for backbone in ('causal_transformer','gated_recurrent'):
            model=PriorDynamics(dimension=16,layers=1,backbone=backbone).eval();b=batch()
            bank=noise_bank(model,rows(),'fixture-model',11,samples=3)
            intact=paired_prediction(model,b,Condition('intact'),bank)
            paired_prediction(model,b,Condition('cut',ascending_gain=0),bank)
            restored=paired_prediction(model,b,Condition('restore',restore=True),bank)
            for key in ('neural_mean','neural_scale','joint_samples'):
                torch.testing.assert_close(intact[key],restored[key],rtol=0,atol=0)

    def test_descending_gate_cannot_change_first_neural_step_before_peripheral_response(self):
        model=PriorDynamics(dimension=16,layers=1).eval();b=batch();bank=noise_bank(model,rows(),'fixture-model',11,samples=3)
        intact=paired_prediction(model,b,Condition('intact'),bank)['neural_mean']
        cut=paired_prediction(model,b,Condition('cut-descending',descending_gain=0),bank)['neural_mean']
        torch.testing.assert_close(intact[:,0],cut[:,0],rtol=0,atol=0)
        self.assertFalse(torch.allclose(intact[:,1:],cut[:,1:]))

    def test_shared_driver_and_brain_only_do_not_acquire_a_body_feedback_route(self):
        for kind in ('shared_driver','brain_only'):
            model=PriorDynamics(dimension=16,layers=1,kind=kind).eval();b=batch()
            bank=noise_bank(model,rows(),'fixture-model',11,samples=3)
            a=paired_prediction(model,b,Condition('intact'),bank)
            c=paired_prediction(model,b,Condition('cut',ascending_gain=0,descending_gain=0),bank)
            torch.testing.assert_close(a['neural_mean'],c['neural_mean'],rtol=0,atol=0)

    def test_future_loss_targets_are_not_read_by_intervention_prediction(self):
        model=PriorDynamics(dimension=16,layers=1).eval();b=batch();bank=noise_bank(model,rows(),'fixture-model',11,samples=3)
        a=paired_prediction(model,b,Condition('intact'),bank)
        b['loss_target_body']=torch.full((2,15,14),999.);b['target_neural']=torch.full((2,15,16),999.)
        c=paired_prediction(model,b,Condition('intact'),bank)
        torch.testing.assert_close(a['joint_samples'],c['joint_samples'],rtol=0,atol=0)

    def test_history_surrogates_preserve_observed_message_vectors(self):
        sequence=torch.randn(2,20,16)
        a=HistoryReplay(sequence,1,15,'replay',4);b=HistoryReplay(sequence,1,15,'permuted',4)
        torch.testing.assert_close(a.values.mean(1),b.values.mean(1),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(a.values.square().mean(1),b.values.square().mean(1),rtol=1e-5,atol=1e-6)
        model=PriorDynamics(dimension=16,layers=1).eval();data=batch();bank=noise_bank(model,rows(),'fixture-model',11,samples=3)
        result=paired_prediction(model,data,Condition('history-permutation',history_surrogate='permuted',surrogate_seed=4),bank)
        self.assertIn('not a fitted autonomous',result['metadata']['message_source'])

    def test_history_mask_is_causal_and_does_not_mutate_source_or_targets(self):
        original=batch();original['history_prior_body']=torch.randn(2,20,128)
        original['history_prior_body_mask']=torch.ones(2,20,1,dtype=torch.bool)
        changed=masked_history(original,5)
        self.assertTrue(original['history_body_mask'].all());self.assertFalse(changed['history_body_mask'][:,-5:].any())
        torch.testing.assert_close(changed['history_body'][:,:-5],original['history_body'][:,:-5],rtol=0,atol=0)
        self.assertFalse(changed['history_prior_body_mask'][:,-5:].any())

    def test_nonrecursive_comparator_does_not_pretend_to_have_a_message_gate(self):
        model=FeedForwardAlternative(context=20,width=16,depth=1).eval();b=batch()
        bank=noise_bank(model,rows(),'fixture-model',11,samples=3)
        result=paired_prediction(model,b,Condition('intact'),bank)
        self.assertEqual(result['metadata']['latent_rollouts'],1)
        with self.assertRaises(ValueError):paired_prediction(model,b,Condition('fake-gate',ascending_gain=0),bank)

    def test_energy_score_has_unbiased_two_sample_distance_and_masks_coordinates(self):
        x=np.asarray([[[[0.,999.]]],[[[2.,999.]]]]).transpose(1,0,2,3)
        y=np.asarray([[[1.,np.nan]]]);mask=np.asarray([[[True,False]]])
        np.testing.assert_allclose(joint_energy_score(x,y,mask),[0.])
        with self.assertRaises(ValueError):joint_energy_score(x,y,np.zeros_like(mask))

    def test_registered_basic_plan_has_one_intact_and_separate_intervention_types(self):
        plan=basic_conditions()
        self.assertEqual(len(plan),18);self.assertEqual(sum(c.name=='intact' for c in plan),1)
        with self.assertRaises(ValueError):Condition('bad',ascending_gain=float('nan')).validate()


if __name__=='__main__':unittest.main()
