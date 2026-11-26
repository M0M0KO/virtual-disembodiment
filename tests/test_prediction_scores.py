from pathlib import Path
import tempfile
import unittest
import numpy as np
from virtual_disembodiment.prediction_scores import checked_arrays,matched_arrays,score_arrays,verify_and_score,freeze_prediction
from virtual_disembodiment.run_io import atomic_npz,atomic_json,file_hash
from virtual_disembodiment.gaussian_baselines import numpy_neural_scores


def fixture():
    rng=np.random.default_rng(123);n=6
    return {'neural_mean':rng.normal(size=(n,15,16)).astype(np.float32),'neural_scale':np.ones((n,15,16),np.float32),
            'observed_neural':rng.normal(size=(n,15,16)).astype(np.float32),'neural_target_mask':np.ones((n,15,16),bool),
            'participant_id':np.array(['a','a','a','a','a','b']),'recording_id':np.array(['r1']*5+['r2']),
            'forecast_origin_sec':np.arange(60,66),'observation_cutoff_sec':np.arange(60,66),
            'supplied_forcing_mg':np.zeros((n,15,2),np.float32)}


class PredictionScoreTests(unittest.TestCase):
    def test_scores_are_person_weighted_not_window_weighted(self):
        a=fixture();scores=score_arrays(a)
        window,_=numpy_neural_scores(a['neural_mean'],a['neural_scale'],a['observed_neural'],a['neural_target_mask'])
        self.assertAlmostEqual(scores['neural_crps'],(window[:5].mean()+window[5])/2,places=6)
        self.assertNotAlmostEqual(scores['neural_crps'],window.mean(),places=4)

    def test_alignment_accepts_reordering_but_rejects_different_windows_or_targets(self):
        a=fixture();b={k:v[::-1].copy() for k,v in a.items()}
        aligned=matched_arrays(a,b)
        np.testing.assert_equal(a['neural_mean'],aligned['neural_mean'])
        b['observed_neural'][0,0,0]+=1
        with self.assertRaises(ValueError):matched_arrays(a,b)
        b={k:v[:-1] for k,v in a.items()}
        with self.assertRaises(ValueError):matched_arrays(a,b)

    def test_duplicate_windows_and_future_cutoff_fail(self):
        for mode in ('duplicate','future'):
            with tempfile.TemporaryDirectory() as t:
                a=fixture()
                if mode=='duplicate':
                    a['forecast_origin_sec'][1]=a['forecast_origin_sec'][0]
                    a['observation_cutoff_sec'][1]=a['observation_cutoff_sec'][0]
                else:a['observation_cutoff_sec'][0]+=1
                p=Path(t)/'predictions.npz';atomic_npz(p,**a)
                with self.assertRaises(ValueError):checked_arrays(p)

    def test_additive_freeze_preserves_numbers_and_original_hash(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);src=root/'source';a=fixture();atomic_npz(src/'predictions.npz',**a)
            before=file_hash(src/'predictions.npz');scores=score_arrays(a)
            atomic_json(src/'scores.json',{**scores,'prediction_file_sha256':before,'created_utc':'2026-10-25T00:00:00+00:00'})
            provenance={'model_sha256':'model','feature_manifest_sha256':'features','configuration_sha256':'config',
                        'code_commit':'test-fixture','evaluation_partition':'selection'}
            freeze_prediction(src,root/'freeze',provenance,['a','b'])
            self.assertEqual(file_hash(src/'predictions.npz'),before)
            with np.load(root/'freeze/predictions.npz',allow_pickle=False) as z:
                for k,v in a.items():np.testing.assert_equal(v,z[k])
                self.assertEqual(str(z['model_sha256']),'model')
                self.assertEqual(str(z['generated_utc']),'2026-10-25T00:00:00+00:00')
            with self.assertRaises(PermissionError):verify_and_score(src,['a'])
            with self.assertRaises(FileExistsError):freeze_prediction(src,root/'freeze',provenance,['a','b'])

    def test_checksum_damage_is_not_treated_as_a_new_result(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a=fixture();atomic_npz(root/'predictions.npz',**a)
            atomic_json(root/'scores.json',{'prediction_file_sha256':'wrong'})
            with self.assertRaises(ValueError):verify_and_score(root)


if __name__=='__main__':unittest.main()
