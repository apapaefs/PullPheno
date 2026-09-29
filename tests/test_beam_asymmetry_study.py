"""Campaign orchestration, immutable preparation and report numerical checks."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scripts import build_beam_asymmetry_report as report
from scripts.pullcut_study_config import event_arguments, comparison_arguments
from scripts.run_beam_asymmetry_study import Study, validate_study
from scripts.build_beam_asymmetry_report import validate_pairwise, validate_scaling, record_selection

ROOT = Path(__file__).resolve().parents[1]
TAG = 'campaign22-beam-asymmetries-20260929'
loader = importlib.util.spec_from_file_location('beam_study_prepare',ROOT/'launchers'/TAG/'prepare.py')
prepare = importlib.util.module_from_spec(loader)
loader.loader.exec_module(prepare)


class BeamStudyTests(unittest.TestCase):
    def setUp(self):
        self.inputs = json.loads((ROOT/'studies'/f'{TAG}.json').read_text())

    def test_nine_fresh_uncut_analyses_and_nine_comparisons(self):
        inputs = validate_study(self.inputs)
        event_jobs = comparisons = 0
        for profile in inputs['profiles']:
            for name in inputs['scenarios']:
                arguments = event_arguments(inputs,name,profile,'uncut')
                self.assertNotIn('--from-run',arguments)
                self.assertNotIn('--pull-magnitude-min',arguments)
                self.assertEqual(arguments[arguments.index('--particle-pt-min')+1],'0.1')
                self.assertEqual(arguments[arguments.index('--workers')+1],'190')
                self.assertEqual(arguments[arguments.index('--event-shards')+1],'384')
                event_jobs += 1
            runs = {name:Path('/runs')/name for name in inputs['scenarios']}
            for group in inputs['comparison_sets']:
                arguments = comparison_arguments(inputs,group,runs)
                self.assertEqual(arguments[1],'/runs/herwig')
                self.assertNotIn('--samples',arguments)
                comparisons += 1
        self.assertEqual((event_jobs,comparisons),(9,9))
        smoke = event_arguments(inputs,'mg5-pythia',inputs['smoke_profile'],'uncut',True)
        self.assertEqual(smoke[smoke.index('--max-events')+1],'5000')
        self.assertEqual(inputs['workers']*inputs['concurrent_analyses'],380)

    def test_no_reused_archives_or_magnitude_threshold(self):
        for change in ({'selections':{'uncut':.0015}}, {'reuse_sources':{'eta2-mll71-111':{'uncut':{'herwig':'old'}}}}, {'workers':191}):
            inputs = copy.deepcopy(self.inputs)
            inputs.update(change)
            with self.assertRaises(ValueError):
                validate_study(inputs)

    def test_phase_prerequisite_and_no_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(batch=root,output_root=root,phase='analysis')
            study = Study(args,self.inputs,SimpleNamespace(ANALYSIS_VERSION='2.10.0'))
            with self.assertRaisesRegex(RuntimeError,'smoke'):
                study.execute()
            study.state['jobs']['recoverable'] = {'phase':'failed'}
            with self.assertRaisesRegex(RuntimeError,'recoverable state'):
                study.invoke('recoverable',['--samples','unused'])
            self.assertEqual(study.state['jobs']['recoverable'],{'phase':'failed'})
            study.state['phases']['analysis']='complete'
            with patch.object(study,'event_group') as group:
                study.execute()
                group.assert_not_called()

    def test_prepare_idempotence_and_frozen_commit_excludes_dirty_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo,output = root/'repo',root/'output'
            repo.mkdir()
            subprocess.run(['git','init','-q',str(repo)],check=True)
            (repo/'analysis.py').write_text('committed = True\n')
            subprocess.run(['git','-C',str(repo),'add','analysis.py'],check=True)
            subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid',
                            'commit','-qm','Fixture'],check=True)
            subprocess.run(['git','-C',str(repo),'remote','add','origin','https://example.invalid/repo'],check=True)
            revision = prepare.git(repo,'rev-parse','HEAD')
            (repo/'analysis.py').write_text('committed = False\n')
            (repo/'not-committed.py').write_text('private = True\n')
            manifests = {}
            inputs = copy.deepcopy(self.inputs)
            for name,scenario in inputs['scenarios'].items():
                path = root/f'{name}.json'
                manifest = {'samples':[{'name':'VBFH','cross_section_pb':.5}]}
                path.write_text(json.dumps(manifest))
                scenario['manifest']=str(path)
                manifests[name]=manifest
            receipt = {'commit':revision,'cached_inputs':[]}
            prepare.prepare(repo,output,inputs,manifests,receipt)
            prepare.prepare(repo,output,inputs,manifests,receipt)
            code=output/'batches'/TAG/'code'
            self.assertEqual((code/'analysis.py').read_text(),'committed = True\n')
            self.assertFalse((code/'not-committed.py').exists())
            provenance = json.loads((code.parent/'source-provenance.json').read_text())
            self.assertEqual(provenance['commit'],revision)
            self.assertFalse(provenance['event_processing_started'])
            (code/'analysis.py').write_text('modified\n')
            with self.assertRaisesRegex(RuntimeError,'Frozen source differs'):
                prepare.prepare(repo,output,inputs,manifests,receipt)

    def test_luminosity_and_signed_stack_closure(self):
        values={'A':[.5,-.2],'mc_covariance':[[.1,0],[0,.2]],
                'process_contributions':{'signal':[.3,-.1],'background':[.2,-.1]},
                'data_covariances':{'300.0':[[.01,0],[0,.02]],'3000.0':[[.001,0],[0,.002]]}}
        payload={'predictions':[{'observables':{'pull_asymmetry_beam':values}}],'ratios':[]}
        self.assertEqual(validate_scaling(payload,[300,3000]),1)
        payload['predictions'][0]['observables']['pull_asymmetry_beam']['process_contributions']['signal'][0]=.4
        with self.assertRaises(AssertionError):
            validate_scaling(payload,[300,3000])

    def test_pairwise_closure_and_ratio_masks(self):
        full={'predictions':[{'strategy':'cutbased','channel':'higgs','scenario':name,'observables':{'A':[.2]}}
                             for name in ('herwig','preco','pythia')], 'ratios':[]}
        subset=copy.deepcopy(full)
        subset['predictions'].pop(1)
        validate_pairwise(full,subset)
        subset['predictions'][1]['observables']['A'][0]=.3
        with self.assertRaisesRegex(AssertionError,'prediction changed'):
            validate_pairwise(full,subset)
        ratios={'predictions':[],'ratios':[{'central':[None,-2.], 'valid':[False,True],
                 'mc_covariance':[[None,None],[None,.1]],'data_covariances':{}}]}
        validate_scaling(ratios,[300,3000])
        ratios['ratios'][0]['valid'][0]=True
        with self.assertRaises(AssertionError):
            validate_scaling(ratios,[300,3000])

    def test_distinct_record_selectors(self):
        records=[{'kind':'asymmetry-reference-stack-plus-total-ratio','view':'full','error_mode':'mc'},
                 {'kind':'reference-stack-plus-total-ratio','view':'full','error_mode':'mc'}]
        self.assertEqual(len(record_selection(records,kind='asymmetry-reference-stack-plus-total-ratio')),1)

    def test_ratio_covariances_respect_each_error_models_numerical_mask(self):
        """A finite ratio can have overflowing MC variance and finite data variance."""
        covariance=[[.1,0,None,None],[0,.2,None,None],
                    [None,None,None,None],[None,None,None,None]]
        ratio={'central':[1.e200,-2.,None,None], 'valid':[True,True,False,True],
               'denominator_valid':[True,True,False,True],
               'numerical_valid':{'mc':[False,True,False,False],
                                  'data':{'300.0':[True,True,False,False],
                                          '3000.0':[True,True,False,False]}},
               'mc_covariance':[[None,0,None,None],[0,.1,None,None],
                                [None,None,None,None],[None,None,None,None]],
               'data_covariances':{'300.0':covariance,
                                   '3000.0':[[None if value is None else value/10 for value in row]
                                              for row in covariance]}}
        payload={'predictions':[],'ratios':[ratio]}
        validate_scaling(payload,[300,3000])
        invalid=copy.deepcopy(payload)
        invalid['ratios'][0]['numerical_valid']['mc'][0]=True
        with self.assertRaises(AssertionError):
            validate_scaling(invalid,[300,3000])
        invalid=copy.deepcopy(payload)
        invalid['ratios'][0]['denominator_valid'][1]=False
        with self.assertRaises(AssertionError):
            validate_scaling(invalid,[300,3000])

    def test_report_builds_twelve_summaries_and_eighteen_galleries(self):
        """Exercise the exact plot record interface without ROOT or rendered analysis runs."""
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            batch=root/'batch'; batch.mkdir()
            status=batch/'status.json'
            state={'phases':{'plots':'complete'},'inputs':self.inputs,'sources':{},'comparisons':{}}
            values={'A':[.1,-.2],'mc_covariance':[[.01,0],[0,.02]],
                    'process_contributions':{'test':[.1,-.2]},
                    'data_covariances':{'300.0':[[.001,0],[0,.002]],'3000.0':[[.0001,0],[0,.0002]]}}
            def artifacts(run,payload,metadata):
                (run/'summaries').mkdir(parents=True)
                Image.new('RGB',(2,2),'white').save(run/'plot.png')
                (run/'plot.pdf').write_bytes(b'%PDF-1.4\n%%EOF\n')
                (run/'summaries/asymmetries.json').write_text(json.dumps(payload))
                (run/'summaries/asymmetries.npz').write_bytes(b'NPZ fixture')
                (run/'run.json').write_text(json.dumps(metadata))
            for profile in self.inputs['profiles']:
                state['sources'][profile]={'uncut':{}}
                for name in self.inputs['scenarios']:
                    run=root/'runs'/f'{profile}-{name}'
                    state['sources'][profile]['uncut'][name]=str(run)
                    records=[{'observable':observable,'channel':'z' if process.endswith('Z') or process=='QCDZjj' else 'higgs',
                              'sample':process,'view':view,'error_mode':'mc','kind':'asymmetry-shape',
                              'png':'plot.png','pdf':'plot.pdf'}
                             for process in ('VBFH','QCDHjj','aajj','VBFZ','QCDZjj')
                             for observable in self.inputs['asymmetry_observables'] for view in ('full','near-zero')]
                    artifacts(run,{'predictions':[{}]*5,'plots':records},{'status':'complete','partial':False})
                for group,members in self.inputs['comparison_sets'].items():
                    run=root/'runs'/f'{profile}-{group}'
                    key=f'{profile}/uncut/{group}'
                    state['comparisons'][key]=str(run)
                    predictions=[{'strategy':'cutbased','channel':channel,
                                  'scenario':self.inputs['scenarios'][name]['identifier'],
                                  'observables':{observable:values for observable in self.inputs['asymmetry_observables']}}
                                 for channel in ('higgs','z') for name in members]
                    records=[{'observable':observable,'channel':channel,'strategy':'cutbased','sample':None,
                              'luminosity_fb':luminosity,'view':view,'error_mode':mode,
                              'kind':'asymmetry-reference-stack-plus-total-ratio','png':'plot.png','pdf':'plot.pdf'}
                             for observable in self.inputs['asymmetry_observables'] for channel in ('higgs','z')
                             for luminosity in (300,3000) for view in ('full','near-zero') for mode in ('mc','data')]
                    metadata={'status':'complete','partial':False,'configuration':{
                        'pull_magnitude_min':None,'particle_pt_min_gev':.1,'cut_profile':profile},
                        'source_runs':[{'run_id':Path(state['sources'][profile]['uncut'][name]).name} for name in members]}
                    artifacts(run,{'predictions':predictions,'ratios':[],'plots':records},metadata)
            status.write_text(json.dumps(state))
            for name,value in (('source-provenance.json',{'commit':'abcdef','code_sha256':{}}),
                               ('inputs.json',self.inputs),('sample-normalization.json',{})):
                (batch/name).write_text(json.dumps(value))
            with patch.object(report.analysis,'load_completed_run',return_value=(
                    SimpleNamespace(pull_magnitude_min=None),[],{'partial':False})), \
                    patch.object(report.analysis,'validate_results'):
                report.build(status,root/'report')
            validation=json.loads((root/'report/validation.json').read_text())
            self.assertEqual(validation['summary_pages'],12)
            self.assertEqual(validation['comparison_galleries'],9)
            self.assertEqual(validation['sample_galleries'],9)
            self.assertEqual(json.loads(status.read_text())['status'],'complete')
            self.assertEqual(len(list((root/'report/summaries').glob('*.html'))),12)
            for path in (root/'report').rglob('*.html'):
                document=path.read_text()
                self.assertIn('.plot-zoom[open]{position:fixed;inset:0',document)
                self.assertIn('if(event.key==="Escape")',document)
                self.assertIn('plot.removeAttribute("open")',document)


if __name__ == '__main__':
    unittest.main()
