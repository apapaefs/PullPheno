#!/usr/bin/env python3
"""Copy only completed study outputs, publishing local run directories atomically."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import HwSimPythonAnalysis as analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote',type=Path,required=True)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    remote=json.loads(args.remote.read_text())
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',remote['host']]
    state=json.loads(subprocess.run(ssh+['cat '+shlex.quote(remote['batch']+'/status.json')],check=True,capture_output=True,text=True).stdout)
    if args.require_complete and state['status']!='complete':
        raise RuntimeError('Study status is '+state['status'])
    status_path=args.remote.with_name('completed-status.json' if state['status']=='complete' else 'latest-status.json')
    status_path.write_text(json.dumps(state,indent=2)+'\n')
    for label,job in state['jobs'].items():
        if 'smoke' in label or job['phase']!='complete': continue
        source=Path(job['run'])
        if source.parent!=Path(remote['output_root'])/'runs':
            raise ValueError('Unexpected source directory: '+str(source))
        destination=args.results/'runs'/source.name
        if destination.exists():
            existing=json.loads((destination/'run.json').read_text())
            assert existing['status']=='complete' and existing['run_id']==source.name
            continue
        incoming=args.results/'.incoming-pullcut'/source.name
        incoming.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-az',remote['host']+':'+str(source)+'/',str(incoming)+'/'],check=True)
        metadata=json.loads((incoming/'run.json').read_text())
        assert metadata['status']=='complete' and metadata['run_id']==source.name
        if metadata['run_type']=='analysis':
            _,results,_=analysis.load_completed_run(incoming)
            analysis.validate_results(results,False)
        incoming.rename(destination)
        print('Available locally:',source.name,flush=True)
    print('Remote status:',state['status'])


if __name__=='__main__': main()
