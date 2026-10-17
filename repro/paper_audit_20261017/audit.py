import csv, hashlib, json, math, re, statistics, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[2]
scratch = Path('/scratch/09032/saini_2/sca/paper_numeric_audit_20261017')
ev = root/'results/evidence'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))
def j(p): return json.loads(p.read_text())
report = {'audit_utc':datetime.now(timezone.utc).isoformat(), 'scope':'Read-only numerical and source audit; no generation, encoder inference, or distribution bootstrap rerun.', 'source_sha256':{'paper/main.tex':sha(root/'paper/main.tex')}}
manifest = j(ev/'manifest.json')
failures = []
for item in manifest['files']:
    if sha(ev/item['archive']) != item['sha256']: failures.append(item['archive'])
if sha(ev/'generation_sidecars.jsonl.gz') != manifest['generation_sidecars_sha256']: failures.append('generation_sidecars.jsonl.gz')
report['archive_integrity'] = {'checked':len(manifest['files'])+1, 'mismatches':failures}
means, errors, coverage = 0, [], []
for folder in sorted((ev/'eval').iterdir()):
    prefix = 'extra_' if folder.name.endswith('_extra') else ''
    source, summary = folder/(prefix+'per_image.csv'), folder/(prefix+'summary.csv')
    if not source.exists() or not summary.exists(): continue
    data = rows(source)
    for s in rows(summary):
        vals = [float(r[s['metric']]) for r in data if r['method']==s['method'] and r['res']==s['res'] and r.get(s['metric'],'')]
        if vals:
            error = abs(float(np.mean(vals))-float(s['value']))
            means += 1
            if error > 1e-8 or not all(map(math.isfinite,vals)): errors.append({'dataset':folder.name,'method':s['method'],'metric':s['metric'],'error':error})
    groups = {}
    for r in data: groups.setdefault(r['method'],[]).append((r['res'],r['seed'],r['prompt_id']))
    coverage.append({'dataset':folder.name,'method_counts':{k:len(v) for k,v in groups.items()}, 'same_ids':len({tuple(sorted(v)) for v in groups.values()})==1,'duplicate_rows':sum(len(v)-len(set(v)) for v in groups.values())})
report['scalar_means'] = {'checked':means,'mismatches':errors,'coverage':coverage}
calibration = []
for p in sorted((ev/'calib').glob('*.npz')):
    if not p.with_suffix('.json').exists(): continue
    z=np.load(p); meta=j(p.with_suffix('.json')); pop=z['populated']; indices=np.flatnonzero(pop)
    nearest=indices[np.abs(np.arange(pop.size)[:,None]-indices[None]).argmin(1)]
    for idx,scale in enumerate(z['scales']):
        stats=z['stats'][idx]; A,B=stats[:2]
        pi=np.clip(np.where(A>0,B/np.where(A>0,A,1),1),0,1)[...,nearest]
        diff=float(np.max(np.abs(pi-z[f's{scale}_pi_head'])))
        rel=float((stats[2]-2*pi*B+pi*pi*A).sum()/stats[4].sum())
        calibration.append({'table':p.name,'scale':int(scale),'pi_max_error':diff,'relative_squared_error':rel,'report_abs_error':abs(rel-meta['scales'][str(scale)]['rel_err']['per_head'])})
report['calibration']=calibration
# Recreate all existing generated appendix tables without touching the manuscript.
out=scratch/'generated_appendix.tex'
subprocess.run([sys.executable,str(root/'paper/figures/src/gen_tables.py'),str(root/'results'),str(out)],check=True,capture_output=True,text=True)
paper=(root/'paper/main.tex').read_text()
generated=out.read_text()
embedded=paper.split('%%% BEGIN GENERATED TABLES',1)[1].split('%%% END GENERATED TABLES',1)[0]
embedded='\n'.join(line for line in embedded.splitlines() if not line.startswith('%')).strip()
report['generated_appendix']={'exact_source_match':embedded==generated.strip(),'generated_sha256':sha(out),'rows_checked':sum('&' in line and '\\\\' in line for line in generated.splitlines())}
out8=scratch/'e2_8192_tables.tex'
cp=subprocess.run([sys.executable,str(root/'paper/figures/src/gen_final_tables.py'),'e2_8192','--out',str(out8)],check=True,capture_output=True,text=True)
report['e2_8192_table']={'exact_source_match':out8.read_bytes()==(root/'paper/figures/snippets/e2_8192_tables.tex').read_bytes(),'generated_sha256':sha(out8),'command_output':cp.stdout.strip()}
lat=j(ev/'latency/audit.json'); latency=[]
for g in lat['groups']:
    size=g['res'].split('x')[0]; idx=(2048,3072,4096,6144,8192).index(int(size))
    name=f'a_y00{idx}_latency_{g["res"]}'+(f'_{g["method"]}' if idx>=3 else '')
    logs=list((ev/'latency').glob(f'task_{name}*.log'))
    logs=[p for p in logs if sha(p)==g['log_sha256']]
    assert len(logs)==1,(g,logs)
    logtext=logs[0].read_text(); events=re.findall(r'saved (\S+) (\S+) (\d+x\d+) seconds=([0-9.]+)',logtext)
    events=[(pid,float(t)) for pid,method,res,t in events if method==g['method'] and res==g['res']]
    assert len(events)==4 and events[0][0]=='sca_latency_warmup',(g,events)
    assert not re.search(rf'skip \S+ {re.escape(g["method"])} {g["res"]}',logtext)
    sidecars={j(p)['config']['prompt_id']:j(p) for p in (ev/'latency'/g['res']/g['method']).glob('*.json')}
    assert len(sidecars)==4
    for pid,t in events:
        d=sidecars[pid]; assert d['complete'] and abs(d['seconds']-t)<0.0006
        assert d['config']['steps']==28 and d['config']['guidance']==4.5 and d['config']['seed']==0
        if pid!='sca_latency_warmup': assert d.get('first_call_in_process') is not True
    median=statistics.median(sidecars[pid]['seconds'] for pid,t in events[1:])
    assert median==g['median_seconds']
    latency.append({'res':g['res'],'method':g['method'],'n_timed':3,'warmup_log_verified':True,'median_seconds':median,'log':str(logs[0].relative_to(root))})
report['latency']={'groups_rechecked':latency,'missing':lat['missing'],'scope':'Saved warm-up and three samples are ordered uninterrupted in each archived task log; protocol and sidecars agree. This does not independently prove node idleness.'}
report['speed_ratios']={'area_over_sca_4k':202.43776634894311/71.06495948787779,'full_over_sca_4k':186.5384593643248/71.06495948787779,'sca_8k_over_2k':287.63978986348957/18.201440702192485}
report['corrected_main_distributions']={}
for metric in ('fid','kid','patch_fid','patch_kid'):
    p=ev/'distributions/checks'/f'final4k_nb_base_rows_images_{metric}_1000.csv'
    report['corrected_main_distributions'][metric]={'source':str(p.relative_to(root)),'rows':rows(p)}
report['delivery_hashes']={}
receipt=j(root/'paper/delivery_receipt.json')
for rel,key in [('paper/main.pdf','paper_sha256'),('paper/sca_overleaf.zip','source_zip_sha256')]:
    report['delivery_hashes'][rel]={'sha256':sha(root/rel),'matches_receipt':sha(root/rel)==receipt[key]}
report['pending_source_locations']=[{'line':i,'source':line.strip()} for i,line in enumerate(paper.splitlines(),1) if '\\todo{' in line and not line.lstrip().startswith('%')]
report['checks_passed']=not failures and not errors and all(c['same_ids'] and c['duplicate_rows']==0 for c in coverage) and all(c['pi_max_error']<1e-12 and c['report_abs_error']<1e-12 for c in calibration) and report['generated_appendix']['exact_source_match'] and report['e2_8192_table']['exact_source_match'] and all(x['matches_receipt'] for x in report['delivery_hashes'].values())
report['scratch_reproduction_script']=str(Path(__file__))
report['scratch_reproduction_sha256']=sha(Path(__file__))
(root/'results/evidence/paper_numeric_audit_20261017.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks_passed':report['checks_passed'],'archive_hashes':report['archive_integrity']['checked'],'scalar_means':means,'calibration_scales':len(calibration),'appendix':report['generated_appendix'],'e2_8k':report['e2_8192_table']['exact_source_match'],'latency_groups':len(latency)},indent=2))
