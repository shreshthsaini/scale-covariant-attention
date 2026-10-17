import csv, json, re, statistics
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'results/evidence/paper_numeric_audit_20261017.json'; report=json.loads(p.read_text()); lines=(root/'paper/main.tex').read_text().splitlines()
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def summary(prefix):
    d={}
    for suffix in ('base','extra'):
        for r in rows(root/'results'/f'{prefix}_{suffix}_summary.csv'): d[r['method'],r['metric']]=float(r['value'])
    for metric in ('fid','patch_fid'):
        for path in (root/'results/evidence/distributions/checks').glob(f'{prefix}_base_rows_rows_{metric}_0.csv'):
            for r in rows(path):d[r['method'],metric]=float(r['value'])
    return d
lat={(r['res'],r['method']):float(r['median_seconds']) for r in rows(root/'results/latency_curve_summary.csv')}
legacy=rows(root/'results/latency_4096_batch1.csv')
lat['4096x4096','sca']=statistics.median(float(r['seconds']) for r in legacy if r['run']=='latency2')
sources={k:summary(k) for k in ('final4k_nb','final4k','final4k_full','e1_baselines','e1_ultraimage','final_2048x4096','final_2048x6144')}
keys=['seconds','ir','aesthetic','clip','pickscore','hpsv2','hpsv21','fid','patch_fid','musiq','musiq_crop1024','clipiqa','arniqa']; digits=[1,2,2,2,2,2,2,1,1,1,1,1,1]
checks=[]; mismatches=[]; res=None
for lineno,line in enumerate(lines,1):
    if lineno<247 or lineno>280:continue
    if 'scale factors (4, 4)' in line:res='4096x4096'
    elif 'scale factors (2, 4)' in line:res='2048x4096'
    elif 'scale factors (2, 6)' in line:res='2048x6144'
    if '&' not in line or line.lstrip().startswith(('Method','\\multicolumn')) or '\\tbd' in line:continue
    cells=[s.strip() for s in line.split('&')];label=cells[0]
    if res=='4096x4096':
        if label.startswith('Full attention'):prefix,method='final4k_full','full'
        elif label.startswith('DyPE'):prefix,method='e1_baselines','dype'
        elif label.startswith('SEGA'):prefix,method='e1_baselines','sega_official'
        elif label.startswith('UltraImage'):prefix,method='e1_ultraimage','ultraimage'
        elif label.startswith('AReA'):prefix,method='final4k_nb','area'
        elif 'band guard' in label:prefix,method='final4k','sca'
        elif label.startswith('\\sca{}'):prefix,method='final4k_nb','sca_nb'
        else:raise RuntimeError(label)
    else:prefix,method='final_'+res,('area' if label.startswith('AReA') else 'sca_nb')
    for idx,(metric,dec) in enumerate(zip(keys,digits),1):
        clean=cells[idx].replace('\\textbf','').replace('{','').replace('}','')
        actual=float(re.search(r'[-+]?\d+(?:\.\d+)?',clean).group())
        expected=sources[prefix][method,metric]
        if metric=='seconds' and res=='4096x4096' and method in ('area','sca_nb','full','sca'):expected=lat[res,method]
        if metric in ('hpsv2','hpsv21','clipiqa','arniqa'):expected*=100
        ok=actual==float(f'{expected:.{dec}f}')
        checks.append({'line':lineno,'method':method,'resolution':res,'metric':metric,'expected_unrounded':expected,'printed':actual,'matches_rounding':ok})
        if not ok:mismatches.append(checks[-1])
report['main_table']={'cells_checked':len(checks),'mismatches':mismatches,'cells':checks}
report['checks_passed']=report['checks_passed'] and not mismatches
p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'cells_checked':len(checks),'mismatches':mismatches},indent=2))
