import csv,json,re,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'results/evidence/paper_numeric_audit_20261017.json';report=json.loads(p.read_text());paper=(root/'paper/main.tex').read_text()
tables={re.search(r'\\label\{([^}]+)\}',t).group(1):t for t in re.findall(r'\\begin\{table\}(.*?)\\end\{table\}',paper,re.S) if re.search(r'\\label\{([^}]+)\}',t)}
profiles={'tab:phase2_baselines':['e1_dype_vs_sca','e1_sega_official_vs_sca','e1_ultraimage_vs_sca'],'tab:phase2_general':['e4_hrgen_sca_nb_vs_area','e5_seed1_sca_vs_area','e5_seed2_sca_vs_area','e8_flux2_sca_vs_full'],'tab:phase2_sensitivity':['e6_sca_cal4_deltas','e6_sca_cal8_deltas','e6_sca_cal16_deltas','e7_sca_g85_deltas','e7_sca_g93_deltas'],'tab:phase2_budget':['budget_sca_s56_vs_sca_nb','budget_sca_g7_vs_sca_nb','budget_sca_s56_vs_ultraimage']}
metrics=['ir','aesthetic','clip','pickscore','hpsv2','hpsv21','musiq','musiq_crop1024','clipiqa','arniqa'];digits=[3,2,2,2,2,2,1,1,1,1]
checked=0;mismatches=[]
for label,files in profiles.items():
    data=[]
    for f in files:
        with (root/'results'/f'{f}.csv').open(newline='') as handle:data.append({r['metric']:r for r in csv.DictReader(handle)})
    table=tables[label]; body=table.split('\\midrule',1)[1].split('\\bottomrule',1)[0]
    lines=[line for line in body.splitlines() if '&' in line]
    assert len(lines)==10
    for metric,dec,line in zip(metrics,digits,lines):
        cells=line.split('&')[1:];assert len(cells)==len(data)
        for filename,source,cell in zip(files,data,cells):
            actual=list(map(float,re.findall(r'[-+]?\d+(?:\.\d+)?',cell)));assert len(actual)==3
            factor=100 if metric in ('hpsv2','hpsv21','clipiqa','arniqa') else 1
            expected=[float(f'{float(source[metric][key])*factor:.{dec}f}') for key in ('delta','ci_low','ci_high')]
            checked+=3
            if actual!=expected:mismatches.append({'table':label,'file':filename,'metric':metric,'actual':actual,'expected':expected})
report['phase2_per_image_tables']={'numeric_values_checked':checked,'mismatches':mismatches,'scope':'Saved 2000-draw paired deltas and endpoints compared after exact display rounding; bootstrap draws were not rerun.'}
report['checks_passed']=report['checks_passed'] and not mismatches
report.setdefault('reproduction_scripts',{})[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['phase2_per_image_tables'],indent=2))
