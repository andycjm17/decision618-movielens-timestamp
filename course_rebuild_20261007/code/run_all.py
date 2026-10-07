"""Sequential entry point for reproducing the saved experiments."""
import argparse,subprocess,sys
from pathlib import Path
from prepare import ROOT

def run(command,log):
    print('Running:',log,flush=True)
    with (ROOT/'results'/log).open('w') as stream:
        subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fresh-cv',action='store_true',help='Rerun all 400 fits; otherwise use the completed fresh CV artifacts.')
    args=parser.parse_args();code=ROOT/'code'
    if not (ROOT/'sources/cv_all_1m.pkl').exists():
        raise FileNotFoundError('Full cache audit requires sources/cv_all_1m.pkl from Canvas file 1807319. Read README.md to retrieve the source through Canvas MCP. The review notebook works with the packaged audited outputs.')
    run([sys.executable,str(code/'prepare.py')],'preparation_run.log')
    for mode in ['teacher','temporal_grid','temporal_final','temporal_crossfit']:
        run(['Rscript',str(code/'fit_softimpute.R'),str(ROOT),mode],mode+'_run.log')
    if args.fresh_cv:run(['Rscript',str(code/'fit_softimpute.R'),str(ROOT),'fresh_cv'],'fresh_cv.log')
    run(['Rscript',str(code/'fit_softimpute.R'),str(ROOT),'fresh_final'],'fresh_selected.log')
    run([sys.executable,str(code/'evaluate_timestamp.py')],'timestamp_models.log')
    run([sys.executable,str(code/'analyze_time.py')],'timestamp_eda.log')
    run(['Rscript',str(code/'audit_models.R'),str(ROOT)],'independent_r_audit.log')
    run([sys.executable,str(code/'audit_analysis.py')],'analysis_audit.log')
    run([sys.executable,str(code/'charts.py')],'charts.log')
    print('Complete. Re-execute MovieLens_course_rebuild.ipynb to refresh its saved presentation.',flush=True)
