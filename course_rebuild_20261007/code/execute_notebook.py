"""Execute with this interpreter through an ephemeral, project-local kernel."""
import json,os,sys,tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from prepare import ROOT

def execute():
    os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'runtime/jupyter')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runtime/mpl-cache'))
    os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'runtime/font-cache'))
    (ROOT/'runtime/jupyter').mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'runtime',prefix='kernel-') as directory:
        kernel=Path(directory)/'decision618';kernel.mkdir()
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Decision 618 ephemeral kernel','language':'python'}))
        manager=KernelManager(kernel_name='decision618',kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
        notebook=nbformat.read(ROOT/'MovieLens_course_rebuild.ipynb',as_version=4)
        client=NotebookClient(notebook,km=manager,timeout=240,resources={'metadata':{'path':str(ROOT)}})
        client.execute(cwd=str(ROOT),cleanup_kc=True);nbformat.validate(notebook)
        nbformat.write(notebook,ROOT/'MovieLens_course_rebuild.ipynb')
    exporter=HTMLExporter(template_name='lab')
    html,_=exporter.from_notebook_node(notebook)
    (ROOT/'MovieLens_course_rebuild.html').write_text(html)
    count=sum(c.cell_type=='code' for c in notebook.cells)
    errors=[o for c in notebook.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
    assert not errors
    (ROOT/'results/notebook_validation.json').write_text(json.dumps({'executed':True,'code_cells':count,'error_outputs':0,'html_exported':True,'interpreter':sys.executable},indent=2))
    print('Executed',count,'code cells; saved notebook and HTML preview.')

if __name__=='__main__':execute()
