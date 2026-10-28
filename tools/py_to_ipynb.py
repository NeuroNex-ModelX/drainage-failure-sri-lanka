"""Convert the '# %%' percent-format .py notebooks in notebooks_py/ into Colab .ipynb files.
Usage: python tools/py_to_ipynb.py notebooks_py notebooks"""
import json, os, re, sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for fn in sorted(os.listdir(src)):
    if not fn.endswith('.py'):
        continue
    text = open(os.path.join(src, fn)).read()
    cells = []
    for block in re.split(r'^# %%', text, flags=re.M)[1:]:
        header, _, body = block.partition('\n')
        if '[markdown]' in header:
            lines = [re.sub(r'^# ?', '', l) for l in body.strip('\n').split('\n')]
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': '\n'.join(lines)})
        else:
            cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                          'outputs': [], 'source': body.strip('\n')})
    nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'colab': {'provenance': []}, 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                       'language_info': {'name': 'python'}}}
    json.dump(nb, open(os.path.join(dst, fn[:-3] + '.ipynb'), 'w'), indent=1)
    print('made', fn[:-3] + '.ipynb', len(cells), 'cells')
