"""Extract the inline <script> from app/web/index.html and syntax-check it with node. Exit 0 if valid."""
import re, subprocess, sys, tempfile
from pathlib import Path

html = (Path(__file__).resolve().parent.parent / 'app/web/index.html').read_text()
scripts = re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', html, re.S)
if not scripts:
    sys.exit('no inline script found')
with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
    f.write('\n'.join(scripts))
r = subprocess.run(['node', '--check', f.name], capture_output=True, text=True)
if r.returncode:
    print(r.stderr[-1500:])
    sys.exit(1)
print('JS syntax OK,', len(html.splitlines()), 'lines')
