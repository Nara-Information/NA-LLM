"""strip_pesudo_html.py 

Strip HTML tags and get unannotated text,
without parsing the string and destroying text 
surrounded by < >. 
Useful as some of the OpenAPI results are not
enocded for HTML even being within a HTML document.
"""

import os
import sys
import re

def _strip(s) -> str:
    
    def _isHTML(s):
        if re.search(r"[가-힣]", s) is None:
            return True 
        elif re.search(r"font.+?[\"\']:\s*[\"\'].{1,5}?[가-힣]", s) is not None:
            # Found Korean character but seems to be font face name 
            return True
        elif re.search(r"[\'\"][가-힣][\w\s]+?[\'\"]", s) is not None:
            # Found Korean character but seems to be a js, css, or html names 
            return True
        else:
            return False
    
    s = re.split(r"(<.+?>)", s)
    s = [e for i, e in enumerate(s) if i%2 == 0 or _isHTML(e) == False]
    s = ''.join(s)
    
    return s 

def strip_file(path, doOverwrite: bool=False):
    with open(path) as f:
        lines = f.read()
    lines = _strip(lines)
    outpath = path if doOverwrite else '-out'.join(os.path.splitext(path)) 
    with open(outpath, 'w') as f:
        f.write(lines)

def main():
    if len(sys.argv) < 2:
        lines = ''.join([l for l in sys.stdin])
        print(_strip(lines))
    else: 
        strip_file(sys.argv[1])

if __name__ == "__main__":
    main()
