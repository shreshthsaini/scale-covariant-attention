"""Write paper/figures/snippets/_test.tex, a scratch document that includes every snippet with the
ICLR style and main.tex's macros, one snippet per page, to check that each compiles and fits.

  <python> make_test_doc.py && cd paper && tectonic -Z search-path=. figures/snippets/_test.tex
"""
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SNIP = os.path.normpath(os.path.join(HERE, "..", "snippets"))
PRE = r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{iclr2026_conference,times}
\input{math_commands.tex}
\usepackage{hyperref}
\usepackage{url}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{etoolbox}
\graphicspath{{../../}}
\newcommand{\sca}{SCA}
\newcommand{\Oloc}{O^{\mathrm{loc}}}
\newcommand{\Oglob}{O^{\mathrm{glob}}}
\newcommand{\Ofull}{O^{\mathrm{full}}}
\newcommand{\pih}{\pi_{h}(\sigma)}
\begin{document}
"""
names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(SNIP, "*.tex")) if not os.path.basename(p).startswith("_"))
body = []
for n in names:
    body.append(r"\noindent\texttt{snippet: %s}\par\input{%s}" % (n.replace("_", r"\_"), n))
    body.append(r"\clearpage")
with open(os.path.join(SNIP, "_test.tex"), "w") as fh:
    fh.write(PRE + "\n".join(body) + "\n\\end{document}\n")
print(names)
