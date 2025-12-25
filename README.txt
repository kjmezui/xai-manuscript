npj Artificial Intelligence - Manuscript Submission
==================================================

Manuscript ID: 
Title: Making AI Understandable, Reliable, and Trustworthy: How to Balance Performance, User-friendliness, and Confidence in Deep Learning Systems
Corresponding Author: Kevin MEZUI (kjmezui@gmail.com)

CONTENTS
--------
This bundle contains the complete LaTeX source files required to compile the submitted manuscript:

1. manuscript_main.tex - Main LaTeX document
2. references.bib      - BibTeX bibliography database
3. figures/            - Directory containing all figure files (PNG format)
4. README.txt          - This file

COMPILATION INSTRUCTIONS
------------------------
The manuscript is prepared for compilation with pdfLaTeX and BibTeX.

Recommended compilation sequence:
   pdflatex manuscript_main.tex
   bibtex manuscript_main
   pdflatex manuscript_main.tex
   pdflatex manuscript_main.tex

Alternatively, use the compile.sh script if included.

DOCUMENT DETAILS
----------------
- Document class: article (12pt, a4paper)
- Bibliography style: APA (via natbib)
- Required packages: All standard packages are included in the preamble
- Special formatting: All editorial statements (Author Contributions, Funding, etc.) are included as separate sections before the References

ADDITIONAL NOTES
----------------
- All tables are properly referenced in the main text
- Figure files are in PNG format, 300 DPI minimum resolution
- The manuscript has been compiled successfully with TeX Live 2024
- No external dependencies beyond standard LaTeX distributions

CONTACT
-------
For questions regarding these source files, please contact the corresponding author.