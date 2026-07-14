# Compilation locale du rapport LaTeX
# Usage :
#   make        -> compile report.pdf (latex + bibtex + latex x2)
#   make quick  -> une seule passe pdflatex (itération rapide, sans biblio)
#   make clean  -> supprime les fichiers auxiliaires
#   make purge  -> clean + supprime le PDF

MAIN    = report
LATEX   = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX  = bibtex

.PHONY: all quick clean purge

# 'all' recompile TOUJOURS (pas de vérification de dates : évite le
# « Rien à faire » quand le PDF est plus récent que les sources).
all:
	$(LATEX) $(MAIN).tex
	-$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex
	@echo "==> $(MAIN).pdf généré."

quick:
	$(LATEX) $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot \
	      *.fls *.fdb_latexmk *.synctex.gz chapters/*.aux

purge: clean
	rm -f $(MAIN).pdf
