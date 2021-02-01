DOC_TEX	= slepc.tex \
	intro.tex \
	eps.tex \
	st.tex \
	svd.tex \
	pep.tex \
	nep.tex \
	mfn.tex \
	extra.tex

.SUFFIXES: .tex .pdf

slepc.pdf: $(DOC_TEX) makefile
	@ latexmk -bibtex- -pdf -silent slepc

clean:
	@ latexmk -bibtex- -pdf -silent -c slepc

