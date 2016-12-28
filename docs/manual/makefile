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
	@ latexmk -bibtex- -pdf slepc

clean:
	@ latexmk -bibtex- -pdf -c slepc

