NOM		= slepc
SHELL		= /bin/sh
DVIPS		= dvips
DVIPDF		= dvipdf
LATEX		= latex
BDIR		= bib
BIBINPUTS	= $(BDIR)

DOC_TEX	= \
	slepc.tex \
	slepc1.tex \
	slepc2.tex \
	slepc3.tex \
	slepc4.tex \
	slepcx.tex 

BIB_FILE = slepc.bib

DOC_GPL	= 

DOC_MP	= 

DOC_FIG	=

DOC_BIB = \
	$(BDIR)/arpack.bib \
	$(BDIR)/hpcn.bib \
	$(BDIR)/eig.bib \
	$(BDIR)/book.bib \
	$(BDIR)/tesi.bib

DOC_EPS	= \
	upv.ps \
	dsic.ps \
	petscwww.ps \
	slepc.eps

OTHER	= slepc.cls engnat.bst srcltx.sty

DOC_SRC	= $(DOC_TEX) $(DOC_GPL) $(DOC_FIG) $(DOC_MP) $(DOC_BIB) \
          $(DOC_EPS) $(OTHER)
DOC_DEP	= $(DOC_TEX) $(DOC_GPL) $(subst .fig,.tex,$(DOC_FIG)) \
	  $(subst .mp,.mpx,$(DOC_MP)) $(DOC_BIB)
DOC_GEN	= $(subst .tex,.aux,$(DOC_TEX)) \
	  $(subst .fig,.tex,$(DOC_FIG)) \
	  $(subst .fig,.eps,$(DOC_FIG)) \
	  $(subst .mp,.mpx,$(DOC_MP)) \
	  $(subst .mp,.log,$(DOC_MP)) \
	  $(subst .mp,.[1-9]*,$(DOC_MP)) \
	  $(NOM).bbl \
	  $(NOM).blg \
	  $(NOM).dvi \
	  $(NOM).idx \
	  $(NOM).ilg \
	  $(NOM).ind \
	  $(NOM).loa \
	  $(NOM).lof \
	  $(NOM).log \
	  $(NOM).lot \
	  $(NOM).toc

default: dvi

#$(BIB_FILE): $(DOC_BIB) makefile
#	@ cat $(DOC_BIB) | bibtool -o $@ \
#	-f "(author){%1n(author)}{%1n(editor)}:%d(year):%+3.1W(title)" \
#	-- "fmt.et.al={}" \
#	-- "ignored.word{and}" \
#	-- "ignored.word{by}" \
#	-- "ignored.word{of}" \
#	-- "ignored.word{an}" \
#	-- "ignored.word{to}" \
#	-- "ignored.word{on}" \
#	-- "ignored.word{with}" \
#	-- "ignored.word{in}" \
#	-- "ignored.word{for}" \
#	-- "sort = {on}" \
#	-- "sort.format={%n(author)%d(year)%10t(title)%d(volume)}" \
#	-- "sort.order={*=author#title}"

###########################################################################

.SUFFIXES: .fig .tex .dvi .ps .pdf

%.mpx: %.mp
	@ mpost $*

.fig.tex:
	@ ( fig2dev -L pstex -p dummy $*.fig > $*.eps )
	@ ( fig2dev -L pstex_t -p $*.eps $*.fig > $*.tex )
#	@ ( fig2dev -L eepic $*.fig > $*.tex )
#	# \usepackage{epic,eepic}

.eps.dvi:
.tex.dvi:
	@ if test ! -f $*.ind; then echo "" > $*.ind; fi
	@ $(LATEX) $*
	@ if ( grep 'Writing index file' $*.log > /dev/null ); \
	then makeindex $* ; $(LATEX) $* ; fi
	@ if ( grep 'LaTeX Warning: Label(s) may' $*.log > /dev/null ); \
	then $(LATEX) $* ; fi
	@ if ( grep 'Warning: Citation' $*.log > /dev/null ); \
	then bibtex $* ; $(LATEX) $* ; fi
	@ if ( grep 'LaTeX Warning: Label(s) may' $*.log > /dev/null ); \
	then $(LATEX) $* ; fi
	@ if ( grep 'Warning: Citation(s) may' $*.log > /dev/null ); \
	then $(LATEX) $* ; fi
	@ if ( grep 'LaTeX Warning: Label(s) may' $*.log > /dev/null ); \
	then $(LATEX) $* ; fi
	@ if ( grep 'LaTeX Warning: Label(s) may' $*.log > /dev/null ); \
	then $(LATEX) $* ; fi

.dvi.ps:
	@ $(DVIPS) -o $*.ps $*

.dvi.pdf:
	@ $(DVIPDF) $*.dvi

###########################################################################

dvi: $(NOM).dvi
ps: $(NOM).ps
pdf: $(NOM).pdf

$(NOM).ps: $(NOM).dvi
$(NOM).dvi: $(DOC_DEP) $(BIB_FILE) makefile

disc:
	@ echo Genere $(NOM)-`date +%d-%b`.tgz
	@ rm -f $(NOM)*.tgz
	@ tar czf $(NOM)-`date +%d-%b`.tgz makefile $(DOC_SRC)
	@ if ( df | grep fd0 > /dev/null ); \
	  then echo Copie a /floppy ; \
	  cp $(NOM)-`date +%d-%b`.tgz /floppy ; \
	  else echo No esta montat el disquet ; fi

clean:
	@ rm -f $(DOC_GEN)

