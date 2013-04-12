LATEX		= pdflatex --shell-escape

DOC_TEX	= slepc.tex \
	intro.tex \
	eps.tex \
	st.tex \
	svd.tex \
	qep.tex \
	extra.tex

DOC_SRC = $(DOC_TEX) slepc.bbl

DOC_GEN	= $(subst .tex,.aux,$(DOC_TEX)) \
	  slepc.blg \
	  slepc.out \
	  slepc.idx \
	  slepc.ilg \
	  slepc.ind \
	  slepc.lof \
	  slepc.log \
	  slepc.lot \
	  slepc.toc \
	  slepc.fold.gnuplot \
	  slepc.fold.table \
	  slepc.sinv1.gnuplot \
	  slepc.sinv1.table \
	  slepc.sinv2.gnuplot \
	  slepc.sinv2.table

LOCDIR  = docs/manual/

default: pdf

.SUFFIXES: .tex .pdf

.tex.pdf:
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

pdf: slepc.pdf

slepc.pdf: $(DOC_SRC) makefile

clean:
	@ rm -f $(DOC_GEN)

