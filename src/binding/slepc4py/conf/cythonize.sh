#!/bin/sh
petsc4py=`python -c "import petsc4py;print(petsc4py.get_include())"`
python -m cython --3str --cleanup 3 -w src -Iinclude -I$petsc4py $@ slepc4py.SLEPc.pyx && \
mv src/slepc4py.SLEPc*.h src/include/slepc4py
