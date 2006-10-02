
CFLAGS	   = 
FFLAGS     =
CPPFLAGS   = 
FPPFLAGS   =
LOCDIR     = src/examples/
EXAMPLESC  = ex1.c ex2.c ex3.c ex4.c ex5.c ex7.c ex8.c ex9.c ex10.c ex11.c ex12.c
EXAMPLESCH = 
EXAMPLESF  = ex1f.F ex6f.F mvmisg.f

TESTEXAMPLES_C  = ex1.PETSc runex1_1 ex1.rm ex2.PETSc runex2_1 ex2.rm \
                  ex4.PETSc runex4_1 ex4.rm ex5.PETSc runex5_1 ex5.rm \
                  ex7.PETSc runex7_1 ex7.rm ex10.PETSc runex10_1 ex10.rm \
                  ex11.PETSc runex11_1 ex11.rm ex12.PETSc runex12_1 ex12.rm

TESTEXAMPLES_FORTRAN  = ex1f.PETSc runex1f_1 ex1f.rm

TESTEXAMPLES_FORTRAN_MPIUNI  = ex6f.PETSc runex6f_1 ex6f.rm

TESTEXAMPLES_C_X11_MPIUNI  = ex3.PETSc runex3_1 ex3.rm

include ${SLEPC_DIR}/bmake/slepc_common

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${SLEPC_LIB}
	${RM} ex1.o

ex1f: ex1f.o chkopts
	-${FLINKER} -o ex1f ex1f.o ${SLEPC_LIB}
	${RM} ex1f.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${SLEPC_LIB}
	${RM} ex2.o

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o ${SLEPC_LIB}
	${RM} ex3.o

ex4: ex4.o chkopts
	-${CLINKER} -o ex4 ex4.o ${SLEPC_LIB}
	${RM} ex4.o

ex5: ex5.o chkopts
	-${CLINKER} -o ex5 ex5.o ${SLEPC_LIB}
	${RM} ex5.o

ex6f: ex6f.o mvmisg.o chkopts
	-${FLINKER} -o ex6f ex6f.o mvmisg.o ${SLEPC_LIB}
	${RM} ex6f.o mvmisg.o

ex7: ex7.o chkopts
	-${CLINKER} -o ex7 ex7.o ${SLEPC_LIB}
	${RM} ex7.o

ex8: ex8.o chkopts
	-${CLINKER} -o ex8 ex8.o ${SLEPC_LIB}
	${RM} ex8.o

ex9: ex9.o chkopts
	-${CLINKER} -o ex9 ex9.o ${SLEPC_LIB}
	${RM} ex9.o

ex10: ex10.o chkopts
	-${CLINKER} -o ex10 ex10.o ${SLEPC_LIB}
	${RM} ex10.o

ex11: ex11.o chkopts
	-${CLINKER} -o ex11 ex11.o ${SLEPC_LIB}
	${RM} ex11.o

ex12: ex12.o chkopts
	-${CLINKER} -o ex12 ex12.o ${SLEPC_LIB}
	${RM} ex12.o

#------------------------------------------------------------------------------------

runex1_1:
	-@${MPIRUN} -np 1 ex1 > ex1_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true; \
	   else echo "Possible problem with ex1_1, diffs above"; fi; \
	   ${RM} -f ex1_1.tmp

runex1f_1:
	-@${MPIRUN} -np 1 ex1f > ex1f_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex1f_1.out ex1f_1.tmp) then true; \
	   else echo "Possible problem with ex1f_1, diffs above"; fi; \
	   ${RM} -f ex1f_1.tmp

runex2_1:
	-@${MPIRUN} -np 1 ex2 > ex2_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true; \
	   else echo "Possible problem with ex2_1, diffs above"; fi; \
	   ${RM} -f ex2_1.tmp

runex3_1:
	-@${MPIRUN} -np 1 ex3 > ex3_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true; \
	   else echo "Possible problem with ex3_1, diffs above"; fi; \
	   ${RM} -f ex3_1.tmp

runex4_1:
	-@${MPIRUN} -np 1 ex4 -file ${SLEPC_DIR}/src/mat/examples/rdb200.petsc > ex4_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true; \
	   else echo "Possible problem with ex4_1, diffs above"; fi; \
	   ${RM} -f ex4_1.tmp

runex5_1:
	-@${MPIRUN} -np 1 ex5 -st_shift 1 > ex5_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp

runex5_2:
	-@${MPIRUN} -np 2 ex5 -st_shift 1 > ex5_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex5_1.out ex5_2.tmp) then true; \
	   else echo "Possible problem with ex5_2, diffs above"; fi; \
	   ${RM} -f ex5_2.tmp

runex6f_1:
	-@${MPIRUN} -np 1 ex6f -st_shift 1 -eps_max_it 1000 -eps_ncv 12 -eps_tol 1e-5 > ex6f_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex6f_1.out ex6f_1.tmp) then true; \
	   else echo "Possible problem with ex6f_1, diffs above"; fi; \
	   ${RM} -f ex6f_1.tmp

runex7_1:
	-@${MPIRUN} -np 1 ex7 -f1 ${SLEPC_DIR}/src/mat/examples/bfw62a.petsc -f2 ${SLEPC_DIR}/src/mat/examples/bfw62b.petsc > ex7_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex7_1.out ex7_1.tmp) then true; \
	   else echo "Possible problem with ex7_1, diffs above"; fi; \
	   ${RM} -f ex7_1.tmp

runex8_1:
	-@${MPIRUN} -np 1 ex8 > ex8_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex8_1.out ex8_1.tmp) then true; \
	   else echo "Possible problem with ex8_1, diffs above"; fi; \
	   ${RM} -f ex8_1.tmp

runex9_1:
	-@${MPIRUN} -np 1 ex9 > ex9_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex9_1.out ex9_1.tmp) then true; \
	   else echo "Possible problem with ex9_1, diffs above"; fi; \
	   ${RM} -f ex9_1.tmp

runex10_1:
	-@${MPIRUN} -np 1 ex10 > ex10_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex10_1.out ex10_1.tmp) then true; \
	   else echo "Possible problem with ex10_1, diffs above"; fi; \
	   ${RM} -f ex10_1.tmp

runex11_1:
	-@${MPIRUN} -np 1 ex11 > ex11_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex11_1.out ex11_1.tmp) then true; \
	   else echo "Possible problem with ex11_1, diffs above"; fi; \
	   ${RM} -f ex11_1.tmp

runex12_1:
	-@${MPIRUN} -np 1 ex12 -eps_type power -st_shift 1 -eps_nev 2 > ex12_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex12_1.out ex12_1.tmp) then true; \
	   else echo "Possible problem with ex12_1, diffs above"; fi; \
	   ${RM} -f ex12_1.tmp

