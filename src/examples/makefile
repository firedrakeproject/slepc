
CFLAGS	   = 
FFLAGS     =
CPPFLAGS   = 
FPPFLAGS   =
LOCDIR     = src/examples/
EXAMPLESC  = ex1.c ex2.c ex3.c ex4.c ex5.c ex7.c ex8.c ex9.c ex10.c
EXAMPLESCH = 
EXAMPLESF  = ex1f.F ex6f.F mvmisg.f

TESTEXAMPLES_1  = ex1.PETSc runex1_1 ex1.rm ex2.PETSc runex2_1 ex2.rm \
                  ex4.PETSc runex4_1 ex4.rm ex5.PETSc runex5_1 runex5_2 ex5.rm \
                  ex7.PETSc runex7_1 ex7.rm ex10.PETSc runex10_1 ex10.rm
TESTEXAMPLES_2  = 
TESTEXAMPLES_3  = ex1f.PETSc runex1f_1 ex1f.rm
TESTEXAMPLES_9  = ex6f.PETSc runex6f_1 ex6f.rm
TESTEXAMPLES_4  = ex3.PETSc runex3_1 ex3.rm
TESTEXAMPLES_5  = 
TESTEXAMPLES_11 = 
TESTEXAMPLES_6  = 
TESTEXAMPLES_7  = 
TESTEXAMPLES_8  = 
TESTEXAMPLES_13 = 

include ${SLEPC_DIR}/bmake/slepc_common

ex1: ex1.o slepc_chkopts
	-${CLINKER} -o ex1 ex1.o ${SLEPC_LIB}
	${RM} ex1.o

ex1f: ex1f.o slepc_chkopts
	-${FLINKER} -o ex1f ex1f.o ${SLEPC_FORTRAN_LIB} ${SLEPC_LIB}
	${RM} ex1f.o

ex2: ex2.o slepc_chkopts
	-${CLINKER} -o ex2 ex2.o ${SLEPC_LIB}
	${RM} ex2.o

ex3: ex3.o slepc_chkopts
	-${CLINKER} -o ex3 ex3.o ${SLEPC_LIB}
	${RM} ex3.o

ex4: ex4.o slepc_chkopts
	-${CLINKER} -o ex4 ex4.o ${SLEPC_LIB}
	${RM} ex4.o

ex5: ex5.o slepc_chkopts
	-${CLINKER} -o ex5 ex5.o ${SLEPC_LIB}
	${RM} ex5.o

ex6f: ex6f.o mvmisg.o slepc_chkopts
	-${FLINKER} -o ex6f ex6f.o mvmisg.o ${SLEPC_FORTRAN_LIB} ${SLEPC_LIB}
	${RM} ex6f.o mvmisg.o

ex7: ex7.o slepc_chkopts
	-${CLINKER} -o ex7 ex7.o ${SLEPC_LIB}
	${RM} ex7.o

ex8: ex8.o slepc_chkopts
	-${CLINKER} -o ex8 ex8.o ${SLEPC_LIB}
	${RM} ex8.o

ex9: ex9.o slepc_chkopts
	-${CLINKER} -o ex9 ex9.o ${SLEPC_LIB}
	${RM} ex9.o

ex10: ex10.o slepc_chkopts
	-${CLINKER} -o ex10 ex10.o ${SLEPC_LIB}
	${RM} ex10.o

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
	-@${MPIRUN} -np 1 ex5 -st_type shift > ex5_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex5_1.out ex5_1.tmp) then true; \
	   else echo "Possible problem with ex5_1, diffs above"; fi; \
	   ${RM} -f ex5_1.tmp

runex5_2:
	-@${MPIRUN} -np 2 ex5 -st_type shift > ex5_2.tmp 2>&1;	  \
	   if (${DIFF} output/ex5_1.out ex5_2.tmp) then true; \
	   else echo "Possible problem with ex5_2, diffs above"; fi; \
	   ${RM} -f ex5_2.tmp

runex6f_1:
	-@${MPIRUN} -np 1 ex6f -st_type shift > ex6f_1.tmp 2>&1;	  \
	   if (${DIFF} output/ex6f_1.out ex6f_1.tmp) then true; \
	   else echo "Possible problem with ex6f_1, diffs above"; fi; \
	   ${RM} -f ex6f_1.tmp

runex7_1:
	-@${MPIRUN} -np 1 ex7 -f1 ${SLEPC_DIR}/src/mat/examples/odep400a.petsc -f2 ${SLEPC_DIR}/src/mat/examples/odep400b.petsc -st_type sinvert -st_shift -1e18 > ex7_1.tmp 2>&1;	  \
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

