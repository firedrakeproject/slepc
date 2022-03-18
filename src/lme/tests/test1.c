/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test LME interface functions, based on ex32.c.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepclme.h>

int main(int argc,char **argv)
{
  Mat                  A,B,C,C1,D;
  LME                  lme;
  PetscReal            tol,errest,error;
  PetscScalar          *u;
  PetscInt             N,n=10,m,Istart,Iend,II,maxit,ncv,i,j;
  PetscErrorCode       ierr;
  PetscBool            flg,testprefix=PETSC_FALSE,viewmatrices=PETSC_FALSE;
  const char           *prefix;
  LMEType              type;
  LMEProblemType       ptype;
  PetscViewerAndFormat *vf;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flg));
  if (!flg) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nLyapunov equation, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_prefix",&testprefix,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-view_matrices",&viewmatrices,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Create the 2-D Laplacian, A
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A,II,II+n,1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,-4.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create a low-rank Mat to store the right-hand side C = C1*C1'
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C1));
  CHKERRQ(MatSetSizes(C1,PETSC_DECIDE,PETSC_DECIDE,N,2));
  CHKERRQ(MatSetType(C1,MATDENSE));
  CHKERRQ(MatSetUp(C1));
  CHKERRQ(MatGetOwnershipRange(C1,&Istart,&Iend));
  CHKERRQ(MatDenseGetArray(C1,&u));
  for (i=Istart;i<Iend;i++) {
    if (i<N/2) u[i-Istart] = 1.0;
    if (i==0) u[i+Iend-2*Istart] = -2.0;
    if (i==1) u[i+Iend-2*Istart] = -1.0;
    if (i==2) u[i+Iend-2*Istart] = -1.0;
  }
  CHKERRQ(MatDenseRestoreArray(C1,&u));
  CHKERRQ(MatAssemblyBegin(C1,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C1,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateLRC(NULL,C1,NULL,NULL,&C));
  CHKERRQ(MatDestroy(&C1));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(LMECreate(PETSC_COMM_WORLD,&lme));
  CHKERRQ(LMESetProblemType(lme,LME_SYLVESTER));
  CHKERRQ(LMEGetProblemType(lme,&ptype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Equation type set to %d\n",ptype));
  CHKERRQ(LMESetProblemType(lme,LME_LYAPUNOV));
  CHKERRQ(LMEGetProblemType(lme,&ptype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Equation type changed to %d\n",ptype));
  CHKERRQ(LMESetCoefficients(lme,A,NULL,NULL,NULL));
  CHKERRQ(LMESetRHS(lme,C));

  /* test prefix usage */
  if (testprefix) {
    CHKERRQ(LMESetOptionsPrefix(lme,"check_"));
    CHKERRQ(LMEAppendOptionsPrefix(lme,"myprefix_"));
    CHKERRQ(LMEGetOptionsPrefix(lme,&prefix));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," LME prefix is currently: %s\n",prefix));
  }

  /* test some interface functions */
  CHKERRQ(LMEGetCoefficients(lme,&B,NULL,NULL,NULL));
  if (viewmatrices) CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(LMEGetRHS(lme,&D));
  if (viewmatrices) CHKERRQ(MatView(D,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(LMESetTolerances(lme,PETSC_DEFAULT,100));
  CHKERRQ(LMESetDimensions(lme,21));
  CHKERRQ(LMESetErrorIfNotConverged(lme,PETSC_TRUE));
  /* test monitors */
  CHKERRQ(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf));
  CHKERRQ(LMEMonitorSet(lme,(PetscErrorCode (*)(LME,PetscInt,PetscReal,void*))LMEMonitorDefault,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy));
  /* CHKERRQ(LMEMonitorCancel(lme)); */
  CHKERRQ(LMESetFromOptions(lme));

  CHKERRQ(LMEGetType(lme,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solver being used: %s\n",type));

  /* query properties and print them */
  CHKERRQ(LMEGetTolerances(lme,&tol,&maxit));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Tolerance: %g, max iterations: %" PetscInt_FMT "\n",(double)tol,maxit));
  CHKERRQ(LMEGetDimensions(lme,&ncv));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Subspace dimension: %" PetscInt_FMT "\n",ncv));
  CHKERRQ(LMEGetErrorIfNotConverged(lme,&flg));
  if (flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Erroring out if convergence fails\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve the matrix equation and compute residual error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(LMESolve(lme));
  CHKERRQ(LMEGetErrorEstimate(lme,&errest));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Error estimate reported by the solver: %.4g\n",(double)errest));
  CHKERRQ(LMEComputeError(lme,&error));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Computed residual norm: %.4g\n\n",(double)error));

  /*
     Free work space
  */
  CHKERRQ(LMEDestroy(&lme));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&C));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -lme_monitor_cancel -lme_converged_reason -lme_view -view_matrices -log_exclude lme,bv
      requires: double
      filter: sed -e "s/4.0[0-9]*e-10/4.03e-10/"

   test:
      suffix: 2
      args: -test_prefix -check_myprefix_lme_monitor
      requires: double
      filter: sed -e "s/estimate [0-9]\.[0-9]*e[+-]\([0-9]*\)/estimate (removed)/g" | sed -e "s/4.0[0-9]*e-10/4.03e-10/"

   test:
      suffix: 3
      args: -lme_monitor_cancel -info -lme_monitor draw::draw_lg -draw_virtual
      requires: double
      filter: sed -e "s/equation = [0-9]\.[0-9]*e[+-]\([0-9]*\)/equation = (removed)/g" | sed -e "s/4.0[0-9]*e-10/4.03e-10/" | grep -v Comm | grep -v machine | grep -v PetscGetHostName | grep -v OpenMP | grep -v Colormap | grep -v "Rank of the Cholesky factor" | grep -v "potrf failed" | grep -v "querying" | grep -v FPTrap | grep -v Device

TEST*/
