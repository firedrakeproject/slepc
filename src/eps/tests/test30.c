/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test changing EPS type.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A;           /* problem matrix */
  EPS            eps;         /* eigenproblem solver context */
  ST             st;
  KSP            ksp;
  PetscInt       n=20,i,Istart,Iend,nrest;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nDiagonal Eigenproblem, n=%" PetscInt_FMT "\n\n",n));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) PetscCall(MatSetValue(A,i,i,i+1,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create eigensolver and view options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
  PetscCall(EPSSetOperators(eps,A,NULL));
  PetscCall(EPSSetProblemType(eps,EPS_HEP));
  PetscCall(EPSSetTarget(eps,4.8));
  PetscCall(EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE));
  PetscCall(EPSSetDimensions(eps,4,PETSC_DEFAULT,PETSC_DEFAULT));

  PetscCall(EPSSetType(eps,EPSRQCG));
  PetscCall(EPSRQCGSetReset(eps,10));
  PetscCall(EPSSetFromOptions(eps));
  PetscCall(EPSRQCGGetReset(eps,&nrest));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"RQCG reset parameter = %" PetscInt_FMT "\n\n",nrest));
  PetscCall(EPSView(eps,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Change eigensolver type and solve problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(EPSSetType(eps,EPSGD));
  PetscCall(EPSGetST(eps,&st));
  PetscCall(STGetKSP(st,&ksp));
  PetscCall(KSPSetType(ksp,KSPPREONLY));
  PetscCall(EPSSolve(eps));
  PetscCall(EPSView(eps,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -eps_harmonic -eps_conv_norm
      filter: grep -v tolerance | sed -e "s/symmetric/hermitian/"

TEST*/
