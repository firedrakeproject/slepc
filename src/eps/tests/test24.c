/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Eigenproblem for the 1-D Laplacian with constraints. "
  "Based on ex1.\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions = matrix dimension.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A;
  EPS            eps;
  EPSType        type;
  Vec            *vi=NULL,*vc=NULL,t;
  PetscInt       n=30,nev=4,i,j,Istart,Iend,nini=0,ncon=0,bs;
  PetscReal      alpha,beta,restart;
  PetscBool      flg,lock;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nini",&nini,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ncon",&ncon,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Laplacian Eigenproblem, n=%" PetscInt_FMT " nini=%" PetscInt_FMT " ncon=%" PetscInt_FMT "\n\n",n,nini,ncon));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,i,i,2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));
  CHKERRQ(EPSSetType(eps,EPSLOBPCG));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
  CHKERRQ(EPSSetConvergenceTest(eps,EPS_CONV_ABS));
  CHKERRQ(EPSSetDimensions(eps,nev,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(EPSLOBPCGSetBlockSize(eps,nev));
  CHKERRQ(EPSLOBPCGSetRestart(eps,0.7));
  CHKERRQ(EPSSetTolerances(eps,1e-8,1200));
  CHKERRQ(EPSSetFromOptions(eps));

  CHKERRQ(MatCreateVecs(A,&t,NULL));
  if (nini) {
    CHKERRQ(VecDuplicateVecs(t,nini,&vi));
    for (i=0;i<nini;i++) CHKERRQ(VecSetRandom(vi[i],NULL));
    CHKERRQ(EPSSetInitialSpace(eps,nini,vi));
  }
  if (ncon) {   /* constraints are exact eigenvectors of lowest eigenvalues */
    alpha = PETSC_PI/(n+1);
    beta  = PetscSqrtReal(2.0/(n+1));
    CHKERRQ(VecDuplicateVecs(t,ncon,&vc));
    for (i=0;i<ncon;i++) {
      for (j=0;j<n;j++) CHKERRQ(VecSetValue(vc[i],j,PetscSinReal(alpha*(j+1)*(i+1))*beta,INSERT_VALUES));
      CHKERRQ(VecAssemblyBegin(vc[i]));
      CHKERRQ(VecAssemblyEnd(vc[i]));
    }
    CHKERRQ(EPSSetDeflationSpace(eps,ncon,vc));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));
  CHKERRQ(EPSGetType(eps,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps,EPSLOBPCG,&flg));
  if (flg) {
    CHKERRQ(EPSLOBPCGGetLocking(eps,&lock));
    if (lock) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Using soft locking\n"));
    CHKERRQ(EPSLOBPCGGetRestart(eps,&restart));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," LOBPCG Restart parameter=%.4g\n",(double)restart));
    CHKERRQ(EPSLOBPCGGetBlockSize(eps,&bs));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," LOBPCG Block size=%" PetscInt_FMT "\n",bs));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroyVecs(nini,&vi));
  CHKERRQ(VecDestroyVecs(ncon,&vc));
  CHKERRQ(VecDestroy(&t));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -ncon 2
      output_file: output/test24_1.out
      test:
         suffix: 1
         requires: !single
      test:
         suffix: 2_cuda
         args: -mat_type aijcusparse
         requires: cuda !single

TEST*/
