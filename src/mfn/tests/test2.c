/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests the case when both arguments of MFNSolve() are the same Vec.\n\n"
  "The command line options are:\n"
  "  -t <sval>, where <sval> = scalar value that multiplies the argument.\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n";

#include <slepcmfn.h>

int main(int argc,char **argv)
{
  Mat            A;           /* problem matrix */
  MFN            mfn;
  FN             f;
  PetscReal      norm;
  PetscScalar    t=0.3;
  PetscInt       N,n=25,m,Istart,Iend,II,i,j;
  PetscBool      flag;
  Vec            v,y;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-t",&t,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nMatrix exponential y=exp(t*A)*e, of the 2-D Laplacian, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                         Build the 2-D Laplacian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) CHKERRQ(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) CHKERRQ(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) CHKERRQ(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) CHKERRQ(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,II,II,4.0,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* set v = ones(n,1) */
  CHKERRQ(MatCreateVecs(A,NULL,&y));
  CHKERRQ(MatCreateVecs(A,NULL,&v));
  CHKERRQ(VecSet(v,1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f));
  CHKERRQ(FNSetType(f,FNEXP));

  CHKERRQ(MFNCreate(PETSC_COMM_WORLD,&mfn));
  CHKERRQ(MFNSetOperator(mfn,A));
  CHKERRQ(MFNSetFN(mfn,f));
  CHKERRQ(MFNSetErrorIfNotConverged(mfn,PETSC_TRUE));
  CHKERRQ(MFNSetFromOptions(mfn));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the problem, y=exp(t*A)*v
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(FNSetScale(f,t,1.0));
  CHKERRQ(MFNSolve(mfn,v,y));
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Computed vector at time t=%.4g has norm %g\n\n",(double)PetscRealPart(t),(double)norm));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Repeat the computation in two steps, overwriting v:
              v=exp(0.5*t*A)*v,  v=exp(0.5*t*A)*v
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(FNSetScale(f,0.5*t,1.0));
  CHKERRQ(MFNSolve(mfn,v,v));
  CHKERRQ(MFNSolve(mfn,v,v));
  /* compute norm of difference */
  CHKERRQ(VecAXPY(y,-1.0,v));
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  if (norm<100*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," The norm of the difference is <100*eps\n\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," The norm of the difference is %g\n\n",(double)norm));
  }

  /*
     Free work space
  */
  CHKERRQ(MFNDestroy(&mfn));
  CHKERRQ(FNDestroy(&f));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&y));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -mfn_type {{krylov expokit}}
      output_file: output/test2_1.out
      test:
         suffix: 1
      test:
         suffix: 1_cuda
         args: -mat_type aijcusparse
         requires: cuda

   test:
      suffix: 3
      args: -mfn_type expokit -t 0.6 -mfn_ncv 35
      requires: !__float128

TEST*/
