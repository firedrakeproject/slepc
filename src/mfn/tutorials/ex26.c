/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Computes the action of the square root of the 2-D Laplacian.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n\n"
  "To draw the solution run with -mfn_view_solution draw -draw_pause -1\n\n";

#include <slepcmfn.h>

int main(int argc,char **argv)
{
  Mat            A;           /* problem matrix */
  MFN            mfn;
  FN             f;
  PetscReal      norm,tol;
  Vec            v,y,z;
  PetscInt       N,n=10,m,Istart,Iend,i,j,II;
  PetscErrorCode ierr;
  PetscBool      flag;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSquare root of Laplacian y=sqrt(A)*e_1, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Compute the discrete 2-D Laplacian, A
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

  /* set symmetry flag so that solver can exploit it */
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));

  /* set v = e_1 */
  CHKERRQ(MatCreateVecs(A,NULL,&v));
  CHKERRQ(VecSetValue(v,0,1.0,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));
  CHKERRQ(VecDuplicate(v,&y));
  CHKERRQ(VecDuplicate(v,&z));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create the solver, set the matrix and the function
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MFNCreate(PETSC_COMM_WORLD,&mfn));
  CHKERRQ(MFNSetOperator(mfn,A));
  CHKERRQ(MFNGetFN(mfn,&f));
  CHKERRQ(FNSetType(f,FNSQRT));
  CHKERRQ(MFNSetErrorIfNotConverged(mfn,PETSC_TRUE));
  CHKERRQ(MFNSetFromOptions(mfn));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      First solve: y=sqrt(A)*v
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MFNSolve(mfn,v,y));
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Intermediate vector has norm %g\n",(double)norm));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Second solve: z=sqrt(A)*y and compare against A*v
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MFNSolve(mfn,y,z));
  CHKERRQ(MFNGetTolerances(mfn,&tol,NULL));

  CHKERRQ(MatMult(A,v,y));   /* overwrite y */
  CHKERRQ(VecAXPY(y,-1.0,z));
  CHKERRQ(VecNorm(y,NORM_2,&norm));

  if (norm<tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Error norm is less than the requested tolerance\n\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Error norm larger than tolerance: %3.1e\n\n",(double)norm));
  }

  /*
     Free work space
  */
  CHKERRQ(MFNDestroy(&mfn));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&z));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -mfn_tol 1e-4

TEST*/
