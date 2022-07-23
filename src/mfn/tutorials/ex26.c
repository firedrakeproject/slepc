/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
  PetscBool      flag;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag));
  if (!flag) m=n;
  N = n*m;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSquare root of Laplacian y=sqrt(A)*e_1, N=%" PetscInt_FMT " (%" PetscInt_FMT "x%" PetscInt_FMT " grid)\n\n",N,n,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Compute the discrete 2-D Laplacian, A
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (II=Istart;II<Iend;II++) {
    i = II/n; j = II-i*n;
    if (i>0) PetscCall(MatSetValue(A,II,II-n,-1.0,INSERT_VALUES));
    if (i<m-1) PetscCall(MatSetValue(A,II,II+n,-1.0,INSERT_VALUES));
    if (j>0) PetscCall(MatSetValue(A,II,II-1,-1.0,INSERT_VALUES));
    if (j<n-1) PetscCall(MatSetValue(A,II,II+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,II,II,4.0,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* set symmetry flag so that solver can exploit it */
  PetscCall(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));

  /* set v = e_1 */
  PetscCall(MatCreateVecs(A,NULL,&v));
  PetscCall(VecSetValue(v,0,1.0,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscCall(VecDuplicate(v,&y));
  PetscCall(VecDuplicate(v,&z));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create the solver, set the matrix and the function
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MFNCreate(PETSC_COMM_WORLD,&mfn));
  PetscCall(MFNSetOperator(mfn,A));
  PetscCall(MFNGetFN(mfn,&f));
  PetscCall(FNSetType(f,FNSQRT));
  PetscCall(MFNSetErrorIfNotConverged(mfn,PETSC_TRUE));
  PetscCall(MFNSetFromOptions(mfn));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      First solve: y=sqrt(A)*v
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MFNSolve(mfn,v,y));
  PetscCall(VecNorm(y,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Intermediate vector has norm %g\n",(double)norm));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Second solve: z=sqrt(A)*y and compare against A*v
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MFNSolve(mfn,y,z));
  PetscCall(MFNGetTolerances(mfn,&tol,NULL));

  PetscCall(MatMult(A,v,y));   /* overwrite y */
  PetscCall(VecAXPY(y,-1.0,z));
  PetscCall(VecNorm(y,NORM_2,&norm));

  if (norm<tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD," Error norm is less than the requested tolerance\n\n"));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD," Error norm larger than tolerance: %3.1e\n\n",(double)norm));

  /*
     Free work space
  */
  PetscCall(MFNDestroy(&mfn));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -mfn_tol 1e-4

TEST*/
