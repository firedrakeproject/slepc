/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Computes exp(t*A)*v for a matrix associated with a Markov model.\n\n"
  "The command line options are:\n"
  "  -t <t>, where <t> = time parameter (multiplies the matrix).\n"
  "  -m <m>, where <m> = number of grid subdivisions in each dimension.\n\n"
  "To draw the solution run with -mfn_view_solution draw -draw_pause -1\n\n";

#include <slepcmfn.h>

/*
   User-defined routines
*/
PetscErrorCode MatMarkovModel(PetscInt m,Mat A);

int main(int argc,char **argv)
{
  Mat                A;           /* problem matrix */
  MFN                mfn;
  FN                 f;
  PetscReal          tol,norm;
  PetscScalar        t=2.0;
  Vec                v,y;
  PetscInt           N,m=15,ncv,maxit,its;
  PetscErrorCode     ierr;
  MFNConvergedReason reason;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-t",&t,NULL));
  N = m*(m+1)/2;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nMarkov y=exp(t*A)*e_1, N=%" PetscInt_FMT " (m=%" PetscInt_FMT ")\n\n",N,m));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            Compute the transition probability matrix, A
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatMarkovModel(m,A));

  /* set v = e_1 */
  CHKERRQ(MatCreateVecs(A,NULL,&y));
  CHKERRQ(MatCreateVecs(A,NULL,&v));
  CHKERRQ(VecSetValue(v,0,1.0,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create matrix function solver context
  */
  CHKERRQ(MFNCreate(PETSC_COMM_WORLD,&mfn));

  /*
     Set operator matrix, the function to compute, and other options
  */
  CHKERRQ(MFNSetOperator(mfn,A));
  CHKERRQ(MFNGetFN(mfn,&f));
  CHKERRQ(FNSetType(f,FNEXP));
  CHKERRQ(FNSetScale(f,t,1.0));
  CHKERRQ(MFNSetTolerances(mfn,1e-07,PETSC_DEFAULT));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(MFNSetFromOptions(mfn));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the problem, y=exp(t*A)*v
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MFNSolve(mfn,v,y));
  CHKERRQ(MFNGetConvergedReason(mfn,&reason));
  PetscCheck(reason>=0,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Solver did not converge");
  CHKERRQ(VecNorm(y,NORM_2,&norm));

  /*
     Optional: Get some information from the solver and display it
  */
  CHKERRQ(MFNGetIterationNumber(mfn,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));
  CHKERRQ(MFNGetDimensions(mfn,&ncv));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Subspace dimension: %" PetscInt_FMT "\n",ncv));
  CHKERRQ(MFNGetTolerances(mfn,&tol,&maxit));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Computed vector at time t=%.4g has norm %g\n\n",(double)PetscRealPart(t),(double)norm));

  /*
     Free work space
  */
  CHKERRQ(MFNDestroy(&mfn));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&y));
  ierr = SlepcFinalize();
  return ierr;
}

/*
    Matrix generator for a Markov model of a random walk on a triangular grid.
    See ex5.c for additional details.
*/
PetscErrorCode MatMarkovModel(PetscInt m,Mat A)
{
  const PetscReal cst = 0.5/(PetscReal)(m-1);
  PetscReal       pd,pu;
  PetscInt        Istart,Iend,i,j,jmax,ix=0;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=1;i<=m;i++) {
    jmax = m-i+1;
    for (j=1;j<=jmax;j++) {
      ix = ix + 1;
      if (ix-1<Istart || ix>Iend) continue;  /* compute only owned rows */
      if (j!=jmax) {
        pd = cst*(PetscReal)(i+j-1);
        /* north */
        if (i==1) {
          CHKERRQ(MatSetValue(A,ix-1,ix,2*pd,INSERT_VALUES));
        } else {
          CHKERRQ(MatSetValue(A,ix-1,ix,pd,INSERT_VALUES));
        }
        /* east */
        if (j==1) {
          CHKERRQ(MatSetValue(A,ix-1,ix+jmax-1,2*pd,INSERT_VALUES));
        } else {
          CHKERRQ(MatSetValue(A,ix-1,ix+jmax-1,pd,INSERT_VALUES));
        }
      }
      /* south */
      pu = 0.5 - cst*(PetscReal)(i+j-3);
      if (j>1) {
        CHKERRQ(MatSetValue(A,ix-1,ix-2,pu,INSERT_VALUES));
      }
      /* west */
      if (i>1) {
        CHKERRQ(MatSetValue(A,ix-1,ix-jmax-2,pu,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      args: -mfn_ncv 6

TEST*/
