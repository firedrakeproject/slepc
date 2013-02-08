/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
      
   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves the same eigenproblem as in example ex5, but computing also left eigenvectors. "
  "It is a Markov model of a random walk on a triangular grid. "
  "A standard nonsymmetric eigenproblem with real eigenvalues. The rightmost eigenvalue is known to be 1.\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = number of grid subdivisions in each dimension.\n\n";

#include <slepceps.h>

/* 
   User-defined routines
*/
PetscErrorCode MatMarkovModel(PetscInt m,Mat A);

#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  Vec            v0,w0;           /* initial vector */
  Vec            *X,*Y;           /* right and left eigenvectors */
  Mat            A;               /* operator matrix */
  EPS            eps;             /* eigenproblem solver context */
  EPSType        type;
  PetscReal      error1,error2,tol,re,im;
  PetscScalar    kr,ki;
  PetscInt       nev,maxit,i,its,nconv,N,m=15;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-m",&m,NULL);CHKERRQ(ierr);
  N = m*(m+1)/2;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nMarkov Model, N=%D (m=%D)\n\n",N,m);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatMarkovModel(m,A);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create eigensolver context
  */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

  /* 
     Set operators. In this case, it is a standard eigenvalue problem.
     Request also left eigenvectors 
  */
  ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr);
  ierr = EPSSetLeftVectorsWanted(eps,PETSC_TRUE);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /*
     Set the initial vector. This is optional, if not done the initial
     vector is set to random values
  */
  ierr = MatGetVecs(A,&v0,&w0);CHKERRQ(ierr);
  ierr = VecSet(v0,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,v0,w0);CHKERRQ(ierr);
  ierr = EPSSetInitialSpace(eps,1,&v0);CHKERRQ(ierr);
  ierr = EPSSetInitialSpaceLeft(eps,1,&w0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(eps,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
  ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4G, maxit=%D\n",tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Get number of converged approximate eigenpairs
  */
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate eigenpairs: %D\n\n",nconv);CHKERRQ(ierr);

  if (nconv>0) {
    /*
       Display eigenvalues and relative errors
    */
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "           k          ||Ax-kx||/||kx||   ||y'A-ky'||/||ky||\n"
         "   ----------------- ------------------ --------------------\n");CHKERRQ(ierr);

    for (i=0;i<nconv;i++) {
      /* 
        Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
        ki (imaginary part)
      */
      ierr = EPSGetEigenvalue(eps,i,&kr,&ki);CHKERRQ(ierr);
      /*
         Compute the relative errors associated to both right and left eigenvectors
      */
      ierr = EPSComputeRelativeError(eps,i,&error1);CHKERRQ(ierr);
      ierr = EPSComputeRelativeErrorLeft(eps,i,&error2);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(kr);
      im = PetscImaginaryPart(kr);
#else
      re = kr;
      im = ki;
#endif 
      if (im!=0.0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD," %9F%+9F j %12G%12G\n",re,im,error1,error2);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12F       %12G       %12G\n",re,error1,error2);CHKERRQ(ierr); 
      }
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

    ierr = VecDuplicateVecs(v0,nconv,&X);
    ierr = VecDuplicateVecs(w0,nconv,&Y);
    for (i=0;i<nconv;i++) {
      ierr = EPSGetEigenvector(eps,i,X[i],NULL);CHKERRQ(ierr);
      ierr = EPSGetEigenvectorLeft(eps,i,Y[i],NULL);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "                   Bi-orthogonality <x,y>                   \n"
         "   ---------------------------------------------------------\n");CHKERRQ(ierr);

    ierr = SlepcCheckOrthogonality(X,nconv,Y,nconv,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
    ierr = VecDestroyVecs(nconv,&X);CHKERRQ(ierr);
    ierr = VecDestroyVecs(nconv,&Y);CHKERRQ(ierr);

  }
  
  /* 
     Free work space
  */
  ierr = VecDestroy(&v0);CHKERRQ(ierr);
  ierr = VecDestroy(&w0);CHKERRQ(ierr);
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "MatMarkovModel"
/*
    Matrix generator for a Markov model of a random walk on a triangular grid.

    This subroutine generates a test matrix that models a random walk on a 
    triangular grid. This test example was used by G. W. Stewart ["{SRRIT} - a 
    FORTRAN subroutine to calculate the dominant invariant subspaces of a real
    matrix", Tech. report. TR-514, University of Maryland (1978).] and in a few
    papers on eigenvalue problems by Y. Saad [see e.g. LAA, vol. 34, pp. 269-295
    (1980) ]. These matrices provide reasonably easy test problems for eigenvalue
    algorithms. The transpose of the matrix  is stochastic and so it is known 
    that one is an exact eigenvalue. One seeks the eigenvector of the transpose 
    associated with the eigenvalue unity. The problem is to calculate the steady
    state probability distribution of the system, which is the eigevector 
    associated with the eigenvalue one and scaled in such a way that the sum all
    the components is equal to one.
    Note: the code will actually compute the transpose of the stochastic matrix
    that contains the transition probabilities.
*/
PetscErrorCode MatMarkovModel(PetscInt m,Mat A)
{
  const PetscReal cst = 0.5/(PetscReal)(m-1);
  PetscReal       pd,pu;
  PetscInt        i,j,jmax,ix=0,Istart,Iend;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=1;i<=m;i++) {
    jmax = m-i+1;
    for (j=1;j<=jmax;j++) {
      ix = ix + 1;
      if (ix-1<Istart || ix>Iend) continue;  /* compute only owned rows */
      if (j!=jmax) {
        pd = cst*(PetscReal)(i+j-1);
        /* north */
        if (i==1) { 
          ierr = MatSetValue(A,ix-1,ix,2*pd,INSERT_VALUES);CHKERRQ(ierr);
        } else {
          ierr = MatSetValue(A,ix-1,ix,pd,INSERT_VALUES);CHKERRQ(ierr);
        }
        /* east */
        if (j==1) { 
          ierr = MatSetValue(A,ix-1,ix+jmax-1,2*pd,INSERT_VALUES);CHKERRQ(ierr);
        } else {
          ierr = MatSetValue(A,ix-1,ix+jmax-1,pd,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      /* south */
      pu = 0.5 - cst*(PetscReal)(i+j-3);
      if (j>1) {
        ierr = MatSetValue(A,ix-1,ix-2,pu,INSERT_VALUES);CHKERRQ(ierr);
      }
      /* west */
      if (i>1) {
        ierr = MatSetValue(A,ix-1,ix-jmax-2,pu,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

