/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

static char help[] = "Generalized Symmetric eigenproblem.\n\n"
  "The problem is Ax = lambda Bx, with:\n"
  "   A = Laplacian operator in 2-D\n"
  "   B = diagonal matrix with all values equal to 4 except nulldim zeros\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n"
  "  -nulldim <k>, where <k> = dimension of the nullspace of B.\n\n";

#include "slepceps.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  Mat         	 A, B;		  /* matrices */
  EPS         	 eps;		  /* eigenproblem solver context */
  ST          	 st;		  /* spectral transformation context */
  const EPSType  type;
  PetscReal   	 error, tol, re, im;
  PetscScalar 	 kr, ki;
  PetscErrorCode ierr;
  PetscInt    	 N, n=10, m, Istart, Iend, II, nev, maxit, i, j, its, nconv, nulldim=0;
  PetscBool   	 flag;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,&flag);CHKERRQ(ierr);
  if(!flag) m=n;
  N = n*m;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nulldim",&nulldim,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized Symmetric Eigenproblem, N=%d (%dx%d grid), null(B)=%d\n\n",N,n,m,nulldim);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Compute the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for( II=Istart; II<Iend; II++ ) { 
    i = II/n; j = II-i*n;  
    if(i>0) { ierr = MatSetValue(A,II,II-n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if(i<m-1) { ierr = MatSetValue(A,II,II+n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if(j>0) { ierr = MatSetValue(A,II,II-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if(j<n-1) { ierr = MatSetValue(A,II,II+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(A,II,II,4.0,INSERT_VALUES);CHKERRQ(ierr);
    if (II>=nulldim) { ierr = MatSetValue(B,II,II,4.0,INSERT_VALUES);CHKERRQ(ierr); }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create eigensolver context
  */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

  /* 
     Set operators. In this case, it is a generalized eigenvalue problem
  */
  ierr = EPSSetOperators(eps,A,B);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_GHEP);CHKERRQ(ierr);

  /* 
     Use shift-and-invert to avoid solving linear systems with a singular B
     in case nulldim>0
  */
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = EPSSetTarget(eps,0.0);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
  ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(eps, &its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %d\n",its);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %d\n",nev);CHKERRQ(ierr);
  ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%d\n",tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Get number of converged approximate eigenpairs
  */
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate eigenpairs: %d\n\n",nconv);
         CHKERRQ(ierr);

  if (nconv>0) {
    /*
       Display eigenvalues and relative errors
    */
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "           k          ||Ax-kBx||/||kx||\n"
         "   ----------------- ------------------\n" );CHKERRQ(ierr);

    for( i=0; i<nconv; i++ ) {
      /* 
        Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
        ki (imaginary part)
      */
      ierr = EPSGetEigenpair(eps,i,&kr,&ki,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      /*
         Compute the relative error associated to each eigenpair
      */
      ierr = EPSComputeRelativeError(eps,i,&error);CHKERRQ(ierr);

#ifdef PETSC_USE_COMPLEX
      re = PetscRealPart(kr);
      im = PetscImaginaryPart(kr);
#else
      re = kr;
      im = ki;
#endif 
      if (im!=0.0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD," %9f%+9f j %12g\n",re,im,error);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12f       %12g\n",re,error);CHKERRQ(ierr); 
      }
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n" );CHKERRQ(ierr);
  }
  
  /* 
     Free work space
  */
  ierr = EPSDestroy(eps);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}

