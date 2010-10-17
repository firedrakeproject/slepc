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

static char help[] = "Illustrates the use of shell spectral transformations. "
  "The problem to be solved is the same as ex1.c and"
  "corresponds to the Laplacian operator in 1 dimension.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions = matrix dimension.\n\n";

#include "slepceps.h"

/* Define context for user-provided spectral transformation */
typedef struct {
  KSP        ksp;
} SampleShellST;

/* Declare routines for user-provided spectral transformation */
PetscErrorCode SampleShellSTCreate(SampleShellST**);
PetscErrorCode SampleShellSTSetUp(SampleShellST*,ST);
PetscErrorCode SampleShellSTApply(ST,Vec,Vec);
PetscErrorCode SampleShellSTBackTransform(ST,PetscInt,PetscScalar*,PetscScalar*);
PetscErrorCode SampleShellSTDestroy(SampleShellST*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  Mat            A;		  /* operator matrix */
  EPS            eps;		  /* eigenproblem solver context */
  ST             st;		  /* spectral transformation context */
  SampleShellST  *shell;	  /* user-defined spectral transform context */
  const EPSType  type;
  PetscReal      error, tol, re, im;
  PetscScalar    kr, ki;
  PetscErrorCode ierr;
  PetscInt       n=30, i, col[3], Istart, Iend, FirstBlock=0, LastBlock=0, nev, maxit, its, nconv;
  PetscScalar    value[3];
  PetscBool      isShell;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n1-D Laplacian Eigenproblem (shell-enabled), n=%d\n\n",n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  if (Istart==0) FirstBlock=PETSC_TRUE;
  if (Iend==n) LastBlock=PETSC_TRUE;
  value[0]=-1.0; value[1]=2.0; value[2]=-1.0;
  for( i=(FirstBlock? Istart+1: Istart); i<(LastBlock? Iend-1: Iend); i++ ) {
    col[0]=i-1; col[1]=i; col[2]=i+1;
    ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (LastBlock) {
    i=n-1; col[0]=n-2; col[1]=n-1;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (FirstBlock) {
    i=0; col[0]=0; col[1]=1; value[0]=2.0; value[1]=-1.0;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create eigensolver context
  */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

  /* 
     Set operators. In this case, it is a standard eigenvalue problem
  */
  ierr = EPSSetOperators(eps,A,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /*
     Initialize shell spectral transformation if selected by user
  */
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)st,STSHELL,&isShell);CHKERRQ(ierr);
  if (isShell) {
    /* (Optional) Create a context for the user-defined spectral tranform;
       this context can be defined to contain any application-specific data. */
    ierr = SampleShellSTCreate(&shell);CHKERRQ(ierr);

    /* (Required) Set the user-defined routine for applying the operator */
    ierr = STShellSetApply(st,SampleShellSTApply);CHKERRQ(ierr);
    ierr = STShellSetContext(st,shell);CHKERRQ(ierr);

    /* (Optional) Set the user-defined routine for back-transformation */
    ierr = STShellSetBackTransform(st,SampleShellSTBackTransform);CHKERRQ(ierr);

    /* (Optional) Set a name for the transformation, used for STView() */
    ierr = PetscObjectSetName((PetscObject)st,"MyTransformation");CHKERRQ(ierr);

    /* (Optional) Do any setup required for the new transformation */
    ierr = SampleShellSTSetUp(shell,st);CHKERRQ(ierr);
  }

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
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %d\n\n",nconv);CHKERRQ(ierr);

  if (nconv>0) {
    /*
       Display eigenvalues and relative errors
    */
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "           k          ||Ax-kx||/||kx||\n"
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
  if (isShell) {
    ierr = SampleShellSTDestroy(shell);CHKERRQ(ierr);
  }
  ierr = EPSDestroy(eps);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}

/***********************************************************************/
/*     Routines for a user-defined shell spectral transformation       */
/***********************************************************************/

#undef __FUNCT__  
#define __FUNCT__ "SampleShellSTCreate"
/*
   SampleShellSTCreate - This routine creates a user-defined
   spectral transformation context.

   Output Parameter:
.  shell - user-defined spectral transformation context
*/
PetscErrorCode SampleShellSTCreate(SampleShellST **shell)
{
  SampleShellST  *newctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = PetscNew(SampleShellST,&newctx);CHKERRQ(ierr);
  ierr   = KSPCreate(PETSC_COMM_WORLD,&newctx->ksp);CHKERRQ(ierr);
  ierr   = KSPAppendOptionsPrefix(newctx->ksp,"st_"); CHKERRQ(ierr);
  *shell = newctx;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SampleShellSTSetUp"
/*
   SampleShellSTSetUp - This routine sets up a user-defined
   spectral transformation context.  

   Input Parameters:
.  shell - user-defined spectral transformation context
.  st    - spectral transformation context containing the operator matrices

   Output Parameter:
.  shell - fully set up user-defined transformation context

   Notes:
   In this example, the user-defined transformation is simply OP=A^-1.
   Therefore, the eigenpairs converge in reversed order. The KSP object
   used for the solution of linear systems with A is handled via the
   user-defined context SampleShellST.
*/
PetscErrorCode SampleShellSTSetUp(SampleShellST *shell,ST st)
{
  Mat            A,B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = STGetOperators(st,&A,&B);CHKERRQ(ierr);
  if (B) { SETERRQ(((PetscObject)st)->comm,0,"Warning: This transformation is not intended for generalized problems"); }
  ierr = KSPSetOperators(shell->ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(shell->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SampleShellSTApply"
/*
   SampleShellSTApply - This routine demonstrates the use of a
   user-provided spectral transformation.

   Input Parameters:
.  ctx - optional user-defined context, as set by STShellSetContext()
.  x - input vector

   Output Parameter:
.  y - output vector

   Notes:
   The transformation implemented in this code is just OP=A^-1 and
   therefore it is of little use, merely as an example of working with
   a STSHELL.
*/
PetscErrorCode SampleShellSTApply(ST st,Vec x,Vec y)
{
  SampleShellST  *shell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = STShellGetContext(st,(void**)&shell);
  ierr = KSPSolve(shell->ksp,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SampleShellSTBackTransform"
/*
   SampleShellSTBackTransform - This routine demonstrates the use of a
   user-provided spectral transformation.

   Input Parameters:
.  ctx  - optional user-defined context, as set by STShellSetContext()
.  eigr - pointer to real part of eigenvalues
.  eigi - pointer to imaginary part of eigenvalues

   Output Parameters:
.  eigr - modified real part of eigenvalues
.  eigi - modified imaginary part of eigenvalues

   Notes:
   This code implements the back transformation of eigenvalues in
   order to retrieve the eigenvalues of the original problem. In this
   example, simply set k_i = 1/k_i.
*/
PetscErrorCode SampleShellSTBackTransform(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscInt j;
  PetscFunctionBegin;
  for (j=0;j<n;j++) {
    eigr[j] = 1.0 / eigr[j];
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SampleShellSTDestroy"
/*
   SampleShellSTDestroy - This routine destroys a user-defined
   spectral transformation context.

   Input Parameter:
.  shell - user-defined spectral transformation context
*/
PetscErrorCode SampleShellSTDestroy(SampleShellST *shell)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy(shell->ksp);CHKERRQ(ierr);
  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


