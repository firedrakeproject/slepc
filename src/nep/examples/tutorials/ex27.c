/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

static char help[] = "Simple nonlinear eigenproblem using the NLEIGS solver.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n"
  "  -split <0/1>, to select the split form in the problem definition (enabled by default)\n";


/*
   Solve T(lambda)x=0 using NLEIGS solver
      with T(lambda) = -D+sqrt(lambda)*I
      where D is the Laplacian operator in 1 dimension
      and with the interpolation interval [.01,16]   
*/

#include <slepcnep.h>

/*
   User-defined routines
*/
PetscErrorCode FormFunction(NEP,PetscScalar,Mat,Mat,void*);
PetscErrorCode ComputeSingularities(NEP,PetscInt*,PetscScalar*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  NEP            nep;             /* nonlinear eigensolver context */
  Mat            F,A[2];             
  NEPType        type;
  PetscInt       n=100,nev,its,Istart,Iend,i;
  PetscErrorCode ierr;
  PetscBool      split=PETSC_TRUE;
  RG             rg;
  FN             f[2];
  PetscScalar    coeffs;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-split",&split,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nNonlinear Eigenproblem, n=%D\n\n",n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear eigensolver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = NEPCreate(PETSC_COMM_WORLD,&nep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Select the NLEIGS solver and set required options for it
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = NEPSetType(nep,NEPNLEIGS);CHKERRQ(ierr);
  ierr = NEPNLEIGSSetSingularitiesFunction(nep,ComputeSingularities,NULL);CHKERRQ(ierr);
  ierr = NEPGetRG(nep,&rg);CHKERRQ(ierr);
  ierr = RGSetType(rg,RGINTERVAL);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = RGIntervalSetEndpoints(rg,0.01,16.0,-0.001,0.001);CHKERRQ(ierr);
#else
  ierr = RGIntervalSetEndpoints(rg,0.01,16.0,0,0);CHKERRQ(ierr);
#endif
  ierr = NEPSetTarget(nep,1.0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Define the nonlinear problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  if (split) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create matrices for the split form 
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = MatCreate(PETSC_COMM_WORLD,&A[0]);CHKERRQ(ierr);
    ierr = MatSetSizes(A[0],PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A[0]);CHKERRQ(ierr);
    ierr = MatSetUp(A[0]);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A[0],&Istart,&Iend);CHKERRQ(ierr);
    for (i=Istart;i<Iend;i++) {
      if (i>0) { ierr = MatSetValue(A[0],i,i-1,1.0,INSERT_VALUES);CHKERRQ(ierr); }
      if (i<n-1) { ierr = MatSetValue(A[0],i,i+1,1.0,INSERT_VALUES);CHKERRQ(ierr); }
      ierr = MatSetValue(A[0],i,i,-2.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD,&A[1]);CHKERRQ(ierr);
    ierr = MatSetSizes(A[1],PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A[1]);CHKERRQ(ierr);
    ierr = MatSetUp(A[1]);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatShift(A[1],1.0);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Define funcions for the split form 
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    ierr = FNCreate(PETSC_COMM_WORLD,&f[0]);CHKERRQ(ierr);
    ierr = FNSetType(f[0],FNRATIONAL);CHKERRQ(ierr);
    coeffs = 1.0;
    ierr = FNRationalSetNumerator(f[0],1,&coeffs);CHKERRQ(ierr);
    ierr = FNCreate(PETSC_COMM_WORLD,&f[1]);CHKERRQ(ierr);
    ierr = FNSetType(f[1],FNSQRT);CHKERRQ(ierr);
    ierr = NEPSetSplitOperator(nep,2,A,f,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);

  } else {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create matrix data structure; set Function evaluation routine
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = MatCreate(PETSC_COMM_WORLD,&F);CHKERRQ(ierr);
    ierr = MatSetSizes(F,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(F);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(F,3,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(F,3,NULL,1,NULL);CHKERRQ(ierr);
    ierr = MatSetUp(F);CHKERRQ(ierr);

    /*
       Set Function matrix data structure and default Function evaluation
       routine
    */
    ierr = NEPSetFunction(nep,F,F,FormFunction,NULL);CHKERRQ(ierr);
  }

  /*
     Set solver parameters at runtime
  */
  ierr = NEPSetFromOptions(nep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = NEPSolve(nep);CHKERRQ(ierr);
  ierr = NEPGetIterationNumber(nep,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of NEP iterations = %D\n\n",its);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = NEPGetType(nep,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type);CHKERRQ(ierr);
  ierr = NEPGetDimensions(nep,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = NEPReasonView(nep,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = NEPErrorView(nep,NEP_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = NEPDestroy(&nep);CHKERRQ(ierr);
  if (split) {
    ierr = MatDestroy(&A[0]);CHKERRQ(ierr);
    ierr = MatDestroy(&A[1]);CHKERRQ(ierr);
    ierr = FNDestroy(&f[0]);CHKERRQ(ierr);
    ierr = FNDestroy(&f[1]);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy(&F);CHKERRQ(ierr);
  }
  ierr = SlepcFinalize();
  return 0;
}


/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
/*
   FormFunction - Computes Function matrix  T(lambda)
*/
PetscErrorCode FormFunction(NEP nep,PetscScalar lambda,Mat fun,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,n,col[3],Istart,Iend;
  PetscBool      FirstBlock=PETSC_FALSE,LastBlock=PETSC_FALSE;
  PetscScalar    value[3],t;

  PetscFunctionBeginUser;
  /*
     Compute Function entries and insert into matrix
  */
  t = PetscSqrtScalar(lambda);
  ierr = MatGetSize(fun,&n,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(fun,&Istart,&Iend);CHKERRQ(ierr);
  if (Istart==0) FirstBlock=PETSC_TRUE;
  if (Iend==n) LastBlock=PETSC_TRUE;
  value[0]=1.0; value[1]=t-2.0; value[2]=1.0;
  for (i=(FirstBlock? Istart+1: Istart); i<(LastBlock? Iend-1: Iend); i++) {
    col[0]=i-1; col[1]=i; col[2]=i+1;
    ierr = MatSetValues(fun,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (LastBlock) {
    i=n-1; col[0]=n-2; col[1]=n-1;
    ierr = MatSetValues(fun,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (FirstBlock) {
    i=0; col[0]=0; col[1]=1; value[0]=t-2.0; value[1]=1.0;
    ierr = MatSetValues(fun,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Assemble matrix
  */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (fun != B) {
    ierr = MatAssemblyBegin(fun,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(fun,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeSingularities"
/*
   ComputeSingularities - Computes maxnp points (at most) in the complex plane where
   the function T(.) is not analytic.

   In this case, we discretize the singularity region (-inf,0)~(-10e+6,-10e-6) 
*/
PetscErrorCode ComputeSingularities(NEP nep,PetscInt *maxnp,PetscScalar *xi,void *pt)
{
  PetscReal h;
  PetscInt  i;

  PetscFunctionBeginUser;
  h = 12.0/(*maxnp-1);
  xi[0] = -1e-6; xi[*maxnp-1] = -1e+6;
  for (i=1;i<*maxnp-1;i++) xi[i] = -PetscPowReal(10,-6+h*i);
  PetscFunctionReturn(0);
}
