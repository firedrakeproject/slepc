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

static char help[] = "Test the NLEIGS solver with shell matrices.\n\n"
  "This is based on ex27 (split form only).\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n";

/*
   Solve T(lambda)x=0 using NLEIGS solver
      with T(lambda) = -D+sqrt(lambda)*I
      where D is the Laplacian operator in 1 dimension
      and with the interpolation interval [.01,16]   
*/

#include <slepcnep.h>

/* User-defined routines */
PetscErrorCode ComputeSingularities(NEP,PetscInt*,PetscScalar*,void*);
PetscErrorCode MatMult_A0(Mat,Vec,Vec);
PetscErrorCode MatGetDiagonal_A0(Mat,Vec);
PetscErrorCode MatDuplicate_A0(Mat,MatDuplicateOption,Mat*);
PetscErrorCode MatMult_A1(Mat,Vec,Vec);
PetscErrorCode MatGetDiagonal_A1(Mat,Vec);
PetscErrorCode MatDuplicate_A1(Mat,MatDuplicateOption,Mat*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  NEP            nep;
  KSP            *ksp;
  PC             pc;
  Mat            A[2];             
  NEPType        type;
  PetscInt       n=100,nev;
  PetscReal      tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscErrorCode ierr;
  RG             rg;
  FN             f[2];
  PetscMPIInt    size;
  PetscBool      terse;
  PetscScalar    coeffs;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSquare root eigenproblem, n=%D\n\n",n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create NEP context, configure NLEIGS with appropriate options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = NEPCreate(PETSC_COMM_WORLD,&nep);CHKERRQ(ierr);
  ierr = NEPSetType(nep,NEPNLEIGS);CHKERRQ(ierr);
  ierr = NEPNLEIGSSetSingularitiesFunction(nep,ComputeSingularities,NULL);CHKERRQ(ierr);
  ierr = NEPGetRG(nep,&rg);CHKERRQ(ierr);
  ierr = RGSetType(rg,RGINTERVAL);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = RGIntervalSetEndpoints(rg,0.01,16.0,-0.001,0.001);CHKERRQ(ierr);
#else
  ierr = RGIntervalSetEndpoints(rg,0.01,16.0,0,0);CHKERRQ(ierr);
#endif
  ierr = NEPSetTarget(nep,1.1);CHKERRQ(ierr);
  ierr = NEPNLEIGSGetKSPs(nep,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp[0],KSPBCGS);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp[0],&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp[0],tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Define the nonlinear problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  /* Create matrix A0 (tridiagonal) */
  ierr = MatCreateShell(PETSC_COMM_WORLD,n,n,n,n,NULL,&A[0]);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A[0],MATOP_MULT,(void(*)())MatMult_A0);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A[0],MATOP_MULT_TRANSPOSE,(void(*)())MatMult_A0);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A[0],MATOP_GET_DIAGONAL,(void(*)())MatGetDiagonal_A0);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A[0],MATOP_DUPLICATE,(void(*)())MatDuplicate_A0);CHKERRQ(ierr);

  /* Create matrix A0 (identity) */
  ierr = MatCreateShell(PETSC_COMM_WORLD,n,n,n,n,NULL,&A[1]);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A[1],MATOP_MULT,(void(*)())MatMult_A1);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A[1],MATOP_MULT_TRANSPOSE,(void(*)())MatMult_A1);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A[1],MATOP_GET_DIAGONAL,(void(*)())MatGetDiagonal_A1);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A[1],MATOP_DUPLICATE,(void(*)())MatDuplicate_A1);CHKERRQ(ierr);

  /* Define funcions for the split form */
  ierr = FNCreate(PETSC_COMM_WORLD,&f[0]);CHKERRQ(ierr);
  ierr = FNSetType(f[0],FNRATIONAL);CHKERRQ(ierr);
  coeffs = 1.0;
  ierr = FNRationalSetNumerator(f[0],1,&coeffs);CHKERRQ(ierr);
  ierr = FNCreate(PETSC_COMM_WORLD,&f[1]);CHKERRQ(ierr);
  ierr = FNSetType(f[1],FNSQRT);CHKERRQ(ierr);
  ierr = NEPSetSplitOperator(nep,2,A,f,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);

  /* Set solver parameters at runtime */
  ierr = NEPSetFromOptions(nep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = NEPSolve(nep);CHKERRQ(ierr);
  ierr = NEPGetType(nep,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type);CHKERRQ(ierr);
  ierr = NEPGetDimensions(nep,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
  if (terse) {
    ierr = NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = NEPReasonView(nep,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = NEPErrorView(nep,NEP_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = NEPDestroy(&nep);CHKERRQ(ierr);
  ierr = MatDestroy(&A[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&A[1]);CHKERRQ(ierr);
  ierr = FNDestroy(&f[0]);CHKERRQ(ierr);
  ierr = FNDestroy(&f[1]);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
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

/* -------------------------------- A0 ----------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "MatMult_A0"
PetscErrorCode MatMult_A0(Mat A,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PetscInt          i,n;
  const PetscScalar *px;
  PetscScalar       *py;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  py[0] = -2.0*px[0]+px[1];
  for (i=1;i<n-1;i++) py[i] = px[i-1]-2.0*px[i]+px[i+1];
  py[n-1] = px[n-2]-2.0*px[n-1];
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_A0"
PetscErrorCode MatGetDiagonal_A0(Mat A,Vec diag)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(diag,-2.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_A0"
PetscErrorCode MatDuplicate_A0(Mat A,MatDuplicateOption op,Mat *B)
{
  PetscInt       n;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&n,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatCreateShell(comm,n,n,n,n,NULL,B);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_MULT,(void(*)())MatMult_A0);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_MULT_TRANSPOSE,(void(*)())MatMult_A0);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_GET_DIAGONAL,(void(*)())MatGetDiagonal_A0);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_DUPLICATE,(void(*)())MatDuplicate_A0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------- A1 ----------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "MatMult_A1"
PetscErrorCode MatMult_A1(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_A1"
PetscErrorCode MatGetDiagonal_A1(Mat A,Vec diag)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(diag,1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_A1"
PetscErrorCode MatDuplicate_A1(Mat A,MatDuplicateOption op,Mat *B)
{
  PetscInt       n;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&n,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatCreateShell(comm,n,n,n,n,NULL,B);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_MULT,(void(*)())MatMult_A1);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_MULT_TRANSPOSE,(void(*)())MatMult_A1);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_GET_DIAGONAL,(void(*)())MatGetDiagonal_A1);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_DUPLICATE,(void(*)())MatDuplicate_A1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

