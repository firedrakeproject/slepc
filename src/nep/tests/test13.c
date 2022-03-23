/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test the NEPProjectOperator() operator.\n\n"
  "This is based on ex22.\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions.\n"
  "  -tau <tau>, where <tau> is the delay parameter.\n";

/*
   Solve parabolic partial differential equation with time delay tau

            u_t = u_xx + a*u(t) + b*u(t-tau)
            u(0,t) = u(pi,t) = 0

   with a = 20 and b(x) = -4.1+x*(1-exp(x-pi)).

   Discretization leads to a DDE of dimension n

            -u' = A*u(t) + B*u(t-tau)

   which results in the nonlinear eigenproblem

            (-lambda*I + A + exp(-tau*lambda)*B)*u = 0
*/

#include <slepcnep.h>

int main(int argc,char **argv)
{
  NEP            nep;
  Mat            Id,A,B,mats[3];
  FN             f1,f2,f3,funs[3];
  BV             V;
  DS             ds;
  Vec            v;
  PetscScalar    coeffs[2],b,*M;
  PetscInt       n=32,Istart,Iend,i,j,k,nc;
  PetscReal      tau=0.001,h,a=20,xi;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Delay Eigenproblem, n=%" PetscInt_FMT ", tau=%g\n",n,(double)tau));
  h = PETSC_PI/(PetscReal)(n+1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear eigensolver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));

  /* Identity matrix */
  CHKERRQ(MatCreateConstantDiagonal(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,1.0,&Id));
  CHKERRQ(MatSetOption(Id,MAT_HERMITIAN,PETSC_TRUE));

  /* A = 1/h^2*tridiag(1,-2,1) + a*I */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(A,i,i-1,1.0/(h*h),INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,1.0/(h*h),INSERT_VALUES));
    CHKERRQ(MatSetValue(A,i,i,-2.0/(h*h)+a,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));

  /* B = diag(b(xi)) */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    xi = (i+1)*h;
    b = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
    CHKERRQ(MatSetValues(B,1,&i,1,&i,&b,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOption(B,MAT_HERMITIAN,PETSC_TRUE));

  /* Functions: f1=-lambda, f2=1.0, f3=exp(-tau*lambda) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f1));
  CHKERRQ(FNSetType(f1,FNRATIONAL));
  coeffs[0] = -1.0; coeffs[1] = 0.0;
  CHKERRQ(FNRationalSetNumerator(f1,2,coeffs));

  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f2));
  CHKERRQ(FNSetType(f2,FNRATIONAL));
  coeffs[0] = 1.0;
  CHKERRQ(FNRationalSetNumerator(f2,1,coeffs));

  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f3));
  CHKERRQ(FNSetType(f3,FNEXP));
  CHKERRQ(FNSetScale(f3,-tau,1.0));

  /* Set the split operator */
  mats[0] = A;  funs[0] = f2;
  mats[1] = Id; funs[1] = f1;
  mats[2] = B;  funs[2] = f3;
  CHKERRQ(NEPSetSplitOperator(nep,3,mats,funs,SUBSET_NONZERO_PATTERN));
  CHKERRQ(NEPSetType(nep,NEPNARNOLDI));
  CHKERRQ(NEPSetFromOptions(nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Project the NEP
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPSetUp(nep));
  CHKERRQ(NEPGetBV(nep,&V));
  CHKERRQ(BVGetSizes(V,NULL,NULL,&nc));
  for (i=0;i<nc;i++) {
    CHKERRQ(BVGetColumn(V,i,&v));
    CHKERRQ(VecSetValue(v,i,1.0,INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(v));
    CHKERRQ(VecAssemblyEnd(v));
    CHKERRQ(BVRestoreColumn(V,i,&v));
  }
  CHKERRQ(NEPGetDS(nep,&ds));
  CHKERRQ(DSSetType(ds,DSNEP));
  CHKERRQ(DSNEPSetFN(ds,3,funs));
  CHKERRQ(DSAllocate(ds,nc));
  CHKERRQ(DSSetDimensions(ds,nc,0,0));
  CHKERRQ(NEPProjectOperator(nep,0,nc));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Display projected matrices and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  for (k=0;k<3;k++) {
    CHKERRQ(DSGetArray(ds,DSMatExtra[k],&M));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nMatrix E%" PetscInt_FMT " = \n",k));
    for (i=0;i<nc;i++) {
      for (j=0;j<nc;j++) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  %.5g",(double)PetscRealPart(M[i+j*nc])));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    }
    CHKERRQ(DSRestoreArray(ds,DSMatExtra[k],&M));
  }

  CHKERRQ(NEPDestroy(&nep));
  CHKERRQ(MatDestroy(&Id));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(FNDestroy(&f1));
  CHKERRQ(FNDestroy(&f2));
  CHKERRQ(FNDestroy(&f3));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -nep_ncv 5

TEST*/
