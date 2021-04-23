/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSGSVD.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  SlepcSC        sc;
  Mat            X;
  Vec            x0;
  PetscReal      sigma,rnorm;
  PetscScalar    *A,*B,*w;
  PetscInt       i,j,k,n=15,m=10,p=10,m1,p1,ld;
  PetscViewer    viewer;
  PetscBool      verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type GSVD - dimension (%D+%D)x%D.\n",n,p,m);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);

  /* Create DS object */
  ierr = DSCreate(PETSC_COMM_WORLD,&ds);CHKERRQ(ierr);
  ierr = DSSetType(ds,DSGSVD);CHKERRQ(ierr);
  ierr = DSSetFromOptions(ds);CHKERRQ(ierr);
  ld   = PetscMax(PetscMax(p,m),n)+2;  /* test leading dimension larger than n */
  ierr = DSAllocate(ds,ld);CHKERRQ(ierr);
  ierr = DSSetDimensions(ds,n,0,0);CHKERRQ(ierr);
  ierr = DSGSVDSetDimensions(ds,m,p);CHKERRQ(ierr);
  ierr = DSGSVDGetDimensions(ds,&m1,&p1);CHKERRQ(ierr);
  if (m1!=m || p1!=p) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Inconsistent dimension values");

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = DSView(ds,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);

  k = PetscMin(n,m);
  /* Fill A with a rectangular Toeplitz matrix */
  ierr = DSGetArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);
  for (i=0;i<k;i++) A[i+i*ld]=1.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { if ((i+j)<m) A[i+(i+j)*ld]=(PetscScalar)(j+1); }
  }
  for (j=1;j<n/2;j++) {
    for (i=0;i<n-j;i++) { if ((i+j)<n && i<m) A[(i+j)+i*ld]=-1.0; }
  }
  ierr = DSRestoreArray(ds,DS_MAT_A,&A);CHKERRQ(ierr);

  k = PetscMin(p,m);
  /* Fill B with a shifted bidiagonal matrix */
  ierr = DSGetArray(ds,DS_MAT_B,&B);CHKERRQ(ierr);
  for (i=m-k;i<m;i++) {
    B[i-m+k+i*ld]=2.0-1.0/(PetscScalar)(i+1);
    if (i) B[i-1-m+k+i*ld]=1.0;
  }
  ierr = DSRestoreArray(ds,DS_MAT_B,&B);CHKERRQ(ierr);

  ierr = DSSetState(ds,DS_STATE_RAW);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n");CHKERRQ(ierr);
  }
  ierr = DSView(ds,viewer);CHKERRQ(ierr);

  /* Solve */
  ierr = PetscMalloc1(m,&w);CHKERRQ(ierr);
  ierr = DSGetSlepcSC(ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  ierr = DSSolve(ds,w,NULL);CHKERRQ(ierr);
  ierr = DSSort(ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DSSynchronize(ds,w,NULL);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }
  /* Print singular values */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed singular values =\n");CHKERRQ(ierr);
  ierr = DSGetDimensions(ds,NULL,NULL,NULL,&k);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    sigma = PetscRealPart(w[i]);
    ierr = PetscViewerASCIIPrintf(viewer,"  %g\n",(double)sigma);CHKERRQ(ierr);
  }

  /* Singular vectors */
  ierr = DSVectors(ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);  /* all singular vectors */
  ierr = DSGetMat(ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = MatCreateVecs(X,NULL,&x0);CHKERRQ(ierr);
  ierr = MatGetColumnVector(X,x0,0);CHKERRQ(ierr);
  ierr = VecNorm(x0,NORM_2,&rnorm);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&x0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st X vector = %.3f\n",(double)rnorm);CHKERRQ(ierr);

  ierr = DSGetMat(ds,DS_MAT_U,&X);CHKERRQ(ierr);
  ierr = MatCreateVecs(X,NULL,&x0);CHKERRQ(ierr);
  ierr = MatGetColumnVector(X,x0,0);CHKERRQ(ierr);
  ierr = VecNorm(x0,NORM_2,&rnorm);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&x0);CHKERRQ(ierr);
  if (PetscAbs(rnorm-1.0)>10*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: the 1st U vector has norm %g\n",(double)rnorm);CHKERRQ(ierr);
  }

  ierr = DSGetMat(ds,DS_MAT_V,&X);CHKERRQ(ierr);
  ierr = MatCreateVecs(X,NULL,&x0);CHKERRQ(ierr);
  ierr = MatGetColumnVector(X,x0,0);CHKERRQ(ierr);
  ierr = VecNorm(x0,NORM_2,&rnorm);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&x0);CHKERRQ(ierr);
  if (PetscAbs(rnorm-1.0)>10*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: the 1st V vector has norm %g\n",(double)rnorm);CHKERRQ(ierr);
  }

  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      output_file: output/test21_1.out
      requires: !single
      nsize: {{1 2 3}}
      filter: grep -v "parallel operation mode" | grep -v "MPI processes"
      test:
         suffix: 1
         args: -ds_parallel redundant
      test:
         suffix: 2
         args: -ds_parallel synchronized

TEST*/
