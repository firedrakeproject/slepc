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
  DS             ds;
  SlepcSC        sc;
  Mat            X;
  Vec            x0;
  PetscReal      sigma,rnorm;
  PetscScalar    *A,*B,*w;
  PetscInt       i,j,k,n=15,m=10,p=10,m1,p1,ld;
  PetscViewer    viewer;
  PetscBool      verbose;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type GSVD - dimension (%" PetscInt_FMT "+%" PetscInt_FMT ")x%" PetscInt_FMT ".\n",n,p,m));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSGSVD));
  CHKERRQ(DSSetFromOptions(ds));
  ld   = PetscMax(PetscMax(p,m),n)+2;  /* test leading dimension larger than n */
  CHKERRQ(DSAllocate(ds,ld));
  CHKERRQ(DSSetDimensions(ds,n,0,0));
  CHKERRQ(DSGSVDSetDimensions(ds,m,p));
  CHKERRQ(DSGSVDGetDimensions(ds,&m1,&p1));
  PetscCheck(m1==m && p1==p,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Inconsistent dimension values");

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(DSView(ds,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));

  k = PetscMin(n,m);
  /* Fill A with a rectangular Toeplitz matrix */
  CHKERRQ(DSGetArray(ds,DS_MAT_A,&A));
  for (i=0;i<k;i++) A[i+i*ld]=1.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { if ((i+j)<m) A[i+(i+j)*ld]=(PetscScalar)(j+1); }
  }
  for (j=1;j<n/2;j++) {
    for (i=0;i<n-j;i++) { if ((i+j)<n && i<m) A[(i+j)+i*ld]=-1.0; }
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_A,&A));

  k = PetscMin(p,m);
  /* Fill B with a shifted bidiagonal matrix */
  CHKERRQ(DSGetArray(ds,DS_MAT_B,&B));
  for (i=m-k;i<m;i++) {
    B[i-m+k+i*ld]=2.0-1.0/(PetscScalar)(i+1);
    if (i) B[i-1-m+k+i*ld]=1.0;
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_B,&B));

  CHKERRQ(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
  }
  CHKERRQ(DSView(ds,viewer));

  /* Solve */
  CHKERRQ(PetscMalloc1(m,&w));
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSolve(ds,w,NULL));
  CHKERRQ(DSSort(ds,w,NULL,NULL,NULL,NULL));
  CHKERRQ(DSSynchronize(ds,w,NULL));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }
  /* Print singular values */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed singular values =\n"));
  CHKERRQ(DSGetDimensions(ds,NULL,NULL,NULL,&k));
  for (i=0;i<k;i++) {
    sigma = PetscRealPart(w[i]);
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %g\n",(double)sigma));
  }

  /* Singular vectors */
  CHKERRQ(DSVectors(ds,DS_MAT_X,NULL,NULL));  /* all singular vectors */
  CHKERRQ(DSGetMat(ds,DS_MAT_X,&X));
  CHKERRQ(MatCreateVecs(X,NULL,&x0));
  CHKERRQ(MatGetColumnVector(X,x0,0));
  CHKERRQ(VecNorm(x0,NORM_2,&rnorm));
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(VecDestroy(&x0));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st X vector = %.3f\n",(double)rnorm));

  CHKERRQ(DSGetMat(ds,DS_MAT_U,&X));
  CHKERRQ(MatCreateVecs(X,NULL,&x0));
  CHKERRQ(MatGetColumnVector(X,x0,0));
  CHKERRQ(VecNorm(x0,NORM_2,&rnorm));
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(VecDestroy(&x0));
  if (PetscAbs(rnorm-1.0)>10*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: the 1st U vector has norm %g\n",(double)rnorm));

  CHKERRQ(DSGetMat(ds,DS_MAT_V,&X));
  CHKERRQ(MatCreateVecs(X,NULL,&x0));
  CHKERRQ(MatGetColumnVector(X,x0,0));
  CHKERRQ(VecNorm(x0,NORM_2,&rnorm));
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(VecDestroy(&x0));
  if (PetscAbs(rnorm-1.0)>10*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: the 1st V vector has norm %g\n",(double)rnorm));

  CHKERRQ(PetscFree(w));
  CHKERRQ(DSDestroy(&ds));
  CHKERRQ(SlepcFinalize());
  return 0;
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
