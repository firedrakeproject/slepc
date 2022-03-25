/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSGSVD with compact storage.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  DS             ds;
  Mat            X;
  Vec            x0;
  SlepcSC        sc;
  PetscReal      *T,*D,sigma,rnorm,aux;
  PetscScalar    *U,*V,*w,d;
  PetscInt       i,n=10,l=0,k=0,ld;
  PetscViewer    viewer;
  PetscBool      verbose,test_dsview,extrarow;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type GSVD with compact storage - dimension %" PetscInt_FMT "x%" PetscInt_FMT ".\n",n,n));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCheck(l<=n && k<=n && l<=k,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Wrong value of dimensions");
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-test_dsview",&test_dsview));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSGSVD));
  CHKERRQ(DSSetFromOptions(ds));
  ld = n+2;  /* test leading dimension larger than n */
  CHKERRQ(DSAllocate(ds,ld));
  CHKERRQ(DSSetDimensions(ds,n,l,k));
  CHKERRQ(DSGSVDSetDimensions(ds,n,PETSC_DECIDE));
  CHKERRQ(DSSetCompact(ds,PETSC_TRUE));
  CHKERRQ(DSSetExtraRow(ds,extrarow));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(DSView(ds,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));

  if (test_dsview) {
    /* Fill A and B with dummy values to test DSView */
    CHKERRQ(DSGetArrayReal(ds,DS_MAT_T,&T));
    CHKERRQ(DSGetArrayReal(ds,DS_MAT_D,&D));
    for (i=0;i<n;i++) { T[i] = i+1; D[i] = -i-1; }
    for (i=0;i<n-1;i++) { T[i+ld] = -1.0; T[i+2*ld] = 1.0; }
    CHKERRQ(DSRestoreArrayReal(ds,DS_MAT_T,&T));
    CHKERRQ(DSRestoreArrayReal(ds,DS_MAT_D,&D));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Fill A and B with upper arrow-bidiagonal matrices
     verifying that [A;B] has orthonormal columns */
  CHKERRQ(DSGetArrayReal(ds,DS_MAT_T,&T));
  CHKERRQ(DSGetArrayReal(ds,DS_MAT_D,&D));
  for (i=0;i<n;i++) T[i] = (PetscReal)(i+1)/(n+1); /* diagonal of matrix A */
  for (i=0;i<k;i++) D[i] = PetscSqrtReal(1.0-T[i]*T[i]);
  for (i=l;i<k;i++) {
    T[i+ld] = PetscSqrtReal((1.0-T[k]*T[k])/(1.0+T[i]*T[i]/(D[i]*D[i])))*0.5*(1.0/k); /* upper diagonal of matrix A */
    T[i+2*ld] = -T[i+ld]*T[i]/D[i]; /* upper diagonal of matrix B */
  }
  aux = 1.0-T[k]*T[k];
  for (i=l;i<k;i++) aux -= T[i+ld]*T[i+ld]+T[i+2*ld]*T[i+2*ld];
  D[k] = PetscSqrtReal(aux);
  for (i=k;i<n-1;i++) {
    T[i+ld] = PetscSqrtReal((1.0-T[i+1]*T[i+1])/(1.0+T[i]*T[i]/(D[i]*D[i])))*0.5; /* upper diagonal of matrix A */
    T[i+2*ld] = -T[i+ld]*T[i]/D[i]; /* upper diagonal of matrix B */
    D[i+1] = PetscSqrtReal(1.0-T[i+1]*T[i+1]-T[ld+i]*T[ld+i]-T[2*ld+i]*T[2*ld+i]); /* diagonal of matrix B */
  }
  if (extrarow) { T[n-1+ld]=-1.0; T[n-1+2*ld]=1.0; }
  /* Fill locked eigenvalues */
  CHKERRQ(PetscMalloc1(n,&w));
  for (i=0;i<l;i++) w[i] = T[i]/D[i];
  CHKERRQ(DSRestoreArrayReal(ds,DS_MAT_T,&T));
  CHKERRQ(DSRestoreArrayReal(ds,DS_MAT_D,&D));
  if (l==0 && k==0) CHKERRQ(DSSetState(ds,DS_STATE_INTERMEDIATE));
  else CHKERRQ(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Solve */
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSolve(ds,w,NULL));
  CHKERRQ(DSSort(ds,w,NULL,NULL,NULL,NULL));
  if (extrarow) CHKERRQ(DSUpdateExtraRow(ds));
  CHKERRQ(DSSynchronize(ds,w,NULL));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Print singular values */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed singular values =\n"));
  for (i=0;i<n;i++) {
    sigma = PetscRealPart(w[i]);
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)sigma));
  }

  if (extrarow) {
    /* Check that extra row is correct */
    CHKERRQ(DSGetArrayReal(ds,DS_MAT_T,&T));
    CHKERRQ(DSGetArray(ds,DS_MAT_U,&U));
    CHKERRQ(DSGetArray(ds,DS_MAT_V,&V));
    d = 0.0;
    for (i=0;i<n;i++) d += T[i+ld]+U[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in A's extra row of %g\n",(double)PetscAbsScalar(d)));
    d = 0.0;
    for (i=0;i<n;i++) d += T[i+2*ld]-V[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in B's extra row of %g\n",(double)PetscAbsScalar(d)));
    CHKERRQ(DSRestoreArrayReal(ds,DS_MAT_T,&T));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_U,&U));
    CHKERRQ(DSRestoreArray(ds,DS_MAT_V,&V));
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
      requires: double
      test:
         suffix: 1
         args: -test_dsview
      test:
         suffix: 2
         args: -l 1 -k 4
      test:
         suffix: 2_extrarow
         filter: sed -e "s/extrarow//"
         args: -l 1 -k 4 -extrarow
         output_file: output/test22_2.out

TEST*/
