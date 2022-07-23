/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSGSVD with compact storage and rectangular matrix A.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  DS             ds;
  Mat            X;
  Vec            x0;
  SlepcSC        sc;
  PetscReal      *T,*D,sigma,rnorm,aux,cond;
  PetscScalar    *U,*V,*w,d;
  PetscInt       i,n=10,m,l=0,k=0,ld;
  PetscViewer    viewer;
  PetscBool      verbose,extrarow;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type GSVD with compact storage - dimension %" PetscInt_FMT "x%" PetscInt_FMT ".\n",n+1,n));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCheck(l<=n && k<=n && l<=k,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Wrong value of dimensions");
  PetscCall(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow));
  m = n+1;

  /* Create DS object */
  PetscCall(DSCreate(PETSC_COMM_WORLD,&ds));
  PetscCall(DSSetType(ds,DSGSVD));
  PetscCall(DSSetFromOptions(ds));
  ld = n+2;  /* test leading dimension larger than n */
  PetscCall(DSAllocate(ds,ld));
  PetscCall(DSSetDimensions(ds,m,l,k));
  PetscCall(DSGSVDSetDimensions(ds,n,n));
  PetscCall(DSSetCompact(ds,PETSC_TRUE));
  PetscCall(DSSetExtraRow(ds,extrarow));

  /* Set up viewer */
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(DSView(ds,viewer));
  PetscCall(PetscViewerPopFormat(viewer));

  /* Fill A and B with lower/upper arrow-bidiagonal matrices
     verifying that [A;B] has orthonormal columns */
  PetscCall(DSGetArrayReal(ds,DS_MAT_T,&T));
  PetscCall(DSGetArrayReal(ds,DS_MAT_D,&D));
  for (i=0;i<n;i++) T[i] = (PetscReal)(i+1)/(n+1); /* diagonal of matrix A */
  for (i=0;i<k;i++) D[i] = PetscSqrtReal(1.0-T[i]*T[i]);
  for (i=l;i<k;i++) {
    T[i+ld] = PetscSqrtReal((1.0-T[k]*T[k])/(1.0+T[i]*T[i]/(D[i]*D[i])))*0.5*(1.0/k); /* upper diagonal of matrix A */
    T[i+2*ld] = -T[i+ld]*T[i]/D[i]; /* upper diagonal of matrix B */
  }
  aux = 1.0-T[k]*T[k];
  for (i=l;i<k;i++) aux -= T[i+ld]*T[i+ld]+T[i+2*ld]*T[i+2*ld];
  T[k+ld] = PetscSqrtReal((1.0-aux)*.1);
  aux -= T[k+ld]*T[k+ld];
  D[k] = PetscSqrtReal(aux);
  for (i=k+1;i<n;i++) {
    T[i-1+2*ld] = -T[i-1+ld]*T[i]/D[i-1]; /* upper diagonal of matrix B */
    aux = 1.0-T[i]*T[i]-T[2*ld+i-1]*T[2*ld+i-1];
    T[i+ld] = PetscSqrtReal(aux)*.1; /* upper diagonal of matrix A */
    D[i] = PetscSqrtReal(aux-T[i+ld]*T[i+ld]);
  }
  if (extrarow) { T[n]=-1.0; T[n-1+2*ld]=1.0; }
  /* Fill locked eigenvalues */
  PetscCall(PetscMalloc1(n,&w));
  for (i=0;i<l;i++) w[i] = T[i]/D[i];
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&T));
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_D,&D));
  if (l==0 && k==0) PetscCall(DSSetState(ds,DS_STATE_INTERMEDIATE));
  else PetscCall(DSSetState(ds,DS_STATE_RAW));
  if (verbose) {
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }

  /* Condition number */
  PetscCall(DSCond(ds,&cond));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Condition number = %.3f\n",(double)cond));

  /* Solve */
  PetscCall(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  PetscCall(DSSolve(ds,w,NULL));
  PetscCall(DSSort(ds,w,NULL,NULL,NULL,NULL));
  if (extrarow) PetscCall(DSUpdateExtraRow(ds));
  PetscCall(DSSynchronize(ds,w,NULL));
  if (verbose) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    PetscCall(DSView(ds,viewer));
  }

  /* Print singular values */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Computed singular values =\n"));
  for (i=0;i<n;i++) {
    sigma = PetscRealPart(w[i]);
    PetscCall(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)sigma));
  }

  if (extrarow) {
    /* Check that extra row is correct */
    PetscCall(DSGetArrayReal(ds,DS_MAT_T,&T));
    PetscCall(DSGetArray(ds,DS_MAT_U,&U));
    PetscCall(DSGetArray(ds,DS_MAT_V,&V));
    d = 0.0;
    for (i=0;i<n;i++) d += T[i+ld]+U[n+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in A's extra row of %g\n",(double)PetscAbsScalar(d)));
    d = 0.0;
    for (i=0;i<n;i++) d += T[i+2*ld]-V[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in B's extra row of %g\n",(double)PetscAbsScalar(d)));
    PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&T));
    PetscCall(DSRestoreArray(ds,DS_MAT_U,&U));
    PetscCall(DSRestoreArray(ds,DS_MAT_V,&V));
  }

  /* Singular vectors */
  PetscCall(DSVectors(ds,DS_MAT_X,NULL,NULL));  /* all singular vectors */
  PetscCall(DSGetMat(ds,DS_MAT_X,&X));
  PetscCall(MatCreateVecs(X,NULL,&x0));
  PetscCall(MatGetColumnVector(X,x0,0));
  PetscCall(VecNorm(x0,NORM_2,&rnorm));
  PetscCall(DSRestoreMat(ds,DS_MAT_X,&X));
  PetscCall(VecDestroy(&x0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of 1st X vector = %.3f\n",(double)rnorm));

  PetscCall(DSGetMat(ds,DS_MAT_U,&X));
  PetscCall(MatCreateVecs(X,NULL,&x0));
  PetscCall(MatGetColumnVector(X,x0,0));
  PetscCall(VecNorm(x0,NORM_2,&rnorm));
  PetscCall(DSRestoreMat(ds,DS_MAT_U,&X));
  PetscCall(VecDestroy(&x0));
  if (PetscAbs(rnorm-1.0)>10*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: the 1st U vector has norm %g\n",(double)rnorm));

  PetscCall(DSGetMat(ds,DS_MAT_V,&X));
  PetscCall(MatCreateVecs(X,NULL,&x0));
  PetscCall(MatGetColumnVector(X,x0,0));
  PetscCall(VecNorm(x0,NORM_2,&rnorm));
  PetscCall(DSRestoreMat(ds,DS_MAT_V,&X));
  PetscCall(VecDestroy(&x0));
  if (PetscAbs(rnorm-1.0)>10*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: the 1st V vector has norm %g\n",(double)rnorm));

  PetscCall(PetscFree(w));
  PetscCall(DSDestroy(&ds));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      requires: double
      output_file: output/test24_1.out
      test:
         suffix: 1
         args: -l 1 -k 4
      test:
         suffix: 1_extrarow
         filter: sed -e "s/extrarow//"
         args: -l 1 -k 4 -extrarow

TEST*/
