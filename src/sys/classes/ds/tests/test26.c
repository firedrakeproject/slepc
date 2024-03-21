/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSHSVD with compact storage.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  DS             ds;
  SlepcSC        sc;
  PetscReal      *T,*D,sigma;
  PetscScalar    *U,*w,d;
  PetscInt       i,n=10,m,l=2,k=5,p=1,ld;
  PetscViewer    viewer;
  PetscBool      verbose,extrarow,reorthog,flg;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  m = n;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type HSVD with compact storage - dimension %" PetscInt_FMT "x%" PetscInt_FMT ".\n",n,m));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  PetscCheck(l<=n && k<=n && l<=k && p>=0 && p<=n-l,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Wrong value of dimensions");
  PetscCall(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-extrarow",&extrarow));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-reorthog",&reorthog));

  /* Create DS object */
  PetscCall(DSCreate(PETSC_COMM_WORLD,&ds));
  PetscCall(DSSetType(ds,DSHSVD));
  PetscCall(DSSetFromOptions(ds));
  ld = n+2;  /* test leading dimension larger than n */
  PetscCall(DSAllocate(ds,ld));
  PetscCall(DSSetDimensions(ds,n,l,k));
  PetscCall(DSHSVDSetDimensions(ds,m));
  PetscCall(DSSetCompact(ds,PETSC_TRUE));
  PetscCall(DSSetExtraRow(ds,extrarow));
  PetscCall(DSHSVDSetReorthogonalize(ds,reorthog));
  PetscCall(DSHSVDGetReorthogonalize(ds,&flg));
  if (flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"reorthogonalizing\n"));

  /* Set up viewer */
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(DSView(ds,viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill upper arrow-bidiagonal matrix and signature matrix */
  PetscCall(DSGetArrayReal(ds,DS_MAT_T,&T));
  for (i=0;i<n;i++) T[i] = (PetscReal)(i+1);
  for (i=l;i<n-1;i++) T[i+ld] = 1.0;
  if (extrarow) T[n-1+ld] = 1.0;
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&T));
  PetscCall(DSGetArrayReal(ds,DS_MAT_D,&D));
  for (i=0;i<n;i++) D[i] = (i>=l && i<l+p)? -1.0: 1.0;
  PetscCall(DSRestoreArrayReal(ds,DS_MAT_D,&D));
  if (l==0 && k==0) PetscCall(DSSetState(ds,DS_STATE_INTERMEDIATE));
  else PetscCall(DSSetState(ds,DS_STATE_RAW));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
  PetscCall(DSView(ds,viewer));

  /* Solve */
  PetscCall(PetscMalloc1(n,&w));
  PetscCall(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestReal;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  PetscCall(DSSolve(ds,w,NULL));
  PetscCall(DSSort(ds,w,NULL,NULL,NULL,NULL));
  if (extrarow) PetscCall(DSUpdateExtraRow(ds));
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
    PetscCall(DSGetArrayReal(ds,DS_MAT_D,&D));
    d = 0.0;
    for (i=l;i<n;i++) d += T[i+ld]-D[i]*U[n-1+i*ld];
    if (PetscAbsScalar(d)>10*PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: there is a mismatch in the extra row of %g\n",(double)PetscAbsScalar(d)));
    PetscCall(DSRestoreArrayReal(ds,DS_MAT_T,&T));
    PetscCall(DSRestoreArray(ds,DS_MAT_U,&U));
    PetscCall(DSRestoreArrayReal(ds,DS_MAT_D,&D));
  }
  PetscCall(PetscFree(w));
  PetscCall(DSDestroy(&ds));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      requires: !single

   test:
      args: -l 0 -k 0
      suffix: 2
      requires: !single

   test:
      args: -extrarow -reorthog {{0 1}}
      suffix: 3
      requires: !single
      filter: grep -v reorthogonalizing

TEST*/
