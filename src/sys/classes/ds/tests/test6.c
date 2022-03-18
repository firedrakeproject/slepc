/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test DSGHIEP with compact storage.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  SlepcSC        sc;
  PetscReal      *T,*s,re,im;
  PetscScalar    *eigr,*eigi;
  PetscInt       i,n=10,l=2,k=5,ld;
  PetscViewer    viewer;
  PetscBool      verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type GHIEP with compact storage - dimension %" PetscInt_FMT ".\n",n));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-l",&l,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCheck(l<=n && k<=n && l<=k,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Wrong value of dimensions");
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));

  /* Create DS object */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSGHIEP));
  CHKERRQ(DSSetFromOptions(ds));
  ld = n+2;  /* test leading dimension larger than n */
  CHKERRQ(DSAllocate(ds,ld));
  CHKERRQ(DSSetDimensions(ds,n,l,k));
  CHKERRQ(DSSetCompact(ds,PETSC_TRUE));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(DSView(ds,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Fill arrow-tridiagonal matrix */
  CHKERRQ(DSGetArrayReal(ds,DS_MAT_T,&T));
  CHKERRQ(DSGetArrayReal(ds,DS_MAT_D,&s));
  for (i=0;i<n;i++) T[i] = (PetscReal)(i+1);
  for (i=k;i<n-1;i++) T[i+ld] = 1.0;
  for (i=l;i<k;i++) T[i+2*ld] = 1.0;
  T[2*ld+l+1] = -7; T[ld+k+1] = -7;
  /* Signature matrix */
  for (i=0;i<n;i++) s[i] = 1.0;
  s[l+1] = -1.0;
  s[k+1] = -1.0;
  CHKERRQ(DSRestoreArrayReal(ds,DS_MAT_T,&T));
  CHKERRQ(DSRestoreArrayReal(ds,DS_MAT_D,&s));
  if (l==0 && k==0) {
    CHKERRQ(DSSetState(ds,DS_STATE_INTERMEDIATE));
  } else {
    CHKERRQ(DSSetState(ds,DS_STATE_RAW));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
  CHKERRQ(DSView(ds,viewer));

  /* Solve */
  CHKERRQ(PetscCalloc2(n,&eigr,n,&eigi));
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSolve(ds,eigr,eigi));
  CHKERRQ(DSSort(ds,eigr,eigi,NULL,NULL,NULL));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Print eigenvalues */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<n;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(eigr[i]);
    im = PetscImaginaryPart(eigr[i]);
#else
    re = eigr[i];
    im = eigi[i];
#endif
    if (PetscAbs(im)<1e-10) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f%+.5fi\n",(double)re,(double)im));
    }
  }
  CHKERRQ(PetscFree2(eigr,eigi));
  CHKERRQ(DSDestroy(&ds));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      requires: !single
      args: -ds_method {{0 1 2}}
      filter: grep -v "solving the problem"

   test:
      suffix: 2
      args: -n 5 -k 4 -l 4 -ds_method {{0 1 2}}
      filter: grep -v "solving the problem"

TEST*/
