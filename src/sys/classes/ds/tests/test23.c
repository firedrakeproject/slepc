/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test interface functions of DSNEP.\n\n";

#include <slepcds.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  FN             f1,f2,f3,funs[3];
  SlepcSC        sc;
  PetscScalar    *Id,*A,*B,*wr,*wi,*X,*W,coeffs[2],auxr,alpha;
  PetscReal      tau=0.001,h,a=20,xi,re,im,nrm,aux;
  PetscInt       i,j,ii,jj,k,n=10,ld,nev,nfun,midx,ip,rits,meth,spls;
  PetscViewer    viewer;
  PetscBool      verbose;
  RG             rg;
  DSMatType      mat[3]={DS_MAT_E0,DS_MAT_E1,DS_MAT_E2};
#if defined(PETSC_USE_COMPLEX)
  PetscBool      flg;
#else
  PetscScalar    auxi;
#endif

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type NEP - dimension %" PetscInt_FMT ", tau=%g.\n",n,(double)tau));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));

  /* Create DS object and set options */
  CHKERRQ(DSCreate(PETSC_COMM_WORLD,&ds));
  CHKERRQ(DSSetType(ds,DSNEP));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(DSSetMethod(ds,1));  /* contour integral */
#endif
  CHKERRQ(DSNEPGetRG(ds,&rg));
  CHKERRQ(RGSetType(rg,RGELLIPSE));
  CHKERRQ(DSNEPSetMinimality(ds,1));
  CHKERRQ(DSNEPSetIntegrationPoints(ds,16));
  CHKERRQ(DSNEPSetRefine(ds,PETSC_DEFAULT,2));
  CHKERRQ(DSNEPSetSamplingSize(ds,25));
  CHKERRQ(DSSetFromOptions(ds));

  /* Print current options */
  CHKERRQ(DSGetMethod(ds,&meth));
#if defined(PETSC_USE_COMPLEX)
  PetscCheck(meth==1,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"This example requires ds_method=1");
  CHKERRQ(RGIsTrivial(rg,&flg));
  PetscCheck(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must at least set the radius of the ellipse");
#endif

  CHKERRQ(DSNEPGetMinimality(ds,&midx));
  CHKERRQ(DSNEPGetIntegrationPoints(ds,&ip));
  CHKERRQ(DSNEPGetRefine(ds,NULL,&rits));
  CHKERRQ(DSNEPGetSamplingSize(ds,&spls));
  if (meth==1) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Contour integral method with %" PetscInt_FMT " integration points, minimality index %" PetscInt_FMT ", and sampling size %" PetscInt_FMT "\n",ip,midx,spls));
    if (rits) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Doing %" PetscInt_FMT " iterations of Newton refinement\n",rits));
    }
  }

  /* Set functions (prior to DSAllocate) */
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

  funs[0] = f1;
  funs[1] = f2;
  funs[2] = f3;
  CHKERRQ(DSNEPSetFN(ds,3,funs));

  /* Set dimensions */
  ld = n;
  CHKERRQ(DSAllocate(ds,ld));
  CHKERRQ(DSSetDimensions(ds,n,0,0));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(PetscViewerPopFormat(viewer));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
  }

  /* Fill matrices */
  CHKERRQ(DSGetArray(ds,DS_MAT_E0,&Id));
  for (i=0;i<n;i++) Id[i+i*ld]=1.0;
  CHKERRQ(DSRestoreArray(ds,DS_MAT_E0,&Id));
  h = PETSC_PI/(PetscReal)(n+1);
  CHKERRQ(DSGetArray(ds,DS_MAT_E1,&A));
  for (i=0;i<n;i++) A[i+i*ld]=-2.0/(h*h)+a;
  for (i=1;i<n;i++) {
    A[i+(i-1)*ld]=1.0/(h*h);
    A[(i-1)+i*ld]=1.0/(h*h);
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_E1,&A));
  CHKERRQ(DSGetArray(ds,DS_MAT_E2,&B));
  for (i=0;i<n;i++) {
    xi = (i+1)*h;
    B[i+i*ld] = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
  }
  CHKERRQ(DSRestoreArray(ds,DS_MAT_E2,&B));

  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }

  /* Solve */
  CHKERRQ(PetscCalloc2(n,&wr,n,&wi));
  CHKERRQ(DSGetSlepcSC(ds,&sc));
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  CHKERRQ(DSSolve(ds,wr,wi));
  CHKERRQ(DSSort(ds,wr,wi,NULL,NULL,NULL));

  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n"));
    CHKERRQ(DSView(ds,viewer));
  }
  CHKERRQ(DSGetDimensions(ds,NULL,NULL,NULL,&nev));

  /* Print computed eigenvalues */
  CHKERRQ(DSNEPGetNumFN(ds,&nfun));
  CHKERRQ(PetscMalloc1(ld*ld,&W));
  CHKERRQ(DSVectors(ds,DS_MAT_X,NULL,NULL));
  CHKERRQ(DSGetArray(ds,DS_MAT_X,&X));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n"));
  for (i=0;i<nev;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(wr[i]);
    im = PetscImaginaryPart(wr[i]);
#else
    re = wr[i];
    im = wi[i];
#endif
    /* Residual */
    CHKERRQ(PetscArrayzero(W,ld*ld));
    for (k=0;k<nfun;k++) {
      CHKERRQ(FNEvaluateFunction(funs[k],wr[i],&alpha));
      CHKERRQ(DSGetArray(ds,mat[k],&A));
      for (jj=0;jj<n;jj++) for (ii=0;ii<n;ii++) W[jj*ld+ii] += alpha*A[jj*ld+ii];
      CHKERRQ(DSRestoreArray(ds,mat[k],&A));
    }
    nrm = 0.0;
    for (k=0;k<n;k++) {
      auxr = 0.0;
#if !defined(PETSC_USE_COMPLEX)
      auxi = 0.0;
#endif
      for (j=0;j<n;j++) {
        auxr += W[k+j*ld]*X[i*ld+j];
#if !defined(PETSC_USE_COMPLEX)
        if (PetscAbs(wi[j])!=0.0) auxi += W[k+j*ld]*X[(i+1)*ld+j];
#endif
      }
      aux = SlepcAbsEigenvalue(auxr,auxi);
      nrm += aux*aux;
    }
    nrm = PetscSqrtReal(nrm);
    if (nrm>1000*n*PETSC_MACHINE_EPSILON) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: the residual norm of the %" PetscInt_FMT "-th computed eigenpair %g\n",i,(double)nrm));
    }
    if (PetscAbs(im)<1e-10) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %.5f%+.5fi\n",(double)re,(double)im));
    }
  }
  CHKERRQ(PetscFree(W));
  CHKERRQ(DSRestoreArray(ds,DS_MAT_X,&X));
  CHKERRQ(DSRestoreArray(ds,DS_MAT_W,&W));
  CHKERRQ(PetscFree2(wr,wi));
  CHKERRQ(FNDestroy(&f1));
  CHKERRQ(FNDestroy(&f2));
  CHKERRQ(FNDestroy(&f3));
  CHKERRQ(DSDestroy(&ds));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      test:
         suffix: 1
         requires: !complex
      test:
         suffix: 2
         args: -ds_nep_rg_ellipse_radius 10
         filter: sed -e "s/[+-]0\.0*i//g" | sed -e "s/37411/37410/"
         requires: complex

TEST*/
