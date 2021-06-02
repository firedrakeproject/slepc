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
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solve a Dense System of type NEP - dimension %D, tau=%g.\n",n,(double)tau);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);

  /* Create DS object and set options */
  ierr = DSCreate(PETSC_COMM_WORLD,&ds);CHKERRQ(ierr);
  ierr = DSSetType(ds,DSNEP);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = DSSetMethod(ds,1);CHKERRQ(ierr);  /* contour integral */
#endif
  ierr = DSNEPGetRG(ds,&rg);CHKERRQ(ierr);
  ierr = RGSetType(rg,RGELLIPSE);CHKERRQ(ierr);
  ierr = DSNEPSetMinimality(ds,1);CHKERRQ(ierr);
  ierr = DSNEPSetIntegrationPoints(ds,16);CHKERRQ(ierr);
  ierr = DSNEPSetRefine(ds,PETSC_DEFAULT,2);CHKERRQ(ierr);
  ierr = DSNEPSetSamplingSize(ds,25);CHKERRQ(ierr);
  ierr = DSSetFromOptions(ds);CHKERRQ(ierr);

  /* Print current options */
  ierr = DSGetMethod(ds,&meth);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  if (meth!=1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"This example requires ds_method=1");
  ierr = RGIsTrivial(rg,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must at least set the radius of the ellipse");
#endif

  ierr = DSNEPGetMinimality(ds,&midx);CHKERRQ(ierr);
  ierr = DSNEPGetIntegrationPoints(ds,&ip);CHKERRQ(ierr);
  ierr = DSNEPGetRefine(ds,NULL,&rits);CHKERRQ(ierr);
  ierr = DSNEPGetSamplingSize(ds,&spls);CHKERRQ(ierr);
  if (meth==1) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Contour integral method with %D integration points, minimality index %D, and sampling size %D\n",ip,midx,spls);CHKERRQ(ierr);
    if (rits) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Doing %D iterations of Newton refinement\n",rits);CHKERRQ(ierr);
    }
  }

  /* Set functions (prior to DSAllocate) */
  ierr = FNCreate(PETSC_COMM_WORLD,&f1);CHKERRQ(ierr);
  ierr = FNSetType(f1,FNRATIONAL);CHKERRQ(ierr);
  coeffs[0] = -1.0; coeffs[1] = 0.0;
  ierr = FNRationalSetNumerator(f1,2,coeffs);CHKERRQ(ierr);

  ierr = FNCreate(PETSC_COMM_WORLD,&f2);CHKERRQ(ierr);
  ierr = FNSetType(f2,FNRATIONAL);CHKERRQ(ierr);
  coeffs[0] = 1.0;
  ierr = FNRationalSetNumerator(f2,1,coeffs);CHKERRQ(ierr);

  ierr = FNCreate(PETSC_COMM_WORLD,&f3);CHKERRQ(ierr);
  ierr = FNSetType(f3,FNEXP);CHKERRQ(ierr);
  ierr = FNSetScale(f3,-tau,1.0);CHKERRQ(ierr);

  funs[0] = f1;
  funs[1] = f2;
  funs[2] = f3;
  ierr = DSNEPSetFN(ds,3,funs);CHKERRQ(ierr);

  /* Set dimensions */
  ld = n;
  ierr = DSAllocate(ds,ld);CHKERRQ(ierr);
  ierr = DSSetDimensions(ds,n,0,0);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  }

  /* Fill matrices */
  ierr = DSGetArray(ds,DS_MAT_E0,&Id);CHKERRQ(ierr);
  for (i=0;i<n;i++) Id[i+i*ld]=1.0;
  ierr = DSRestoreArray(ds,DS_MAT_E0,&Id);CHKERRQ(ierr);
  h = PETSC_PI/(PetscReal)(n+1);
  ierr = DSGetArray(ds,DS_MAT_E1,&A);CHKERRQ(ierr);
  for (i=0;i<n;i++) A[i+i*ld]=-2.0/(h*h)+a;
  for (i=1;i<n;i++) {
    A[i+(i-1)*ld]=1.0/(h*h);
    A[(i-1)+i*ld]=1.0/(h*h);
  }
  ierr = DSRestoreArray(ds,DS_MAT_E1,&A);CHKERRQ(ierr);
  ierr = DSGetArray(ds,DS_MAT_E2,&B);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    xi = (i+1)*h;
    B[i+i*ld] = -4.1+xi*(1.0-PetscExpReal(xi-PETSC_PI));
  }
  ierr = DSRestoreArray(ds,DS_MAT_E2,&B);CHKERRQ(ierr);

  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Solve */
  ierr = PetscCalloc2(n,&wr,n,&wi);CHKERRQ(ierr);
  ierr = DSGetSlepcSC(ds,&sc);CHKERRQ(ierr);
  sc->comparison    = SlepcCompareLargestMagnitude;
  sc->comparisonctx = NULL;
  sc->map           = NULL;
  sc->mapobj        = NULL;
  ierr = DSSolve(ds,wr,wi);CHKERRQ(ierr);
  ierr = DSSort(ds,wr,wi,NULL,NULL,NULL);CHKERRQ(ierr);

  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"After solve - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }
  ierr = DSGetDimensions(ds,NULL,NULL,NULL,&nev);CHKERRQ(ierr);

  /* Print computed eigenvalues */
  ierr = DSNEPGetNumFN(ds,&nfun);CHKERRQ(ierr);
  ierr = PetscMalloc1(ld*ld,&W);CHKERRQ(ierr);
  ierr = DSVectors(ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  ierr = DSGetArray(ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues =\n");CHKERRQ(ierr);
  for (i=0;i<nev;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(wr[i]);
    im = PetscImaginaryPart(wr[i]);
#else
    re = wr[i];
    im = wi[i];
#endif
    /* Residual */
    ierr = PetscArrayzero(W,ld*ld);CHKERRQ(ierr);
    for (k=0;k<nfun;k++) {
      ierr = FNEvaluateFunction(funs[k],wr[i],&alpha);CHKERRQ(ierr);
      ierr = DSGetArray(ds,mat[k],&A);CHKERRQ(ierr);
      for (jj=0;jj<n;jj++) for (ii=0;ii<n;ii++) W[jj*ld+ii] += alpha*A[jj*ld+ii];
      ierr = DSRestoreArray(ds,mat[k],&A);CHKERRQ(ierr);
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
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: the residual norm of the %D-th computed eigenpair %g\n",i,(double)nrm);CHKERRQ(ierr);
    }
    if (PetscAbs(im)<1e-10) {
      ierr = PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  %.5f%+.5fi\n",(double)re,(double)im);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(W);CHKERRQ(ierr);
  ierr = DSRestoreArray(ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = DSRestoreArray(ds,DS_MAT_W,&W);CHKERRQ(ierr);
  ierr = PetscFree2(wr,wi);CHKERRQ(ierr);
  ierr = FNDestroy(&f1);CHKERRQ(ierr);
  ierr = FNDestroy(&f2);CHKERRQ(ierr);
  ierr = FNDestroy(&f3);CHKERRQ(ierr);
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
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
