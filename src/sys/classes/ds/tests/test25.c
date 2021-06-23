/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test for DSPEP and DSNEP.\n\n";

#include <slepcds.h>

#define NMAT 5

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DS             ds;
  FN             f[NMAT],qfun;
  SlepcSC        sc;
  PetscScalar    *A,*wr,*wi,*X,*y,*r,numer[NMAT],alpha;
  PetscReal      c[10] = { 0.6, 1.3, 1.3, 0.1, 0.1, 1.2, 1.0, 1.0, 1.2, 1.0 };
  PetscReal      tol,radius=1.5,re,im,nrm;
  PetscInt       i,j,ii,jj,II,k,m=3,n,ld,nev,nfun,d,*inside;
  PetscViewer    viewer;
  PetscBool      verbose,isnep=PETSC_FALSE;
  RG             rg;
  DSMatType      mat[5]={DS_MAT_E0,DS_MAT_E1,DS_MAT_E2,DS_MAT_E3,DS_MAT_E4};
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *yi,*ri,alphai=0.0,t;
#endif

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-isnep",&isnep,NULL);CHKERRQ(ierr);
  n = m*m;
  k = 10;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nButterfly problem, n=%D (m=%D)\n\n",n,m);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-radius",&radius,NULL);CHKERRQ(ierr);

  /* Create DS object */
  ierr = DSCreate(PETSC_COMM_WORLD,&ds);CHKERRQ(ierr);
  tol  = 1000*n*PETSC_MACHINE_EPSILON;
  if (isnep) {
    ierr = DSSetType(ds,DSNEP);CHKERRQ(ierr);
    ierr = DSSetMethod(ds,1);CHKERRQ(ierr);
    ierr = DSNEPSetRefine(ds,tol,PETSC_DECIDE);CHKERRQ(ierr);
  } else {
    ierr = DSSetType(ds,DSPEP);CHKERRQ(ierr);
  }
  ierr = DSSetFromOptions(ds);CHKERRQ(ierr);

  /* Set functions (prior to DSAllocate) f_i=x^i */
  if (isnep) {
    numer[0] = 1.0;
    for (j=1;j<NMAT;j++) numer[j] = 0.0;
    for (i=0;i<NMAT;i++) {
      ierr = FNCreate(PETSC_COMM_WORLD,&f[i]);CHKERRQ(ierr);
      ierr = FNSetType(f[i],FNRATIONAL);CHKERRQ(ierr);
      ierr = FNRationalSetNumerator(f[i],i+1,numer);CHKERRQ(ierr);
    }
    ierr = DSNEPSetFN(ds,NMAT,f);CHKERRQ(ierr);
  } else {
    ierr = DSPEPSetDegree(ds,NMAT-1);CHKERRQ(ierr);
  }

  /* Set dimensions */
  ld = n+2;  /* test leading dimension larger than n */
  ierr = DSAllocate(ds,ld);CHKERRQ(ierr);
  ierr = DSSetDimensions(ds,n,0,0);CHKERRQ(ierr);

  /* Set region (used only in method=1) */
  ierr = RGCreate(PETSC_COMM_WORLD,&rg);CHKERRQ(ierr);
  ierr = RGSetType(rg,RGELLIPSE);CHKERRQ(ierr);
  ierr = RGEllipseSetParameters(rg,1.5,radius,.5);CHKERRQ(ierr);
  ierr = RGSetFromOptions(rg);CHKERRQ(ierr);
  if (isnep) {
    ierr = DSNEPSetRG(ds,rg);CHKERRQ(ierr);
  }

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = DSViewFromOptions(ds,NULL,"-ds_view");CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    /* Show info about functions */
    if (isnep) {
      ierr = DSNEPGetNumFN(ds,&nfun);CHKERRQ(ierr);
      for (i=0;i<nfun;i++) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Function %D:\n",i);CHKERRQ(ierr);
        ierr = DSNEPGetFN(ds,i,&qfun);CHKERRQ(ierr);
        ierr = FNView(qfun,NULL);CHKERRQ(ierr);
      }
    }
  }

  /* Fill matrices */
  /* A0 */
  ierr = DSGetArray(ds,DS_MAT_E0,&A);CHKERRQ(ierr);
  for (II=0;II<n;II++) {
    i = II/m; j = II-i*m;
    A[II+II*ld] = 4.0*c[0]/6.0+4.0*c[1]/6.0;
    if (j>0) A[II+(II-1)*ld] = c[0]/6.0;
    if (j<m-1) A[II+ld*(II+1)] = c[0]/6.0;
    if (i>0) A[II+ld*(II-m)] = c[1]/6.0;
    if (i<m-1) A[II+ld*(II+m)] = c[1]/6.0;
  }
  ierr = DSRestoreArray(ds,DS_MAT_E0,&A);CHKERRQ(ierr);

  /* A1 */
  ierr = DSGetArray(ds,DS_MAT_E1,&A);CHKERRQ(ierr);
  for (II=0;II<n;II++) {
    i = II/m; j = II-i*m;
    if (j>0) A[II+ld*(II-1)] = c[2];
    if (j<m-1) A[II+ld*(II+1)] = -c[2];
    if (i>0) A[II+ld*(II-m)] = c[3];
    if (i<m-1) A[II+ld*(II+m)] = -c[3];
  }
  ierr = DSRestoreArray(ds,DS_MAT_E1,&A);CHKERRQ(ierr);

  /* A2 */
  ierr = DSGetArray(ds,DS_MAT_E2,&A);CHKERRQ(ierr);
  for (II=0;II<n;II++) {
    i = II/m; j = II-i*m;
    A[II+ld*II] = -2.0*c[4]-2.0*c[5];
    if (j>0) A[II+ld*(II-1)] = c[4];
    if (j<m-1) A[II+ld*(II+1)] = c[4];
    if (i>0) A[II+ld*(II-m)] = c[5];
    if (i<m-1) A[II+ld*(II+m)] = c[5];
  }
  ierr = DSRestoreArray(ds,DS_MAT_E2,&A);CHKERRQ(ierr);

  /* A3 */
  ierr = DSGetArray(ds,DS_MAT_E3,&A);CHKERRQ(ierr);
  for (II=0;II<n;II++) {
    i = II/m; j = II-i*m;
    if (j>0) A[II+ld*(II-1)] = c[6];
    if (j<m-1) A[II+ld*(II+1)] = -c[6];
    if (i>0) A[II+ld*(II-m)] = c[7];
    if (i<m-1) A[II+ld*(II+m)] = -c[7];
  }
  ierr = DSRestoreArray(ds,DS_MAT_E3,&A);CHKERRQ(ierr);

  /* A4 */
  ierr = DSGetArray(ds,DS_MAT_E4,&A);CHKERRQ(ierr);
  for (II=0;II<n;II++) {
    i = II/m; j = II-i*m;
    A[II+ld*II] = 2.0*c[8]+2.0*c[9];
    if (j>0) A[II+ld*(II-1)] = -c[8];
    if (j<m-1) A[II+ld*(II+1)] = -c[8];
    if (i>0) A[II+ld*(II-m)] = -c[9];
    if (i<m-1) A[II+ld*(II+m)] = -c[9];
  }
  ierr = DSRestoreArray(ds,DS_MAT_E4,&A);CHKERRQ(ierr);

  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial - - - - - - - - -\n");CHKERRQ(ierr);
    ierr = DSView(ds,viewer);CHKERRQ(ierr);
  }

  /* Solve */
  if (isnep) {
    ierr = DSNEPGetMinimality(ds,&d);CHKERRQ(ierr);
  } else {
    ierr = DSPEPGetDegree(ds,&d);CHKERRQ(ierr);
  }
  ierr = PetscCalloc3(n*d,&wr,n*d,&wi,n*d,&inside);CHKERRQ(ierr);
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
  if (isnep) {
    ierr = DSGetDimensions(ds,NULL,NULL,NULL,&nev);CHKERRQ(ierr);
    for (i=0;i<nev;i++) inside[i] = i;
  } else {
    ierr = RGCheckInside(rg,d*n,wr,wi,inside);CHKERRQ(ierr);
    nev = 0;
    for (i=0;i<d*n;i++) if (inside[i]>0) inside[nev++] = i;
  }

  /* Print computed eigenvalues */
  ierr = PetscMalloc2(ld,&y,ld,&r);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc2(ld,&yi,ld,&ri);CHKERRQ(ierr);
#endif
  ierr = DSVectors(ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  ierr = DSGetArray(ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed eigenvalues in the region: %D\n",nev);CHKERRQ(ierr);
  for (i=0;i<nev;i++) {
#if defined(PETSC_USE_COMPLEX)
    re = PetscRealPart(wr[inside[i]]);
    im = PetscImaginaryPart(wr[inside[i]]);
#else
    re = wr[inside[i]];
    im = wi[inside[i]];
#endif
    ierr = PetscArrayzero(r,n);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscArrayzero(ri,n);CHKERRQ(ierr);
#endif
    /* Residual */
    alpha = 1.0;
    for (k=0;k<NMAT;k++) {
      ierr = DSGetArray(ds,mat[k],&A);CHKERRQ(ierr);
      for (ii=0;ii<n;ii++) {
        y[ii] = 0.0;
        for (jj=0;jj<n;jj++) y[ii] += A[jj*ld+ii]*X[inside[i]*ld+jj];
      }
#if !defined(PETSC_USE_COMPLEX)
      for (ii=0;ii<n;ii++) {
        yi[ii] = 0.0;
        for (jj=0;jj<n;jj++) yi[ii] += A[jj*ld+ii]*X[inside[i+1]*ld+jj];
      }
#endif
      ierr = DSRestoreArray(ds,mat[k],&A);CHKERRQ(ierr);
      if (isnep) {
        ierr = FNEvaluateFunction(f[k],wr[inside[i]],&alpha);CHKERRQ(ierr);
      }
      for (ii=0;ii<n;ii++) r[ii] += alpha*y[ii];
#if !defined(PETSC_USE_COMPLEX)
      for (ii=0;ii<n;ii++) r[ii]  -= alphai*yi[ii];
      for (ii=0;ii<n;ii++) ri[ii] += alpha*yi[ii]+alphai*y[ii];
#endif
      if (!isnep) {
#if defined(PETSC_USE_COMPLEX)
        alpha *= wr[inside[i]];
#else
        t      = alpha;
        alpha  = alpha*re-alphai*im;
        alphai = alphai*re+t*im;
#endif
      }
    }
    nrm = 0.0;
    for (k=0;k<n;k++) {
#if !defined(PETSC_USE_COMPLEX)
      nrm += r[k]*r[k]+ri[k]*ri[k];
#else
      nrm += PetscRealPart(r[k]*PetscConj(r[k]));
#endif
    }
    nrm = PetscSqrtReal(nrm);
    if (nrm/SlepcAbsEigenvalue(wr[inside[i]],wi[inside[i]])>tol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: the residual norm of the %D-th computed eigenpair %g\n",i,(double)nrm);CHKERRQ(ierr);
    }
    if (PetscAbs(im)<1e-10) {
      ierr = PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  %.5f%+.5fi\n",(double)re,(double)im);CHKERRQ(ierr);
    }
#if !defined(PETSC_USE_COMPLEX)
    if (im!=0.0) i++;
    if (PetscAbs(im)<1e-10) {
      ierr = PetscViewerASCIIPrintf(viewer,"  %.5f\n",(double)re);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  %.5f%+.5fi\n",(double)re,(double)-im);CHKERRQ(ierr);
    }
#endif
  }
  ierr = DSRestoreArray(ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = PetscFree3(wr,wi,inside);CHKERRQ(ierr);
  ierr = PetscFree2(y,r);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscFree2(yi,ri);CHKERRQ(ierr);
#endif
  if (isnep) {
    for (i=0;i<NMAT;i++) {
      ierr = FNDestroy(&f[i]);CHKERRQ(ierr);
    }
  }
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
  ierr = RGDestroy(&rg);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      filter: sed -e "s/[+-]\([0-9]\.[0-9]*i\)/+-\\1/" | sed -e "s/56808/56807/" | sed -e "s/34719/34720/"
      output_file: output/test25_1.out
      test:
         suffix: 1
      test:
         suffix: 2
         args: -isnep
         requires: complex !single

TEST*/
