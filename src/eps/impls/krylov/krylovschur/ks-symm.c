/*

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur for symmetric eigenproblems

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/epsimpl.h>
#include "krylovschur.h"

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_KrylovSchur_Symm"
PetscErrorCode EPSSolve_KrylovSchur_Symm(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        k,l,ld,nv,nconv;
  Mat             U;
  PetscReal       *a,*b,beta;
  PetscBool       breakdown;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);

  /* Get the starting Lanczos vector */
  ierr = EPSGetStartVector(eps,0,NULL);CHKERRQ(ierr);
  l = 0;

  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    b = a + ld;
    ierr = EPSFullLanczos(eps,a,b,eps->nconv+l,&nv,&breakdown);CHKERRQ(ierr);
    beta = b[nv-1];
    ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    ierr = DSSetDimensions(eps->ds,nv,0,eps->nconv,eps->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(eps->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(eps->V,eps->nconv,nv);CHKERRQ(ierr);

    /* Solve projected problem */
    ierr = DSSolve(eps->ds,eps->eigr,NULL);CHKERRQ(ierr);
    if (eps->arbitrary) { ierr = EPSGetArbitraryValues(eps,eps->rr,eps->ri);CHKERRQ(ierr); }
    ierr = DSSort(eps->ds,eps->eigr,NULL,eps->rr,eps->ri,NULL);CHKERRQ(ierr);
    ierr = DSUpdateExtraRow(eps->ds);CHKERRQ(ierr);

    /* Check convergence */
    ierr = EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,nv-eps->nconv,beta,1.0,&k);CHKERRQ(ierr);
    ierr = (*eps->stopping)(eps,eps->its,eps->max_it,k,eps->nev,&eps->reason,eps->stoppingctx);CHKERRQ(ierr);
    nconv = k;

    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
    if (!ctx->lock && l>0) { l += k; k = 0; } /* non-locking variant: reset no. of converged pairs */

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Lanczos factorization */
        ierr = PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%D norm=%g)\n",eps->its,(double)beta);CHKERRQ(ierr);
        if (k<eps->nev) {
          ierr = EPSGetStartVector(eps,k,&breakdown);CHKERRQ(ierr);
          if (breakdown) {
            eps->reason = EPS_DIVERGED_BREAKDOWN;
            ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
          }
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        ierr = DSTruncate(eps->ds,k+l);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = DSGetMat(eps->ds,DS_MAT_Q,&U);CHKERRQ(ierr);
    ierr = BVMultInPlace(eps->V,U,eps->nconv,k+l);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);

    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = BVCopyColumn(eps->V,nv,k+l);CHKERRQ(ierr);
    }

    eps->nconv = k;
    ierr = EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

