/*                       

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur for symmetric eigenproblems

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KrylovSchur_Symm"
PetscErrorCode EPSSolve_KrylovSchur_Symm(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,l,ld,nv,m;
  Vec            u=eps->work[0];
  PetscScalar    *Q;
  PetscReal      *a,*b,beta;
  PetscBool      breakdown;

  PetscFunctionBegin;
  ierr = PSGetLeadingDimension(eps->ps,&ld);CHKERRQ(ierr);

  /* Get the starting Lanczos vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    m = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = PSGetArrayReal(eps->ps,PS_MAT_T,&a);CHKERRQ(ierr);
    b = a + ld;
    ierr = EPSFullLanczos(eps,a+l,b+l,eps->V,eps->nconv+l,&m,u,&breakdown);CHKERRQ(ierr);
    nv = m - eps->nconv;
    beta = b[nv-1];
    ierr = PSRestoreArrayReal(eps->ps,PS_MAT_T,&a);CHKERRQ(ierr);
    ierr = PSSetDimensions(eps->ps,nv,0,l);CHKERRQ(ierr);
    if (l==0) {
      ierr = PSSetState(eps->ps,PS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = PSSetState(eps->ps,PS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Solve projected problem */ 
    ierr = PSSolve(eps->ps,eps->eigr+eps->nconv,PETSC_NULL);CHKERRQ(ierr);
    ierr = PSSort(eps->ps,eps->eigr+eps->nconv,PETSC_NULL,eps->which_func,eps->which_ctx);CHKERRQ(ierr);

    /* Check convergence */
    ierr = PSGetArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = EPSKrylovConvergence(eps,PETSC_TRUE,PETSC_FALSE,eps->nconv,nv,eps->V+eps->nconv,nv,beta,1.0,&k);CHKERRQ(ierr);
    ierr = PSRestoreArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else l = (eps->nconv+nv-k)/2;

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Lanczos factorization */
        ierr = PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%D norm=%G)\n",eps->its,beta);CHKERRQ(ierr);
        ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        ierr = PSGetArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
        ierr = PSGetArrayReal(eps->ps,PS_MAT_T,&a);CHKERRQ(ierr);
        b = a + ld;
        for (i=0;i<l;i++) {
          a[i] = PetscRealPart(eps->eigr[i+k]);
          b[i] = PetscRealPart(Q[nv-1+(i+k-eps->nconv)*ld]*beta);
        }
        ierr = PSRestoreArrayReal(eps->ps,PS_MAT_T,&a);CHKERRQ(ierr);
        ierr = PSRestoreArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = PSGetArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv,eps->V+eps->nconv,0,k+l-eps->nconv,Q,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PSRestoreArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecAXPBY(eps->V[k+l],1.0/beta,0.0,u);CHKERRQ(ierr);
    }

    ierr = EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv+eps->nconv);CHKERRQ(ierr);
    eps->nconv = k;
  } 
  PetscFunctionReturn(0);
}

