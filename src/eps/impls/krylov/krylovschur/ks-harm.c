/*                       

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur with harmonic extraction

   References:

       [1] "Practical Implementation of Harmonic Krylov-Schur", SLEPc Technical
            Report STR-9, available at http://www.grycap.upv.es/slepc.

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
#define __FUNCT__ "EPSSolve_KrylovSchur_Harmonic"
PetscErrorCode EPSSolve_KrylovSchur_Harmonic(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,l,lwork,nv,ld;
  Vec            u=eps->work[0];
  PetscScalar    *S,*Q,*g,*work;
  PetscReal      beta,gamma=1.0;
  PetscBool      breakdown;

  PetscFunctionBegin;
  lwork = 7*eps->ncv;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*sizeof(PetscScalar),&g);CHKERRQ(ierr);
  ierr = PSGetLeadingDimension(eps->ps,&ld);CHKERRQ(ierr);

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = PSSetDimensions(eps->ps,nv,eps->nconv,l);CHKERRQ(ierr);
    ierr = PSGetArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
    ierr = EPSBasicArnoldi(eps,PETSC_FALSE,S,ld,eps->V,eps->nconv+l,&nv,u,&beta,&breakdown);CHKERRQ(ierr);
    ierr = VecScale(u,1.0/beta);CHKERRQ(ierr);
    ierr = PSRestoreArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
    if (l==0) {
      ierr = PSSetState(eps->ps,PS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = PSSetState(eps->ps,PS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Compute translation of Krylov decomposition */ 
    ierr = PSTranslateHarmonic(eps->ps,eps->target,beta,PETSC_FALSE,g,&gamma);CHKERRQ(ierr);

    /* Solve projected problem and compute residual norm estimates */ 
    ierr = PSSolve(eps->ps,eps->eigr,eps->eigi);CHKERRQ(ierr);
    ierr = PSSort(eps->ps,eps->eigr,eps->eigi,eps->which_func,eps->which_ctx);CHKERRQ(ierr);

    /* Check convergence */ 
    ierr = PSGetArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
    ierr = PSGetArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = EPSKrylovConvergence(eps,PETSC_FALSE,eps->trackall,eps->nconv,nv-eps->nconv,S,ld,Q,ld,eps->V,nv,beta,gamma,&k,work);CHKERRQ(ierr);
    ierr = PSRestoreArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
    ierr = PSRestoreArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    ierr = PSGetArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
    ierr = PSGetArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);

    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = (nv-k)/2;
#if !defined(PETSC_USE_COMPLEX)
      if (S[k+l+(k+l-1)*ld] != 0.0) {
        if (k+l<nv-1) l = l+1;
        else l = l-1;
      }
#endif
    }
           
    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Arnoldi factorization */
        ierr = PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%D norm=%G)\n",eps->its,beta);CHKERRQ(ierr);
        ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        ierr = PSSetDimensions(eps->ps,nv,k,l);CHKERRQ(ierr);
        ierr = PSTranslateHarmonic(eps->ps,0.0,beta,PETSC_TRUE,g,&gamma);CHKERRQ(ierr);
        for (i=k;i<k+l;i++) {
          S[k+l+i*ld] = Q[nv-1+i*ld]*beta*gamma;
        }
        /* gamma u^ = u - U*g~ */
        ierr = SlepcVecMAXPBY(u,1.0,-1.0,ld,g,eps->V);CHKERRQ(ierr);        
        ierr = VecScale(u,1.0/gamma);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->V,eps->nconv,k+l,Q,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PSRestoreArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
    ierr = PSRestoreArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
    
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecCopy(u,eps->V[k+l]);CHKERRQ(ierr);
    }
    eps->nconv = k;
    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
  } 

  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(g);CHKERRQ(ierr);
  ierr = PSGetArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
  ierr = PetscMemcpy(eps->T,S,sizeof(PetscScalar)*ld*ld);CHKERRQ(ierr);
  ierr = PSRestoreArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

