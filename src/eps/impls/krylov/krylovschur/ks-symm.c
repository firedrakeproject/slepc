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
#define __FUNCT__ "EPSGetArbitraryValues"
PetscErrorCode EPSGetArbitraryValues(EPS eps,PetscScalar *rr,PetscScalar *ri)
{
  PetscErrorCode ierr;
  PetscInt       i,ld,n,l;
  Vec            xr=eps->work[1],xi=eps->work[2];
  PetscScalar    *X;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = DSGetDimensions(eps->ds,&n,PETSC_NULL,&l,PETSC_NULL);CHKERRQ(ierr);
  ierr = DSVectors(eps->ds,DS_MAT_X,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DSGetArray(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = VecSet(xi,0.0);CHKERRQ(ierr);
  for (i=l;i<n;i++) {
    ierr = VecSet(xr,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(xr,n,X+i*ld,eps->V);CHKERRQ(ierr);    
    ierr = (*eps->arbit_func)(eps->eigr[i],eps->eigi[i],xr,xi,rr+i,ri+i,eps->arbit_ctx);CHKERRQ(ierr);    
  }
  ierr = DSRestoreArray(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KrylovSchur_Symm"
PetscErrorCode EPSSolve_KrylovSchur_Symm(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       k,l,ld,nv;
  Vec            u=eps->work[0];
  PetscScalar    *Q,*rr=PETSC_NULL,*ri=PETSC_NULL;
  PetscReal      *a,*b,beta;
  PetscBool      breakdown;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  if (eps->arbit_func) {
    ierr = PetscMalloc(ld*sizeof(PetscScalar),&rr);CHKERRQ(ierr);
    ierr = PetscMalloc(ld*sizeof(PetscScalar),&ri);CHKERRQ(ierr);
  }

  /* Get the starting Lanczos vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    b = a + ld;
    ierr = EPSFullLanczos(eps,a,b,eps->V,eps->nconv+l,&nv,u,&breakdown);CHKERRQ(ierr);
    beta = b[nv-1];
    ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    ierr = DSSetDimensions(eps->ds,nv,PETSC_IGNORE,eps->nconv,eps->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(eps->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Solve projected problem */ 
    ierr = DSSolve(eps->ds,eps->eigr,PETSC_NULL);CHKERRQ(ierr);
    if (eps->arbit_func) { ierr = EPSGetArbitraryValues(eps,rr,ri);CHKERRQ(ierr); }
    ierr = DSSort(eps->ds,eps->eigr,PETSC_NULL,rr,ri,PETSC_NULL);CHKERRQ(ierr);
    ierr = DSUpdateExtraRow(eps->ds);CHKERRQ(ierr);

    /* Check convergence */
    ierr = EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,nv-eps->nconv,eps->V,nv,beta,1.0,&k);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else l = (nv-k)/2;

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
        ierr = DSTruncate(eps->ds,k+l);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = DSGetArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv,eps->V,eps->nconv,k+l,Q,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecAXPBY(eps->V[k+l],1.0/beta,0.0,u);CHKERRQ(ierr);
    }

    ierr = EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
    eps->nconv = k;
  } 
  if (eps->arbit_func) {
    ierr = PetscFree(rr);CHKERRQ(ierr);
    ierr = PetscFree(ri);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

