/*                       

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur for symmetric-indefinite eigenproblems

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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
#include "krylovschur.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSFullLanczosIndef"
static PetscErrorCode EPSFullLanczosIndef(EPS eps,PetscReal *alpha,PetscReal *beta,PetscReal *omega,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       j,m = *M;
  PetscReal      norm,min=1.0;
  PetscScalar    *hwork,lhwork[100];

  PetscFunctionBegin;
  if (m > 100) {
    ierr = PetscMalloc((eps->nds+m)*sizeof(PetscScalar),&hwork);CHKERRQ(ierr);
  } else {
    hwork = lhwork;
  }

  for (j=k;j<m-1;j++) {
    ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr);
    ierr = IPPseudoOrthogonalize(eps->ip,j+1,V,omega,V[j+1],hwork,&norm,breakdown);CHKERRQ(ierr);
    alpha[j] = PetscRealPart(hwork[j]);
    beta[j] = PetscAbsReal(norm);
    min = PetscMin(min,beta[j]);
    omega[j+1] = (norm<0.0)?-1:1;
    
    if (breakdown && *breakdown ) {
      *M = j+1;
      if (m > 100) {
        ierr = PetscFree(hwork);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    } else {
      ierr = VecScale(V[j+1],1.0/norm);CHKERRQ(ierr);
    }
  }
  ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr);
  ierr = IPPseudoOrthogonalize(eps->ip,m,V,omega,f,hwork,&norm,PETSC_NULL);CHKERRQ(ierr);
  alpha[m-1] = PetscRealPart(hwork[m-1]); 
  beta[m-1] =PetscAbsReal(norm);
  omega[m] = (norm<0.0)?-1:1;
  if (m > 100) {
    ierr = PetscFree(hwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KrylovSchur_Indefinite"
PetscErrorCode EPSSolve_KrylovSchur_Indefinite(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_KRYLOVSCHUR *ctx = (EPS_KRYLOVSCHUR*)eps->data;
  PetscInt        i,k,l,ld,nv;
  Vec             u=eps->work[0];
  PetscScalar     *Q;
  PetscReal       *a,*b,*r,beta,norm,*omega;
  PetscBool       breakdown=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  /* Get the starting Lanczos vector */
  ierr = SlepcVecSetRandom(eps->V[0],eps->rand);CHKERRQ(ierr);
  ierr = IPNorm(eps->ip,eps->V[0],&norm);
  if (norm==0.0) SETERRQ(((PetscObject)eps)->comm,1,"Initial vector is zero or belongs to the deflation space"); 
  ierr = DSGetArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
  omega[0] = (norm > 0)?1.0:-1.0;
  beta = PetscAbsReal(norm);
  ierr = DSRestoreArrayReal(eps->ds,DS_MAT_D,&omega);
  ierr = VecScale(eps->V[0],1.0/norm);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    b = a + ld;
    ierr = DSGetArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
    ierr = EPSFullLanczosIndef(eps,a,b,omega,eps->V,eps->nconv+l,&nv,u,PETSC_NULL);CHKERRQ(ierr);
    beta = b[nv-1];
    ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(eps->ds,DS_MAT_D,&omega);
    ierr = DSSetDimensions(eps->ds,nv,PETSC_IGNORE,eps->nconv,eps->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(eps->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Solve projected problem */
    ierr = DSSolve(eps->ds,eps->eigr,eps->eigi);CHKERRQ(ierr);
    ierr = DSSort(eps->ds,eps->eigr,eps->eigi,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

    /* Check convergence */
    ierr = EPSKrylovConvergence(eps,PETSC_FALSE,eps->nconv,nv-eps->nconv,eps->V,nv,beta,1.0,&k);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
      ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
      if(*(a+ld+k+l-1)!=0){
        if (k+l<nv-1) l = l+1;
        else l = l-1;
      }
      ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    }
    
    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        SETERRQ1(((PetscObject)eps)->comm,PETSC_ERR_CONV_FAILED,"Breakdown in Indefinite Krylov-Schur (beta=%g)",beta);
      } else {
        /* Prepare the Rayleigh quotient for restart */
        ierr = DSGetArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSGetArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
        r = a + 2*ld;
        for (i=k;i<k+l;i++) {
          r[i] = PetscRealPart(Q[nv-1+i*ld]*beta);
        }
        omega[k+l] = omega[nv];
        ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        ierr = DSRestoreArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = DSGetArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv,eps->V,eps->nconv,k+l,Q,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DSRestoreArray(eps->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);

    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = DSGetArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
      ierr = VecAXPBY(eps->V[k+l],1.0/(beta*omega[k+l]),0.0,u);CHKERRQ(ierr);
      ierr = DSRestoreArrayReal(eps->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
    }

    ierr = EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
    eps->nconv = k;

  }
  ierr = DSSetDimensions(eps->ds,eps->nconv,PETSC_IGNORE,0,0);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

