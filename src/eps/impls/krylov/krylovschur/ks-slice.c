/*                       

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur with spectrum slicing for symmetric eigenproblems

   References:

       [1] R.G. Grimes et al., "A shifted block Lanczos algorithm for solving
           sparse symmetric generalized eigenproblems", SIAM J. Matrix Analysis
           and App., 15(1), pp. 228–272, 1994.

       [2] G.W. Stewart, "A Krylov-Schur Algorithm for Large Eigenproblems",
           SIAM J. Matrix Analysis and App., 23(3), pp. 601-614, 2001. 

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include <private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

extern PetscErrorCode EPSProjectedKSSym(EPS,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*);

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KrylovSchur_Slice"
PetscErrorCode EPSSolve_KrylovSchur_Slice(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,l,lds,lt,nv,m,*iwork;
  Vec            u=eps->work[0];
  PetscScalar    *Q;
  PetscReal      *a,*b,*work,beta;
  PetscBool      breakdown;

  PetscFunctionBegin;
  lds = PetscMin(eps->mpd,eps->ncv);
  ierr = PetscMalloc(lds*lds*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(lds*lds*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(2*lds*sizeof(PetscInt),&iwork);CHKERRQ(ierr);
  lt = PetscMin(eps->nev+eps->mpd,eps->ncv);
  ierr = PetscMalloc(lt*sizeof(PetscReal),&a);CHKERRQ(ierr);  
  ierr = PetscMalloc(lt*sizeof(PetscReal),&b);CHKERRQ(ierr);  

  /* Get the starting Lanczos vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Lanczos factorization */
    m = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = EPSFullLanczos(eps,a+l,b+l,eps->V,eps->nconv+l,&m,u,&breakdown);CHKERRQ(ierr);
    nv = m - eps->nconv;
    beta = b[nv-1];

    /* Solve projected problem and compute residual norm estimates */ 
    ierr = EPSProjectedKSSym(eps,nv,l,a,b,eps->eigr+eps->nconv,Q,work,iwork);CHKERRQ(ierr);

    /* Check convergence */
    ierr = EPSKrylovConvergence(eps,PETSC_TRUE,eps->nconv,nv,PETSC_NULL,nv,Q,eps->V+eps->nconv,nv,beta,1.0,&k,PETSC_NULL);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else l = (eps->nconv+nv-k)/2;

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Lanczos factorization */
        PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%D norm=%G)\n",eps->its,beta);
        ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          PetscInfo(eps,"Unable to generate more start vectors\n");
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        for (i=0;i<l;i++) {
          a[i] = PetscRealPart(eps->eigr[i+k]);
          b[i] = PetscRealPart(Q[nv-1+(i+k-eps->nconv)*nv]*beta);
        }
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->V+eps->nconv,0,k+l-eps->nconv,Q,nv,PETSC_FALSE);CHKERRQ(ierr);
    /* Normalize u and append it to V */
    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecAXPBY(eps->V[k+l],1.0/beta,0.0,u);CHKERRQ(ierr);
    }

    ierr = EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv+eps->nconv);CHKERRQ(ierr);
    eps->nconv = k;
  } 
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

