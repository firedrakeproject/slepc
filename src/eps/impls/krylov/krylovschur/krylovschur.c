/*                       

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur

   Algorithm:

       Single-vector Krylov-Schur method for both symmetric and non-symmetric
       problems.

   References:

       [1] "Krylov-Schur Methods in SLEPc", SLEPc Technical Report STR-7, 
           available at http://www.grycap.upv.es/slepc.

       [2] G.W. Stewart, "A Krylov-Schur Algorithm for Large Eigenproblems",
           SIAM J. Matrix Analysis and App., 23(3), pp. 601-614, 2001. 

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

#include <private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

PetscErrorCode EPSSolve_KrylovSchur_Default(EPS);
extern PetscErrorCode EPSSolve_KrylovSchur_Harmonic(EPS);
extern PetscErrorCode EPSSolve_KrylovSchur_Symm(EPS);
extern PetscErrorCode EPSSolve_KrylovSchur_Slice(EPS);

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_KrylovSchur"
PetscErrorCode EPSSetUp_KrylovSchur(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      issinv;

  PetscFunctionBegin;
  /* spectrum slicing requires special treatment of default values */
  if (eps->which==EPS_ALL) {
    if (eps->inta==0.0 && eps->intb==0.0) SETERRQ(((PetscObject)eps)->comm,1,"Must define a computational interval when using EPS_ALL"); 
    if (!eps->ishermitian) SETERRQ(((PetscObject)eps)->comm,1,"Spectrum slicing only available for symmetric/Hermitian eigenproblems"); 
    if (!((PetscObject)(eps->OP))->type_name) { /* default to shift-and-invert */
      ierr = STSetType(eps->OP,STSINVERT);CHKERRQ(ierr);
    }
    ierr = PetscTypeCompare((PetscObject)eps->OP,STSINVERT,&issinv);CHKERRQ(ierr);
    if (!issinv) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Shift-and-invert ST is needed for spectrum slicing");
    if (eps->intb >= PETSC_MAX_REAL) { /* right-open interval */
      if (eps->inta <= PETSC_MIN_REAL) SETERRQ(((PetscObject)eps)->comm,1,"The defined computational interval should have at least one of their sides bounded");
      ierr = STSetDefaultShift(eps->OP,eps->inta);CHKERRQ(ierr);
    }
    else { ierr = STSetDefaultShift(eps->OP,eps->intb);CHKERRQ(ierr); }

    if (eps->nev==1) eps->nev = 40;  /* nev not set, use default value */
    if (eps->nev<10) SETERRQ(((PetscObject)eps)->comm,1,"nev cannot be less than 10 in spectrum slicing runs"); 
    eps->ops->backtransform = PETSC_NULL;
  }

  /* proceed with the general case */
  if (eps->ncv) { /* ncv set */
    if (eps->ncv<eps->nev) SETERRQ(((PetscObject)eps)->comm,1,"The value of ncv must be at least nev"); 
  } else if (eps->mpd) { /* mpd set */
    eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd);
  } else { /* neither set: defaults depend on nev being small or large */
    if (eps->nev<500) eps->ncv = PetscMin(eps->n,PetscMax(2*eps->nev,eps->nev+15));
    else { eps->mpd = 500; eps->ncv = PetscMin(eps->n,eps->nev+eps->mpd); }
  }
  if (!eps->mpd) eps->mpd = eps->ncv;
  if (eps->ncv>eps->nev+eps->mpd) SETERRQ(((PetscObject)eps)->comm,1,"The value of ncv must not be larger than nev+mpd"); 
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) eps->which = EPS_LARGEST_MAGNITUDE;
  if (eps->ishermitian && (eps->which==EPS_LARGEST_IMAGINARY || eps->which==EPS_SMALLEST_IMAGINARY))
    SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of eps->which");

  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->extraction!=EPS_RITZ && eps->extraction!=EPS_HARMONIC) {
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Unsupported extraction type");
  }

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  if (!eps->ishermitian || eps->extraction==EPS_HARMONIC) {
    ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  }
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Left vectors not supported in this solver");
  if (eps->ishermitian) {
    if (eps->which==EPS_ALL) eps->ops->solve = EPSSolve_KrylovSchur_Slice;
    else {
      switch (eps->extraction) {
        case EPS_RITZ:     eps->ops->solve = EPSSolve_KrylovSchur_Symm; break;
        case EPS_HARMONIC: eps->ops->solve = EPSSolve_KrylovSchur_Harmonic; break;
        default: SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Unsupported extraction type");
      }
    }
  } else {
    switch (eps->extraction) {
      case EPS_RITZ:     eps->ops->solve = EPSSolve_KrylovSchur_Default; break;
      case EPS_HARMONIC: eps->ops->solve = EPSSolve_KrylovSchur_Harmonic; break;
      default: SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Unsupported extraction type");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSProjectedKSNonsym"
/*
   EPSProjectedKSNonsym - Solves the projected eigenproblem in the Krylov-Schur
   method (non-symmetric case).

   On input:
     l is the number of vectors kept in previous restart (0 means first restart)
     S is the projected matrix (leading dimension is lds)

   On output:
     S has (real) Schur form with diagonal blocks sorted appropriately
     Q contains the corresponding Schur vectors (order n, leading dimension n)
*/
PetscErrorCode EPSProjectedKSNonsym(EPS eps,PetscInt l,PetscScalar *S,PetscInt lds,PetscScalar *Q,PetscInt n)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (l==0) {
    ierr = PetscMemzero(Q,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<n;i++) 
      Q[i*(n+1)] = 1.0;
  } else {
    /* Reduce S to Hessenberg form, S <- Q S Q' */
    ierr = EPSDenseHessenberg(n,eps->nconv,S,lds,Q);CHKERRQ(ierr);
  }
  /* Reduce S to (quasi-)triangular form, S <- Q S Q' */
  ierr = EPSDenseSchur(n,eps->nconv,S,lds,Q,eps->eigr,eps->eigi);CHKERRQ(ierr);
  /* Sort the remaining columns of the Schur form */
  ierr = EPSSortDenseSchur(eps,n,eps->nconv,S,lds,Q,eps->eigr,eps->eigi);CHKERRQ(ierr);    
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KrylovSchur_Default"
PetscErrorCode EPSSolve_KrylovSchur_Default(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,l,lwork,nv;
  Vec            u=eps->work[0];
  PetscScalar    *S=eps->T,*Q,*work;
  PetscReal      beta;
  PetscBool      breakdown;

  PetscFunctionBegin;
  ierr = PetscMemzero(S,eps->ncv*eps->ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  lwork = 7*eps->ncv;
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);

  /* Get the starting Arnoldi vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  l = 0;
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;

    /* Compute an nv-step Arnoldi factorization */
    nv = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = EPSBasicArnoldi(eps,PETSC_FALSE,S,eps->ncv,eps->V,eps->nconv+l,&nv,u,&beta,&breakdown);CHKERRQ(ierr);
    ierr = VecScale(u,1.0/beta);CHKERRQ(ierr);

    /* Solve projected problem */ 
    ierr = EPSProjectedKSNonsym(eps,l,S,eps->ncv,Q,nv);CHKERRQ(ierr);

    /* Check convergence */ 
    ierr = EPSKrylovConvergence(eps,PETSC_FALSE,eps->trackall,eps->nconv,nv-eps->nconv,S,eps->ncv,Q,eps->V,nv,beta,1.0,&k,work);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
    
    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = (nv-k)/2;
#if !defined(PETSC_USE_COMPLEX)
      if (S[(k+l-1)*(eps->ncv+1)+1] != 0.0) {
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
        for (i=k;i<k+l;i++) {
          S[i*eps->ncv+k+l] = Q[(i+1)*nv-1]*beta;
        }
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->V,eps->nconv,k+l,Q,nv,PETSC_FALSE);CHKERRQ(ierr);

    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecCopy(u,eps->V[k+l]);CHKERRQ(ierr);
    }
    eps->nconv = k;

    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
  } 
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSReset_KrylovSchur"
PetscErrorCode EPSReset_KrylovSchur(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  ierr = EPSReset_Default(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_KrylovSchur"
PetscErrorCode EPSCreate_KrylovSchur(EPS eps)
{
  PetscFunctionBegin;
  eps->ops->setup          = EPSSetUp_KrylovSchur;
  eps->ops->reset          = EPSReset_KrylovSchur;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->computevectors = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}
EXTERN_C_END

