/*                       

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur

   Algorithm:

       Single-vector Krylov-Schur method for non-symmetric problems,
       including harmonic extraction.

   References:

       [1] "Krylov-Schur Methods in SLEPc", SLEPc Technical Report STR-7, 
           available at http://www.grycap.upv.es/slepc.

       [2] G.W. Stewart, "A Krylov-Schur Algorithm for Large Eigenproblems",
           SIAM J. Matrix Analysis and App., 23(3), pp. 601-614, 2001. 

       [3] "Practical Implementation of Harmonic Krylov-Schur", SLEPc Technical
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

PetscErrorCode EPSSolve_KrylovSchur_Default(EPS);
extern PetscErrorCode EPSSolve_KrylovSchur_Symm(EPS);
extern PetscErrorCode EPSSolve_KrylovSchur_Slice(EPS);
extern PetscErrorCode EPSSolve_KrylovSchur_Indefinite(EPS);

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_KrylovSchur"
PetscErrorCode EPSSetUp_KrylovSchur(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      issinv;
  enum { EPS_KS_DEFAULT, EPS_KS_SYMM, EPS_KS_SLICE, EPS_KS_INDEF } variant;

  PetscFunctionBegin;
  /* spectrum slicing requires special treatment of default values */
  if (eps->which==EPS_ALL) {
    if (eps->inta==0.0 && eps->intb==0.0) SETERRQ(((PetscObject)eps)->comm,1,"Must define a computational interval when using EPS_ALL"); 
    if (!eps->ishermitian) SETERRQ(((PetscObject)eps)->comm,1,"Spectrum slicing only available for symmetric/Hermitian eigenproblems"); 
    if (!((PetscObject)(eps->OP))->type_name) { /* default to shift-and-invert */
      ierr = STSetType(eps->OP,STSINVERT);CHKERRQ(ierr);
    }
    ierr = PetscTypeCompareAny((PetscObject)eps->OP,&issinv,STSINVERT,STCAYLEY,"");CHKERRQ(ierr);
    if (!issinv) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Shift-and-invert or Cayley ST is needed for spectrum slicing");
#if defined(PETSC_USE_REAL_DOUBLE)
    if (eps->tol==PETSC_DEFAULT) eps->tol = 1e-10;  /* use tighter tolerance */
#endif
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
  if (!eps->max_it) {
    if (eps->which==EPS_ALL) eps->max_it = 100;  /* special case for spectrum slicing */
    else eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  }
  if (!eps->which) { ierr = EPSDefaultSetWhich(eps);CHKERRQ(ierr); }
  if (eps->ishermitian && (eps->which==EPS_LARGEST_IMAGINARY || eps->which==EPS_SMALLEST_IMAGINARY)) SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of eps->which");

  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->extraction!=EPS_RITZ && eps->extraction!=EPS_HARMONIC)
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Unsupported extraction type");

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  if (!eps->ishermitian || eps->extraction==EPS_HARMONIC) {
    ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  }
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Left vectors not supported in this solver");
  if (eps->ishermitian) {
    if (eps->which==EPS_ALL) {
      if (eps->isgeneralized && !eps->ispositive) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Spectrum slicing not implemented for indefinite problems");

      else variant = EPS_KS_SLICE;
    } else if (eps->isgeneralized && !eps->ispositive) {
      variant = EPS_KS_INDEF;
    } else {
      switch (eps->extraction) {
        case EPS_RITZ:     variant = EPS_KS_SYMM; break;
        case EPS_HARMONIC: variant = EPS_KS_DEFAULT; break;
        default: SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Unsupported extraction type");
      }
    }
  } else {
    switch (eps->extraction) {
      case EPS_RITZ:     variant = EPS_KS_DEFAULT; break;
      case EPS_HARMONIC: variant = EPS_KS_DEFAULT; break;
      default: SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Unsupported extraction type");
    }
  }
  switch (variant) {
    case EPS_KS_DEFAULT:
      eps->ops->solve = EPSSolve_KrylovSchur_Default;
      ierr = PSSetType(eps->ps,PSNHEP);CHKERRQ(ierr);
      break;
    case EPS_KS_SYMM:
      eps->ops->solve = EPSSolve_KrylovSchur_Symm;
      ierr = PSSetType(eps->ps,PSHEP);CHKERRQ(ierr);
      ierr = PSSetCompact(eps->ps,PETSC_TRUE);CHKERRQ(ierr);
      break;
    case EPS_KS_SLICE:
      eps->ops->solve = EPSSolve_KrylovSchur_Slice;
      ierr = PSSetType(eps->ps,PSHEP);CHKERRQ(ierr);
      ierr = PSSetCompact(eps->ps,PETSC_TRUE);CHKERRQ(ierr);
      break;
    case EPS_KS_INDEF:
      eps->ops->solve = EPSSolve_KrylovSchur_Indefinite;
      ierr = PSSetType(eps->ps,PSNHEP);CHKERRQ(ierr);
      break;
    default: SETERRQ(((PetscObject)eps)->comm,1,"Unexpected error");
  }
  ierr = PSAllocate(eps->ps,eps->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_KrylovSchur_Default"
PetscErrorCode EPSSolve_KrylovSchur_Default(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,k,l,nv,ld;
  Vec            u=eps->work[0];
  PetscScalar    *S,*Q,*g,*work;
  PetscReal      beta,gamma=1.0;
  PetscBool      breakdown,harmonic;

  PetscFunctionBegin;
  ierr = PSGetLeadingDimension(eps->ps,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc(7*ld*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  harmonic = (eps->extraction==EPS_HARMONIC || eps->extraction==EPS_REFINED_HARMONIC)?PETSC_TRUE:PETSC_FALSE;
  if (harmonic) { ierr = PetscMalloc(ld*sizeof(PetscScalar),&g);CHKERRQ(ierr); }

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

    /* Compute translation of Krylov decomposition if harmonic extraction used */ 
    if (harmonic) {
      ierr = PSTranslateHarmonic(eps->ps,eps->target,beta,PETSC_FALSE,g,&gamma);CHKERRQ(ierr);
    }

    /* Solve projected problem */ 
    ierr = PSSolve(eps->ps,eps->eigr,eps->eigi);CHKERRQ(ierr);
    ierr = PSSort(eps->ps,eps->eigr,eps->eigi,eps->which_func,eps->which_ctx);CHKERRQ(ierr);

    /* Check convergence */ 
    ierr = PSGetArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
    ierr = PSGetArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = EPSKrylovConvergence(eps,PETSC_FALSE,PETSC_FALSE,eps->nconv,nv-eps->nconv,S,ld,Q,ld,eps->V,nv,beta,gamma,&k,work);CHKERRQ(ierr);
    ierr = PSRestoreArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
    ierr = PSRestoreArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (k >= eps->nev) eps->reason = EPS_CONVERGED_TOL;

    /* Update l */
    if (eps->reason != EPS_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = (nv-k)/2;
#if !defined(PETSC_USE_COMPLEX)
      ierr = PSGetArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
      if (S[k+l+(k+l-1)*ld] != 0.0) {
        if (k+l<nv-1) l = l+1;
        else l = l-1;
      }
      ierr = PSRestoreArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
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
        /* Undo translation of Krylov decomposition */ 
        if (harmonic) {
          ierr = PSSetDimensions(eps->ps,nv,k,l);CHKERRQ(ierr);
          ierr = PSTranslateHarmonic(eps->ps,0.0,beta,PETSC_TRUE,g,&gamma);CHKERRQ(ierr);
          /* gamma u^ = u - U*g~ */
          ierr = SlepcVecMAXPBY(u,1.0,-1.0,ld,g,eps->V);CHKERRQ(ierr);        
          ierr = VecScale(u,1.0/gamma);CHKERRQ(ierr);
        }
        /* Prepare the Rayleigh quotient for restart */
        ierr = PSGetArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
        ierr = PSGetArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
        for (i=k;i<k+l;i++) {
          S[k+l+i*ld] = Q[nv-1+i*ld]*beta*gamma;
        }
        ierr = PSRestoreArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
        ierr = PSRestoreArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = PSGetArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv,eps->V,eps->nconv,k+l,Q,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PSRestoreArray(eps->ps,PS_MAT_Q,&Q);CHKERRQ(ierr);

    if (eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecCopy(u,eps->V[k+l]);CHKERRQ(ierr);
    }
    eps->nconv = k;
    ierr = EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
  } 

  ierr = PetscFree(work);CHKERRQ(ierr);
  if (harmonic) { ierr = PetscFree(g);CHKERRQ(ierr); }
  ierr = PSGetArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
  ierr = PetscMemcpy(eps->T,S,sizeof(PetscScalar)*ld*ld);CHKERRQ(ierr);
  ierr = PSRestoreArray(eps->ps,PS_MAT_A,&S);CHKERRQ(ierr);
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

