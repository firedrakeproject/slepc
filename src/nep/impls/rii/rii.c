/*

   SLEPc nonlinear eigensolver: "rii"

   Method: Residual inverse iteration

   Algorithm:

       Simple residual inverse iteration with varying shift.

   References:

       [1] A. Neumaier, "Residual inverse iteration for the nonlinear
           eigenvalue problem", SIAM J. Numer. Anal. 22(5):914-923, 1985.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/nepimpl.h>

#undef __FUNCT__
#define __FUNCT__ "NEPSetUp_RII"
PetscErrorCode NEPSetUp_RII(NEP nep)
{
  PetscErrorCode ierr;
  PetscBool      istrivial;

  PetscFunctionBegin;
  if (nep->ncv) { /* ncv set */
    if (nep->ncv<nep->nev) SETERRQ(PetscObjectComm((PetscObject)nep),1,"The value of ncv must be at least nev");
  } else if (nep->mpd) { /* mpd set */
    nep->ncv = PetscMin(nep->n,nep->nev+nep->mpd);
  } else { /* neither set: defaults depend on nev being small or large */
    if (nep->nev<500) nep->ncv = PetscMin(nep->n,PetscMax(2*nep->nev,nep->nev+15));
    else {
      nep->mpd = 500;
      nep->ncv = PetscMin(nep->n,nep->nev+nep->mpd);
    }
  }
  if (!nep->mpd) nep->mpd = nep->ncv;
  if (nep->ncv>nep->nev+nep->mpd) SETERRQ(PetscObjectComm((PetscObject)nep),1,"The value of ncv must not be larger than nev+mpd");
  if (nep->nev>1) { ierr = PetscInfo(nep,"Warning: requested more than one eigenpair but RII can only compute one\n");CHKERRQ(ierr); }
  if (!nep->max_it) nep->max_it = PetscMax(5000,2*nep->n/nep->ncv);
  if (!nep->max_funcs) nep->max_funcs = nep->max_it;

  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This solver does not support region filtering");

  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  ierr = NEPSetWorkVecs(nep,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSolve_RII"
PetscErrorCode NEPSolve_RII(NEP nep)
{
  PetscErrorCode     ierr;
  Mat                T=nep->function,Tp=nep->jacobian,Tsigma;
  Vec                u,r=nep->work[0],delta=nep->work[1];
  PetscScalar        lambda,a1,a2;
  PetscReal          relerr;
  PetscBool          hascopy;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  /* get initial approximation of eigenvalue and eigenvector */
  ierr = NEPGetDefaultShift(nep,&lambda);CHKERRQ(ierr);
  if (!nep->nini) {
    ierr = BVSetRandomColumn(nep->V,0,nep->rand);CHKERRQ(ierr);
  }
  ierr = BVGetColumn(nep->V,0,&u);CHKERRQ(ierr);

  /* correct eigenvalue approximation: lambda = lambda - (u'*T*u)/(u'*Tp*u) */
  ierr = NEPComputeFunction(nep,lambda,T,T);CHKERRQ(ierr);
  ierr = MatMult(T,u,r);CHKERRQ(ierr);
  ierr = VecDot(u,r,&a1);CHKERRQ(ierr);
  ierr = NEPApplyJacobian(nep,lambda,u,delta,r,Tp);CHKERRQ(ierr);
  ierr = VecDot(u,r,&a2);CHKERRQ(ierr);
  lambda = lambda - a1/a2;

  /* prepare linear solver */
  ierr = MatDuplicate(T,MAT_COPY_VALUES,&Tsigma);CHKERRQ(ierr);
  ierr = KSPSetOperators(nep->ksp,Tsigma,Tsigma);CHKERRQ(ierr);

  /* Restart loop */
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;

    /* update preconditioner and set adaptive tolerance */
    if (nep->lag && !(nep->its%nep->lag) && nep->its>2*nep->lag && relerr<1e-2) {
      ierr = MatHasOperation(T,MATOP_COPY,&hascopy);CHKERRQ(ierr);
      if (hascopy) {
        ierr = MatCopy(T,Tsigma,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ierr = MatDestroy(&Tsigma);CHKERRQ(ierr);
        ierr = MatDuplicate(T,MAT_COPY_VALUES,&Tsigma);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(nep->ksp,Tsigma,Tsigma);CHKERRQ(ierr);
    }
    if (!nep->cctol) {
      nep->ktol = PetscMax(nep->ktol/2.0,PETSC_MACHINE_EPSILON*10.0);
      ierr = KSPSetTolerances(nep->ksp,nep->ktol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    }

    /* form residual,  r = T(lambda)*u */
    ierr = NEPApplyFunction(nep,lambda,u,delta,r,T,T);CHKERRQ(ierr);

    /* convergence test */
    ierr = VecNorm(r,NORM_2,&relerr);CHKERRQ(ierr);
    nep->errest[nep->nconv] = relerr;
    nep->eigr[nep->nconv] = lambda;
    if (relerr<=nep->rtol) {
      nep->nconv = nep->nconv + 1;
      nep->reason = NEP_CONVERGED_FNORM_RELATIVE;
    }
    ierr = NEPMonitor(nep,nep->its,nep->nconv,nep->eigr,nep->errest,1);CHKERRQ(ierr);

    if (!nep->nconv) {
      /* eigenvector correction: delta = T(sigma)\r */
      ierr = NEP_KSPSolve(nep,r,delta);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(nep->ksp,&kspreason);CHKERRQ(ierr);
      if (kspreason<0) {
        ierr = PetscInfo1(nep,"iter=%D, linear solve failed, stopping solve\n",nep->its);CHKERRQ(ierr);
        nep->reason = NEP_DIVERGED_LINEAR_SOLVE;
        break;
      }

      /* update eigenvector: u = u - delta */
      ierr = VecAXPY(u,-1.0,delta);CHKERRQ(ierr);

      /* normalize eigenvector */
      ierr = VecNormalize(u,NULL);CHKERRQ(ierr);

      /* correct eigenvalue: lambda = lambda - (u'*T*u)/(u'*Tp*u) */
      ierr = NEPApplyFunction(nep,lambda,u,delta,r,T,T);CHKERRQ(ierr);
      ierr = VecDot(u,r,&a1);CHKERRQ(ierr);
      ierr = NEPApplyJacobian(nep,lambda,u,delta,r,Tp);CHKERRQ(ierr);
      ierr = VecDot(u,r,&a2);CHKERRQ(ierr);
      lambda = lambda - a1/a2;
    }
    if (nep->its >= nep->max_it) nep->reason = NEP_DIVERGED_MAX_IT;
  }
  ierr = MatDestroy(&Tsigma);CHKERRQ(ierr);
  ierr = BVRestoreColumn(nep->V,0,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCreate_RII"
PETSC_EXTERN PetscErrorCode NEPCreate_RII(NEP nep)
{
  PetscFunctionBegin;
  nep->ops->solve        = NEPSolve_RII;
  nep->ops->setup        = NEPSetUp_RII;
  PetscFunctionReturn(0);
}

