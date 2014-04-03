/*

   SLEPc nonlinear eigensolver: "narnoldi"

   Method: Nonlinear Arnoldi

   Algorithm:

       Arnoldi for nonlinear eigenproblems.

   References:

       [1] H. Voss, "An Arnoldi method for nonlinear eigenvalue problems",
           BIT 44:387-401, 2004.

   Last update: Mar 2013

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/nepimpl.h>         /*I "slepcnep.h" I*/

#undef __FUNCT__
#define __FUNCT__ "NEPSetUp_NARNOLDI"
PetscErrorCode NEPSetUp_NARNOLDI(NEP nep)
{
  PetscErrorCode ierr;

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
  if (!nep->max_it) nep->max_it = PetscMax(5000,2*nep->n/nep->ncv);
  if (!nep->max_funcs) nep->max_funcs = nep->max_it;
  if (!nep->split) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"NARNOLDI only available for split operator");

  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  ierr = NEPSetWorkVecs(nep,3);CHKERRQ(ierr);

  /* set-up DS and transfer split operator functions */
  ierr = DSSetType(nep->ds,DSNEP);CHKERRQ(ierr);
  ierr = DSSetFN(nep->ds,nep->nt,nep->f);CHKERRQ(ierr);
  ierr = DSAllocate(nep->ds,nep->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSolve_NARNOLDI"
PetscErrorCode NEPSolve_NARNOLDI(NEP nep)
{
  PetscErrorCode     ierr;
  Mat                T=nep->function,Tsigma;
  Vec                f,u=nep->V[0],r=nep->work[0],x=nep->work[1],w=nep->work[2];
  PetscScalar        *X,lambda;
  PetscReal          beta,resnorm=0.0;
  PetscInt           n;
  PetscBool          breakdown;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  /* get initial space and shift */
  ierr = NEPGetDefaultShift(nep,&lambda);CHKERRQ(ierr);
  if (!nep->nini) {
    ierr = SlepcVecSetRandom(u,nep->rand);CHKERRQ(ierr);
    ierr = VecNormalize(u,NULL);CHKERRQ(ierr);
    n = 1;
  } else n = nep->nini;

  /* build projected matrices for initial space */
  ierr = NEPProjectOperator(nep,0,n,r);CHKERRQ(ierr);

  /* prepare linear solver */
  ierr = NEPComputeFunction(nep,lambda,T,T);CHKERRQ(ierr);
  ierr = MatDuplicate(T,MAT_COPY_VALUES,&Tsigma);CHKERRQ(ierr);
  ierr = KSPSetOperators(nep->ksp,Tsigma,Tsigma);CHKERRQ(ierr);

  /* Restart loop */
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;

    /* solve projected problem */
    ierr = DSSetDimensions(nep->ds,n,0,0,0);CHKERRQ(ierr);
    ierr = DSSetState(nep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    ierr = DSSolve(nep->ds,nep->eig,NULL);CHKERRQ(ierr);
    lambda = nep->eig[0];

    /* compute Ritz vector, x = V*s */
    ierr = DSGetArray(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
    ierr = SlepcVecMAXPBY(x,0.0,1.0,n,X,nep->V);CHKERRQ(ierr);
    ierr = DSRestoreArray(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);

    /* compute the residual, r = T(lambda)*x */
    ierr = NEPApplyFunction(nep,lambda,x,w,r,NULL,NULL);CHKERRQ(ierr);

    /* convergence test */
    ierr = VecNorm(r,NORM_2,&resnorm);CHKERRQ(ierr);
    nep->errest[nep->nconv] = resnorm;
    if (resnorm<=nep->rtol) {
      ierr = VecCopy(x,nep->V[nep->nconv]);CHKERRQ(ierr);
      nep->nconv = nep->nconv + 1;
      nep->reason = NEP_CONVERGED_FNORM_RELATIVE;
    }
    ierr = NEPMonitor(nep,nep->its,nep->nconv,nep->eig,nep->errest,1);CHKERRQ(ierr);

    if (nep->reason == NEP_CONVERGED_ITERATING) {

      /* continuation vector: f = T(sigma)\r */
      f = nep->V[n];
      ierr = NEP_KSPSolve(nep,r,f);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(nep->ksp,&kspreason);CHKERRQ(ierr);
      if (kspreason<0) {
        ierr = PetscInfo1(nep,"iter=%D, linear solve failed, stopping solve\n",nep->its);CHKERRQ(ierr);
        nep->reason = NEP_DIVERGED_LINEAR_SOLVE;
        break;
      }

      /* orthonormalize */
      ierr = IPOrthogonalize(nep->ip,0,NULL,n,NULL,nep->V,f,NULL,&beta,&breakdown);CHKERRQ(ierr);
      if (breakdown || beta==0.0) {
        ierr = PetscInfo1(nep,"iter=%D, orthogonalization failed, stopping solve\n",nep->its);CHKERRQ(ierr);
        nep->reason = NEP_DIVERGED_BREAKDOWN;
        break;
      }
      ierr = VecScale(f,1.0/beta);CHKERRQ(ierr);

      /* update projected matrices */
      ierr = NEPProjectOperator(nep,n,n+1,r);CHKERRQ(ierr);
      n++;
    }
    if (nep->its >= nep->max_it) nep->reason = NEP_DIVERGED_MAX_IT;
  }
  ierr = MatDestroy(&Tsigma);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCreate_NARNOLDI"
PETSC_EXTERN PetscErrorCode NEPCreate_NARNOLDI(NEP nep)
{
  PetscFunctionBegin;
  nep->ops->solve          = NEPSolve_NARNOLDI;
  nep->ops->setup          = NEPSetUp_NARNOLDI;
  nep->ops->reset          = NEPReset_Default;
  PetscFunctionReturn(0);
}

