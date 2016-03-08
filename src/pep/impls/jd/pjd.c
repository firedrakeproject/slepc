/*

   SLEPc polynomial eigensolver: "jd"

   Method: Jacobi-Davidson

   Algorithm:

       Jacobi-Davidson for polynomial eigenvalue problems.
       Based on code contributed by the authors of [2] below.

   References:

       [1] G.L.G. Sleijpen et al., "Jacobi-Davidson type methods for
           generalized eigenproblems and polynomial eigenproblems", BIT
           36(3):595-633, 1996.

       [2] Feng-Nan Hwang, Zih-Hao Wei, Tsung-Ming Huang, Weichung Wang,
           "A Parallel Additive Schwarz Preconditioned Jacobi-Davidson
           Algorithm for Polynomial Eigenvalue Problems in Quantum Dot
           Simulation", J. Comput. Phys. 229(8):2932-2947, 2010.

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

#include <slepc/private/pepimpl.h>    /*I "slepcpep.h" I*/
#include <slepc/private/dsimpl.h>
#include "pjdp.h"

#undef __FUNCT__
#define __FUNCT__ "PEPSetUp_JD"
PetscErrorCode PEPSetUp_JD(PEP pep)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscBool      isshift,flg;
  PetscInt       i;

  PetscFunctionBegin;
  pep->lineariz = PETSC_FALSE;
  ierr = PEPSetDimensions_Default(pep,pep->nev,&pep->ncv,&pep->mpd);CHKERRQ(ierr);
  if (!pep->max_it) pep->max_it = PetscMax(100,2*pep->n/pep->ncv);
  if (!pep->which) pep->which = PEP_LARGEST_MAGNITUDE;
  if (pep->nev>1) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Polynomial JD does not support nev>1 yet");

  /* Set STSHIFT as the default ST */
  if (!((PetscObject)pep->st)->type_name) {
    ierr = STSetType(pep->st,STSHIFT);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSHIFT,&isshift);CHKERRQ(ierr);
  if (!isshift) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"JD only works with shift spectral transformation");

  if (pep->basis!=PEP_BASIS_MONOMIAL) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver not implemented for non-monomial bases");
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver requires the ST transformation flag unset, see STSetTransform()");

  if (!pjd->keep) pjd->keep = 0.5;

  ierr = PEPAllocateSolution(pep,0);CHKERRQ(ierr);
  ierr = PEPSetWorkVecs(pep,4);CHKERRQ(ierr);
  ierr = PetscMalloc1(pep->nmat,&pjd->W);CHKERRQ(ierr);
  for (i=0;i<pep->nmat;i++) {
    ierr = BVDuplicate(pep->V,pjd->W+i);CHKERRQ(ierr);
  }
  ierr = DSSetType(pep->ds,DSPEP);CHKERRQ(ierr);
  ierr = DSPEPSetDegree(pep->ds,pep->nmat-1);CHKERRQ(ierr);
  ierr = DSAllocate(pep->ds,pep->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDPurgeDuplicates"
/*
   Check for multiple eigenvalues.
*/
static PetscErrorCode PEPJDPurgeDuplicates(PEP pep)
{
  PEP_JD   *pjd = (PEP_JD*)pep->data;
  PetscInt i,k;

  PetscFunctionBegin;
  k = pep->nconv;  /* TODO: should have a while loop here */
  for (i=0;i<pep->nconv;i++) {
    if (SlepcAbsEigenvalue(pep->eigr[i]-pep->eigr[k],pep->eigi[i]-pep->eigi[k])<pjd->mtol) {
      pep->eigr[k] = PETSC_INFINITY;
      pep->eigi[k] = PETSC_INFINITY;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDDiffMatMult"
/*
   Multiplication of derivative of P, i.e.
      P'(\lambda) x = \sum_{i=1}^{n} (i*\lambda^{i-1} A_i)x 
*/
static PetscErrorCode PEPJDDiffMatMult(PEP pep,PetscScalar theta,Vec x,Vec y,Vec w)
{
  PetscErrorCode ierr;
  PetscScalar    fact=1.0;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  for (i=1;i<pep->nmat;i++) {
    ierr = MatMult(pep->A[i],x,w);CHKERRQ(ierr);
    ierr = VecAXPY(y,fact*(PetscReal)i,w);CHKERRQ(ierr);
    fact *= theta;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCShellApply_PEPJD"
/*
   Application of shell preconditioner:
      y = B\x - eta*B\p,  with eta = (u'*B\x)/(u'*B\p)
*/
static PetscErrorCode PCShellApply_PEPJD(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PetscScalar    eta;
  PEP_JD_PCSHELL *pcctx;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&pcctx);CHKERRQ(ierr);

  /* y = B\x */
  ierr = PCApply(pcctx->pc,x,y);CHKERRQ(ierr);

  /* Compute eta = u'*y / u'*Bp */
  ierr = VecDot(y,pcctx->u,&eta);CHKERRQ(ierr);
  eta /= pcctx->gamma;
  
  /* y = y - eta*Bp */
  ierr = VecAXPY(y,-eta,pcctx->Bp);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSolve_JD"
PetscErrorCode PEPSolve_JD(PEP pep)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PEP_JD_PCSHELL *pcctx;
  PetscInt       k,nv,ld,minv,low,high;
  PetscScalar    theta,*pX;
  PetscReal      norm;
  PetscBool      lindep;
  Vec            t,u=pep->work[0],p=pep->work[1],r=pep->work[2],w=pep->work[3];
  Mat            G,X,Ptheta;
  KSP            ksp;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(pep->ds,&ld);CHKERRQ(ierr);
  if (pep->nini==0) {  
    nv = 1;
    ierr = BVSetRandomColumn(pep->V,0,pep->rand);CHKERRQ(ierr);
    ierr = BVNormColumn(pep->V,0,NORM_2,&norm);CHKERRQ(ierr);
    ierr = BVScaleColumn(pep->V,0,1.0/norm);CHKERRQ(ierr);
  } else nv = pep->nini;

  /* Restart loop */
  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;

    low = (pjd->flglk || pjd->flgre)? 0: nv-1;
    high = nv;
    ierr = DSSetDimensions(pep->ds,nv,0,0,0);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pep->V,low,high);CHKERRQ(ierr);
    for (k=0;k<pep->nmat;k++) {
      ierr = BVSetActiveColumns(pjd->W[k],low,high);CHKERRQ(ierr);
      ierr = BVMatMult(pep->V,pep->A[k],pjd->W[k]);CHKERRQ(ierr);
      ierr = DSGetMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
      ierr = BVMatProject(pjd->W[k],NULL,pep->V,G);CHKERRQ(ierr);
      ierr = DSRestoreMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(pep->V,0,nv);CHKERRQ(ierr);

    /* Solve projected problem */
    ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    ierr = DSSolve(pep->ds,pep->eigr+pep->nconv,pep->eigi+pep->nconv);CHKERRQ(ierr);
    ierr = DSSort(pep->ds,pep->eigr+pep->nconv,pep->eigi+pep->nconv,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = PEPJDPurgeDuplicates(pep);CHKERRQ(ierr);
    ierr = DSSort(pep->ds,pep->eigr+pep->nconv,pep->eigi+pep->nconv,NULL,NULL,NULL);CHKERRQ(ierr);
    theta = pep->eigr[pep->nconv];
#if !defined(PETSC_USE_COMPLEX)
    if (PetscAbsScalar(pep->eigi[pep->nconv])!=0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PJD solver not implemented for complex Ritz values in real arithmetic");
#endif

    /* Compute Ritz vector u=V*X(:,1) */
    ierr = DSGetArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(pep->V,0,nv);CHKERRQ(ierr);
    ierr = BVMultVec(pep->V,1.0,0.0,u,pX);CHKERRQ(ierr);
    ierr = DSRestoreArray(pep->ds,DS_MAT_X,&pX);CHKERRQ(ierr);

    /* Compute p=P'(theta)*u  */
    ierr = PEPJDDiffMatMult(pep,theta,u,p,w);CHKERRQ(ierr);

    /* Form matrix P(theta) and compute residual r=P(theta)*u */
    ierr = STMatSetUp(pep->st,theta,NULL);CHKERRQ(ierr);
    ierr = STGetKSP(pep->st,&ksp);CHKERRQ(ierr);
    ierr = KSPGetOperators(ksp,&Ptheta,NULL);CHKERRQ(ierr);
    ierr = MatMult(Ptheta,u,r);CHKERRQ(ierr);

    /* Replace preconditioner with one containing projectors */
    if (!pjd->pcshell) {
      ierr = PCCreate(PetscObjectComm((PetscObject)ksp),&pjd->pcshell);CHKERRQ(ierr);
      ierr = PCSetType(pjd->pcshell,PCSHELL);CHKERRQ(ierr);
      ierr = PCShellSetName(pjd->pcshell,"PCPEPJD");
      ierr = PCShellSetApply(pjd->pcshell,PCShellApply_PEPJD);CHKERRQ(ierr);
      ierr = PetscNew(&pcctx);CHKERRQ(ierr);
      ierr = PCShellSetContext(pjd->pcshell,pcctx);CHKERRQ(ierr);
      ierr = PCSetOperators(pjd->pcshell,Ptheta,Ptheta);CHKERRQ(ierr);
      ierr = VecDuplicate(u,&pcctx->Bp);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pcctx->pc);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)pcctx->pc);CHKERRQ(ierr);
    } else {
      ierr = KSPGetPC(ksp,&pcctx->pc);CHKERRQ(ierr);
    }
    ierr = KSPSetPC(ksp,pjd->pcshell);CHKERRQ(ierr);
    pcctx->u = u;

    /* Check convergence */
    ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
    ierr = (*pep->converged)(pep,theta,0,norm,&pep->errest[pep->nconv],pep->convergedctx);CHKERRQ(ierr);
    if (pep->its >= pep->max_it) pep->reason = PEP_DIVERGED_ITS;

    if (pep->errest[pep->nconv]<pep->tol) {

      /* Ritz pair converged */
      minv = PetscMin(nv,(PetscInt)pjd->keep*pep->ncv);
      ierr = DSOrthogonalize(pep->ds,DS_MAT_X,nv,NULL);CHKERRQ(ierr);
      ierr = DSGetMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = BVMultInPlace(pep->V,X,pep->nconv,minv);CHKERRQ(ierr);
      ierr = DSRestoreMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      pep->nconv++;
      if (pep->nconv >= pep->nev) pep->reason = PEP_CONVERGED_TOL;
      else nv = minv + pep->nconv;
      pjd->flglk = PETSC_TRUE;

    } else if (nv==pep->ncv-1) {

      /* Basis full, force restart */
      minv = PetscMin(nv,(PetscInt)pjd->keep*pep->ncv);
      ierr = DSOrthogonalize(pep->ds,DS_MAT_X,nv,NULL);CHKERRQ(ierr);
      ierr = DSGetMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = BVMultInPlace(pep->V,X,pep->nconv,minv);CHKERRQ(ierr);
      ierr = DSRestoreMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      nv = minv + pep->nconv;
      pjd->flgre = PETSC_TRUE;

    } else {

      /* Solve correction equation to expand basis */
      ierr = PCApply(pcctx->pc,p,pcctx->Bp);CHKERRQ(ierr);
      ierr = VecScale(r,-1.0);CHKERRQ(ierr);
      ierr = VecDot(pcctx->Bp,u,&pcctx->gamma);CHKERRQ(ierr);
      ierr = BVGetColumn(pep->V,nv,&t);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,r,t);CHKERRQ(ierr);
      ierr = BVRestoreColumn(pep->V,nv,&t);CHKERRQ(ierr);
      ierr = BVOrthogonalizeColumn(pep->V,nv,NULL,&norm,&lindep);CHKERRQ(ierr);
      if (lindep) SETERRQ(PETSC_COMM_SELF,1,"Linearly dependent continuation vector");
      ierr = BVScaleColumn(pep->V,nv,1.0/norm);CHKERRQ(ierr);
      nv++;
      pjd->flglk = PETSC_FALSE;
      pjd->flgre = PETSC_FALSE;
    }

    /* Restore preconditioner */
    ierr = KSPGetPC(ksp,&pjd->pcshell);CHKERRQ(ierr);
    ierr = KSPSetPC(ksp,pcctx->pc);CHKERRQ(ierr);

    ierr = PEPMonitor(pep,pep->its,pep->nconv,pep->eigr,pep->eigi,pep->errest,nv);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&pcctx->Bp);CHKERRQ(ierr);
  ierr = PCDestroy(&pcctx->pc);CHKERRQ(ierr);
  ierr = PetscFree(pcctx);CHKERRQ(ierr);
  ierr = PCDestroy(&pjd->pcshell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPComputeVectors_JD"
PetscErrorCode PEPComputeVectors_JD(PEP pep)
{
  PetscErrorCode ierr;
  PetscInt       k;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  Mat            G,X;

  PetscFunctionBegin;
  ierr = DSSetDimensions(pep->ds,pep->nconv,0,0,0);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(pep->V,0,pep->nconv);CHKERRQ(ierr);
  for (k=0;k<pep->nmat;k++) {
    ierr = BVSetActiveColumns(pjd->W[k],0,pep->nconv);CHKERRQ(ierr);
    ierr = BVMatMult(pep->V,pep->A[k],pjd->W[k]);CHKERRQ(ierr);
    ierr = DSGetMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
    ierr = BVMatProject(pjd->W[k],NULL,pep->V,G);CHKERRQ(ierr);
    ierr = DSRestoreMat(pep->ds,DSMatExtra[k],&G);CHKERRQ(ierr);
  }

  /* Solve projected problem */
  ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
  ierr = DSSolve(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
  ierr = DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);

  /* Compute Ritz vectors */
  ierr = DSGetMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = BVMultInPlace(pep->V,X,0,pep->nconv);CHKERRQ(ierr);
  ierr = DSRestoreMat(pep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPReset_JD"
PetscErrorCode PEPReset_JD(PEP pep)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0;i<pep->nmat;i++) {
    ierr = BVDestroy(pjd->W+i);CHKERRQ(ierr);
  }
  ierr = PetscFree(pjd->W);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPDestroy_JD"
PetscErrorCode PEPDestroy_JD(PEP pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetTolerances_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetTolerances_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPCreate_JD"
PETSC_EXTERN PetscErrorCode PEPCreate_JD(PEP pep)
{
  PEP_JD         *pjd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pep,&pjd);CHKERRQ(ierr);
  pep->data = (void*)pjd;

  pjd->keep = 0;
  pjd->mtol = 1e-5;
  pjd->htol = 1e-2;
  pjd->stol = 1e-2;

  pep->ops->solve          = PEPSolve_JD;
  pep->ops->setup          = PEPSetUp_JD;
  pep->ops->setfromoptions = PEPSetFromOptions_JD;
  pep->ops->reset          = PEPReset_JD;
  pep->ops->destroy        = PEPDestroy_JD;
  pep->ops->view           = PEPView_JD;
  pep->ops->computevectors = PEPComputeVectors_JD;
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetRestart_C",PEPJDSetRestart_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetRestart_C",PEPJDGetRestart_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetTolerances_C",PEPJDSetTolerances_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetTolerances_C",PEPJDGetTolerances_JD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

