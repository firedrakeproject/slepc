/*
   Explicit linearization for polynomial eigenproblems.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/pepimpl.h>         /*I "slepcpep.h" I*/
#include "linearp.h"

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_Shift"
static PetscErrorCode MatMult_Linear_Shift(Mat M,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  PEP               pep;
  const PetscScalar *px;
  PetscScalar       *py,a,sigma=0.0;
  PetscInt          nmat,deg,i,m;
  Vec               x1,x2,x3,y1,aux;
  PetscReal         *ca,*cb,*cg;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
  pep = ctx->pep;
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = STGetShift(pep->st,&sigma);CHKERRQ(ierr);
  }
  nmat = pep->nmat;
  deg = nmat-1;
  m = pep->nloc;
  ca = pep->pbc;
  cb = pep->pbc+nmat;
  cg = pep->pbc+2*nmat;
  x1=ctx->w[0];x2=ctx->w[1];x3=ctx->w[2];y1=ctx->w[3];aux=ctx->w[4];
  
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  a = 1.0;

  /* first block */
  ierr = VecPlaceArray(x2,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(x3,px+m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  ierr = VecAXPY(y1,cb[0]-sigma,x2);CHKERRQ(ierr);
  ierr = VecAXPY(y1,ca[0],x3);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(x3);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);

  /* inner blocks */
  for (i=1;i<deg-1;i++) {
    ierr = VecPlaceArray(x1,px+(i-1)*m);CHKERRQ(ierr);
    ierr = VecPlaceArray(x2,px+i*m);CHKERRQ(ierr);
    ierr = VecPlaceArray(x3,px+(i+1)*m);CHKERRQ(ierr);
    ierr = VecPlaceArray(y1,py+i*m);CHKERRQ(ierr);
    ierr = VecAXPY(y1,cg[i],x1);CHKERRQ(ierr);
    ierr = VecAXPY(y1,cb[i]-sigma,x2);CHKERRQ(ierr);
    ierr = VecAXPY(y1,ca[i],x3);CHKERRQ(ierr);
    ierr = VecResetArray(x1);CHKERRQ(ierr);
    ierr = VecResetArray(x2);CHKERRQ(ierr);
    ierr = VecResetArray(x3);CHKERRQ(ierr);
    ierr = VecResetArray(y1);CHKERRQ(ierr);
  }

  /* last block */
  ierr = VecPlaceArray(y1,py+(deg-1)*m);CHKERRQ(ierr);
  for (i=0;i<deg;i++) {
    ierr = VecPlaceArray(x1,px+i*m);CHKERRQ(ierr);
    ierr = STMatMult(pep->st,i,x1,aux);CHKERRQ(ierr);
    ierr = VecAXPY(y1,a,aux);CHKERRQ(ierr);
    ierr = VecResetArray(x1);CHKERRQ(ierr);
    a *= pep->sfactor;
  }
  ierr = VecCopy(y1,aux);CHKERRQ(ierr);
  ierr = STMatSolve(pep->st,aux,y1);CHKERRQ(ierr);
  ierr = VecScale(y1,-ca[deg-1]/a);CHKERRQ(ierr);
  ierr = VecPlaceArray(x1,px+(deg-2)*m);CHKERRQ(ierr);
  ierr = VecPlaceArray(x2,px+(deg-1)*m);CHKERRQ(ierr);
  ierr = VecAXPY(y1,cg[deg-1],x1);CHKERRQ(ierr);
  ierr = VecAXPY(y1,cb[deg-1]-sigma,x2);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(x2);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Linear_Sinvert"
static PetscErrorCode MatMult_Linear_Sinvert(Mat M,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PEP_LINEAR        *ctx;
  PEP               pep;
  const PetscScalar *px;
  PetscScalar       *py,a,sigma,t=1.0,tp=0.0,tt;
  PetscInt          nmat,deg,i,m;
  Vec               x1,y1,y2,y3,aux,aux2;
  PetscReal         *ca,*cb,*cg;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
  pep = ctx->pep;
  nmat = pep->nmat;
  deg = nmat-1;
  m = pep->nloc;
  ca = pep->pbc;
  cb = pep->pbc+nmat;
  cg = pep->pbc+2*nmat;
  x1=ctx->w[0];y1=ctx->w[1];y2=ctx->w[2];y3=ctx->w[3];aux=ctx->w[4];aux2=ctx->w[5];
  ierr = EPSGetTarget(ctx->eps,&sigma);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  a = pep->sfactor;

  /* first block */
  ierr = VecPlaceArray(x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(y1,py+m);CHKERRQ(ierr);
  ierr = VecCopy(x1,y1);CHKERRQ(ierr);
  ierr = VecScale(y1,1.0/ca[0]);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(y1);CHKERRQ(ierr);

  /* second block */
  if (deg>2) {
    ierr = VecPlaceArray(x1,px+m);CHKERRQ(ierr);
    ierr = VecPlaceArray(y1,py+m);CHKERRQ(ierr);
    ierr = VecPlaceArray(y2,py+2*m);CHKERRQ(ierr);
    ierr = VecCopy(x1,y2);CHKERRQ(ierr);
    ierr = VecAXPY(y2,sigma-cb[1],y1);CHKERRQ(ierr);
    ierr = VecScale(y2,1.0/ca[1]);CHKERRQ(ierr);
    ierr = VecResetArray(x1);CHKERRQ(ierr);
    ierr = VecResetArray(y1);CHKERRQ(ierr);
    ierr = VecResetArray(y2);CHKERRQ(ierr);
  }

  /* inner blocks */
  for (i=2;i<deg-1;i++) {
    ierr = VecPlaceArray(x1,px+i*m);CHKERRQ(ierr);
    ierr = VecPlaceArray(y1,py+(i-1)*m);CHKERRQ(ierr);
    ierr = VecPlaceArray(y2,py+i*m);CHKERRQ(ierr);
    ierr = VecPlaceArray(y3,py+(i+1)*m);CHKERRQ(ierr);
    ierr = VecCopy(x1,y3);CHKERRQ(ierr);
    ierr = VecAXPY(y3,sigma-cb[i],y2);CHKERRQ(ierr);
    ierr = VecAXPY(y3,-cg[i],y1);CHKERRQ(ierr);
    ierr = VecScale(y3,1.0/ca[i]);CHKERRQ(ierr);
    ierr = VecResetArray(x1);CHKERRQ(ierr);
    ierr = VecResetArray(y1);CHKERRQ(ierr);
    ierr = VecResetArray(y2);CHKERRQ(ierr);
    ierr = VecResetArray(y3);CHKERRQ(ierr);
  }

  /* last block */
  ierr = VecPlaceArray(y1,py);CHKERRQ(ierr);
  for (i=0;i<deg-2;i++) {
    ierr = VecPlaceArray(y2,py+(i+1)*m);CHKERRQ(ierr);
    ierr = STMatMult(pep->st,i+1,y2,aux);CHKERRQ(ierr);
    ierr = VecAXPY(y1,a,aux);CHKERRQ(ierr);
    ierr = VecResetArray(y2);CHKERRQ(ierr);
    a *= pep->sfactor;
  }
  i = deg-2;
  ierr = VecPlaceArray(y2,py+(i+1)*m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y3,py+i*m);CHKERRQ(ierr);
  ierr = VecCopy(y2,aux2);CHKERRQ(ierr);
  ierr = VecAXPY(aux2,cg[i+1]/ca[i+1],y3);CHKERRQ(ierr);
  ierr = STMatMult(pep->st,i+1,aux2,aux);CHKERRQ(ierr);
  ierr = VecAXPY(y1,a,aux);CHKERRQ(ierr);
  ierr = VecResetArray(y2);CHKERRQ(ierr);
  ierr = VecResetArray(y3);CHKERRQ(ierr);
  a *= pep->sfactor;
  i = deg-1;
  ierr = VecPlaceArray(x1,px+i*m);CHKERRQ(ierr);
  ierr = VecPlaceArray(y3,py+i*m);CHKERRQ(ierr);
  ierr = VecCopy(x1,aux2);CHKERRQ(ierr);
  ierr = VecAXPY(aux2,sigma-cb[i],y3);CHKERRQ(ierr);
  ierr = VecScale(aux2,1.0/ca[i]);CHKERRQ(ierr);
  ierr = STMatMult(pep->st,i+1,aux2,aux);CHKERRQ(ierr);
  ierr = VecAXPY(y1,a,aux);CHKERRQ(ierr);
  ierr = VecResetArray(x1);CHKERRQ(ierr);
  ierr = VecResetArray(y3);CHKERRQ(ierr);

  ierr = VecCopy(y1,aux);CHKERRQ(ierr);
  ierr = STMatSolve(pep->st,aux,y1);CHKERRQ(ierr);
  ierr = VecScale(y1,-1.0);CHKERRQ(ierr);

  /* final update */
  for (i=1;i<deg;i++) {
    ierr = VecPlaceArray(y2,py+i*m);CHKERRQ(ierr);
    tt = t;
    t = ((sigma-cb[i-1])*t-cg[i-1]*tp)/ca[i-1]; /* i-th basis polynomial */
    tp = tt;
    ierr = VecAXPY(y2,t,y1);CHKERRQ(ierr);
    ierr = VecResetArray(y2);CHKERRQ(ierr);
  }
  ierr = VecResetArray(y1);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BackTransform_Linear"
static PetscErrorCode BackTransform_Linear(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;
  ST             stctx;

  PetscFunctionBegin;
  ierr = STShellGetContext(st,(void**)&ctx);CHKERRQ(ierr);
  ierr = PEPGetST(ctx->pep,&stctx);CHKERRQ(ierr);
  ierr = STBackTransform(stctx,n,eigr,eigi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Apply_Linear"
static PetscErrorCode Apply_Linear(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;

  PetscFunctionBegin;
  ierr = STShellGetContext(st,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatMult(ctx->A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetUp_Linear"
PetscErrorCode PEPSetUp_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  ST             st;
  PetscInt       i=0,deg=pep->nmat-1;
  EPSWhich       which;
  EPSProblemType ptype;
  PetscBool      trackall,istrivial,transf,shift,sinv,ks;
  PetscScalar    sigma,*epsarray,*peparray;
  Vec            veps;
  /* function tables */
  PetscErrorCode (*fcreate[][2])(MPI_Comm,PEP_LINEAR*,Mat*) = {
    { MatCreateExplicit_Linear_N1A, MatCreateExplicit_Linear_N1B },   /* N1 */
    { MatCreateExplicit_Linear_N2A, MatCreateExplicit_Linear_N2B },   /* N2 */
    { MatCreateExplicit_Linear_S1A, MatCreateExplicit_Linear_S1B },   /* S1 */
    { MatCreateExplicit_Linear_S2A, MatCreateExplicit_Linear_S2B },   /* S2 */
    { MatCreateExplicit_Linear_H1A, MatCreateExplicit_Linear_H1B },   /* H1 */
    { MatCreateExplicit_Linear_H2A, MatCreateExplicit_Linear_H2B }    /* H2 */
  };

  PetscFunctionBegin;
  if (pep->stopping!=PEPStoppingBasic) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"User-defined stopping test not supported");
  pep->lineariz = PETSC_TRUE;
  if (!ctx->cform) ctx->cform = 1;
  ierr = STGetTransform(pep->st,&transf);CHKERRQ(ierr);
  /* Set STSHIFT as the default ST */
  if (!((PetscObject)pep->st)->type_name) {
    ierr = STSetType(pep->st,STSHIFT);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSHIFT,&shift);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv);CHKERRQ(ierr);
  if (!shift && !sinv) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Only STSHIFT and STSINVERT spectral transformations can be used");
  if (!pep->which) {
    if (sinv) pep->which = PEP_TARGET_MAGNITUDE;
    else pep->which = PEP_LARGEST_MAGNITUDE;
  }
  ierr = STSetUp(pep->st);CHKERRQ(ierr);
  if (!ctx->eps) { ierr = PEPLinearGetEPS(pep,&ctx->eps);CHKERRQ(ierr); }
  ierr = EPSGetST(ctx->eps,&st);CHKERRQ(ierr);
  if (!transf) { ierr = EPSSetTarget(ctx->eps,pep->target);CHKERRQ(ierr); }
  if (sinv && !transf) { ierr = STSetDefaultShift(st,pep->target);CHKERRQ(ierr); }
  /* compute scale factor if not set by user */
  ierr = PEPComputeScaleFactor(pep);CHKERRQ(ierr);

  if (ctx->explicitmatrix) {
    if (transf) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Explicit matrix option is not implemented with st-transform flag active");
    if (pep->nmat!=3) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Explicit matrix option only available for quadratic problems");
    if (pep->basis!=PEP_BASIS_MONOMIAL) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Explicit matrix option not implemented for non-monomial bases");
    if (pep->scale==PEP_SCALE_DIAGONAL || pep->scale==PEP_SCALE_BOTH) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Diagonal scaling not allowed in PEPLINEAR with explicit matrices");
    if (sinv && !transf) { ierr = STSetType(st,STSINVERT);CHKERRQ(ierr); }
    ierr = RGPushScale(pep->rg,1.0/pep->sfactor);CHKERRQ(ierr);
    ierr = STGetTOperators(pep->st,0,&ctx->K);CHKERRQ(ierr);
    ierr = STGetTOperators(pep->st,1,&ctx->C);CHKERRQ(ierr);
    ierr = STGetTOperators(pep->st,2,&ctx->M);CHKERRQ(ierr);
    ctx->sfactor = pep->sfactor;
    ctx->dsfactor = pep->dsfactor;
  
    ierr = MatDestroy(&ctx->A);CHKERRQ(ierr);
    ierr = MatDestroy(&ctx->B);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->w[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->w[1]);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->w[2]);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->w[3]);CHKERRQ(ierr);
  
    switch (pep->problem_type) {
      case PEP_GENERAL:    i = 0; break;
      case PEP_HERMITIAN:  i = 2; break;
      case PEP_GYROSCOPIC: i = 4; break;
      default: SETERRQ(PetscObjectComm((PetscObject)pep),1,"Wrong value of pep->problem_type");
    }
    i += ctx->cform-1;

    ierr = (*fcreate[i][0])(PetscObjectComm((PetscObject)pep),ctx,&ctx->A);CHKERRQ(ierr);
    ierr = (*fcreate[i][1])(PetscObjectComm((PetscObject)pep),ctx,&ctx->B);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->A);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->B);CHKERRQ(ierr);

  } else {   /* implicit matrix */
    if (pep->problem_type!=PEP_GENERAL) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Must use the explicit matrix option if problem type is not general");
    if (!((PetscObject)(ctx->eps))->type_name) {
      ierr = EPSSetType(ctx->eps,EPSKRYLOVSCHUR);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectTypeCompare((PetscObject)ctx->eps,EPSKRYLOVSCHUR,&ks);CHKERRQ(ierr);
      if (!ks) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Implicit matrix option only implemented for Krylov-Schur");
    }
    if (ctx->cform!=1) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Implicit matrix option not available for 2nd companion form");
    ierr = STSetType(st,STSHELL);CHKERRQ(ierr);
    ierr = STShellSetContext(st,(PetscObject)ctx);CHKERRQ(ierr);
    if (!transf) { ierr = STShellSetBackTransform(st,BackTransform_Linear);CHKERRQ(ierr); }
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&ctx->w[0]);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&ctx->w[1]);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&ctx->w[2]);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&ctx->w[3]);CHKERRQ(ierr);
    ierr = MatCreateVecs(pep->A[0],&ctx->w[4],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(pep->A[0],&ctx->w[5],NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(pep,6,ctx->w);CHKERRQ(ierr);
    ierr = MatCreateShell(PetscObjectComm((PetscObject)pep),deg*pep->nloc,deg*pep->nloc,deg*pep->n,deg*pep->n,ctx,&ctx->A);CHKERRQ(ierr);
    if (sinv && !transf) {
      ierr = MatShellSetOperation(ctx->A,MATOP_MULT,(void(*)(void))MatMult_Linear_Sinvert);CHKERRQ(ierr);
    } else {
      ierr = MatShellSetOperation(ctx->A,MATOP_MULT,(void(*)(void))MatMult_Linear_Shift);CHKERRQ(ierr);
    }
    ierr = STShellSetApply(st,Apply_Linear);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->A);CHKERRQ(ierr);
    ctx->pep = pep;

    ierr = PEPBasisCoefficients(pep,pep->pbc);CHKERRQ(ierr);
    if (!transf) {
      ierr = PetscMalloc1(pep->nmat,&pep->solvematcoeffs);CHKERRQ(ierr);
      if (sinv) {
        ierr = PEPEvaluateBasis(pep,pep->target,0,pep->solvematcoeffs,NULL);CHKERRQ(ierr);
      } else {
        for (i=0;i<deg;i++) pep->solvematcoeffs[i] = 0.0;
        pep->solvematcoeffs[deg] = 1.0;
      }
      ierr = STScaleShift(pep->st,1.0/pep->sfactor);CHKERRQ(ierr);
      ierr = RGPushScale(pep->rg,1.0/pep->sfactor);CHKERRQ(ierr);
    }
    if (pep->sfactor!=1.0) {
      for (i=0;i<pep->nmat;i++) {
        pep->pbc[pep->nmat+i] /= pep->sfactor;
        pep->pbc[2*pep->nmat+i] /= pep->sfactor*pep->sfactor; 
      }
    }
  }

  ierr = EPSSetOperators(ctx->eps,ctx->A,ctx->B);CHKERRQ(ierr);
  ierr = EPSGetProblemType(ctx->eps,&ptype);CHKERRQ(ierr);
  if (!ptype) {
    if (ctx->explicitmatrix) {
      ierr = EPSSetProblemType(ctx->eps,EPS_GNHEP);CHKERRQ(ierr);
    } else {
      ierr = EPSSetProblemType(ctx->eps,EPS_NHEP);CHKERRQ(ierr);
    }
  }
  if (transf) which = EPS_LARGEST_MAGNITUDE;
  else {
    switch (pep->which) {
        case PEP_LARGEST_MAGNITUDE:  which = EPS_LARGEST_MAGNITUDE; break;
        case PEP_SMALLEST_MAGNITUDE: which = EPS_SMALLEST_MAGNITUDE; break;
        case PEP_LARGEST_REAL:       which = EPS_LARGEST_REAL; break;
        case PEP_SMALLEST_REAL:      which = EPS_SMALLEST_REAL; break;
        case PEP_LARGEST_IMAGINARY:  which = EPS_LARGEST_IMAGINARY; break;
        case PEP_SMALLEST_IMAGINARY: which = EPS_SMALLEST_IMAGINARY; break;
        case PEP_TARGET_MAGNITUDE:   which = EPS_TARGET_MAGNITUDE; break;
        case PEP_TARGET_REAL:        which = EPS_TARGET_REAL; break;
        case PEP_TARGET_IMAGINARY:   which = EPS_TARGET_IMAGINARY; break;
        case PEP_WHICH_USER:         which = EPS_WHICH_USER;
          ierr = EPSSetEigenvalueComparison(ctx->eps,pep->sc->comparison,pep->sc->comparisonctx);CHKERRQ(ierr);
          break;
        default: SETERRQ(PetscObjectComm((PetscObject)pep),1,"Wrong value of which");
    }
  }
  ierr = EPSSetWhichEigenpairs(ctx->eps,which);CHKERRQ(ierr);

  ierr = EPSSetDimensions(ctx->eps,pep->nev,pep->ncv?pep->ncv:PETSC_DEFAULT,pep->mpd?pep->mpd:PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSSetTolerances(ctx->eps,pep->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:pep->tol,pep->max_it?pep->max_it:PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = RGIsTrivial(pep->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) {
    if (transf) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"PEPLINEAR does not support a nontrivial region with st-transform");
    ierr = EPSSetRG(ctx->eps,pep->rg);CHKERRQ(ierr);
  }
  /* Transfer the trackall option from pep to eps */
  ierr = PEPGetTrackAll(pep,&trackall);CHKERRQ(ierr);
  ierr = EPSSetTrackAll(ctx->eps,trackall);CHKERRQ(ierr);

  /* temporary change of target */
  if (pep->sfactor!=1.0) {
    ierr = EPSGetTarget(ctx->eps,&sigma);CHKERRQ(ierr);
    ierr = EPSSetTarget(ctx->eps,sigma/pep->sfactor);CHKERRQ(ierr);
  }

  /* process initial vector */
  if (pep->nini<=-deg) {
    ierr = VecCreateMPI(PetscObjectComm((PetscObject)ctx->eps),deg*pep->nloc,deg*pep->n,&veps);CHKERRQ(ierr);
    ierr = VecGetArray(veps,&epsarray);CHKERRQ(ierr);
    for (i=0;i<deg;i++) {
      ierr = VecGetArray(pep->IS[i],&peparray);CHKERRQ(ierr);
      ierr = PetscMemcpy(epsarray+i*pep->nloc,peparray,pep->nloc*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = VecRestoreArray(pep->IS[i],&peparray);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(veps,&epsarray);CHKERRQ(ierr);
    ierr = EPSSetInitialSpace(ctx->eps,1,&veps);CHKERRQ(ierr);
    ierr = VecDestroy(&veps);CHKERRQ(ierr);
  }
  if (pep->nini<0) {
    ierr = SlepcBasisDestroy_Private(&pep->nini,&pep->IS);CHKERRQ(ierr);
  }

  ierr = EPSSetUp(ctx->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(ctx->eps,NULL,&pep->ncv,&pep->mpd);CHKERRQ(ierr);
  ierr = EPSGetTolerances(ctx->eps,NULL,&pep->max_it);CHKERRQ(ierr);
  if (pep->nini>0) { ierr = PetscInfo(pep,"Ignoring initial vectors\n");CHKERRQ(ierr); }
  ierr = PEPAllocateSolution(pep,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearExtract_Residual"
/*
   PEPLinearExtract_Residual - Auxiliary routine that copies the solution of the
   linear eigenproblem to the PEP object. The eigenvector of the generalized
   problem is supposed to be
                               z = [  x  ]
                                   [ l*x ]
   The eigenvector is taken from z(1:n) or z(n+1:2*n) depending on the explicitly
   computed residual norm.
   Finally, x is normalized so that ||x||_2 = 1.
*/
static PetscErrorCode PEPLinearExtract_Residual(PEP pep,EPS eps)
{
  PetscErrorCode    ierr;
  PetscInt          i,k;
  const PetscScalar *px;
  PetscScalar       *er=pep->eigr,*ei=pep->eigi;
  PetscReal         rn1,rn2;
  Vec               xr,xi=NULL,wr;
  Mat               A;
#if !defined(PETSC_USE_COMPLEX)
  Vec               wi;
  const PetscScalar *py;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ierr = PEPSetWorkVecs(pep,2);CHKERRQ(ierr);
#else
  ierr = PEPSetWorkVecs(pep,4);CHKERRQ(ierr);
#endif
  ierr = EPSGetOperators(eps,&A,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&xr,NULL);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&wr);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = VecDuplicate(xr,&xi);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&wi);CHKERRQ(ierr);
#endif
  for (i=0;i<pep->nconv;i++) {
    ierr = EPSGetEigenpair(eps,i,NULL,NULL,xr,xi);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    if (ei[i]!=0.0) {   /* complex conjugate pair */
      ierr = VecGetArrayRead(xr,&px);CHKERRQ(ierr);
      ierr = VecGetArrayRead(xi,&py);CHKERRQ(ierr);
      ierr = VecPlaceArray(wr,px);CHKERRQ(ierr);
      ierr = VecPlaceArray(wi,py);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(wr,wi,PETSC_TRUE,NULL);CHKERRQ(ierr);
      ierr = PEPComputeResidualNorm_Private(pep,er[i],ei[i],wr,wi,pep->work,&rn1);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i,wr);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i+1,wi);CHKERRQ(ierr);
      for (k=1;k<pep->nmat-1;k++) {
        ierr = VecResetArray(wr);CHKERRQ(ierr);
        ierr = VecResetArray(wi);CHKERRQ(ierr);
        ierr = VecPlaceArray(wr,px+k*pep->nloc);CHKERRQ(ierr);
        ierr = VecPlaceArray(wi,py+k*pep->nloc);CHKERRQ(ierr);
        ierr = SlepcVecNormalize(wr,wi,PETSC_TRUE,NULL);CHKERRQ(ierr);
        ierr = PEPComputeResidualNorm_Private(pep,er[i],ei[i],wr,wi,pep->work,&rn2);CHKERRQ(ierr);
        if (rn1>rn2) {
          ierr = BVInsertVec(pep->V,i,wr);CHKERRQ(ierr);
          ierr = BVInsertVec(pep->V,i+1,wi);CHKERRQ(ierr);
          rn1 = rn2;
        }
      }
      ierr = VecResetArray(wr);CHKERRQ(ierr);
      ierr = VecResetArray(wi);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xr,&px);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xi,&py);CHKERRQ(ierr);
      i++;
    } else   /* real eigenvalue */
#endif
    {
      ierr = VecGetArrayRead(xr,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(wr,px);CHKERRQ(ierr);
      ierr = SlepcVecNormalize(wr,NULL,PETSC_FALSE,NULL);CHKERRQ(ierr);
      ierr = PEPComputeResidualNorm_Private(pep,er[i],ei[i],wr,NULL,pep->work,&rn1);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i,wr);CHKERRQ(ierr);
      for (k=1;k<pep->nmat-1;k++) {
        ierr = VecResetArray(wr);CHKERRQ(ierr);
        ierr = VecPlaceArray(wr,px+k*pep->nloc);CHKERRQ(ierr);
        ierr = SlepcVecNormalize(wr,NULL,PETSC_FALSE,NULL);CHKERRQ(ierr);
        ierr = PEPComputeResidualNorm_Private(pep,er[i],ei[i],wr,NULL,pep->work,&rn2);CHKERRQ(ierr);
        if (rn1>rn2) {
          ierr = BVInsertVec(pep->V,i,wr);CHKERRQ(ierr);
          rn1 = rn2;
        }
      }
      ierr = VecResetArray(wr);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xr,&px);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&wr);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = VecDestroy(&wi);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearExtract_None"
/*
   PEPLinearExtract_None - Same as PEPLinearExtract_Norm but always takes
   the first block.
*/
static PetscErrorCode PEPLinearExtract_None(PEP pep,EPS eps)
{
  PetscErrorCode    ierr;
  PetscInt          i;
  const PetscScalar *px;
  Mat               A;
  Vec               xr,xi,w;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar       *ei=pep->eigi;
#endif

  PetscFunctionBegin;
  ierr = EPSGetOperators(eps,&A,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&xr,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(xr,&xi);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&w);CHKERRQ(ierr);
  for (i=0;i<pep->nconv;i++) {
    ierr = EPSGetEigenpair(eps,i,NULL,NULL,xr,xi);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    if (ei[i]!=0.0) {   /* complex conjugate pair */
      ierr = VecGetArrayRead(xr,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xr,&px);CHKERRQ(ierr);
      ierr = VecGetArrayRead(xi,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i+1,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xi,&px);CHKERRQ(ierr);
      i++;
    } else   /* real eigenvalue */
#endif
    {
      ierr = VecGetArrayRead(xr,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xr,&px);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearExtract_Norm"
/*
   PEPLinearExtract_Norm - Auxiliary routine that copies the solution of the
   linear eigenproblem to the PEP object. The eigenvector of the generalized
   problem is supposed to be
                               z = [  x  ]
                                   [ l*x ]
   If |l|<1.0, the eigenvector is taken from z(1:n), otherwise from z(n+1:2*n).
   Finally, x is normalized so that ||x||_2 = 1.
*/
static PetscErrorCode PEPLinearExtract_Norm(PEP pep,EPS eps)
{
  PetscErrorCode    ierr;
  PetscInt          i,offset;
  const PetscScalar *px;
  PetscScalar       *er=pep->eigr;
  Mat               A;
  Vec               xr,xi=NULL,w;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar       *ei=pep->eigi;
#endif

  PetscFunctionBegin;
  ierr = EPSGetOperators(eps,&A,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&xr,NULL);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = VecDuplicate(xr,&xi);CHKERRQ(ierr);
#endif
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pep),1,pep->nloc,pep->n,NULL,&w);CHKERRQ(ierr);
  for (i=0;i<pep->nconv;i++) {
    ierr = EPSGetEigenpair(eps,i,NULL,NULL,xr,xi);CHKERRQ(ierr);
    if (SlepcAbsEigenvalue(er[i],ei[i])>1.0) offset = (pep->nmat-2)*pep->nloc;
    else offset = 0;
#if !defined(PETSC_USE_COMPLEX)
    if (ei[i]!=0.0) {   /* complex conjugate pair */
      ierr = VecGetArrayRead(xr,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px+offset);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xr,&px);CHKERRQ(ierr);
      ierr = VecGetArrayRead(xi,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px+offset);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i+1,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xi,&px);CHKERRQ(ierr);
      i++;
    } else /* real eigenvalue */
#endif
    {
      ierr = VecGetArrayRead(xr,&px);CHKERRQ(ierr);
      ierr = VecPlaceArray(w,px+offset);CHKERRQ(ierr);
      ierr = BVInsertVec(pep->V,i,w);CHKERRQ(ierr);
      ierr = VecResetArray(w);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xr,&px);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPExtractVectors_Linear"
PetscErrorCode PEPExtractVectors_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  
  PetscFunctionBegin;
  switch (pep->extract) {
  case PEP_EXTRACT_NONE:
    ierr = PEPLinearExtract_None(pep,ctx->eps);CHKERRQ(ierr);
    break;
  case PEP_EXTRACT_NORM:
    ierr = PEPLinearExtract_Norm(pep,ctx->eps);CHKERRQ(ierr);
    break;
  case PEP_EXTRACT_RESIDUAL:
    ierr = PEPLinearExtract_Residual(pep,ctx->eps);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Extraction not implemented in this solver");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSolve_Linear"
PetscErrorCode PEPSolve_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  PetscScalar    sigma;
  PetscBool      flg;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = EPSSolve(ctx->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(ctx->eps,&pep->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(ctx->eps,&pep->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(ctx->eps,(EPSConvergedReason*)&pep->reason);CHKERRQ(ierr);

  /* recover eigenvalues */
  for (i=0;i<pep->nconv;i++) {
    ierr = EPSGetEigenpair(ctx->eps,i,&pep->eigr[i],&pep->eigi[i],NULL,NULL);CHKERRQ(ierr);
    pep->eigr[i] *= pep->sfactor;
    pep->eigi[i] *= pep->sfactor;
  }

  /* restore target */
  ierr = EPSGetTarget(ctx->eps,&sigma);CHKERRQ(ierr);
  ierr = EPSSetTarget(ctx->eps,sigma*pep->sfactor);CHKERRQ(ierr);

  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (flg && pep->ops->backtransform) {
    ierr = (*pep->ops->backtransform)(pep);CHKERRQ(ierr);
  }
  if (pep->sfactor!=1.0) {
    /* Restore original values */
    for (i=0;i<pep->nmat;i++){
      pep->pbc[pep->nmat+i] *= pep->sfactor;
      pep->pbc[2*pep->nmat+i] *= pep->sfactor*pep->sfactor;
    }
    if (!flg && !ctx->explicitmatrix) {
      ierr = STScaleShift(pep->st,pep->sfactor);CHKERRQ(ierr);
    } 
  }
  if (ctx->explicitmatrix) {
    ierr = RGPopScale(pep->rg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSMonitor_Linear"
static PetscErrorCode EPSMonitor_Linear(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PEP            pep = (PEP)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PEPMonitor(pep,its,nconv,eigr,eigi,errest,nest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetFromOptions_Linear"
PetscErrorCode PEPSetFromOptions_Linear(PetscOptionItems *PetscOptionsObject,PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  PetscInt       i;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PEP Linear Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pep_linear_cform","Number of the companion form","PEPLinearSetCompanionForm",ctx->cform,&i,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PEPLinearSetCompanionForm(pep,i);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-pep_linear_explicitmatrix","Use explicit matrix in linearization","PEPLinearSetExplicitMatrix",ctx->explicitmatrix,&val,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PEPLinearSetExplicitMatrix(pep,val);CHKERRQ(ierr);
  }
  if (!ctx->eps) { ierr = PEPLinearGetEPS(pep,&ctx->eps);CHKERRQ(ierr); }
  ierr = EPSSetFromOptions(ctx->eps);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetCompanionForm_Linear"
static PetscErrorCode PEPLinearSetCompanionForm_Linear(PEP pep,PetscInt cform)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  if (!cform) PetscFunctionReturn(0);
  if (cform==PETSC_DECIDE || cform==PETSC_DEFAULT) ctx->cform = 1;
  else {
    if (cform!=1 && cform!=2) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Invalid value of argument 'cform'");
    ctx->cform = cform;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetCompanionForm"
/*@
   PEPLinearSetCompanionForm - Choose between the two companion forms available
   for the linearization of a quadratic eigenproblem.

   Logically Collective on PEP

   Input Parameters:
+  pep   - polynomial eigenvalue solver
-  cform - 1 or 2 (first or second companion form)

   Options Database Key:
.  -pep_linear_cform <int> - Choose the companion form

   Level: advanced

.seealso: PEPLinearGetCompanionForm()
@*/
PetscErrorCode PEPLinearSetCompanionForm(PEP pep,PetscInt cform)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(pep,cform,2);
  ierr = PetscTryMethod(pep,"PEPLinearSetCompanionForm_C",(PEP,PetscInt),(pep,cform));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetCompanionForm_Linear"
static PetscErrorCode PEPLinearGetCompanionForm_Linear(PEP pep,PetscInt *cform)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  *cform = ctx->cform;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetCompanionForm"
/*@
   PEPLinearGetCompanionForm - Returns the number of the companion form that
   will be used for the linearization of a quadratic eigenproblem.

   Not Collective

   Input Parameter:
.  pep  - polynomial eigenvalue solver

   Output Parameter:
.  cform - the companion form number (1 or 2)

   Level: advanced

.seealso: PEPLinearSetCompanionForm()
@*/
PetscErrorCode PEPLinearGetCompanionForm(PEP pep,PetscInt *cform)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidIntPointer(cform,2);
  ierr = PetscUseMethod(pep,"PEPLinearGetCompanionForm_C",(PEP,PetscInt*),(pep,cform));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetExplicitMatrix_Linear"
static PetscErrorCode PEPLinearSetExplicitMatrix_Linear(PEP pep,PetscBool explicitmatrix)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  ctx->explicitmatrix = explicitmatrix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetExplicitMatrix"
/*@
   PEPLinearSetExplicitMatrix - Indicate if the matrices A and B for the
   linearization of the problem must be built explicitly.

   Logically Collective on PEP

   Input Parameters:
+  pep      - polynomial eigenvalue solver
-  explicit - boolean flag indicating if the matrices are built explicitly

   Options Database Key:
.  -pep_linear_explicitmatrix <boolean> - Indicates the boolean flag

   Level: advanced

.seealso: PEPLinearGetExplicitMatrix()
@*/
PetscErrorCode PEPLinearSetExplicitMatrix(PEP pep,PetscBool explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(pep,explicitmatrix,2);
  ierr = PetscTryMethod(pep,"PEPLinearSetExplicitMatrix_C",(PEP,PetscBool),(pep,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetExplicitMatrix_Linear"
static PetscErrorCode PEPLinearGetExplicitMatrix_Linear(PEP pep,PetscBool *explicitmatrix)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  *explicitmatrix = ctx->explicitmatrix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetExplicitMatrix"
/*@
   PEPLinearGetExplicitMatrix - Returns the flag indicating if the matrices
   A and B for the linearization are built explicitly.

   Not Collective

   Input Parameter:
.  pep  - polynomial eigenvalue solver

   Output Parameter:
.  explicitmatrix - the mode flag

   Level: advanced

.seealso: PEPLinearSetExplicitMatrix()
@*/
PetscErrorCode PEPLinearGetExplicitMatrix(PEP pep,PetscBool *explicitmatrix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(explicitmatrix,2);
  ierr = PetscUseMethod(pep,"PEPLinearGetExplicitMatrix_C",(PEP,PetscBool*),(pep,explicitmatrix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetEPS_Linear"
static PetscErrorCode PEPLinearSetEPS_Linear(PEP pep,EPS eps)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(&ctx->eps);CHKERRQ(ierr);
  ctx->eps = eps;
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->eps);CHKERRQ(ierr);
  pep->state = PEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearSetEPS"
/*@
   PEPLinearSetEPS - Associate an eigensolver object (EPS) to the
   polynomial eigenvalue solver.

   Collective on PEP

   Input Parameters:
+  pep - polynomial eigenvalue solver
-  eps - the eigensolver object

   Level: advanced

.seealso: PEPLinearGetEPS()
@*/
PetscErrorCode PEPLinearSetEPS(PEP pep,EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(pep,1,eps,2);
  ierr = PetscTryMethod(pep,"PEPLinearSetEPS_C",(PEP,EPS),(pep,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetEPS_Linear"
static PetscErrorCode PEPLinearGetEPS_Linear(PEP pep,EPS *eps)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  ST             st;

  PetscFunctionBegin;
  if (!ctx->eps) {
    ierr = EPSCreate(PetscObjectComm((PetscObject)pep),&ctx->eps);CHKERRQ(ierr);
    ierr = EPSSetOptionsPrefix(ctx->eps,((PetscObject)pep)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(ctx->eps,"pep_linear_");CHKERRQ(ierr);
    ierr = EPSGetST(ctx->eps,&st);CHKERRQ(ierr);
    ierr = STSetOptionsPrefix(st,((PetscObject)ctx->eps)->prefix);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->eps,(PetscObject)pep,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->eps);CHKERRQ(ierr);
    ierr = EPSMonitorSet(ctx->eps,EPSMonitor_Linear,pep,NULL);CHKERRQ(ierr);
  }
  *eps = ctx->eps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPLinearGetEPS"
/*@
   PEPLinearGetEPS - Retrieve the eigensolver object (EPS) associated
   to the polynomial eigenvalue solver.

   Not Collective

   Input Parameter:
.  pep - polynomial eigenvalue solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: PEPLinearSetEPS()
@*/
PetscErrorCode PEPLinearGetEPS(PEP pep,EPS *eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(eps,2);
  ierr = PetscUseMethod(pep,"PEPLinearGetEPS_C",(PEP,EPS*),(pep,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPView_Linear"
PetscErrorCode PEPView_Linear(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!ctx->eps) { ierr = PEPLinearGetEPS(pep,&ctx->eps);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPrintf(viewer,"  Linear: %s matrices\n",ctx->explicitmatrix? "explicit": "implicit");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Linear: %s companion form\n",ctx->cform==1? "1st": "2nd");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = EPSView(ctx->eps,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPReset_Linear"
PetscErrorCode PEPReset_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  if (!ctx->eps) { ierr = EPSReset(ctx->eps);CHKERRQ(ierr); }
  ierr = MatDestroy(&ctx->A);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->B);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->w[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->w[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->w[2]);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->w[3]);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->w[4]);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->w[5]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPDestroy_Linear"
PetscErrorCode PEPDestroy_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  ierr = EPSDestroy(&ctx->eps);CHKERRQ(ierr);
  ierr = PetscFree(pep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetCompanionForm_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetCompanionForm_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetEPS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetExplicitMatrix_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPCreate_Linear"
PETSC_EXTERN PetscErrorCode PEPCreate_Linear(PEP pep)
{
  PetscErrorCode ierr;
  PEP_LINEAR     *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(pep,&ctx);CHKERRQ(ierr);
  ctx->explicitmatrix = PETSC_FALSE;
  pep->data = (void*)ctx;

  pep->ops->solve          = PEPSolve_Linear;
  pep->ops->setup          = PEPSetUp_Linear;
  pep->ops->setfromoptions = PEPSetFromOptions_Linear;
  pep->ops->destroy        = PEPDestroy_Linear;
  pep->ops->reset          = PEPReset_Linear;
  pep->ops->view           = PEPView_Linear;
  pep->ops->backtransform  = PEPBackTransform_Default;
  pep->ops->computevectors = PEPComputeVectors_Default;
  pep->ops->extractvectors = PEPExtractVectors_Linear;
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetCompanionForm_C",PEPLinearSetCompanionForm_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetCompanionForm_C",PEPLinearGetCompanionForm_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetEPS_C",PEPLinearSetEPS_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetEPS_C",PEPLinearGetEPS_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetExplicitMatrix_C",PEPLinearSetExplicitMatrix_Linear);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetExplicitMatrix_C",PEPLinearGetExplicitMatrix_Linear);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

