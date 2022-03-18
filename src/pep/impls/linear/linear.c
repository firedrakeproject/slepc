/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Explicit linearization for polynomial eigenproblems
*/

#include <slepc/private/pepimpl.h>         /*I "slepcpep.h" I*/
#include "linear.h"

static PetscErrorCode MatMult_Linear_Shift(Mat M,Vec x,Vec y)
{
  PEP_LINEAR        *ctx;
  PEP               pep;
  const PetscScalar *px;
  PetscScalar       *py,a,sigma=0.0;
  PetscInt          nmat,deg,i,m;
  Vec               x1,x2,x3,y1,aux;
  PetscReal         *ca,*cb,*cg;
  PetscBool         flg;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  pep = ctx->pep;
  CHKERRQ(STGetTransform(pep->st,&flg));
  if (!flg) {
    CHKERRQ(STGetShift(pep->st,&sigma));
  }
  nmat = pep->nmat;
  deg = nmat-1;
  m = pep->nloc;
  ca = pep->pbc;
  cb = pep->pbc+nmat;
  cg = pep->pbc+2*nmat;
  x1=ctx->w[0];x2=ctx->w[1];x3=ctx->w[2];y1=ctx->w[3];aux=ctx->w[4];

  CHKERRQ(VecSet(y,0.0));
  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArray(y,&py));
  a = 1.0;

  /* first block */
  CHKERRQ(VecPlaceArray(x2,px));
  CHKERRQ(VecPlaceArray(x3,px+m));
  CHKERRQ(VecPlaceArray(y1,py));
  CHKERRQ(VecAXPY(y1,cb[0]-sigma,x2));
  CHKERRQ(VecAXPY(y1,ca[0],x3));
  CHKERRQ(VecResetArray(x2));
  CHKERRQ(VecResetArray(x3));
  CHKERRQ(VecResetArray(y1));

  /* inner blocks */
  for (i=1;i<deg-1;i++) {
    CHKERRQ(VecPlaceArray(x1,px+(i-1)*m));
    CHKERRQ(VecPlaceArray(x2,px+i*m));
    CHKERRQ(VecPlaceArray(x3,px+(i+1)*m));
    CHKERRQ(VecPlaceArray(y1,py+i*m));
    CHKERRQ(VecAXPY(y1,cg[i],x1));
    CHKERRQ(VecAXPY(y1,cb[i]-sigma,x2));
    CHKERRQ(VecAXPY(y1,ca[i],x3));
    CHKERRQ(VecResetArray(x1));
    CHKERRQ(VecResetArray(x2));
    CHKERRQ(VecResetArray(x3));
    CHKERRQ(VecResetArray(y1));
  }

  /* last block */
  CHKERRQ(VecPlaceArray(y1,py+(deg-1)*m));
  for (i=0;i<deg;i++) {
    CHKERRQ(VecPlaceArray(x1,px+i*m));
    CHKERRQ(STMatMult(pep->st,i,x1,aux));
    CHKERRQ(VecAXPY(y1,a,aux));
    CHKERRQ(VecResetArray(x1));
    a *= pep->sfactor;
  }
  CHKERRQ(VecCopy(y1,aux));
  CHKERRQ(STMatSolve(pep->st,aux,y1));
  CHKERRQ(VecScale(y1,-ca[deg-1]/a));
  CHKERRQ(VecPlaceArray(x1,px+(deg-2)*m));
  CHKERRQ(VecPlaceArray(x2,px+(deg-1)*m));
  CHKERRQ(VecAXPY(y1,cg[deg-1],x1));
  CHKERRQ(VecAXPY(y1,cb[deg-1]-sigma,x2));
  CHKERRQ(VecResetArray(x1));
  CHKERRQ(VecResetArray(x2));
  CHKERRQ(VecResetArray(y1));

  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Linear_Sinvert(Mat M,Vec x,Vec y)
{
  PEP_LINEAR        *ctx;
  PEP               pep;
  const PetscScalar *px;
  PetscScalar       *py,a,sigma,t=1.0,tp=0.0,tt;
  PetscInt          nmat,deg,i,m;
  Vec               x1,y1,y2,y3,aux,aux2;
  PetscReal         *ca,*cb,*cg;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  pep = ctx->pep;
  nmat = pep->nmat;
  deg = nmat-1;
  m = pep->nloc;
  ca = pep->pbc;
  cb = pep->pbc+nmat;
  cg = pep->pbc+2*nmat;
  x1=ctx->w[0];y1=ctx->w[1];y2=ctx->w[2];y3=ctx->w[3];aux=ctx->w[4];aux2=ctx->w[5];
  CHKERRQ(EPSGetTarget(ctx->eps,&sigma));
  CHKERRQ(VecSet(y,0.0));
  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArray(y,&py));
  a = pep->sfactor;

  /* first block */
  CHKERRQ(VecPlaceArray(x1,px));
  CHKERRQ(VecPlaceArray(y1,py+m));
  CHKERRQ(VecCopy(x1,y1));
  CHKERRQ(VecScale(y1,1.0/ca[0]));
  CHKERRQ(VecResetArray(x1));
  CHKERRQ(VecResetArray(y1));

  /* second block */
  if (deg>2) {
    CHKERRQ(VecPlaceArray(x1,px+m));
    CHKERRQ(VecPlaceArray(y1,py+m));
    CHKERRQ(VecPlaceArray(y2,py+2*m));
    CHKERRQ(VecCopy(x1,y2));
    CHKERRQ(VecAXPY(y2,sigma-cb[1],y1));
    CHKERRQ(VecScale(y2,1.0/ca[1]));
    CHKERRQ(VecResetArray(x1));
    CHKERRQ(VecResetArray(y1));
    CHKERRQ(VecResetArray(y2));
  }

  /* inner blocks */
  for (i=2;i<deg-1;i++) {
    CHKERRQ(VecPlaceArray(x1,px+i*m));
    CHKERRQ(VecPlaceArray(y1,py+(i-1)*m));
    CHKERRQ(VecPlaceArray(y2,py+i*m));
    CHKERRQ(VecPlaceArray(y3,py+(i+1)*m));
    CHKERRQ(VecCopy(x1,y3));
    CHKERRQ(VecAXPY(y3,sigma-cb[i],y2));
    CHKERRQ(VecAXPY(y3,-cg[i],y1));
    CHKERRQ(VecScale(y3,1.0/ca[i]));
    CHKERRQ(VecResetArray(x1));
    CHKERRQ(VecResetArray(y1));
    CHKERRQ(VecResetArray(y2));
    CHKERRQ(VecResetArray(y3));
  }

  /* last block */
  CHKERRQ(VecPlaceArray(y1,py));
  for (i=0;i<deg-2;i++) {
    CHKERRQ(VecPlaceArray(y2,py+(i+1)*m));
    CHKERRQ(STMatMult(pep->st,i+1,y2,aux));
    CHKERRQ(VecAXPY(y1,a,aux));
    CHKERRQ(VecResetArray(y2));
    a *= pep->sfactor;
  }
  i = deg-2;
  CHKERRQ(VecPlaceArray(y2,py+(i+1)*m));
  CHKERRQ(VecPlaceArray(y3,py+i*m));
  CHKERRQ(VecCopy(y2,aux2));
  CHKERRQ(VecAXPY(aux2,cg[i+1]/ca[i+1],y3));
  CHKERRQ(STMatMult(pep->st,i+1,aux2,aux));
  CHKERRQ(VecAXPY(y1,a,aux));
  CHKERRQ(VecResetArray(y2));
  CHKERRQ(VecResetArray(y3));
  a *= pep->sfactor;
  i = deg-1;
  CHKERRQ(VecPlaceArray(x1,px+i*m));
  CHKERRQ(VecPlaceArray(y3,py+i*m));
  CHKERRQ(VecCopy(x1,aux2));
  CHKERRQ(VecAXPY(aux2,sigma-cb[i],y3));
  CHKERRQ(VecScale(aux2,1.0/ca[i]));
  CHKERRQ(STMatMult(pep->st,i+1,aux2,aux));
  CHKERRQ(VecAXPY(y1,a,aux));
  CHKERRQ(VecResetArray(x1));
  CHKERRQ(VecResetArray(y3));

  CHKERRQ(VecCopy(y1,aux));
  CHKERRQ(STMatSolve(pep->st,aux,y1));
  CHKERRQ(VecScale(y1,-1.0));

  /* final update */
  for (i=1;i<deg;i++) {
    CHKERRQ(VecPlaceArray(y2,py+i*m));
    tt = t;
    t = ((sigma-cb[i-1])*t-cg[i-1]*tp)/ca[i-1]; /* i-th basis polynomial */
    tp = tt;
    CHKERRQ(VecAXPY(y2,t,y1));
    CHKERRQ(VecResetArray(y2));
  }
  CHKERRQ(VecResetArray(y1));

  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  PetscFunctionReturn(0);
}

static PetscErrorCode BackTransform_Linear(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  PEP_LINEAR     *ctx;
  ST             stctx;

  PetscFunctionBegin;
  CHKERRQ(STShellGetContext(st,&ctx));
  CHKERRQ(PEPGetST(ctx->pep,&stctx));
  CHKERRQ(STBackTransform(stctx,n,eigr,eigi));
  PetscFunctionReturn(0);
}

/*
   Dummy backtransform operation
 */
static PetscErrorCode BackTransform_Skip(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode Apply_Linear(ST st,Vec x,Vec y)
{
  PEP_LINEAR     *ctx;

  PetscFunctionBegin;
  CHKERRQ(STShellGetContext(st,&ctx));
  CHKERRQ(MatMult(ctx->A,x,y));
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSetUp_Linear(PEP pep)
{
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  ST             st;
  PetscInt       i=0,deg=pep->nmat-1;
  EPSWhich       which = EPS_LARGEST_MAGNITUDE;
  EPSProblemType ptype;
  PetscBool      trackall,istrivial,transf,sinv,ks;
  PetscScalar    sigma,*epsarray,*peparray;
  Vec            veps,w=NULL;
  /* function tables */
  PetscErrorCode (*fcreate[][2])(MPI_Comm,PEP_LINEAR*,Mat*) = {
    { MatCreateExplicit_Linear_NA, MatCreateExplicit_Linear_NB },
    { MatCreateExplicit_Linear_SA, MatCreateExplicit_Linear_SB },
    { MatCreateExplicit_Linear_HA, MatCreateExplicit_Linear_HB },
  };

  PetscFunctionBegin;
  PEPCheckShiftSinvert(pep);
  PEPCheckUnsupported(pep,PEP_FEATURE_STOPPING);
  PEPCheckIgnored(pep,PEP_FEATURE_CONVERGENCE);
  CHKERRQ(STGetTransform(pep->st,&transf));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv));
  if (!pep->which) CHKERRQ(PEPSetWhichEigenpairs_Default(pep));
  PetscCheck(pep->which!=PEP_ALL,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"This solver does not support computing all eigenvalues");
  CHKERRQ(STSetUp(pep->st));
  if (!ctx->eps) CHKERRQ(PEPLinearGetEPS(pep,&ctx->eps));
  CHKERRQ(EPSGetST(ctx->eps,&st));
  if (!transf && !ctx->usereps) CHKERRQ(EPSSetTarget(ctx->eps,pep->target));
  if (sinv && !transf && !ctx->usereps) CHKERRQ(STSetDefaultShift(st,pep->target));
  /* compute scale factor if not set by user */
  CHKERRQ(PEPComputeScaleFactor(pep));

  if (ctx->explicitmatrix) {
    PEPCheckQuadraticCondition(pep,PETSC_TRUE," (with explicit matrix)");
    PEPCheckUnsupportedCondition(pep,PEP_FEATURE_NONMONOMIAL,PETSC_TRUE," (with explicit matrix)");
    PetscCheck(!transf,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Explicit matrix option is not implemented with st-transform flag active");
    PetscCheck(pep->scale!=PEP_SCALE_DIAGONAL && pep->scale!=PEP_SCALE_BOTH,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Diagonal scaling not allowed in PEPLINEAR with explicit matrices");
    if (sinv && !transf) CHKERRQ(STSetType(st,STSINVERT));
    CHKERRQ(RGPushScale(pep->rg,1.0/pep->sfactor));
    CHKERRQ(STGetMatrixTransformed(pep->st,0,&ctx->K));
    CHKERRQ(STGetMatrixTransformed(pep->st,1,&ctx->C));
    CHKERRQ(STGetMatrixTransformed(pep->st,2,&ctx->M));
    ctx->sfactor = pep->sfactor;
    ctx->dsfactor = pep->dsfactor;

    CHKERRQ(MatDestroy(&ctx->A));
    CHKERRQ(MatDestroy(&ctx->B));
    CHKERRQ(VecDestroy(&ctx->w[0]));
    CHKERRQ(VecDestroy(&ctx->w[1]));
    CHKERRQ(VecDestroy(&ctx->w[2]));
    CHKERRQ(VecDestroy(&ctx->w[3]));

    switch (pep->problem_type) {
      case PEP_GENERAL:    i = 0; break;
      case PEP_HERMITIAN:
      case PEP_HYPERBOLIC: i = 1; break;
      case PEP_GYROSCOPIC: i = 2; break;
    }

    CHKERRQ((*fcreate[i][0])(PetscObjectComm((PetscObject)pep),ctx,&ctx->A));
    CHKERRQ((*fcreate[i][1])(PetscObjectComm((PetscObject)pep),ctx,&ctx->B));
    CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->A));
    CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->B));

  } else {   /* implicit matrix */
    PetscCheck(pep->problem_type==PEP_GENERAL,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Must use the explicit matrix option if problem type is not general");
    if (!((PetscObject)(ctx->eps))->type_name) {
      CHKERRQ(EPSSetType(ctx->eps,EPSKRYLOVSCHUR));
    } else {
      CHKERRQ(PetscObjectTypeCompare((PetscObject)ctx->eps,EPSKRYLOVSCHUR,&ks));
      PetscCheck(ks,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Implicit matrix option only implemented for Krylov-Schur");
    }
    PetscCheck(ctx->alpha==1.0 && ctx->beta==0.0,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Implicit matrix option does not support setting alpha,beta parameters of the linearization");
    CHKERRQ(STSetType(st,STSHELL));
    CHKERRQ(STShellSetContext(st,ctx));
    if (!transf) CHKERRQ(STShellSetBackTransform(st,BackTransform_Linear));
    else CHKERRQ(STShellSetBackTransform(st,BackTransform_Skip));
    CHKERRQ(MatCreateVecsEmpty(pep->A[0],&ctx->w[0],&ctx->w[1]));
    CHKERRQ(MatCreateVecsEmpty(pep->A[0],&ctx->w[2],&ctx->w[3]));
    CHKERRQ(MatCreateVecs(pep->A[0],&ctx->w[4],&ctx->w[5]));
    CHKERRQ(PetscLogObjectParents(pep,6,ctx->w));
    CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)pep),deg*pep->nloc,deg*pep->nloc,deg*pep->n,deg*pep->n,ctx,&ctx->A));
    if (sinv && !transf) {
      CHKERRQ(MatShellSetOperation(ctx->A,MATOP_MULT,(void(*)(void))MatMult_Linear_Sinvert));
    } else {
      CHKERRQ(MatShellSetOperation(ctx->A,MATOP_MULT,(void(*)(void))MatMult_Linear_Shift));
    }
    CHKERRQ(STShellSetApply(st,Apply_Linear));
    CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->A));
    ctx->pep = pep;

    CHKERRQ(PEPBasisCoefficients(pep,pep->pbc));
    if (!transf) {
      CHKERRQ(PetscMalloc1(pep->nmat,&pep->solvematcoeffs));
      CHKERRQ(PetscLogObjectMemory((PetscObject)pep,pep->nmat*sizeof(PetscScalar)));
      if (sinv) {
        CHKERRQ(PEPEvaluateBasis(pep,pep->target,0,pep->solvematcoeffs,NULL));
      } else {
        for (i=0;i<deg;i++) pep->solvematcoeffs[i] = 0.0;
        pep->solvematcoeffs[deg] = 1.0;
      }
      CHKERRQ(STScaleShift(pep->st,1.0/pep->sfactor));
      CHKERRQ(RGPushScale(pep->rg,1.0/pep->sfactor));
    }
    if (pep->sfactor!=1.0) {
      for (i=0;i<pep->nmat;i++) {
        pep->pbc[pep->nmat+i] /= pep->sfactor;
        pep->pbc[2*pep->nmat+i] /= pep->sfactor*pep->sfactor;
      }
    }
  }

  CHKERRQ(EPSSetOperators(ctx->eps,ctx->A,ctx->B));
  CHKERRQ(EPSGetProblemType(ctx->eps,&ptype));
  if (!ptype) {
    if (ctx->explicitmatrix) {
      CHKERRQ(EPSSetProblemType(ctx->eps,EPS_GNHEP));
    } else {
      CHKERRQ(EPSSetProblemType(ctx->eps,EPS_NHEP));
    }
  }
  if (!ctx->usereps) {
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
        case PEP_ALL:                which = EPS_ALL; break;
        case PEP_WHICH_USER:         which = EPS_WHICH_USER;
          CHKERRQ(EPSSetEigenvalueComparison(ctx->eps,pep->sc->comparison,pep->sc->comparisonctx));
          break;
      }
    }
    CHKERRQ(EPSSetWhichEigenpairs(ctx->eps,which));

    CHKERRQ(EPSSetDimensions(ctx->eps,pep->nev,pep->ncv,pep->mpd));
    CHKERRQ(EPSSetTolerances(ctx->eps,SlepcDefaultTol(pep->tol),pep->max_it));
  }
  CHKERRQ(RGIsTrivial(pep->rg,&istrivial));
  if (!istrivial) {
    PetscCheck(!transf,PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"PEPLINEAR does not support a nontrivial region with st-transform");
    CHKERRQ(EPSSetRG(ctx->eps,pep->rg));
  }
  /* Transfer the trackall option from pep to eps */
  CHKERRQ(PEPGetTrackAll(pep,&trackall));
  CHKERRQ(EPSSetTrackAll(ctx->eps,trackall));

  /* temporary change of target */
  if (pep->sfactor!=1.0) {
    CHKERRQ(EPSGetTarget(ctx->eps,&sigma));
    CHKERRQ(EPSSetTarget(ctx->eps,sigma/pep->sfactor));
  }

  /* process initial vector */
  if (pep->nini<0) {
    CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)ctx->eps),deg*pep->nloc,deg*pep->n,&veps));
    CHKERRQ(VecGetArray(veps,&epsarray));
    for (i=0;i<deg;i++) {
      if (i<-pep->nini) {
        CHKERRQ(VecGetArray(pep->IS[i],&peparray));
        CHKERRQ(PetscArraycpy(epsarray+i*pep->nloc,peparray,pep->nloc));
        CHKERRQ(VecRestoreArray(pep->IS[i],&peparray));
      } else {
        if (!w) CHKERRQ(VecDuplicate(pep->IS[0],&w));
        CHKERRQ(VecSetRandom(w,NULL));
        CHKERRQ(VecGetArray(w,&peparray));
        CHKERRQ(PetscArraycpy(epsarray+i*pep->nloc,peparray,pep->nloc));
        CHKERRQ(VecRestoreArray(w,&peparray));
      }
    }
    CHKERRQ(VecRestoreArray(veps,&epsarray));
    CHKERRQ(EPSSetInitialSpace(ctx->eps,1,&veps));
    CHKERRQ(VecDestroy(&veps));
    CHKERRQ(VecDestroy(&w));
    CHKERRQ(SlepcBasisDestroy_Private(&pep->nini,&pep->IS));
  }

  CHKERRQ(EPSSetUp(ctx->eps));
  CHKERRQ(EPSGetDimensions(ctx->eps,NULL,&pep->ncv,&pep->mpd));
  CHKERRQ(EPSGetTolerances(ctx->eps,NULL,&pep->max_it));
  CHKERRQ(PEPAllocateSolution(pep,0));
  PetscFunctionReturn(0);
}

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
  CHKERRQ(PEPSetWorkVecs(pep,2));
#else
  CHKERRQ(PEPSetWorkVecs(pep,4));
#endif
  CHKERRQ(EPSGetOperators(eps,&A,NULL));
  CHKERRQ(MatCreateVecs(A,&xr,NULL));
  CHKERRQ(MatCreateVecsEmpty(pep->A[0],&wr,NULL));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(VecDuplicate(xr,&xi));
  CHKERRQ(VecDuplicateEmpty(wr,&wi));
#endif
  for (i=0;i<pep->nconv;i++) {
    CHKERRQ(EPSGetEigenpair(eps,i,NULL,NULL,xr,xi));
#if !defined(PETSC_USE_COMPLEX)
    if (ei[i]!=0.0) {   /* complex conjugate pair */
      CHKERRQ(VecGetArrayRead(xr,&px));
      CHKERRQ(VecGetArrayRead(xi,&py));
      CHKERRQ(VecPlaceArray(wr,px));
      CHKERRQ(VecPlaceArray(wi,py));
      CHKERRQ(VecNormalizeComplex(wr,wi,PETSC_TRUE,NULL));
      CHKERRQ(PEPComputeResidualNorm_Private(pep,er[i],ei[i],wr,wi,pep->work,&rn1));
      CHKERRQ(BVInsertVec(pep->V,i,wr));
      CHKERRQ(BVInsertVec(pep->V,i+1,wi));
      for (k=1;k<pep->nmat-1;k++) {
        CHKERRQ(VecResetArray(wr));
        CHKERRQ(VecResetArray(wi));
        CHKERRQ(VecPlaceArray(wr,px+k*pep->nloc));
        CHKERRQ(VecPlaceArray(wi,py+k*pep->nloc));
        CHKERRQ(VecNormalizeComplex(wr,wi,PETSC_TRUE,NULL));
        CHKERRQ(PEPComputeResidualNorm_Private(pep,er[i],ei[i],wr,wi,pep->work,&rn2));
        if (rn1>rn2) {
          CHKERRQ(BVInsertVec(pep->V,i,wr));
          CHKERRQ(BVInsertVec(pep->V,i+1,wi));
          rn1 = rn2;
        }
      }
      CHKERRQ(VecResetArray(wr));
      CHKERRQ(VecResetArray(wi));
      CHKERRQ(VecRestoreArrayRead(xr,&px));
      CHKERRQ(VecRestoreArrayRead(xi,&py));
      i++;
    } else   /* real eigenvalue */
#endif
    {
      CHKERRQ(VecGetArrayRead(xr,&px));
      CHKERRQ(VecPlaceArray(wr,px));
      CHKERRQ(VecNormalizeComplex(wr,NULL,PETSC_FALSE,NULL));
      CHKERRQ(PEPComputeResidualNorm_Private(pep,er[i],ei[i],wr,NULL,pep->work,&rn1));
      CHKERRQ(BVInsertVec(pep->V,i,wr));
      for (k=1;k<pep->nmat-1;k++) {
        CHKERRQ(VecResetArray(wr));
        CHKERRQ(VecPlaceArray(wr,px+k*pep->nloc));
        CHKERRQ(VecNormalizeComplex(wr,NULL,PETSC_FALSE,NULL));
        CHKERRQ(PEPComputeResidualNorm_Private(pep,er[i],ei[i],wr,NULL,pep->work,&rn2));
        if (rn1>rn2) {
          CHKERRQ(BVInsertVec(pep->V,i,wr));
          rn1 = rn2;
        }
      }
      CHKERRQ(VecResetArray(wr));
      CHKERRQ(VecRestoreArrayRead(xr,&px));
    }
  }
  CHKERRQ(VecDestroy(&wr));
  CHKERRQ(VecDestroy(&xr));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(VecDestroy(&wi));
  CHKERRQ(VecDestroy(&xi));
#endif
  PetscFunctionReturn(0);
}

/*
   PEPLinearExtract_None - Same as PEPLinearExtract_Norm but always takes
   the first block.
*/
static PetscErrorCode PEPLinearExtract_None(PEP pep,EPS eps)
{
  PetscInt          i;
  const PetscScalar *px;
  Mat               A;
  Vec               xr,xi=NULL,w;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar       *ei=pep->eigi;
#endif

  PetscFunctionBegin;
  CHKERRQ(EPSGetOperators(eps,&A,NULL));
  CHKERRQ(MatCreateVecs(A,&xr,NULL));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(VecDuplicate(xr,&xi));
#endif
  CHKERRQ(MatCreateVecsEmpty(pep->A[0],&w,NULL));
  for (i=0;i<pep->nconv;i++) {
    CHKERRQ(EPSGetEigenvector(eps,i,xr,xi));
#if !defined(PETSC_USE_COMPLEX)
    if (ei[i]!=0.0) {   /* complex conjugate pair */
      CHKERRQ(VecGetArrayRead(xr,&px));
      CHKERRQ(VecPlaceArray(w,px));
      CHKERRQ(BVInsertVec(pep->V,i,w));
      CHKERRQ(VecResetArray(w));
      CHKERRQ(VecRestoreArrayRead(xr,&px));
      CHKERRQ(VecGetArrayRead(xi,&px));
      CHKERRQ(VecPlaceArray(w,px));
      CHKERRQ(BVInsertVec(pep->V,i+1,w));
      CHKERRQ(VecResetArray(w));
      CHKERRQ(VecRestoreArrayRead(xi,&px));
      i++;
    } else   /* real eigenvalue */
#endif
    {
      CHKERRQ(VecGetArrayRead(xr,&px));
      CHKERRQ(VecPlaceArray(w,px));
      CHKERRQ(BVInsertVec(pep->V,i,w));
      CHKERRQ(VecResetArray(w));
      CHKERRQ(VecRestoreArrayRead(xr,&px));
    }
  }
  CHKERRQ(VecDestroy(&w));
  CHKERRQ(VecDestroy(&xr));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(VecDestroy(&xi));
#endif
  PetscFunctionReturn(0);
}

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
  PetscInt          i,offset;
  const PetscScalar *px;
  PetscScalar       *er=pep->eigr;
  Mat               A;
  Vec               xr,xi=NULL,w;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar       *ei=pep->eigi;
#endif

  PetscFunctionBegin;
  CHKERRQ(EPSGetOperators(eps,&A,NULL));
  CHKERRQ(MatCreateVecs(A,&xr,NULL));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(VecDuplicate(xr,&xi));
#endif
  CHKERRQ(MatCreateVecsEmpty(pep->A[0],&w,NULL));
  for (i=0;i<pep->nconv;i++) {
    CHKERRQ(EPSGetEigenpair(eps,i,NULL,NULL,xr,xi));
    if (SlepcAbsEigenvalue(er[i],ei[i])>1.0) offset = (pep->nmat-2)*pep->nloc;
    else offset = 0;
#if !defined(PETSC_USE_COMPLEX)
    if (ei[i]!=0.0) {   /* complex conjugate pair */
      CHKERRQ(VecGetArrayRead(xr,&px));
      CHKERRQ(VecPlaceArray(w,px+offset));
      CHKERRQ(BVInsertVec(pep->V,i,w));
      CHKERRQ(VecResetArray(w));
      CHKERRQ(VecRestoreArrayRead(xr,&px));
      CHKERRQ(VecGetArrayRead(xi,&px));
      CHKERRQ(VecPlaceArray(w,px+offset));
      CHKERRQ(BVInsertVec(pep->V,i+1,w));
      CHKERRQ(VecResetArray(w));
      CHKERRQ(VecRestoreArrayRead(xi,&px));
      i++;
    } else /* real eigenvalue */
#endif
    {
      CHKERRQ(VecGetArrayRead(xr,&px));
      CHKERRQ(VecPlaceArray(w,px+offset));
      CHKERRQ(BVInsertVec(pep->V,i,w));
      CHKERRQ(VecResetArray(w));
      CHKERRQ(VecRestoreArrayRead(xr,&px));
    }
  }
  CHKERRQ(VecDestroy(&w));
  CHKERRQ(VecDestroy(&xr));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(VecDestroy(&xi));
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PEPExtractVectors_Linear(PEP pep)
{
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  switch (pep->extract) {
  case PEP_EXTRACT_NONE:
    CHKERRQ(PEPLinearExtract_None(pep,ctx->eps));
    break;
  case PEP_EXTRACT_NORM:
    CHKERRQ(PEPLinearExtract_Norm(pep,ctx->eps));
    break;
  case PEP_EXTRACT_RESIDUAL:
    CHKERRQ(PEPLinearExtract_Residual(pep,ctx->eps));
    break;
  case PEP_EXTRACT_STRUCTURED:
    SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Extraction not implemented in this solver");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSolve_Linear(PEP pep)
{
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  PetscScalar    sigma;
  PetscBool      flg;
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(EPSSolve(ctx->eps));
  CHKERRQ(EPSGetConverged(ctx->eps,&pep->nconv));
  CHKERRQ(EPSGetIterationNumber(ctx->eps,&pep->its));
  CHKERRQ(EPSGetConvergedReason(ctx->eps,(EPSConvergedReason*)&pep->reason));

  /* recover eigenvalues */
  for (i=0;i<pep->nconv;i++) {
    CHKERRQ(EPSGetEigenpair(ctx->eps,i,&pep->eigr[i],&pep->eigi[i],NULL,NULL));
    pep->eigr[i] *= pep->sfactor;
    pep->eigi[i] *= pep->sfactor;
  }

  /* restore target */
  CHKERRQ(EPSGetTarget(ctx->eps,&sigma));
  CHKERRQ(EPSSetTarget(ctx->eps,sigma*pep->sfactor));

  CHKERRQ(STGetTransform(pep->st,&flg));
  if (flg && pep->ops->backtransform) {
    CHKERRQ((*pep->ops->backtransform)(pep));
  }
  if (pep->sfactor!=1.0) {
    /* Restore original values */
    for (i=0;i<pep->nmat;i++) {
      pep->pbc[pep->nmat+i] *= pep->sfactor;
      pep->pbc[2*pep->nmat+i] *= pep->sfactor*pep->sfactor;
    }
    if (!flg && !ctx->explicitmatrix) {
      CHKERRQ(STScaleShift(pep->st,pep->sfactor));
    }
  }
  if (ctx->explicitmatrix || !flg) {
    CHKERRQ(RGPopScale(pep->rg));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSMonitor_Linear(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PEP            pep = (PEP)ctx;

  PetscFunctionBegin;
  CHKERRQ(PEPMonitor(pep,its,nconv,eigr,eigi,errest,nest));
  PetscFunctionReturn(0);
}

PetscErrorCode PEPSetFromOptions_Linear(PetscOptionItems *PetscOptionsObject,PEP pep)
{
  PetscBool      set,val;
  PetscInt       k;
  PetscReal      array[2]={0,0};
  PetscBool      flg;
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"PEP Linear Options"));

    k = 2;
    CHKERRQ(PetscOptionsRealArray("-pep_linear_linearization","Parameters of the linearization","PEPLinearSetLinearization",array,&k,&flg));
    if (flg) {
      CHKERRQ(PEPLinearSetLinearization(pep,array[0],array[1]));
    }

    CHKERRQ(PetscOptionsBool("-pep_linear_explicitmatrix","Use explicit matrix in linearization","PEPLinearSetExplicitMatrix",ctx->explicitmatrix,&val,&set));
    if (set) CHKERRQ(PEPLinearSetExplicitMatrix(pep,val));

  CHKERRQ(PetscOptionsTail());

  if (!ctx->eps) CHKERRQ(PEPLinearGetEPS(pep,&ctx->eps));
  CHKERRQ(EPSSetFromOptions(ctx->eps));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPLinearSetLinearization_Linear(PEP pep,PetscReal alpha,PetscReal beta)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  PetscCheck(beta!=0.0 || alpha!=0.0,PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONG,"Parameters alpha and beta cannot be zero simultaneously");
  ctx->alpha = alpha;
  ctx->beta  = beta;
  PetscFunctionReturn(0);
}

/*@
   PEPLinearSetLinearization - Set the coefficients that define
   the linearization of a quadratic eigenproblem.

   Logically Collective on pep

   Input Parameters:
+  pep   - polynomial eigenvalue solver
.  alpha - first parameter of the linearization
-  beta  - second parameter of the linearization

   Options Database Key:
.  -pep_linear_linearization <alpha,beta> - Sets the coefficients

   Notes:
   Cannot pass zero for both alpha and beta. The default values are
   alpha=1 and beta=0.

   Level: advanced

.seealso: PEPLinearGetLinearization()
@*/
PetscErrorCode PEPLinearSetLinearization(PEP pep,PetscReal alpha,PetscReal beta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,alpha,2);
  PetscValidLogicalCollectiveReal(pep,beta,3);
  CHKERRQ(PetscTryMethod(pep,"PEPLinearSetLinearization_C",(PEP,PetscReal,PetscReal),(pep,alpha,beta)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPLinearGetLinearization_Linear(PEP pep,PetscReal *alpha,PetscReal *beta)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  if (alpha) *alpha = ctx->alpha;
  if (beta)  *beta  = ctx->beta;
  PetscFunctionReturn(0);
}

/*@
   PEPLinearGetLinearization - Returns the coefficients that define
   the linearization of a quadratic eigenproblem.

   Not Collective

   Input Parameter:
.  pep  - polynomial eigenvalue solver

   Output Parameters:
+  alpha - the first parameter of the linearization
-  beta  - the second parameter of the linearization

   Level: advanced

.seealso: PEPLinearSetLinearization()
@*/
PetscErrorCode PEPLinearGetLinearization(PEP pep,PetscReal *alpha,PetscReal *beta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  CHKERRQ(PetscUseMethod(pep,"PEPLinearGetLinearization_C",(PEP,PetscReal*,PetscReal*),(pep,alpha,beta)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPLinearSetExplicitMatrix_Linear(PEP pep,PetscBool explicitmatrix)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  if (ctx->explicitmatrix != explicitmatrix) {
    ctx->explicitmatrix = explicitmatrix;
    pep->state = PEP_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   PEPLinearSetExplicitMatrix - Indicate if the matrices A and B for the
   linearization of the problem must be built explicitly.

   Logically Collective on pep

   Input Parameters:
+  pep         - polynomial eigenvalue solver
-  explicitmat - boolean flag indicating if the matrices are built explicitly

   Options Database Key:
.  -pep_linear_explicitmatrix <boolean> - Indicates the boolean flag

   Level: advanced

.seealso: PEPLinearGetExplicitMatrix()
@*/
PetscErrorCode PEPLinearSetExplicitMatrix(PEP pep,PetscBool explicitmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(pep,explicitmat,2);
  CHKERRQ(PetscTryMethod(pep,"PEPLinearSetExplicitMatrix_C",(PEP,PetscBool),(pep,explicitmat)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPLinearGetExplicitMatrix_Linear(PEP pep,PetscBool *explicitmat)
{
  PEP_LINEAR *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  *explicitmat = ctx->explicitmatrix;
  PetscFunctionReturn(0);
}

/*@
   PEPLinearGetExplicitMatrix - Returns the flag indicating if the matrices
   A and B for the linearization are built explicitly.

   Not Collective

   Input Parameter:
.  pep  - polynomial eigenvalue solver

   Output Parameter:
.  explicitmat - the mode flag

   Level: advanced

.seealso: PEPLinearSetExplicitMatrix()
@*/
PetscErrorCode PEPLinearGetExplicitMatrix(PEP pep,PetscBool *explicitmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidBoolPointer(explicitmat,2);
  CHKERRQ(PetscUseMethod(pep,"PEPLinearGetExplicitMatrix_C",(PEP,PetscBool*),(pep,explicitmat)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPLinearSetEPS_Linear(PEP pep,EPS eps)
{
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)eps));
  CHKERRQ(EPSDestroy(&ctx->eps));
  ctx->eps = eps;
  ctx->usereps = PETSC_TRUE;
  CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->eps));
  pep->state = PEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   PEPLinearSetEPS - Associate an eigensolver object (EPS) to the
   polynomial eigenvalue solver.

   Collective on pep

   Input Parameters:
+  pep - polynomial eigenvalue solver
-  eps - the eigensolver object

   Level: advanced

.seealso: PEPLinearGetEPS()
@*/
PetscErrorCode PEPLinearSetEPS(PEP pep,EPS eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(pep,1,eps,2);
  CHKERRQ(PetscTryMethod(pep,"PEPLinearSetEPS_C",(PEP,EPS),(pep,eps)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PEPLinearGetEPS_Linear(PEP pep,EPS *eps)
{
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  if (!ctx->eps) {
    CHKERRQ(EPSCreate(PetscObjectComm((PetscObject)pep),&ctx->eps));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ctx->eps,(PetscObject)pep,1));
    CHKERRQ(EPSSetOptionsPrefix(ctx->eps,((PetscObject)pep)->prefix));
    CHKERRQ(EPSAppendOptionsPrefix(ctx->eps,"pep_linear_"));
    CHKERRQ(PetscLogObjectParent((PetscObject)pep,(PetscObject)ctx->eps));
    CHKERRQ(PetscObjectSetOptions((PetscObject)ctx->eps,((PetscObject)pep)->options));
    CHKERRQ(EPSMonitorSet(ctx->eps,EPSMonitor_Linear,pep,NULL));
  }
  *eps = ctx->eps;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(eps,2);
  CHKERRQ(PetscUseMethod(pep,"PEPLinearGetEPS_C",(PEP,EPS*),(pep,eps)));
  PetscFunctionReturn(0);
}

PetscErrorCode PEPView_Linear(PEP pep,PetscViewer viewer)
{
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    if (!ctx->eps) CHKERRQ(PEPLinearGetEPS(pep,&ctx->eps));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %s matrices\n",ctx->explicitmatrix? "explicit": "implicit"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  linearization parameters: alpha=%g beta=%g\n",(double)ctx->alpha,(double)ctx->beta));
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(EPSView(ctx->eps,viewer));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PEPReset_Linear(PEP pep)
{
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  if (!ctx->eps) CHKERRQ(EPSReset(ctx->eps));
  CHKERRQ(MatDestroy(&ctx->A));
  CHKERRQ(MatDestroy(&ctx->B));
  CHKERRQ(VecDestroy(&ctx->w[0]));
  CHKERRQ(VecDestroy(&ctx->w[1]));
  CHKERRQ(VecDestroy(&ctx->w[2]));
  CHKERRQ(VecDestroy(&ctx->w[3]));
  CHKERRQ(VecDestroy(&ctx->w[4]));
  CHKERRQ(VecDestroy(&ctx->w[5]));
  PetscFunctionReturn(0);
}

PetscErrorCode PEPDestroy_Linear(PEP pep)
{
  PEP_LINEAR     *ctx = (PEP_LINEAR*)pep->data;

  PetscFunctionBegin;
  CHKERRQ(EPSDestroy(&ctx->eps));
  CHKERRQ(PetscFree(pep->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetLinearization_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetLinearization_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetEPS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetEPS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetExplicitMatrix_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetExplicitMatrix_C",NULL));
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode PEPCreate_Linear(PEP pep)
{
  PEP_LINEAR     *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pep,&ctx));
  pep->data = (void*)ctx;

  pep->lineariz       = PETSC_TRUE;
  ctx->explicitmatrix = PETSC_FALSE;
  ctx->alpha          = 1.0;
  ctx->beta           = 0.0;

  pep->ops->solve          = PEPSolve_Linear;
  pep->ops->setup          = PEPSetUp_Linear;
  pep->ops->setfromoptions = PEPSetFromOptions_Linear;
  pep->ops->destroy        = PEPDestroy_Linear;
  pep->ops->reset          = PEPReset_Linear;
  pep->ops->view           = PEPView_Linear;
  pep->ops->backtransform  = PEPBackTransform_Default;
  pep->ops->computevectors = PEPComputeVectors_Default;
  pep->ops->extractvectors = PEPExtractVectors_Linear;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetLinearization_C",PEPLinearSetLinearization_Linear));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetLinearization_C",PEPLinearGetLinearization_Linear));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetEPS_C",PEPLinearSetEPS_Linear));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetEPS_C",PEPLinearGetEPS_Linear));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearSetExplicitMatrix_C",PEPLinearSetExplicitMatrix_Linear));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pep,"PEPLinearGetExplicitMatrix_C",PEPLinearGetExplicitMatrix_Linear));
  PetscFunctionReturn(0);
}
