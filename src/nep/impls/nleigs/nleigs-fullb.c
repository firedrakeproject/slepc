/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Full basis for the linearization of the rational approximation of non-linear eigenproblems
*/

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include "nleigs.h"

static PetscErrorCode MatMult_FullBasis_Sinvert(Mat M,Vec x,Vec y)
{
  NEP_NLEIGS        *ctx;
  NEP               nep;
  const PetscScalar *px;
  PetscScalar       *beta,*s,*xi,*t,*py,sigma;
  PetscInt          nmat,d,i,k,m;
  Vec               xx,xxx,yy,yyy,w,ww,www;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&nep));
  ctx = (NEP_NLEIGS*)nep->data;
  beta = ctx->beta; s = ctx->s; xi = ctx->xi;
  sigma = ctx->shifts[0];
  nmat = ctx->nmat;
  d = nmat-1;
  m = nep->nloc;
  CHKERRQ(PetscMalloc1(ctx->nmat,&t));
  xx = ctx->w[0]; xxx = ctx->w[1]; yy = ctx->w[2]; yyy=ctx->w[3];
  w = nep->work[0]; ww = nep->work[1]; www = nep->work[2];
  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArray(y,&py));
  CHKERRQ(VecPlaceArray(xx,px+(d-1)*m));
  CHKERRQ(VecPlaceArray(xxx,px+(d-2)*m));
  CHKERRQ(VecPlaceArray(yy,py+(d-2)*m));
  CHKERRQ(VecCopy(xxx,yy));
  CHKERRQ(VecAXPY(yy,beta[d-1]/xi[d-2],xx));
  CHKERRQ(VecScale(yy,1.0/(s[d-2]-sigma)));
  CHKERRQ(VecResetArray(xx));
  CHKERRQ(VecResetArray(xxx));
  CHKERRQ(VecResetArray(yy));
  for (i=d-3;i>=0;i--) {
    CHKERRQ(VecPlaceArray(xx,px+(i+1)*m));
    CHKERRQ(VecPlaceArray(xxx,px+i*m));
    CHKERRQ(VecPlaceArray(yy,py+i*m));
    CHKERRQ(VecPlaceArray(yyy,py+(i+1)*m));
    CHKERRQ(VecCopy(xxx,yy));
    CHKERRQ(VecAXPY(yy,beta[i+1]/xi[i],xx));
    CHKERRQ(VecAXPY(yy,-beta[i+1]*(1.0-sigma/xi[i]),yyy));
    CHKERRQ(VecScale(yy,1.0/(s[i]-sigma)));
    CHKERRQ(VecResetArray(xx));
    CHKERRQ(VecResetArray(xxx));
    CHKERRQ(VecResetArray(yy));
    CHKERRQ(VecResetArray(yyy));
  }
  if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
    CHKERRQ(VecZeroEntries(w));
    for (k=0;k<nep->nt;k++) {
      CHKERRQ(VecZeroEntries(ww));
      CHKERRQ(VecPlaceArray(xx,px+(d-1)*m));
      CHKERRQ(VecAXPY(ww,-ctx->coeffD[k+nep->nt*d]/beta[d],xx));
      CHKERRQ(VecResetArray(xx));
      for (i=0;i<d-1;i++) {
        CHKERRQ(VecPlaceArray(yy,py+i*m));
        CHKERRQ(VecAXPY(ww,-ctx->coeffD[nep->nt*i+k],yy));
        CHKERRQ(VecResetArray(yy));
      }
      CHKERRQ(MatMult(nep->A[k],ww,www));
      CHKERRQ(VecAXPY(w,1.0,www));
    }
  } else {
    CHKERRQ(VecPlaceArray(xx,px+(d-1)*m));
    CHKERRQ(MatMult(ctx->D[d],xx,w));
    CHKERRQ(VecScale(w,-1.0/beta[d]));
    CHKERRQ(VecResetArray(xx));
    for (i=0;i<d-1;i++) {
      CHKERRQ(VecPlaceArray(yy,py+i*m));
      CHKERRQ(MatMult(ctx->D[i],yy,ww));
      CHKERRQ(VecResetArray(yy));
      CHKERRQ(VecAXPY(w,-1.0,ww));
    }
  }
  CHKERRQ(VecPlaceArray(yy,py+(d-1)*m));
  CHKERRQ(KSPSolve(ctx->ksp[0],w,yy));
  CHKERRQ(NEPNLEIGSEvalNRTFunct(nep,d-1,sigma,t));
  for (i=0;i<d-1;i++) {
    CHKERRQ(VecPlaceArray(yyy,py+i*m));
    CHKERRQ(VecAXPY(yyy,t[i],yy));
    CHKERRQ(VecResetArray(yyy));
  }
  CHKERRQ(VecScale(yy,t[d-1]));
  CHKERRQ(VecResetArray(yy));
  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  CHKERRQ(PetscFree(t));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_FullBasis_Sinvert(Mat M,Vec x,Vec y)
{
  NEP_NLEIGS        *ctx;
  NEP               nep;
  const PetscScalar *px;
  PetscScalar       *beta,*s,*xi,*t,*py,sigma;
  PetscInt          nmat,d,i,k,m;
  Vec               xx,yy,yyy,w,z0;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&nep));
  ctx = (NEP_NLEIGS*)nep->data;
  beta = ctx->beta; s = ctx->s; xi = ctx->xi;
  sigma = ctx->shifts[0];
  nmat = ctx->nmat;
  d = nmat-1;
  m = nep->nloc;
  CHKERRQ(PetscMalloc1(ctx->nmat,&t));
  xx = ctx->w[0]; yy = ctx->w[1]; yyy=ctx->w[2];
  w = nep->work[0]; z0 = nep->work[1];
  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArray(y,&py));
  CHKERRQ(NEPNLEIGSEvalNRTFunct(nep,d,sigma,t));
  CHKERRQ(VecPlaceArray(xx,px+(d-1)*m));
  CHKERRQ(VecCopy(xx,w));
  CHKERRQ(VecScale(w,t[d-1]));
  CHKERRQ(VecResetArray(xx));
  for (i=0;i<d-1;i++) {
    CHKERRQ(VecPlaceArray(xx,px+i*m));
    CHKERRQ(VecAXPY(w,t[i],xx));
    CHKERRQ(VecResetArray(xx));
  }
  CHKERRQ(KSPSolveTranspose(ctx->ksp[0],w,z0));

  CHKERRQ(VecPlaceArray(yy,py));
  if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
    CHKERRQ(VecZeroEntries(yy));
    for (k=0;k<nep->nt;k++) {
      CHKERRQ(MatMult(nep->A[k],z0,w));
      CHKERRQ(VecAXPY(yy,ctx->coeffD[k],w));
    }
  } else {
    CHKERRQ(MatMultTranspose(ctx->D[0],z0,yy));
  }
  CHKERRQ(VecPlaceArray(xx,px));
  CHKERRQ(VecAXPY(yy,-1.0,xx));
  CHKERRQ(VecResetArray(xx));
  CHKERRQ(VecScale(yy,-1.0/(s[0]-sigma)));
  CHKERRQ(VecResetArray(yy));
  for (i=2;i<d;i++) {
    CHKERRQ(VecPlaceArray(yy,py+(i-1)*m));
    if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
      CHKERRQ(VecZeroEntries(yy));
      for (k=0;k<nep->nt;k++) {
        CHKERRQ(MatMult(nep->A[k],z0,w));
        CHKERRQ(VecAXPY(yy,ctx->coeffD[k+(i-1)*nep->nt],w));
      }
    } else {
      CHKERRQ(MatMultTranspose(ctx->D[i-1],z0,yy));
    }
    CHKERRQ(VecPlaceArray(yyy,py+(i-2)*m));
    CHKERRQ(VecAXPY(yy,beta[i-1]*(1.0-sigma/xi[i-2]),yyy));
    CHKERRQ(VecResetArray(yyy));
    CHKERRQ(VecPlaceArray(xx,px+(i-1)*m));
    CHKERRQ(VecAXPY(yy,-1.0,xx));
    CHKERRQ(VecResetArray(xx));
    CHKERRQ(VecScale(yy,-1.0/(s[i-1]-sigma)));
    CHKERRQ(VecResetArray(yy));
  }
  CHKERRQ(VecPlaceArray(yy,py+(d-1)*m));
  if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
    CHKERRQ(VecZeroEntries(yy));
    for (k=0;k<nep->nt;k++) {
      CHKERRQ(MatMult(nep->A[k],z0,w));
      CHKERRQ(VecAXPY(yy,ctx->coeffD[k+d*nep->nt],w));
    }
  } else {
    CHKERRQ(MatMultTranspose(ctx->D[d],z0,yy));
  }
  CHKERRQ(VecScale(yy,-1.0/beta[d]));
  CHKERRQ(VecPlaceArray(yyy,py+(d-2)*m));
  CHKERRQ(VecAXPY(yy,beta[d-1]/xi[d-2],yyy));
  CHKERRQ(VecResetArray(yyy));
  CHKERRQ(VecResetArray(yy));

  for (i=d-2;i>0;i--) {
    CHKERRQ(VecPlaceArray(yyy,py+(i-1)*m));
    CHKERRQ(VecPlaceArray(yy,py+i*m));
    CHKERRQ(VecAXPY(yy,beta[i]/xi[i-1],yyy));
    CHKERRQ(VecResetArray(yyy));
    CHKERRQ(VecResetArray(yy));
  }

  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  CHKERRQ(PetscFree(t));
  PetscFunctionReturn(0);
}

static PetscErrorCode BackTransform_FullBasis(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  NEP            nep;

  PetscFunctionBegin;
  CHKERRQ(STShellGetContext(st,&nep));
  CHKERRQ(NEPNLEIGSBackTransform((PetscObject)nep,n,eigr,eigi));
  PetscFunctionReturn(0);
}

static PetscErrorCode Apply_FullBasis(ST st,Vec x,Vec y)
{
  NEP            nep;
  NEP_NLEIGS     *ctx;

  PetscFunctionBegin;
  CHKERRQ(STShellGetContext(st,&nep));
  ctx = (NEP_NLEIGS*)nep->data;
  CHKERRQ(MatMult(ctx->A,x,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode ApplyTranspose_FullBasis(ST st,Vec x,Vec y)
{
  NEP            nep;
  NEP_NLEIGS     *ctx;

  PetscFunctionBegin;
  CHKERRQ(STShellGetContext(st,&nep));
  ctx = (NEP_NLEIGS*)nep->data;
  CHKERRQ(MatMultTranspose(ctx->A,x,y));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSetUp_NLEIGS_FullBasis(NEP nep)
{
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  ST             st;
  Mat            Q;
  PetscInt       i=0,deg=ctx->nmat-1;
  PetscBool      trackall,istrivial,ks;
  PetscScalar    *epsarray,*neparray;
  Vec            veps,w=NULL;
  EPSWhich       which;

  PetscFunctionBegin;
  PetscCheck(ctx->nshifts==0,PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"The full-basis option is not supported with rational Krylov");
  if (!ctx->eps) CHKERRQ(NEPNLEIGSGetEPS(nep,&ctx->eps));
  CHKERRQ(EPSGetST(ctx->eps,&st));
  CHKERRQ(EPSSetTarget(ctx->eps,nep->target));
  CHKERRQ(STSetDefaultShift(st,nep->target));
  if (!((PetscObject)(ctx->eps))->type_name) {
    CHKERRQ(EPSSetType(ctx->eps,EPSKRYLOVSCHUR));
  } else {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)ctx->eps,EPSKRYLOVSCHUR,&ks));
    PetscCheck(ks,PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Full-basis option only implemented for Krylov-Schur");
  }
  CHKERRQ(STSetType(st,STSHELL));
  CHKERRQ(STShellSetContext(st,nep));
  CHKERRQ(STShellSetBackTransform(st,BackTransform_FullBasis));
  CHKERRQ(KSPGetOperators(ctx->ksp[0],&Q,NULL));
  CHKERRQ(MatCreateVecsEmpty(Q,&ctx->w[0],&ctx->w[1]));
  CHKERRQ(MatCreateVecsEmpty(Q,&ctx->w[2],&ctx->w[3]));
  CHKERRQ(PetscLogObjectParents(nep,6,ctx->w));
  CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)nep),deg*nep->nloc,deg*nep->nloc,deg*nep->n,deg*nep->n,nep,&ctx->A));
  CHKERRQ(MatShellSetOperation(ctx->A,MATOP_MULT,(void(*)(void))MatMult_FullBasis_Sinvert));
  CHKERRQ(MatShellSetOperation(ctx->A,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_FullBasis_Sinvert));
  CHKERRQ(STShellSetApply(st,Apply_FullBasis));
  CHKERRQ(STShellSetApplyTranspose(st,ApplyTranspose_FullBasis));
  CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->A));
  CHKERRQ(EPSSetOperators(ctx->eps,ctx->A,NULL));
  CHKERRQ(EPSSetProblemType(ctx->eps,EPS_NHEP));
  switch (nep->which) {
    case NEP_TARGET_MAGNITUDE:   which = EPS_TARGET_MAGNITUDE; break;
    case NEP_TARGET_REAL:        which = EPS_TARGET_REAL; break;
    case NEP_TARGET_IMAGINARY:   which = EPS_TARGET_IMAGINARY; break;
    case NEP_WHICH_USER:         which = EPS_WHICH_USER;
      CHKERRQ(EPSSetEigenvalueComparison(ctx->eps,nep->sc->comparison,nep->sc->comparisonctx));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Should set a target selection in NEPSetWhichEigenpairs()");
  }
  CHKERRQ(EPSSetWhichEigenpairs(ctx->eps,which));
  CHKERRQ(RGIsTrivial(nep->rg,&istrivial));
  if (!istrivial) CHKERRQ(EPSSetRG(ctx->eps,nep->rg));
  CHKERRQ(EPSSetDimensions(ctx->eps,nep->nev,nep->ncv,nep->mpd));
  CHKERRQ(EPSSetTolerances(ctx->eps,SlepcDefaultTol(nep->tol),nep->max_it));
  CHKERRQ(EPSSetTwoSided(ctx->eps,nep->twosided));
  /* Transfer the trackall option from pep to eps */
  CHKERRQ(NEPGetTrackAll(nep,&trackall));
  CHKERRQ(EPSSetTrackAll(ctx->eps,trackall));

  /* process initial vector */
  if (nep->nini<0) {
    CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject)ctx->eps),deg*nep->nloc,deg*nep->n,&veps));
    CHKERRQ(VecGetArray(veps,&epsarray));
    for (i=0;i<deg;i++) {
      if (i<-nep->nini) {
        CHKERRQ(VecGetArray(nep->IS[i],&neparray));
        CHKERRQ(PetscArraycpy(epsarray+i*nep->nloc,neparray,nep->nloc));
        CHKERRQ(VecRestoreArray(nep->IS[i],&neparray));
      } else {
        if (!w) CHKERRQ(VecDuplicate(nep->IS[0],&w));
        CHKERRQ(VecSetRandom(w,NULL));
        CHKERRQ(VecGetArray(w,&neparray));
        CHKERRQ(PetscArraycpy(epsarray+i*nep->nloc,neparray,nep->nloc));
        CHKERRQ(VecRestoreArray(w,&neparray));
      }
    }
    CHKERRQ(VecRestoreArray(veps,&epsarray));
    CHKERRQ(EPSSetInitialSpace(ctx->eps,1,&veps));
    CHKERRQ(VecDestroy(&veps));
    CHKERRQ(VecDestroy(&w));
    CHKERRQ(SlepcBasisDestroy_Private(&nep->nini,&nep->IS));
  }

  CHKERRQ(EPSSetUp(ctx->eps));
  CHKERRQ(EPSGetDimensions(ctx->eps,NULL,&nep->ncv,&nep->mpd));
  CHKERRQ(EPSGetTolerances(ctx->eps,NULL,&nep->max_it));
  CHKERRQ(NEPAllocateSolution(nep,0));
  PetscFunctionReturn(0);
}

/*
   NEPNLEIGSExtract_None - Extracts the first block of the basis
   and normalizes the columns.
*/
static PetscErrorCode NEPNLEIGSExtract_None(NEP nep,EPS eps)
{
  PetscInt          i,k,m,d;
  const PetscScalar *px;
  PetscScalar       sigma=nep->target,*b;
  Mat               A;
  Vec               xxr,xxi=NULL,w,t,xx;
  PetscReal         norm;
  NEP_NLEIGS        *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  d = ctx->nmat-1;
  CHKERRQ(EPSGetOperators(eps,&A,NULL));
  CHKERRQ(MatCreateVecs(A,&xxr,NULL));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(VecDuplicate(xxr,&xxi));
#endif
  w = nep->work[0];
  for (i=0;i<nep->nconv;i++) {
    CHKERRQ(EPSGetEigenvector(eps,i,xxr,xxi));
    CHKERRQ(VecGetArrayRead(xxr,&px));
    CHKERRQ(VecPlaceArray(w,px));
    CHKERRQ(BVInsertVec(nep->V,i,w));
    CHKERRQ(BVNormColumn(nep->V,i,NORM_2,&norm));
    CHKERRQ(BVScaleColumn(nep->V,i,1.0/norm));
    CHKERRQ(VecResetArray(w));
    CHKERRQ(VecRestoreArrayRead(xxr,&px));
  }
  if (nep->twosided) {
    CHKERRQ(PetscMalloc1(ctx->nmat,&b));
    CHKERRQ(NEPNLEIGSEvalNRTFunct(nep,d,sigma,b));
    m = nep->nloc;
    xx = ctx->w[0];
    w = nep->work[0]; t = nep->work[1];
    for (k=0;k<nep->nconv;k++) {
      CHKERRQ(EPSGetLeftEigenvector(eps,k,xxr,xxi));
      CHKERRQ(VecGetArrayRead(xxr,&px));
      CHKERRQ(VecPlaceArray(xx,px+(d-1)*m));
      CHKERRQ(VecCopy(xx,w));
      CHKERRQ(VecScale(w,PetscConj(b[d-1])));
      CHKERRQ(VecResetArray(xx));
      for (i=0;i<d-1;i++) {
        CHKERRQ(VecPlaceArray(xx,px+i*m));
        CHKERRQ(VecAXPY(w,PetscConj(b[i]),xx));
        CHKERRQ(VecResetArray(xx));
      }
      CHKERRQ(VecConjugate(w));
      CHKERRQ(KSPSolveTranspose(ctx->ksp[0],w,t));
      CHKERRQ(VecConjugate(t));
      CHKERRQ(BVInsertVec(nep->W,k,t));
      CHKERRQ(BVNormColumn(nep->W,k,NORM_2,&norm));
      CHKERRQ(BVScaleColumn(nep->W,k,1.0/norm));
      CHKERRQ(VecRestoreArrayRead(xxr,&px));
    }
    CHKERRQ(PetscFree(b));
  }
  CHKERRQ(VecDestroy(&xxr));
#if !defined(PETSC_USE_COMPLEX)
  CHKERRQ(VecDestroy(&xxi));
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSolve_NLEIGS_FullBasis(NEP nep)
{
  NEP_NLEIGS     *ctx = (NEP_NLEIGS*)nep->data;
  PetscInt       i;
  PetscScalar    eigi=0.0;

  PetscFunctionBegin;
  CHKERRQ(EPSSolve(ctx->eps));
  CHKERRQ(EPSGetConverged(ctx->eps,&nep->nconv));
  CHKERRQ(EPSGetIterationNumber(ctx->eps,&nep->its));
  CHKERRQ(EPSGetConvergedReason(ctx->eps,(EPSConvergedReason*)&nep->reason));

  /* recover eigenvalues */
  for (i=0;i<nep->nconv;i++) {
    CHKERRQ(EPSGetEigenpair(ctx->eps,i,&nep->eigr[i],&eigi,NULL,NULL));
#if !defined(PETSC_USE_COMPLEX)
    PetscCheck(eigi==0.0,PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Complex value requires complex arithmetic");
#endif
  }
  CHKERRQ(NEPNLEIGSExtract_None(nep,ctx->eps));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPNLEIGSSetEPS_NLEIGS(NEP nep,EPS eps)
{
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)eps));
  CHKERRQ(EPSDestroy(&ctx->eps));
  ctx->eps = eps;
  CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->eps));
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   NEPNLEIGSSetEPS - Associate an eigensolver object (EPS) to the NLEIGS solver.

   Collective on nep

   Input Parameters:
+  nep - nonlinear eigenvalue solver
-  eps - the eigensolver object

   Level: advanced

.seealso: NEPNLEIGSGetEPS()
@*/
PetscErrorCode NEPNLEIGSSetEPS(NEP nep,EPS eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(nep,1,eps,2);
  CHKERRQ(PetscTryMethod(nep,"NEPNLEIGSSetEPS_C",(NEP,EPS),(nep,eps)));
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSMonitor_NLEIGS(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  NEP            nep = (NEP)ctx;
  PetscInt       i,nv = PetscMin(nest,nep->ncv);

  PetscFunctionBegin;
  for (i=0;i<nv;i++) {
    nep->eigr[i]   = eigr[i];
    nep->eigi[i]   = eigi[i];
    nep->errest[i] = errest[i];
  }
  CHKERRQ(NEPNLEIGSBackTransform((PetscObject)nep,nv,nep->eigr,nep->eigi));
  CHKERRQ(NEPMonitor(nep,its,nconv,nep->eigr,nep->eigi,nep->errest,nest));
  PetscFunctionReturn(0);
}

PetscErrorCode NEPNLEIGSGetEPS_NLEIGS(NEP nep,EPS *eps)
{
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  if (!ctx->eps) {
    CHKERRQ(EPSCreate(PetscObjectComm((PetscObject)nep),&ctx->eps));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ctx->eps,(PetscObject)nep,1));
    CHKERRQ(EPSSetOptionsPrefix(ctx->eps,((PetscObject)nep)->prefix));
    CHKERRQ(EPSAppendOptionsPrefix(ctx->eps,"nep_nleigs_"));
    CHKERRQ(PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->eps));
    CHKERRQ(PetscObjectSetOptions((PetscObject)ctx->eps,((PetscObject)nep)->options));
    CHKERRQ(EPSMonitorSet(ctx->eps,EPSMonitor_NLEIGS,nep,NULL));
  }
  *eps = ctx->eps;
  PetscFunctionReturn(0);
}

/*@
   NEPNLEIGSGetEPS - Retrieve the eigensolver object (EPS) associated
   to the nonlinear eigenvalue solver.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: NEPNLEIGSSetEPS()
@*/
PetscErrorCode NEPNLEIGSGetEPS(NEP nep,EPS *eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(eps,2);
  CHKERRQ(PetscUseMethod(nep,"NEPNLEIGSGetEPS_C",(NEP,EPS*),(nep,eps)));
  PetscFunctionReturn(0);
}
