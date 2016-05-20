/*

   SLEPc nonlinear eigensolver: "interpol"

   Method: Polynomial interpolation

   Algorithm:

       Uses a PEP object to solve the interpolated NEP. Currently supports
       only Chebyshev interpolation on an interval.

   References:

       [1] C. Effenberger and D. Kresser, "Chebyshev interpolation for
           nonlinear eigenvalue problems", BIT 52:933-951, 2012.

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

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include <slepc/private/pepimpl.h>         /*I "slepcpep.h" I*/

typedef struct {
  PEP       pep;
  PetscInt  deg;
} NEP_INTERPOL;

#undef __FUNCT__
#define __FUNCT__ "NEPSetUp_Interpol"
PetscErrorCode NEPSetUp_Interpol(NEP nep)
{
  PetscErrorCode ierr;
  NEP_INTERPOL   *ctx = (NEP_INTERPOL*)nep->data;
  ST             st;
  RG             rg;
  PetscReal      a,b,c,d,s,tol;
  PetscScalar    zero=0.0;
  PetscBool      flg,istrivial,trackall;
  PetscInt       its,in;

  PetscFunctionBegin;
  ierr = NEPSetDimensions_Default(nep,nep->nev,&nep->ncv,&nep->mpd);CHKERRQ(ierr);
  if (nep->ncv>nep->nev+nep->mpd) SETERRQ(PetscObjectComm((PetscObject)nep),1,"The value of ncv must not be larger than nev+mpd");
  if (!nep->max_it) nep->max_it = PetscMax(5000,2*nep->n/nep->ncv);
  if (nep->fui!=NEP_USER_INTERFACE_SPLIT) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"NEPINTERPOL only available for split operator");
  if (nep->stopping!=NEPStoppingBasic) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"This solver does not support user-defined stopping test");

  /* transfer PEP options */
  if (!ctx->pep) { ierr = NEPInterpolGetPEP(nep,&ctx->pep);CHKERRQ(ierr); }
  ierr = PEPSetBV(ctx->pep,nep->V);CHKERRQ(ierr);
  ierr = PEPSetBasis(ctx->pep,PEP_BASIS_CHEBYSHEV1);CHKERRQ(ierr);
  ierr = PEPSetWhichEigenpairs(ctx->pep,PEP_TARGET_MAGNITUDE);CHKERRQ(ierr);
  ierr = PEPGetST(ctx->pep,&st);CHKERRQ(ierr);
  ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);
  ierr = PEPSetDimensions(ctx->pep,nep->nev,nep->ncv?nep->ncv:PETSC_DEFAULT,nep->mpd?nep->mpd:PETSC_DEFAULT);CHKERRQ(ierr);
  tol=ctx->pep->tol;
  if (tol==PETSC_DEFAULT) tol = (nep->tol==PETSC_DEFAULT)?SLEPC_DEFAULT_TOL/10.0:nep->tol/10.0;
  its=ctx->pep->max_it;
  if (!its) its = nep->max_it?nep->max_it:PETSC_DEFAULT;
  ierr = PEPSetTolerances(ctx->pep,tol,its);CHKERRQ(ierr);
  ierr = NEPGetTrackAll(nep,&trackall);CHKERRQ(ierr);
  ierr = PEPSetTrackAll(ctx->pep,trackall);CHKERRQ(ierr);

  /* transfer region options */
  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (istrivial) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"NEPINTERPOL requires a nontrivial region");
  ierr = PetscObjectTypeCompare((PetscObject)nep->rg,RGINTERVAL,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Only implemented for interval regions");
  ierr = RGIntervalGetEndpoints(nep->rg,&a,&b,&c,&d);CHKERRQ(ierr);
  if (a<=-PETSC_MAX_REAL || b>=PETSC_MAX_REAL) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Only implemented for bounded intervals");
  ierr = PEPGetRG(ctx->pep,&rg);CHKERRQ(ierr);
  ierr = RGSetType(rg,RGINTERVAL);CHKERRQ(ierr);
  if (a==b) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Only implemented for intervals on the real axis");
  s = 2.0/(b-a);
  c = c*s;
  d = d*s;
  ierr = RGIntervalSetEndpoints(rg,-1.0,1.0,c,d);CHKERRQ(ierr);
  ierr = RGCheckInside(nep->rg,1,&nep->target,&zero,&in);CHKERRQ(ierr);
  if (in<0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"The target is not inside the target set");
  ierr = PEPSetTarget(ctx->pep,(nep->target-(a+b)/2)*s);CHKERRQ(ierr);

  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ChebyshevNodes"
/*
  Input: 
    d, number of nodes to compute
    a,b, interval extrems
  Output:
    *x, array containing the d Chebyshev nodes of the interval [a,b]
    *dct2, coefficients to compute a discrete cosine transformation (DCT-II)
*/
static PetscErrorCode ChebyshevNodes(PetscInt d,PetscReal a,PetscReal b,PetscScalar *x,PetscReal *dct2)
{
  PetscInt  j,i;
  PetscReal t;

  PetscFunctionBegin;
  for (j=0;j<d+1;j++) {
    t = ((2*j+1)*PETSC_PI)/(2*(d+1));
    x[j] = (a+b)/2.0+((b-a)/2.0)*PetscCosReal(t);
    for (i=0;i<d+1;i++) dct2[j*(d+1)+i] = PetscCosReal(i*t);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSolve_Interpol"
PetscErrorCode NEPSolve_Interpol(NEP nep)
{
  PetscErrorCode ierr;
  NEP_INTERPOL   *ctx = (NEP_INTERPOL*)nep->data;
  Mat            *A;   /*T=nep->function,Tp=nep->jacobian;*/
  PetscScalar    *x,*fx,t;
  PetscReal      *cs,a,b,s;
  PetscInt       i,j,k,deg=ctx->deg;

  PetscFunctionBegin;
  ierr = PetscMalloc4(deg+1,&A,(deg+1)*(deg+1),&cs,deg+1,&x,(deg+1)*nep->nt,&fx);CHKERRQ(ierr);
  ierr = RGIntervalGetEndpoints(nep->rg,&a,&b,NULL,NULL);CHKERRQ(ierr);
  ierr = ChebyshevNodes(deg,a,b,x,cs);CHKERRQ(ierr);
  for (j=0;j<nep->nt;j++) {
    for (i=0;i<=deg;i++) {
      ierr = FNEvaluateFunction(nep->f[j],x[i],&fx[i+j*(deg+1)]);CHKERRQ(ierr);
    }
  }

  /* Polynomial coefficients */
  for (k=0;k<=deg;k++) {
    ierr = MatDuplicate(nep->A[0],MAT_COPY_VALUES,&A[k]);CHKERRQ(ierr);
    t = 0.0;
    for (i=0;i<deg+1;i++) t += fx[i]*cs[i*(deg+1)+k];
    t *= 2.0/(deg+1); 
    if (k==0) t /= 2.0;
    ierr = MatScale(A[k],t);CHKERRQ(ierr);
    for (j=1;j<nep->nt;j++) {
      t = 0.0;
      for (i=0;i<deg+1;i++) t += fx[i+j*(deg+1)]*cs[i*(deg+1)+k];
      t *= 2.0/(deg+1); 
      if (k==0) t /= 2.0;
      ierr = MatAXPY(A[k],t,nep->A[j],nep->mstr);CHKERRQ(ierr);
    }
  }

  ierr = PEPSetOperators(ctx->pep,deg+1,A);CHKERRQ(ierr);
  for (k=0;k<=deg;k++) {
    ierr = MatDestroy(&A[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(A,cs,x,fx);CHKERRQ(ierr);

  /* Solve polynomial eigenproblem */
  ierr = PEPSolve(ctx->pep);CHKERRQ(ierr);
  ierr = PEPGetConverged(ctx->pep,&nep->nconv);CHKERRQ(ierr);
  ierr = PEPGetIterationNumber(ctx->pep,&nep->its);CHKERRQ(ierr);
  ierr = PEPGetConvergedReason(ctx->pep,(PEPConvergedReason*)&nep->reason);CHKERRQ(ierr);
  s = 2.0/(b-a);
  for (i=0;i<nep->nconv;i++) {
    ierr = PEPGetEigenpair(ctx->pep,i,&nep->eigr[i],&nep->eigi[i],NULL,NULL);CHKERRQ(ierr);
    nep->eigr[i] /= s;
    nep->eigr[i] += (a+b)/2.0;
    nep->eigi[i] /= s;
  }
  nep->state = NEP_STATE_EIGENVECTORS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPMonitor_Interpol"
static PetscErrorCode PEPMonitor_Interpol(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  PetscInt       i,n;
  NEP            nep = (NEP)ctx;
  PetscReal      a,b,s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  n = PetscMin(nest,nep->ncv);
  for (i=0;i<n;i++) {
    nep->eigr[i]   = eigr[i];
    nep->eigi[i]   = eigi[i];
    nep->errest[i] = errest[i];
  }
  ierr = STBackTransform(pep->st,n,nep->eigr,nep->eigi);CHKERRQ(ierr);
  ierr = RGIntervalGetEndpoints(nep->rg,&a,&b,NULL,NULL);CHKERRQ(ierr);
  s = 2.0/(b-a);
  for (i=0;i<n;i++) {
    nep->eigr[i] /= s;
    nep->eigr[i] += (a+b)/2.0;
    nep->eigi[i] /= s;
  }  
  ierr = NEPMonitor(nep,its,nconv,nep->eigr,nep->eigi,nep->errest,nest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetFromOptions_Interpol"
PetscErrorCode NEPSetFromOptions_Interpol(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  PetscErrorCode ierr;
  NEP_INTERPOL   *ctx = (NEP_INTERPOL*)nep->data;

  PetscFunctionBegin;
  if (!ctx->pep) { ierr = NEPInterpolGetPEP(nep,&ctx->pep);CHKERRQ(ierr); }
  ierr = PEPSetFromOptions(ctx->pep);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"NEP Interpol Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nep_interpol_degree","Degree of interpolation polynomial","NEPInterpolSetDegree",ctx->deg,&ctx->deg,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPInterpolSetDegree_Interpol"
static PetscErrorCode NEPInterpolSetDegree_Interpol(NEP nep,PetscInt deg)
{
  NEP_INTERPOL *ctx = (NEP_INTERPOL*)nep->data;

  PetscFunctionBegin;
  ctx->deg = deg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPInterpolSetDegree"
/*@
   NEPInterpolSetDegree - Sets the degree of the interpolation polynomial.

   Collective on NEP

   Input Parameters:
+  nep - nonlinear eigenvalue solver
-  deg - polynomial degree

   Level: advanced

.seealso: NEPInterpolGetDegree()
@*/
PetscErrorCode NEPInterpolSetDegree(NEP nep,PetscInt deg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,deg,2);
  ierr = PetscTryMethod(nep,"NEPInterpolSetDegree_C",(NEP,PetscInt),(nep,deg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPInterpolGetDegree_Interpol"
static PetscErrorCode NEPInterpolGetDegree_Interpol(NEP nep,PetscInt *deg)
{
  NEP_INTERPOL *ctx = (NEP_INTERPOL*)nep->data;

  PetscFunctionBegin;
  *deg = ctx->deg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPInterpolGetDegree"
/*@
   NEPInterpolGetDegree - Gets the degree of the interpolation polynomial.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  deg - the polynomial degree

   Level: advanced

.seealso: NEPInterpolSetDegree()
@*/
PetscErrorCode NEPInterpolGetDegree(NEP nep,PetscInt *deg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(deg,2);
  ierr = PetscUseMethod(nep,"NEPInterpolGetDegree_C",(NEP,PetscInt*),(nep,deg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPInterpolSetPEP_Interpol"
static PetscErrorCode NEPInterpolSetPEP_Interpol(NEP nep,PEP pep)
{
  PetscErrorCode ierr;
  NEP_INTERPOL   *ctx = (NEP_INTERPOL*)nep->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)pep);CHKERRQ(ierr);
  ierr = PEPDestroy(&ctx->pep);CHKERRQ(ierr);
  ctx->pep = pep;
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->pep);CHKERRQ(ierr);
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPInterpolSetPEP"
/*@
   NEPInterpolSetPEP - Associate a polynomial eigensolver object (PEP) to the
   nonlinear eigenvalue solver.

   Collective on NEP

   Input Parameters:
+  nep - nonlinear eigenvalue solver
-  pep - the polynomial eigensolver object

   Level: advanced

.seealso: NEPInterpolGetPEP()
@*/
PetscErrorCode NEPInterpolSetPEP(NEP nep,PEP pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(pep,PEP_CLASSID,2);
  PetscCheckSameComm(nep,1,pep,2);
  ierr = PetscTryMethod(nep,"NEPInterpolSetPEP_C",(NEP,PEP),(nep,pep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPInterpolGetPEP_Interpol"
static PetscErrorCode NEPInterpolGetPEP_Interpol(NEP nep,PEP *pep)
{
  PetscErrorCode ierr;
  NEP_INTERPOL   *ctx = (NEP_INTERPOL*)nep->data;
  ST             st;

  PetscFunctionBegin;
  if (!ctx->pep) {
    ierr = PEPCreate(PetscObjectComm((PetscObject)nep),&ctx->pep);CHKERRQ(ierr);
    ierr = PEPSetOptionsPrefix(ctx->pep,((PetscObject)nep)->prefix);CHKERRQ(ierr);
    ierr = PEPAppendOptionsPrefix(ctx->pep,"nep_interpol_");CHKERRQ(ierr);
    ierr = PEPGetST(ctx->pep,&st);CHKERRQ(ierr);
    ierr = STSetOptionsPrefix(st,((PetscObject)ctx->pep)->prefix);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->pep,(PetscObject)nep,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->pep);CHKERRQ(ierr);
    ierr = PEPMonitorSet(ctx->pep,PEPMonitor_Interpol,nep,NULL);CHKERRQ(ierr);
  }
  *pep = ctx->pep;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPInterpolGetPEP"
/*@
   NEPInterpolGetPEP - Retrieve the polynomial eigensolver object (PEP)
   associated with the nonlinear eigenvalue solver.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  pep - the polynomial eigensolver object

   Level: advanced

.seealso: NEPInterpolSetPEP()
@*/
PetscErrorCode NEPInterpolGetPEP(NEP nep,PEP *pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(pep,2);
  ierr = PetscUseMethod(nep,"NEPInterpolGetPEP_C",(NEP,PEP*),(nep,pep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPView_Interpol"
PetscErrorCode NEPView_Interpol(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_INTERPOL   *ctx = (NEP_INTERPOL*)nep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!ctx->pep) { ierr = NEPInterpolGetPEP(nep,&ctx->pep);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPrintf(viewer,"  Interpol: polynomial degree %D\n",ctx->deg);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PEPView(ctx->pep,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPReset_Interpol"
PetscErrorCode NEPReset_Interpol(NEP nep)
{
  PetscErrorCode ierr;
  NEP_INTERPOL   *ctx = (NEP_INTERPOL*)nep->data;

  PetscFunctionBegin;
  if (!ctx->pep) { ierr = PEPReset(ctx->pep);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPDestroy_Interpol"
PetscErrorCode NEPDestroy_Interpol(NEP nep)
{
  PetscErrorCode ierr;
  NEP_INTERPOL   *ctx = (NEP_INTERPOL*)nep->data;

  PetscFunctionBegin;
  ierr = PEPDestroy(&ctx->pep);CHKERRQ(ierr);
  ierr = PetscFree(nep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPInterpolSetDegree_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPInterpolGetDegree_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPInterpolSetPEP_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPInterpolGetPEP_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCreate_Interpol"
PETSC_EXTERN PetscErrorCode NEPCreate_Interpol(NEP nep)
{
  PetscErrorCode ierr;
  NEP_INTERPOL   *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(nep,&ctx);CHKERRQ(ierr);
  ctx->deg  = 5;
  nep->data = (void*)ctx;

  nep->ops->solve          = NEPSolve_Interpol;
  nep->ops->setup          = NEPSetUp_Interpol;
  nep->ops->setfromoptions = NEPSetFromOptions_Interpol;
  nep->ops->reset          = NEPReset_Interpol;
  nep->ops->destroy        = NEPDestroy_Interpol;
  nep->ops->view           = NEPView_Interpol;
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPInterpolSetDegree_C",NEPInterpolSetDegree_Interpol);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPInterpolGetDegree_C",NEPInterpolGetDegree_Interpol);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPInterpolSetPEP_C",NEPInterpolSetPEP_Interpol);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPInterpolGetPEP_C",NEPInterpolGetPEP_Interpol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

