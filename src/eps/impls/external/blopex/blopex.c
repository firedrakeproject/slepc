/*
   This file implements a wrapper to the BLOPEX package

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>
#include <slepc-private/stimpl.h>
#include "slepc-interface.h"
#include <blopex_lobpcg.h>
#include <blopex_interpreter.h>
#include <blopex_multivector.h>
#include <blopex_temp_multivector.h>

PetscErrorCode EPSSolve_BLOPEX(EPS);

typedef struct {
  lobpcg_Tolerance           tol;
  lobpcg_BLASLAPACKFunctions blap_fn;
  mv_MultiVectorPtr          eigenvectors;
  mv_MultiVectorPtr          Y;
  mv_InterfaceInterpreter    ii;
  ST                         st;
  Vec                        w;
} EPS_BLOPEX;

#undef __FUNCT__
#define __FUNCT__ "Precond_FnSingleVector"
static void Precond_FnSingleVector(void *data,void *x,void *y)
{
  PetscErrorCode ierr;
  EPS            eps = (EPS)data;
  EPS_BLOPEX     *blopex = (EPS_BLOPEX*)eps->data;

  PetscFunctionBegin;
  ierr = KSPSolve(blopex->st->ksp,(Vec)x,(Vec)y);CHKERRABORT(PetscObjectComm((PetscObject)eps),ierr);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "Precond_FnMultiVector"
static void Precond_FnMultiVector(void *data,void *x,void *y)
{
  EPS        eps = (EPS)data;
  EPS_BLOPEX *blopex = (EPS_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(Precond_FnSingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "OperatorASingleVector"
static void OperatorASingleVector(void *data,void *x,void *y)
{
  PetscErrorCode ierr;
  EPS            eps = (EPS)data;
  EPS_BLOPEX     *blopex = (EPS_BLOPEX*)eps->data;
  Mat            A,B;
  PetscScalar    sigma;
  PetscInt       nmat;

  PetscFunctionBegin;
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRABORT(PetscObjectComm((PetscObject)eps),ierr);
  ierr = STGetOperators(eps->st,0,&A);CHKERRABORT(PetscObjectComm((PetscObject)eps),ierr);
  if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRABORT(PetscObjectComm((PetscObject)eps),ierr); }
  ierr = MatMult(A,(Vec)x,(Vec)y);CHKERRABORT(PetscObjectComm((PetscObject)eps),ierr);
  ierr = STGetShift(eps->st,&sigma);CHKERRABORT(PetscObjectComm((PetscObject)eps),ierr);
  if (sigma != 0.0) {
    if (nmat>1) {
      ierr = MatMult(B,(Vec)x,blopex->w);CHKERRABORT(PetscObjectComm((PetscObject)eps),ierr);
    } else {
      ierr = VecCopy((Vec)x,blopex->w);CHKERRABORT(PetscObjectComm((PetscObject)eps),ierr);
    }
    ierr = VecAXPY((Vec)y,-sigma,blopex->w);CHKERRABORT(PetscObjectComm((PetscObject)eps),ierr);
  }
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "OperatorAMultiVector"
static void OperatorAMultiVector(void *data,void *x,void *y)
{
  EPS        eps = (EPS)data;
  EPS_BLOPEX *blopex = (EPS_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(OperatorASingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "OperatorBSingleVector"
static void OperatorBSingleVector(void *data,void *x,void *y)
{
  PetscErrorCode ierr;
  EPS            eps = (EPS)data;
  Mat            B;

  PetscFunctionBegin;
  ierr = STGetOperators(eps->st,1,&B);CHKERRABORT(PetscObjectComm((PetscObject)eps),ierr);
  ierr = MatMult(B,(Vec)x,(Vec)y);CHKERRABORT(PetscObjectComm((PetscObject)eps),ierr);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "OperatorBMultiVector"
static void OperatorBMultiVector(void *data,void *x,void *y)
{
  EPS        eps = (EPS)data;
  EPS_BLOPEX *blopex = (EPS_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(OperatorBSingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_BLOPEX"
PetscErrorCode EPSSetUp_BLOPEX(EPS eps)
{
#if defined(PETSC_MISSING_LAPACK_POTRF) || defined(PETSC_MISSING_LAPACK_SYGV)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRF/SYGV - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  EPS_BLOPEX     *blopex = (EPS_BLOPEX*)eps->data;
  PetscBool      isPrecond,istrivial,flg;
  BV             Y;
  PetscInt       k;

  PetscFunctionBegin;
  if (!eps->ishermitian) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"blopex only works for hermitian problems");
  if (!eps->which) eps->which = EPS_SMALLEST_REAL;
  if (eps->which!=EPS_SMALLEST_REAL) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");

  ierr = STSetUp(eps->st);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STPRECOND,&isPrecond);CHKERRQ(ierr);
  if (!isPrecond) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"blopex only works with STPRECOND");
  blopex->st = eps->st;

  eps->ncv = eps->nev = PetscMin(eps->nev,eps->n);
  if (eps->mpd) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");

  /* blopex only works with BVVECS or BVCONTIGUOUS, if different set to CONTIGUOUS */
  if (!eps->V) { ierr = EPSGetBV(eps,&eps->V);CHKERRQ(ierr); }
  ierr = PetscObjectTypeCompareAny((PetscObject)eps->V,&flg,BVVECS,BVCONTIGUOUS,"");CHKERRQ(ierr);
  if (!flg) {
    ierr = BVSetType(eps->V,BVCONTIGUOUS);CHKERRQ(ierr);
  }

  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,1);CHKERRQ(ierr);

  if (eps->converged == EPSConvergedEigRelative) {
    blopex->tol.absolute = 0.0;
    blopex->tol.relative = eps->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:eps->tol;
  } else if (eps->converged == EPSConvergedAbsolute) {
    blopex->tol.absolute = eps->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:eps->tol;
    blopex->tol.relative = 0.0;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Convergence test not supported in this solver");
  }

  SLEPCSetupInterpreter(&blopex->ii);
  blopex->eigenvectors = mv_MultiVectorCreateFromSampleVector(&blopex->ii,eps->ncv,eps->V);

  ierr = BVGetVec(eps->V,&blopex->w);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)blopex->w);CHKERRQ(ierr);
  if (eps->nds<0) {
    k = -eps->nds;
    ierr = BVCreate(PetscObjectComm((PetscObject)eps),&Y);CHKERRQ(ierr);
    ierr = BVSetSizesFromVec(Y,blopex->w,k);CHKERRQ(ierr);
    ierr = BVSetType(Y,BVVECS);CHKERRQ(ierr);
    ierr = BVInsertVecs(Y,0,&k,eps->defl,PETSC_FALSE);CHKERRQ(ierr);
    ierr = SlepcBasisDestroy_Private(&eps->nds,&eps->defl);CHKERRQ(ierr);
    blopex->Y = mv_MultiVectorCreateFromSampleVector(&blopex->ii,k,Y);
    ierr = BVDestroy(&Y);CHKERRQ(ierr);
  } else blopex->Y = NULL;

#if defined(PETSC_USE_COMPLEX)
  blopex->blap_fn.zpotrf = PETSC_zpotrf_interface;
  blopex->blap_fn.zhegv = PETSC_zsygv_interface;
#else
  blopex->blap_fn.dpotrf = PETSC_dpotrf_interface;
  blopex->blap_fn.dsygv = PETSC_dsygv_interface;
#endif

  if (eps->extraction) { ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr); }
  ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support region filtering");

  /* dispatch solve method */
  eps->ops->solve = EPSSolve_BLOPEX;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_BLOPEX"
PetscErrorCode EPSSolve_BLOPEX(EPS eps)
{
  EPS_BLOPEX     *blopex = (EPS_BLOPEX*)eps->data;
  PetscScalar    sigma;
  int            i,j,info,its,nconv;
  double         *residhist=NULL;
  PetscErrorCode ierr;
#if defined(PETSC_USE_COMPLEX)
  komplex        *lambdahist=NULL;
#else
  double         *lambdahist=NULL;
#endif

  PetscFunctionBegin;
  /* Complete the initial basis with random vectors */
  for (i=eps->nini;i<eps->ncv;i++) {
    ierr = BVSetRandomColumn(eps->V,i,eps->rand);CHKERRQ(ierr);
  }

  if (eps->numbermonitors>0) {
    ierr = PetscMalloc2(eps->ncv*(eps->max_it+1),&lambdahist,eps->ncv*(eps->max_it+1),&residhist);CHKERRQ(ierr);
  }

#if defined(PETSC_USE_COMPLEX)
  info = lobpcg_solve_complex(blopex->eigenvectors,eps,OperatorAMultiVector,
        eps->isgeneralized?eps:NULL,eps->isgeneralized?OperatorBMultiVector:NULL,
        eps,Precond_FnMultiVector,blopex->Y,
        blopex->blap_fn,blopex->tol,eps->max_it,0,&its,
        (komplex*)eps->eigr,lambdahist,eps->ncv,eps->errest,residhist,eps->ncv);
#else
  info = lobpcg_solve_double(blopex->eigenvectors,eps,OperatorAMultiVector,
        eps->isgeneralized?eps:NULL,eps->isgeneralized?OperatorBMultiVector:NULL,
        eps,Precond_FnMultiVector,blopex->Y,
        blopex->blap_fn,blopex->tol,eps->max_it,0,&its,
        eps->eigr,lambdahist,eps->ncv,eps->errest,residhist,eps->ncv);
#endif
  if (info>0) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Error in blopex (code=%d)",info);

  if (eps->numbermonitors>0) {
    for (i=0;i<its;i++) {
      nconv = 0;
      for (j=0;j<eps->ncv;j++) {
        if (residhist[j+i*eps->ncv]>eps->tol) break;
        else nconv++;
      }
      ierr = EPSMonitor(eps,i,nconv,(PetscScalar*)lambdahist+i*eps->ncv,eps->eigi,residhist+i*eps->ncv,eps->ncv);CHKERRQ(ierr);
    }
    ierr = PetscFree2(lambdahist,residhist);CHKERRQ(ierr);
  }

  eps->its = its;
  eps->nconv = eps->ncv;
  ierr = STGetShift(eps->st,&sigma);CHKERRQ(ierr);
  if (sigma != 0.0) {
    for (i=0;i<eps->nconv;i++) eps->eigr[i]+=sigma;
  }
  if (info==-1) eps->reason = EPS_DIVERGED_ITS;
  else eps->reason = EPS_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSReset_BLOPEX"
PetscErrorCode EPSReset_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;
  EPS_BLOPEX     *blopex = (EPS_BLOPEX*)eps->data;

  PetscFunctionBegin;
  mv_MultiVectorDestroy(blopex->eigenvectors);
  mv_MultiVectorDestroy(blopex->Y);
  ierr = VecDestroy(&blopex->w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_BLOPEX"
PetscErrorCode EPSDestroy_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  LOBPCG_DestroyRandomContext();
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetFromOptions_BLOPEX"
PetscErrorCode EPSSetFromOptions_BLOPEX(EPS eps)
{
  PetscErrorCode  ierr;
  KSP             ksp;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EPS BLOPEX Options");CHKERRQ(ierr);
  LOBPCG_SetFromOptionsRandomContext();

  /* Set STPrecond as the default ST */
  if (!((PetscObject)eps->st)->type_name) {
    ierr = STSetType(eps->st,STPRECOND);CHKERRQ(ierr);
  }
  ierr = STPrecondSetKSPHasMat(eps->st,PETSC_TRUE);CHKERRQ(ierr);

  /* Set the default options of the KSP */
  ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_BLOPEX"
PETSC_EXTERN PetscErrorCode EPSCreate_BLOPEX(EPS eps)
{
  EPS_BLOPEX     *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  eps->ops->setup                = EPSSetUp_BLOPEX;
  eps->ops->setfromoptions       = EPSSetFromOptions_BLOPEX;
  eps->ops->destroy              = EPSDestroy_BLOPEX;
  eps->ops->reset                = EPSReset_BLOPEX;
  eps->ops->backtransform        = EPSBackTransform_Default;
  LOBPCG_InitRandomContext(PetscObjectComm((PetscObject)eps),eps->rand);
  PetscFunctionReturn(0);
}

