/*
   This file implements a wrapper to the FEAST package

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>        /*I "slepceps.h" I*/
#include <../src/eps/impls/external/feast/feastp.h>

PetscErrorCode EPSSolve_FEAST(EPS);

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_FEAST"
PetscErrorCode EPSSetUp_FEAST(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       ncv;
  PetscBool      issinv;
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;

  PetscFunctionBegin;
  if (eps->ncv) {
    if (eps->ncv<eps->nev+2) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The value of ncv must be at least nev+2");
  } else eps->ncv = PetscMin(PetscMax(20,2*eps->nev+1),eps->n); /* set default value of ncv */
  if (eps->mpd) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (!eps->max_it) eps->max_it = PetscMax(300,(PetscInt)(2*eps->n/eps->ncv));
  if (!eps->which) eps->which = EPS_ALL;

  ncv = eps->ncv;
  ierr = PetscFree(ctx->work1);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->nloc*ncv*sizeof(PetscScalar),&ctx->work1);CHKERRQ(ierr);
  ierr = PetscFree(ctx->work2);CHKERRQ(ierr);
  ierr = PetscMalloc(2*eps->nloc*ncv*sizeof(PetscReal),&ctx->work2);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(eps,3*eps->nloc*ncv*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscFree(ctx->Aq);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&ctx->Aq);CHKERRQ(ierr);
  ierr = PetscFree(ctx->Bq);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&ctx->Bq);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(eps,2*ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);

  if (!((PetscObject)(eps->st))->type_name) { /* default to shift-and-invert */
    ierr = STSetType(eps->st,STSINVERT);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompareAny((PetscObject)eps->st,&issinv,STSINVERT,STCAYLEY,"");CHKERRQ(ierr);
  if (!issinv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Shift-and-invert or Cayley ST is needed for FEAST");

  if (eps->extraction) { ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr); }

  if (eps->which!=EPS_ALL || (eps->inta==0.0 && eps->intb==0.0)) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"FEAST must be used with a computational interval");
  if (!eps->ishermitian) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"FEAST only available for symmetric/Hermitian eigenproblems");
  if (eps->balance!=EPS_BALANCE_NONE) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Balancing not supported in the Arpack interface");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,1);CHKERRQ(ierr);

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Left vectors not supported in this solver");
  eps->ops->solve = EPSSolve_FEAST;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_FEAST"
PetscErrorCode EPSSolve_FEAST(EPS eps)
{
  PetscErrorCode ierr;
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;
  PetscBLASInt   n,fpm[64],ijob,info,nev,ncv,loop;
  PetscReal      *evals,epsout;
  PetscInt       i,k,nmat;
  PetscScalar    *pV,Ze;
  Vec            x,y,w = eps->work[0];
  Mat            A,B;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(eps->nev,&nev);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(eps->ncv,&ncv);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(eps->nloc,&n);CHKERRQ(ierr);

  /* parameters */
  FEASTinit_(fpm);
  fpm[0] = (eps->numbermonitors>0)? 1: 0;                      /* runtime comments */
  ierr = PetscBLASIntCast(eps->max_it,&fpm[3]);CHKERRQ(ierr);  /* refinement loops */
#if !defined(PETSC_HAVE_MPIUNI)
  ierr = PetscBLASIntCast(MPI_Comm_c2f(PetscObjectComm((PetscObject)eps)),&fpm[8]);CHKERRQ(ierr);
#endif

  ierr = PetscMalloc(eps->ncv*sizeof(PetscReal),&evals);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)eps),1,eps->nloc,PETSC_DECIDE,NULL,&x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)eps),1,eps->nloc,PETSC_DECIDE,NULL,&y);CHKERRQ(ierr);
  ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);

  ijob = -1;           /* first call to reverse communication interface */
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
  if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr); }
  else B = NULL;

  do {

    PetscStackCall("FEASTrci",FEASTrci_(&ijob,&n,&Ze,ctx->work1,ctx->work2,ctx->Aq,ctx->Bq,fpm,&epsout,&loop,&eps->inta,&eps->intb,&eps->ncv,evals,pV,&eps->nconv,eps->errest,&info));

    if (ncv!=eps->ncv) SETERRQ1(PetscObjectComm((PetscObject)eps),1,"FEAST changed value of ncv to %d",ncv);
    if (ijob == 10 || ijob == 20) {
      /* set new quadrature point */
      ierr = STSetShift(eps->st,-Ze);CHKERRQ(ierr);
    } else if (ijob == 11 || ijob == 21) {
      /* linear solve (A-sigma*B)\work2, overwrite work2 */
      for (k=0;k<ncv;k++) {
        ierr = VecPlaceArray(x,ctx->work2+eps->nloc*k);CHKERRQ(ierr);
        if (ijob == 11) {
          ierr = STMatSolve(eps->st,1,x,w);CHKERRQ(ierr);
        } else {
          ierr = STMatSolveTranspose(eps->st,1,x,w);CHKERRQ(ierr);
        }
        ierr = VecCopy(w,x);CHKERRQ(ierr);
        ierr = VecScale(x,-1.0);CHKERRQ(ierr);
        ierr = VecResetArray(x);CHKERRQ(ierr);
      }
    } else if (ijob == 30 || ijob == 40) {
      /* multiplication A*V or B*V, result in work1 */
      for (k=0;k<fpm[24];k++) {
        ierr = VecPlaceArray(x,&pV[(fpm[23]+k-1)*eps->nloc]);CHKERRQ(ierr);
        ierr = VecPlaceArray(y,&ctx->work1[(fpm[23]+k-1)*eps->nloc]);CHKERRQ(ierr);
        ierr = MatMult((ijob==30)?A:B,x,y);CHKERRQ(ierr);
        ierr = VecResetArray(x);CHKERRQ(ierr);
        ierr = VecResetArray(y);CHKERRQ(ierr);
      }
    } else if (ijob != 0) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Internal error in FEAST reverse comunication interface (ijob=%d)",ijob);

  } while (ijob != 0);

  if (info!=0) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Error reported by FEAST (%d)",info);

  eps->its = loop;
  for (i=0;i<eps->nconv;i++) eps->eigr[i] = evals[i];
  if (eps->nconv >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  ierr = VecRestoreArray(eps->V[0],&pV);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFree(evals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSReset_FEAST"
PetscErrorCode EPSReset_FEAST(EPS eps)
{
  PetscErrorCode ierr;
  EPS_FEAST      *ctx = (EPS_FEAST*)eps->data;

  PetscFunctionBegin;
  ierr = PetscFree(ctx->work1);CHKERRQ(ierr);
  ierr = PetscFree(ctx->work2);CHKERRQ(ierr);
  ierr = PetscFree(ctx->Aq);CHKERRQ(ierr);
  ierr = PetscFree(ctx->Bq);CHKERRQ(ierr);
  ierr = EPSReset_Default(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_FEAST"
PetscErrorCode EPSDestroy_FEAST(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_FEAST"
PETSC_EXTERN PetscErrorCode EPSCreate_FEAST(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,EPS_FEAST,&eps->data);CHKERRQ(ierr);
  eps->ops->setup                = EPSSetUp_FEAST;
  eps->ops->destroy              = EPSDestroy_FEAST;
  eps->ops->reset                = EPSReset_FEAST;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  PetscFunctionReturn(0);
}

