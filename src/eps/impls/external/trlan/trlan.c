/*
   This file implements a wrapper to the TRLAN package

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

#include <slepc/private/epsimpl.h>
#include <../src/eps/impls/external/trlan/trlanp.h>

PetscErrorCode EPSSolve_TRLAN(EPS);

/* Nasty global variable to access EPS data from TRLan_ */
static EPS globaleps;

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_TRLAN"
PetscErrorCode EPSSetUp_TRLAN(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      istrivial;
  EPS_TRLAN      *tr = (EPS_TRLAN*)eps->data;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(PetscMax(7,eps->nev+PetscMin(eps->nev,6)),&tr->maxlan);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(PetscObjectComm((PetscObject)eps),1,"The value of ncv must be at least nev");
  } else eps->ncv = tr->maxlan;
  if (eps->mpd) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (!eps->max_it) eps->max_it = PetscMax(1000,eps->n);

  if (!eps->ishermitian) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");

  if (eps->isgeneralized) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Requested method is not available for generalized problems");

  if (!eps->which) eps->which = EPS_LARGEST_REAL;
  if (eps->which!=EPS_LARGEST_REAL && eps->which!=EPS_SMALLEST_REAL && eps->which!=EPS_TARGET_REAL) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");
  if (eps->stopping!=EPSStoppingBasic) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"External packages do not support user-defined stopping test");

  tr->restart = 0;
  if (tr->maxlan+1-eps->ncv<=0) {
    ierr = PetscBLASIntCast(tr->maxlan*(tr->maxlan+10),&tr->lwork);CHKERRQ(ierr);
  } else {
    ierr = PetscBLASIntCast(eps->nloc*(tr->maxlan+1-eps->ncv) + tr->maxlan*(tr->maxlan+10),&tr->lwork);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(tr->lwork,&tr->work);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)eps,tr->lwork*sizeof(PetscReal));CHKERRQ(ierr);

  if (eps->extraction) { ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr); }
  ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support region filtering");

  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);

  /* dispatch solve method */
  eps->ops->solve = EPSSolve_TRLAN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_TRLAN"
static PetscBLASInt MatMult_TRLAN(PetscBLASInt *n,PetscBLASInt *m,PetscReal *xin,PetscBLASInt *ldx,PetscReal *yout,PetscBLASInt *ldy)
{
  PetscErrorCode ierr;
  Vec            x,y;
  PetscBLASInt   i;

  PetscFunctionBegin;
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)globaleps),1,*n,PETSC_DECIDE,NULL,&x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)globaleps),1,*n,PETSC_DECIDE,NULL,&y);CHKERRQ(ierr);
  for (i=0;i<*m;i++) {
    ierr = VecPlaceArray(x,(PetscScalar*)xin+i*(*ldx));CHKERRQ(ierr);
    ierr = VecPlaceArray(y,(PetscScalar*)yout+i*(*ldy));CHKERRQ(ierr);
    ierr = STApply(globaleps->st,x,y);CHKERRQ(ierr);
    ierr = BVOrthogonalizeVec(globaleps->V,y,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = VecResetArray(x);CHKERRQ(ierr);
    ierr = VecResetArray(y);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_TRLAN"
PetscErrorCode EPSSolve_TRLAN(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   ipar[32],n,lohi,stat,ncv;
  EPS_TRLAN      *tr = (EPS_TRLAN*)eps->data;
  PetscScalar    *pV;
  Vec            v0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(eps->ncv,&ncv);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(eps->nloc,&n);CHKERRQ(ierr);

  if (eps->which==EPS_LARGEST_REAL || eps->which==EPS_TARGET_REAL) lohi = 1;
  else if (eps->which==EPS_SMALLEST_REAL) lohi = -1;
  else SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");

  globaleps = eps;

  ipar[0]  = 0;            /* stat: error flag */
  ipar[1]  = lohi;         /* smallest (lohi<0) or largest eigenvalues (lohi>0) */
  ierr = PetscBLASIntCast(eps->nev,&ipar[2]);CHKERRQ(ierr); /* number of desired eigenpairs */
  ipar[3]  = 0;            /* number of eigenpairs already converged */
  ipar[4]  = tr->maxlan;   /* maximum Lanczos basis size */
  ipar[5]  = tr->restart;  /* restarting scheme */
  ierr = PetscBLASIntCast(eps->max_it,&ipar[6]);CHKERRQ(ierr); /* maximum number of MATVECs */
#if !defined(PETSC_HAVE_MPIUNI)
  ierr = PetscBLASIntCast(MPI_Comm_c2f(PetscObjectComm((PetscObject)eps)),&ipar[7]);CHKERRQ(ierr);
#endif
  ipar[8]  = 0;            /* verboseness */
  ipar[9]  = 99;           /* Fortran IO unit number used to write log messages */
  ipar[10] = 1;            /* use supplied starting vector */
  ipar[11] = 0;            /* checkpointing flag */
  ipar[12] = 98;           /* Fortran IO unit number used to write checkpoint files */
  ipar[13] = 0;            /* number of flops per matvec per PE (not used) */
  tr->work[0] = eps->tol;  /* relative tolerance on residual norms */

  for (i=0;i<eps->ncv;i++) eps->eigr[i]=0.0;
  ierr = EPSGetStartVector(eps,0,NULL);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(eps->V,0,0);CHKERRQ(ierr);  /* just for deflation space */
  ierr = BVGetColumn(eps->V,0,&v0);CHKERRQ(ierr);
  ierr = VecGetArray(v0,&pV);CHKERRQ(ierr);

  PetscStackCall("TRLan",TRLan_(MatMult_TRLAN,ipar,&n,&ncv,eps->eigr,pV,&n,tr->work,&tr->lwork));

  ierr = VecRestoreArray(v0,&pV);CHKERRQ(ierr);
  ierr = BVRestoreColumn(eps->V,0,&v0);CHKERRQ(ierr);

  stat        = ipar[0];
  eps->nconv  = ipar[3];
  eps->its    = ipar[25];
  eps->reason = EPS_CONVERGED_TOL;

  if (stat!=0) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Error in TRLAN (code=%d)",stat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSReset_TRLAN"
PetscErrorCode EPSReset_TRLAN(EPS eps)
{
  PetscErrorCode ierr;
  EPS_TRLAN      *tr = (EPS_TRLAN*)eps->data;

  PetscFunctionBegin;
  ierr = PetscFree(tr->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_TRLAN"
PetscErrorCode EPSDestroy_TRLAN(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_TRLAN"
PETSC_EXTERN PetscErrorCode EPSCreate_TRLAN(EPS eps)
{
  EPS_TRLAN      *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  eps->ops->setup                = EPSSetUp_TRLAN;
  eps->ops->destroy              = EPSDestroy_TRLAN;
  eps->ops->reset                = EPSReset_TRLAN;
  eps->ops->backtransform        = EPSBackTransform_Default;
  PetscFunctionReturn(0);
}

