/*
       This file implements a wrapper to the TRLAN package

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include <private/epsimpl.h>          /*I "slepceps.h" I*/
#include <../src/eps/impls/external/trlan/trlanp.h>

PetscErrorCode EPSSolve_TRLAN(EPS);

/* Nasty global variable to access EPS data from TRLan_ */
static EPS globaleps;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_TRLAN"
PetscErrorCode EPSSetUp_TRLAN(EPS eps)
{
  PetscErrorCode ierr;
  EPS_TRLAN      *tr = (EPS_TRLAN *)eps->data;

  PetscFunctionBegin;
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(((PetscObject)eps)->comm,1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = eps->nev;
  if (eps->mpd) PetscInfo(eps,"Warning: parameter mpd ignored\n");
  if (!eps->max_it) eps->max_it = PetscMax(1000,eps->n);
  
  if (!eps->ishermitian)
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");

  if (eps->isgeneralized)
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Requested method is not available for generalized problems");

  if (!eps->which) eps->which = EPS_LARGEST_REAL;
  if (eps->which!=EPS_LARGEST_REAL && eps->which!=EPS_SMALLEST_REAL && eps->which!=EPS_TARGET_REAL) 
    SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of eps->which");

  tr->restart = 0;
  tr->maxlan = PetscBLASIntCast(eps->nev+PetscMin(eps->nev,6));
  if (tr->maxlan+1-eps->ncv<=0) { tr->lwork = PetscBLASIntCast(tr->maxlan*(tr->maxlan+10)); }
  else { tr->lwork = PetscBLASIntCast(eps->nloc*(tr->maxlan+1-eps->ncv) + tr->maxlan*(tr->maxlan+10)); }
  ierr = PetscMalloc(tr->lwork*sizeof(PetscReal),&tr->work);CHKERRQ(ierr);

  if (eps->extraction) {
     ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr);
  }

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Left vectors not supported in this solver");
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
  ierr = VecCreateMPIWithArray(((PetscObject)globaleps)->comm,*n,PETSC_DECIDE,PETSC_NULL,&x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(((PetscObject)globaleps)->comm,*n,PETSC_DECIDE,PETSC_NULL,&y);CHKERRQ(ierr);
  for (i=0;i<*m;i++) {
    ierr = VecPlaceArray(x,(PetscScalar*)xin+i*(*ldx));CHKERRQ(ierr);
    ierr = VecPlaceArray(y,(PetscScalar*)yout+i*(*ldy));CHKERRQ(ierr);
    ierr = STApply(globaleps->OP,x,y);CHKERRQ(ierr);
    ierr = IPOrthogonalize(globaleps->ip,0,PETSC_NULL,globaleps->nds,PETSC_NULL,globaleps->DS,y,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
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
  EPS_TRLAN      *tr = (EPS_TRLAN *)eps->data;   
  PetscScalar    *pV;
  
  PetscFunctionBegin;
  ncv = PetscBLASIntCast(eps->ncv);
  n = PetscBLASIntCast(eps->nloc);
  
  if (eps->which==EPS_LARGEST_REAL || eps->which==EPS_TARGET_REAL) lohi = 1;
  else if (eps->which==EPS_SMALLEST_REAL) lohi = -1;
  else SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of eps->which");

  globaleps = eps;

  ipar[0]  = 0;            /* stat: error flag */
  ipar[1]  = lohi;         /* smallest (lohi<0) or largest eigenvalues (lohi>0) */
  ipar[2]  = PetscBLASIntCast(eps->nev); /* number of desired eigenpairs */
  ipar[3]  = 0;            /* number of eigenpairs already converged */
  ipar[4]  = tr->maxlan;   /* maximum Lanczos basis size */
  ipar[5]  = tr->restart;  /* restarting scheme */
  ipar[6]  = PetscBLASIntCast(eps->max_it); /* maximum number of MATVECs */
  ipar[7]  = PetscBLASIntCast(MPI_Comm_c2f(((PetscObject)eps)->comm)); /* communicator */
  ipar[8]  = 0;            /* verboseness */
  ipar[9]  = 99;           /* Fortran IO unit number used to write log messages */
  ipar[10] = 1;            /* use supplied starting vector */
  ipar[11] = 0;            /* checkpointing flag */
  ipar[12] = 98;           /* Fortran IO unit number used to write checkpoint files */
  ipar[13] = 0;            /* number of flops per matvec per PE (not used) */
  tr->work[0] = eps->tol;  /* relative tolerance on residual norms */

  for (i=0;i<eps->ncv;i++) eps->eigr[i]=0.0;
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetArray(eps->V[0],&pV);CHKERRQ(ierr);

  TRLan_(MatMult_TRLAN,ipar,&n,&ncv,eps->eigr,pV,&n,tr->work,&tr->lwork);

  ierr = VecRestoreArray(eps->V[0],&pV);CHKERRQ(ierr);

  stat        = ipar[0];
  eps->nconv  = ipar[3];
  eps->its    = ipar[25];
  eps->reason = EPS_CONVERGED_TOL;
  
  if (stat!=0) { SETERRQ1(((PetscObject)eps)->comm,PETSC_ERR_LIB,"Error in TRLAN (code=%d)",stat);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_TRLAN"
PetscErrorCode EPSDestroy_TRLAN(EPS eps)
{
  PetscErrorCode ierr;
  EPS_TRLAN      *tr = (EPS_TRLAN *)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscFree(tr->work);CHKERRQ(ierr);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_TRLAN"
PetscErrorCode EPSCreate_TRLAN(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,EPS_TRLAN,&eps->data);CHKERRQ(ierr);
  eps->ops->setup                = EPSSetUp_TRLAN;
  eps->ops->destroy              = EPSDestroy_TRLAN;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END
