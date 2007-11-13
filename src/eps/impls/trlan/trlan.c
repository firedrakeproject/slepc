/*
       This file implements a wrapper to the TRLAN package

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/eps/impls/trlan/trlanp.h"

/* Nasty global variable to access EPS data from TRLan_ */
static EPS globaleps;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_TRLAN"
PetscErrorCode EPSSetUp_TRLAN(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       n;
  EPS_TRLAN      *tr = (EPS_TRLAN *)eps->data;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&n);CHKERRQ(ierr); 
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = eps->nev;
  if (!eps->max_it) eps->max_it = PetscMax(1000,n);
  
  if (!eps->ishermitian)
    SETERRQ(PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");

  if (eps->isgeneralized)
    SETERRQ(PETSC_ERR_SUP,"Requested method is not available for generalized problems");

  tr->restart = 0;
  ierr = VecGetLocalSize(eps->vec_initial,&n); CHKERRQ(ierr);
  tr->maxlan = eps->nev+PetscMin(eps->nev,6);
  if (tr->maxlan+1-eps->ncv<=0) tr->lwork = tr->maxlan*(tr->maxlan+10);
  else tr->lwork = n*(tr->maxlan+1-eps->ncv) + tr->maxlan*(tr->maxlan+10);
  ierr = PetscMalloc(tr->lwork*sizeof(PetscReal),&tr->work);CHKERRQ(ierr);

  if (eps->projection) {
     ierr = PetscInfo(eps,"Warning: projection type ignored\n");CHKERRQ(ierr);
  }

  ierr = EPSAllocateSolutionContiguous(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_TRLAN"
static int MatMult_TRLAN(int *n,int *m,PetscReal *xin,int *ldx,PetscReal *yout,int *ldy)
{
  PetscErrorCode ierr;
  Vec            x,y;
  int            i;

  PetscFunctionBegin;
  ierr = VecCreateMPIWithArray(((PetscObject)globaleps)->comm,*n,PETSC_DECIDE,PETSC_NULL,&x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(((PetscObject)globaleps)->comm,*n,PETSC_DECIDE,PETSC_NULL,&y);CHKERRQ(ierr);
  for (i=0;i<*m;i++) {
    ierr = VecPlaceArray(x,(PetscScalar*)xin+i*(*ldx));CHKERRQ(ierr);
    ierr = VecPlaceArray(y,(PetscScalar*)yout+i*(*ldy));CHKERRQ(ierr);
    ierr = STApply(globaleps->OP,x,y);CHKERRQ(ierr);
    ierr = IPOrthogonalize(globaleps->ip,globaleps->nds,PETSC_NULL,globaleps->DS,y,PETSC_NULL,PETSC_NULL,PETSC_NULL,globaleps->work[0]);CHKERRQ(ierr);
    ierr = VecResetArray(x);CHKERRQ(ierr);
    ierr = VecResetArray(y);CHKERRQ(ierr);	
  }
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_TRLAN"
PetscErrorCode EPSSolve_TRLAN(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       nn;				   
  int            ipar[32], i, n, lohi, stat; 
  EPS_TRLAN      *tr = (EPS_TRLAN *)eps->data;	   
  PetscScalar    *pV;				   
  
  PetscFunctionBegin;

  ierr = VecGetLocalSize(eps->vec_initial,&nn); CHKERRQ(ierr);
  n = nn;
  
  if (eps->which==EPS_LARGEST_REAL) lohi = 1;
  else if (eps->which==EPS_SMALLEST_REAL) lohi = -1;
  else SETERRQ(1,"Wrong value of eps->which");

  globaleps = eps;

  ipar[0]  = 0;            /* stat: error flag */
  ipar[1]  = lohi;         /* smallest (lohi<0) or largest eigenvalues (lohi>0) */
  ipar[2]  = eps->nev;     /* number of desired eigenpairs */
  ipar[3]  = 0;            /* number of eigenpairs already converged */
  ipar[4]  = tr->maxlan;   /* maximum Lanczos basis size */
  ipar[5]  = tr->restart;  /* restarting scheme */
  ipar[6]  = eps->max_it;  /* maximum number of MATVECs */
  ipar[7]  = MPI_Comm_c2f(((PetscObject)eps)->comm);    /* communicator */
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

  TRLan_ ( MatMult_TRLAN, ipar, &n, &eps->ncv, eps->eigr, pV, &n, tr->work, &tr->lwork );

  ierr = VecRestoreArray( eps->V[0], &pV );CHKERRQ(ierr);

  stat        = ipar[0];
  eps->nconv  = ipar[3];
  eps->its    = ipar[25];
  eps->reason = EPS_CONVERGED_TOL;
  
  if (stat!=0) { SETERRQ1(PETSC_ERR_LIB,"Error in TRLAN (code=%d)",stat);}

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_TRLAN"
PetscErrorCode EPSDestroy_TRLAN(EPS eps)
{
  PetscErrorCode ierr;
  EPS_TRLAN      *tr = (EPS_TRLAN *)eps->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscFree(tr->work);CHKERRQ(ierr);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = EPSFreeSolutionContiguous(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_TRLAN"
PetscErrorCode EPSCreate_TRLAN(EPS eps)
{
  PetscErrorCode ierr;
  EPS_TRLAN      *trlan;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_TRLAN,&trlan);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_TRLAN));
  eps->data                      = (void *) trlan;
  eps->ops->solve                = EPSSolve_TRLAN;
  eps->ops->setup                = EPSSetUp_TRLAN;
  eps->ops->destroy              = EPSDestroy_TRLAN;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  eps->which = EPS_LARGEST_REAL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
