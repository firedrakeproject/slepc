
/*                       
       This file implements a wrapper to the PLANSO package
*/
#include "src/eps/impls/planso/plansop.h"

/* Nasty global variable to access EPS data from PLANop_ and PLANopm_ */
static EPS globaleps;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_PLANSO"
static int EPSSetUp_PLANSO(EPS eps)
{
  int        ierr, n;
  EPS_PLANSO *pl = (EPS_PLANSO *)eps->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_ERR_SUP,"Requested method is not available for complex problems");
#endif
  if (!eps->ishermitian)
    SETERRQ(PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");

  ierr = VecGetLocalSize(eps->vec_initial,&n); CHKERRQ(ierr);
  pl->lwork = 5*n+1+4*eps->ncv+PetscMax(n,eps->ncv+1);
  ierr = PetscMalloc(pl->lwork*sizeof(PetscReal),&pl->work);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetDefaults_PLANSO"
static int EPSSetDefaults_PLANSO(EPS eps)
{
  int         ierr, N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = eps->nev;
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PLANop_"
int  PLANop_(int *n,PetscReal *s, PetscReal *q, PetscReal *p)
{
  Vec    x,y;
  int    ierr;

  PetscFunctionBegin;
  ierr = VecCreateMPIWithArray(globaleps->comm,*n,PETSC_DECIDE,(PetscScalar*)q,&x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(globaleps->comm,*n,PETSC_DECIDE,(PetscScalar*)p,&y);CHKERRQ(ierr);
  ierr = STApply(globaleps->OP,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PLANopm_"
int  PLANopm_(int *n,PetscReal *q, PetscReal *s)
{
  Vec    x,y;
  int    ierr;

  PetscFunctionBegin;
  ierr = VecCreateMPIWithArray(globaleps->comm,*n,PETSC_DECIDE,(PetscScalar*)q,&x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(globaleps->comm,*n,PETSC_DECIDE,(PetscScalar*)s,&y);CHKERRQ(ierr);
  ierr = STApplyB(globaleps->OP,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_PLANSO"
static int  EPSSolve_PLANSO(EPS eps)
{
  int        i, n, msglvl, lohi, ierr;
  PetscReal  condm;
  EPS_PLANSO *pl = (EPS_PLANSO *)eps->data;
  MPI_Fint    fcomm;
  
  PetscFunctionBegin;

  ierr = VecGetLocalSize(eps->vec_initial,&n); CHKERRQ(ierr);
  
  if (eps->which==EPS_LARGEST_REAL) lohi = 1;
  else if (eps->which==EPS_SMALLEST_REAL) lohi = -1;
  else SETERRQ(1,"Wrong value of eps->which");

  condm = 1.0;         /* estimated condition number: we have no information */
  msglvl = 0;
  globaleps = eps;
  fcomm = MPI_Comm_c2f(eps->comm);

  PLANdr2_( &n, &eps->ncv, &eps->nev, &lohi, &condm, &eps->tol, &eps->its, &eps->nconv, 
            eps->eigr, eps->eigi, pl->work, &pl->lwork, &ierr, &msglvl, &fcomm );

  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
  eps->reason = EPS_CONVERGED_TOL;
  
  if (ierr!=0) { SETERRQ1(PETSC_ERR_LIB,"Error in PLANSO (code=%d)",ierr);}

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_PLANSO"
/*
  EPSDestroy_PLANSO - Destroys the context variable for PLANSO.

  Input Parameter: 
. eps - the iterative context
*/
int EPSDestroy_PLANSO(EPS eps)
{
  EPS_PLANSO *pl = (EPS_PLANSO *)eps->data;
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (pl->work)  { ierr = PetscFree(pl->work);CHKERRQ(ierr); }
  if (eps->data) { ierr = PetscFree(eps->data);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_PLANSO"
int EPSCreate_PLANSO(EPS eps)
{
  EPS_PLANSO *planso;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_PLANSO,&planso);CHKERRQ(ierr);
  PetscMemzero(planso,sizeof(EPS_PLANSO));
  PetscLogObjectMemory(eps,sizeof(EPS_PLANSO));
  eps->data                      = (void *) planso;
  eps->ops->setup                = EPSSetUp_PLANSO;
  eps->ops->setdefaults          = EPSSetDefaults_PLANSO;
  eps->ops->solve                = EPSSolve_PLANSO;
  eps->ops->destroy              = EPSDestroy_PLANSO;
  eps->ops->view                 = 0;
  eps->which = EPS_LARGEST_REAL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
