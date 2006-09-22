
/*                       
       This file implements a wrapper to the LOBPCG solver from HYPRE package
*/
#include "src/eps/epsimpl.h"
#include "src/contrib/blopex/petsc-interface/petsc-interface.h"
#include "lobpcg.h"
#include "interpreter.h"
#include "multivector.h"
#include "temp_multivector.h"

typedef struct {
  lobpcg_Tolerance           tol;
  lobpcg_BLASLAPACKFunctions blap_fn;
  mv_MultiVectorPtr          eigenvectors;
  mv_InterfaceInterpreter    ii;
  KSP                        ksp;
} EPS_LOBPCG;

#undef __FUNCT__  
#define __FUNCT__ "Precond_FnSingleVector"
static void Precond_FnSingleVector(void *data,void *x,void *y)
{
  PetscErrorCode  ierr;
  EPS             eps = (EPS)data;
  EPS_LOBPCG      *lobpcg = (EPS_LOBPCG*)eps->data;
      
  PetscFunctionBegin;
  ierr = KSPSolve(lobpcg->ksp,(Vec)x,(Vec)y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "Precond_FnMultiVector"
static void Precond_FnMultiVector(void *data,void *x,void *y)
{
  EPS             eps = (EPS)data;
  EPS_LOBPCG      *lobpcg = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  lobpcg->ii.Eval(Precond_FnSingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "OperatorASingleVector"
static void OperatorASingleVector(void *data,void *x,void *y)
{
  PetscErrorCode  ierr;
  EPS		  eps = (EPS)data;
  
  PetscFunctionBegin;
  ierr = STApply(eps->OP,(Vec)x,(Vec)y);
  ierr = EPSOrthogonalize(eps,eps->nds,PETSC_NULL,eps->DS,(Vec)y,PETSC_NULL,PETSC_NULL,PETSC_NULL); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "OperatorAMultiVector"
static void OperatorAMultiVector(void *data,void *x,void *y)
{
  EPS		  eps = (EPS)data;
  EPS_LOBPCG	  *lobpcg = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  lobpcg->ii.Eval(OperatorASingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_LOBPCG"
PetscErrorCode EPSSetUp_LOBPCG(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_LOBPCG      *lobpcg = (EPS_LOBPCG *)eps->data;
  Mat             A;
  int             N;
  PetscTruth      isShift;

  PetscFunctionBegin;
  if (!eps->ishermitian || eps->isgeneralized) { 
    SETERRQ(PETSC_ERR_SUP,"LOBPCG only works for standard symmetric problems"); 
  }
  ierr = PetscTypeCompare((PetscObject)eps->OP,STSHIFT,&isShift);CHKERRQ(ierr);
  if (!isShift) {
    SETERRQ(PETSC_ERR_SUP,"LOBPCG only works with shift spectral transformation"); 
  }
  if (eps->which!=EPS_SMALLEST_REAL) {
    SETERRQ(1,"Wrong value of eps->which");
  }
  ierr = STGetOperators(eps->OP,&A,PETSC_NULL);CHKERRQ(ierr);
  ierr = KSPSetOperators(lobpcg->ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetUp(lobpcg->ksp);CHKERRQ(ierr);

  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  eps->ncv = eps->nev = PetscMin(eps->nev,N);
  
  if (!eps->tol) eps->tol = 1.e-7;
  lobpcg->tol.absolute = eps->tol;
  lobpcg->tol.relative = 1e-50;
  
  LOBPCG_InitRandomContext();
  PETSCSetupInterpreter(&lobpcg->ii);
  lobpcg->eigenvectors = mv_MultiVectorCreateFromSampleVector(&lobpcg->ii,eps->ncv,eps->vec_initial);
  mv_MultiVectorSetRandom(lobpcg->eigenvectors,1234);
  
  lobpcg->blap_fn.dpotrf = PETSC_dpotrf_interface;
  lobpcg->blap_fn.dsygv = PETSC_dsygv_interface;

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_LOBPCG"
PetscErrorCode EPSSolve_LOBPCG(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_LOBPCG      *lobpcg = (EPS_LOBPCG *)eps->data;
  int             info,i;
  mv_TempMultiVector *mv;
  
  PetscFunctionBegin;
  info = lobpcg_solve(lobpcg->eigenvectors,eps,OperatorAMultiVector,
  	PETSC_NULL,PETSC_NULL,eps,Precond_FnMultiVector,NULL,
        lobpcg->blap_fn,lobpcg->tol,eps->max_it,0,&eps->its,
        eps->eigr,PETSC_NULL,0,eps->errest,PETSC_NULL,0);
  if (info>0) SETERRQ1(PETSC_ERR_LIB,"Error in LOBPCG (code=%d)",info); 

  eps->nconv = eps->ncv;
  if (info==-1) eps->reason = EPS_DIVERGED_ITS;
  else eps->reason = EPS_CONVERGED_TOL;
  
  mv = mv_MultiVectorGetData(lobpcg->eigenvectors);
  for (i=0;i<eps->nconv;i++) {
    ierr = VecCopy(mv->vector[i],eps->V[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_LOBPCG"
PetscErrorCode EPSSetFromOptions_LOBPCG(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LOBPCG     *lobpcg = (EPS_LOBPCG *)eps->data;

  PetscFunctionBegin;
  ierr = KSPSetFromOptions(lobpcg->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_LOBPCG"
PetscErrorCode EPSDestroy_LOBPCG(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LOBPCG     *lobpcg = (EPS_LOBPCG *)eps->data;

  PetscFunctionBegin;
  ierr = KSPDestroy(lobpcg->ksp);CHKERRQ(ierr);
  LOBPCG_DestroyRandomContext();
  mv_MultiVectorDestroy(lobpcg->eigenvectors);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_LOBPCG"
PetscErrorCode EPSCreate_LOBPCG(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LOBPCG     *lobpcg;
  const char*    prefix;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_LOBPCG,&lobpcg);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_LOBPCG));
  ierr = KSPCreate(eps->comm,&lobpcg->ksp);CHKERRQ(ierr);
  ierr = EPSGetOptionsPrefix(eps,&prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(lobpcg->ksp,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(lobpcg->ksp,"eps_lobpcg_");CHKERRQ(ierr);
  eps->data                      = (void *) lobpcg;
  eps->ops->solve                = EPSSolve_LOBPCG;
  eps->ops->setup                = EPSSetUp_LOBPCG;
  eps->ops->setfromoptions       = EPSSetFromOptions_LOBPCG;
  eps->ops->destroy              = EPSDestroy_LOBPCG;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  eps->which = EPS_SMALLEST_REAL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
