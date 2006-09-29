
/*                       
       This file implements a wrapper to the BLOPEX solver
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
} EPS_BLOPEX;

#undef __FUNCT__  
#define __FUNCT__ "Precond_FnSingleVector"
static void Precond_FnSingleVector(void *data,void *x,void *y)
{
  PetscErrorCode  ierr;
  EPS             eps = (EPS)data;
  EPS_BLOPEX      *blopex = (EPS_BLOPEX*)eps->data;
      
  PetscFunctionBegin;
  ierr = KSPSolve(blopex->ksp,(Vec)x,(Vec)y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "Precond_FnMultiVector"
static void Precond_FnMultiVector(void *data,void *x,void *y)
{
  EPS             eps = (EPS)data;
  EPS_BLOPEX      *blopex = (EPS_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(Precond_FnSingleVector,data,x,y);
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
  EPS_BLOPEX	  *blopex = (EPS_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(OperatorASingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_BLOPEX"
PetscErrorCode EPSSetUp_BLOPEX(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_BLOPEX      *blopex = (EPS_BLOPEX *)eps->data;
  Mat             A;
  int             N;
  PetscTruth      isShift;

  PetscFunctionBegin;
  if (!eps->ishermitian || eps->isgeneralized) { 
    SETERRQ(PETSC_ERR_SUP,"blopex only works for standard symmetric problems"); 
  }
  ierr = PetscTypeCompare((PetscObject)eps->OP,STSHIFT,&isShift);CHKERRQ(ierr);
  if (!isShift) {
    SETERRQ(PETSC_ERR_SUP,"blopex only works with shift spectral transformation"); 
  }
  if (eps->which!=EPS_SMALLEST_REAL) {
    SETERRQ(1,"Wrong value of eps->which");
  }
  ierr = STGetOperators(eps->OP,&A,PETSC_NULL);CHKERRQ(ierr);
  ierr = KSPSetOperators(blopex->ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetUp(blopex->ksp);CHKERRQ(ierr);

  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  eps->ncv = eps->nev = PetscMin(eps->nev,N);
  if (!eps->max_it) eps->max_it = PetscMax(100,2*N/eps->ncv);
  
  blopex->tol.absolute = eps->tol;
  blopex->tol.relative = 1e-50;
  
  LOBPCG_InitRandomContext();
  PETSCSetupInterpreter(&blopex->ii);
  blopex->eigenvectors = mv_MultiVectorCreateFromSampleVector(&blopex->ii,eps->ncv,eps->vec_initial);
  mv_MultiVectorSetRandom(blopex->eigenvectors,1234);
  
  blopex->blap_fn.dpotrf = PETSC_dpotrf_interface;
  blopex->blap_fn.dsygv = PETSC_dsygv_interface;

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_BLOPEX"
PetscErrorCode EPSSolve_BLOPEX(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_BLOPEX      *blopex = (EPS_BLOPEX *)eps->data;
  int             info,i;
  mv_TempMultiVector *mv;
  
  PetscFunctionBegin;
  info = lobpcg_solve(blopex->eigenvectors,eps,OperatorAMultiVector,
  	PETSC_NULL,PETSC_NULL,eps,Precond_FnMultiVector,NULL,
        blopex->blap_fn,blopex->tol,eps->max_it,0,&eps->its,
        eps->eigr,PETSC_NULL,0,eps->errest,PETSC_NULL,0);
  if (info>0) SETERRQ1(PETSC_ERR_LIB,"Error in blopex (code=%d)",info); 

  eps->nconv = eps->ncv;
  if (info==-1) eps->reason = EPS_DIVERGED_ITS;
  else eps->reason = EPS_CONVERGED_TOL;
  
  mv = (mv_TempMultiVector*)mv_MultiVectorGetData(blopex->eigenvectors);
  for (i=0;i<eps->nconv;i++) {
    ierr = VecCopy((Vec)mv->vector[i],eps->V[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_BLOPEX"
PetscErrorCode EPSSetFromOptions_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;
  EPS_BLOPEX     *blopex = (EPS_BLOPEX *)eps->data;

  PetscFunctionBegin;
  ierr = KSPSetFromOptions(blopex->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_BLOPEX"
PetscErrorCode EPSDestroy_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;
  EPS_BLOPEX     *blopex = (EPS_BLOPEX *)eps->data;

  PetscFunctionBegin;
  ierr = KSPDestroy(blopex->ksp);CHKERRQ(ierr);
  LOBPCG_DestroyRandomContext();
  mv_MultiVectorDestroy(blopex->eigenvectors);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_BLOPEX"
PetscErrorCode EPSCreate_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;
  EPS_BLOPEX     *blopex;
  const char*    prefix;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_BLOPEX,&blopex);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_BLOPEX));
  ierr = KSPCreate(eps->comm,&blopex->ksp);CHKERRQ(ierr);
  ierr = EPSGetOptionsPrefix(eps,&prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(blopex->ksp,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(blopex->ksp,"eps_blopex_");CHKERRQ(ierr);
  eps->data                      = (void *) blopex;
  eps->ops->solve                = EPSSolve_BLOPEX;
  eps->ops->setup                = EPSSetUp_BLOPEX;
  eps->ops->setfromoptions       = EPSSetFromOptions_BLOPEX;
  eps->ops->destroy              = EPSDestroy_BLOPEX;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  eps->which = EPS_SMALLEST_REAL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
