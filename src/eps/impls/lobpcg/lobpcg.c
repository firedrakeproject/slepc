
/*                       
       This file implements a wrapper to the LOBPCG solver from HYPRE package
*/
#include "src/eps/epsimpl.h"
EXTERN_C_BEGIN
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_lobpcg.h"
#include "IJ_mv.h"
EXTERN_C_END

EXTERN PetscErrorCode VecHYPRE_IJVectorCreate(Vec,HYPRE_IJVector*);
EXTERN PetscErrorCode VecHYPRE_IJVectorCopy(Vec v,HYPRE_IJVector ij);
EXTERN PetscErrorCode VecHYPRE_IJVectorCopyFrom(HYPRE_IJVector ij,Vec v);

typedef struct {
  HYPRE_LobpcgData data;
  int              bsize;
  HYPRE_ParVector  *eigenvector;
  double           *eigval;
  KSP              ksp;
  Vec              x,b;
  HYPRE_IJVector   ijx,ijb;
} EPS_LOBPCG;

/* Nasty global variable to access EPS data from FunctSolver and FunctA */
static EPS globaleps;

#undef __FUNCT__  
#define __FUNCT__ "EPSFunctSolver"
int EPSFunctSolver(HYPRE_ParVector b,HYPRE_ParVector x)
{ /* solves A*x=b */
  PetscErrorCode  ierr;
  EPS_LOBPCG      *lobpcg = (EPS_LOBPCG *)globaleps->data;
  HYPRE_ParVector px,pb;

  PetscFunctionBegin;

  HYPRE_IJVectorGetObject(lobpcg->ijb,(void**)&pb);
  HYPRE_ParVectorCopy(b,pb);
  ierr = VecHYPRE_IJVectorCopyFrom(lobpcg->ijb,lobpcg->b);CHKERRQ(ierr);
  
  ierr = KSPSolve(lobpcg->ksp,lobpcg->b,lobpcg->x);CHKERRQ(ierr);CHKERRQ(ierr);
  
  ierr = VecHYPRE_IJVectorCopy(lobpcg->x,lobpcg->ijx);CHKERRQ(ierr);
  HYPRE_IJVectorGetObject(lobpcg->ijx,(void**)&px);
  HYPRE_ParVectorCopy(px,x);
    
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSFunctA"
int EPSFunctA(HYPRE_ParVector x,HYPRE_ParVector y)
{ /* computes y=A*x */
  PetscErrorCode  ierr;
  EPS_LOBPCG      *lobpcg = (EPS_LOBPCG *)globaleps->data;
  Mat             A;
  HYPRE_ParVector px,py;
  PetscReal       shift;

  PetscFunctionBegin;

  HYPRE_IJVectorGetObject(lobpcg->ijx,(void**)&px);
  HYPRE_ParVectorCopy(x,px);
  ierr = VecHYPRE_IJVectorCopyFrom(lobpcg->ijx,lobpcg->x);CHKERRQ(ierr);
  
  ierr = STGetOperators(globaleps->OP,&A,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMult(A,lobpcg->x,lobpcg->b);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = STGetShift(globaleps->OP,&shift);CHKERRQ(ierr);
  if (shift != 0) {
    ierr = VecAXPY(&shift,lobpcg->x,lobpcg->b);CHKERRQ(ierr);
  }
  
  ierr = VecHYPRE_IJVectorCopy(lobpcg->b,lobpcg->ijb);CHKERRQ(ierr);
  HYPRE_IJVectorGetObject(lobpcg->ijb,(void**)&py);
  HYPRE_ParVectorCopy(py,y);
    
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_LOBPCG"
PetscErrorCode EPSSetUp_LOBPCG(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LOBPCG     *lobpcg = (EPS_LOBPCG *)eps->data;
  Mat            A;
  HYPRE_IJVector ijv;
  int            i, N;
  PetscTruth     isShift;

  PetscFunctionBegin;
  if (!eps->ishermitian || eps->isgeneralized) { 
    SETERRQ(PETSC_ERR_SUP,"LOBPCG only works for standard symmetric problems"); 
  }
  ierr = PetscTypeCompare((PetscObject)eps->OP,STSHIFT,&isShift);CHKERRQ(ierr);
  if (!isShift) {
    SETERRQ(PETSC_ERR_SUP,"LOBPCG only works with shift spectral transformation"); 
  }
  ierr = STGetOperators(eps->OP,&A,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&lobpcg->x,&lobpcg->b);
  ierr = VecHYPRE_IJVectorCreate(lobpcg->x,&lobpcg->ijx);
  ierr = VecHYPRE_IJVectorCreate(lobpcg->b,&lobpcg->ijb);
  ierr = KSPSetOperators(lobpcg->ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetUp(lobpcg->ksp);CHKERRQ(ierr);
  ierr = VecGetSize(lobpcg->x,&N);CHKERRQ(ierr);
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  HYPRE_LobpcgSetTolerance(lobpcg->data,eps->tol);
  if (!eps->tol) eps->tol = 1.e-7;
  HYPRE_LobpcgSetMaxIterations(lobpcg->data,eps->max_it);
  eps->ncv = eps->nev;
  lobpcg->bsize = eps->nev;
  HYPRE_LobpcgSetBlocksize(lobpcg->data,lobpcg->bsize);
  ierr = PetscMalloc(lobpcg->bsize*sizeof(HYPRE_ParVector),&lobpcg->eigenvector);CHKERRQ(ierr);
  for (i=0;i<lobpcg->bsize;i++) {
    ierr = VecHYPRE_IJVectorCreate(lobpcg->x,&ijv);CHKERRQ(ierr);
    if (i==0) { 
      ierr = VecHYPRE_IJVectorCopy(eps->vec_initial,ijv);CHKERRQ(ierr);
    } else {
      ierr = SlepcVecSetRandom(lobpcg->x);CHKERRQ(ierr);
      ierr = VecHYPRE_IJVectorCopy(lobpcg->x,ijv);CHKERRQ(ierr);
    }
    HYPRE_IJVectorGetObject(ijv,(void**)&lobpcg->eigenvector[i]);    
  }
  HYPRE_LobpcgSetVerbose(lobpcg->data,1);
  HYPRE_LobpcgSetSolverFunction(lobpcg->data,EPSFunctSolver);
  HYPRE_LobpcgSetup(lobpcg->data);
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_LOBPCG"
PetscErrorCode EPSSolve_LOBPCG(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LOBPCG     *lobpcg = (EPS_LOBPCG *)eps->data;
  int            i;

  PetscFunctionBegin;
  globaleps = eps;
  HYPRE_LobpcgSolve(lobpcg->data,EPSFunctA,lobpcg->eigenvector,&lobpcg->eigval);
  for (i=0;i<lobpcg->bsize;i++) { eps->eigr[i] = lobpcg->eigval[i]; }
  eps->nconv = lobpcg->bsize;
  eps->reason = EPS_CONVERGED_TOL;
  HYPRE_LobpcgGetIterations(lobpcg->data,&eps->its);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeVectors_LOBPCG"
PetscErrorCode EPSComputeVectors_LOBPCG(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_LOBPCG      *lobpcg = (EPS_LOBPCG *)eps->data;
  HYPRE_ParVector px;
  int             i;

  PetscFunctionBegin;
  HYPRE_IJVectorGetObject(lobpcg->ijx,(void**)&px);
  for (i=0;i<lobpcg->bsize;i++) {
    HYPRE_ParVectorCopy(lobpcg->eigenvector[i],px);
    ierr = VecHYPRE_IJVectorCopyFrom(lobpcg->ijx,eps->AV[i]);CHKERRQ(ierr);
  }
  eps->evecsavailable = PETSC_TRUE;
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
  int            i;

  PetscFunctionBegin;
  HYPRE_LobpcgDestroy(lobpcg->data);
  HYPRE_IJVectorDestroy(lobpcg->ijx);
  HYPRE_IJVectorDestroy(lobpcg->ijb);
  for (i=0;i<lobpcg->bsize;i++) {
    HYPRE_ParVectorDestroy(lobpcg->eigenvector[i]);
  }
  ierr = KSPDestroy(lobpcg->ksp);CHKERRQ(ierr);
  ierr = VecDestroy(lobpcg->x);CHKERRQ(ierr);
  ierr = VecDestroy(lobpcg->b);CHKERRQ(ierr);
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
  char*          prefix;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_LOBPCG,&lobpcg);CHKERRQ(ierr);
  PetscMemzero(lobpcg,sizeof(EPS_LOBPCG));
  PetscLogObjectMemory(eps,sizeof(EPS_LOBPCG));
  HYPRE_LobpcgCreate(&lobpcg->data);
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
  eps->ops->computevectors       = EPSComputeVectors_LOBPCG;
  PetscFunctionReturn(0);
}
EXTERN_C_END
