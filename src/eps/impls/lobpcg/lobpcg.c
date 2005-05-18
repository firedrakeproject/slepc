
/*                       
       This file implements a wrapper to the LOBPCG solver from HYPRE package
*/
#include "src/eps/epsimpl.h"
EXTERN_C_BEGIN
#include "HYPRE_lobpcg.h"
#include "HYPRE_parcsr_int.h"

#include "multivector.h"
/* #include "temp_multivector.h" */
typedef struct
{
 long   numVectors;
 int*   mask;   void** vector;
 int    ownsVectors;
 int    ownsMask;
                                                                                                             
 HYPRE_InterfaceInterpreter* interpreter;
                                                                                                             
} hypre_TempMultiVector;

EXTERN_C_END

EXTERN PetscErrorCode VecHYPRE_IJVectorCreate(Vec,HYPRE_IJVector*);
EXTERN PetscErrorCode VecHYPRE_IJVectorCopy(Vec v,HYPRE_IJVector ij);
EXTERN PetscErrorCode VecHYPRE_IJVectorCopyFrom(HYPRE_IJVector ij,Vec v);

typedef struct {
  HYPRE_InterfaceInterpreter interpreter;
  HYPRE_Solver               solver;
  int                        bsize;
  hypre_MultiVectorPtr       eigenvectors;
  double                     *eigval;
  KSP                        ksp;
  Vec                        x,b;
  HYPRE_IJVector             ijx,ijb;
} EPS_LOBPCG;

/* Nasty global variable to access EPS data from FunctSolver and FunctA */
static EPS globaleps;

#undef __FUNCT__  
#define __FUNCT__ "EPSFunctSolver"
int EPSFunctSolver(HYPRE_Solver s, HYPRE_Matrix m, HYPRE_Vector x, HYPRE_Vector y)
{ /* solves A*x=b */

  PetscErrorCode  ierr;
  EPS_LOBPCG      *lobpcg = (EPS_LOBPCG *)globaleps->data;
  HYPRE_ParVector px,pb;

  PetscFunctionBegin;

  HYPRE_IJVectorGetObject(lobpcg->ijb,(void**)&pb);
  HYPRE_ParVectorCopy((HYPRE_ParVector)x,pb);
  ierr = VecHYPRE_IJVectorCopyFrom(lobpcg->ijb,lobpcg->b);CHKERRQ(ierr);
  
  ierr = KSPSolve(lobpcg->ksp,lobpcg->b,lobpcg->x);CHKERRQ(ierr);CHKERRQ(ierr);
  
  ierr = VecHYPRE_IJVectorCopy(lobpcg->x,lobpcg->ijx);CHKERRQ(ierr);
  HYPRE_IJVectorGetObject(lobpcg->ijx,(void**)&px);
  HYPRE_ParVectorCopy(px,(HYPRE_ParVector)y);
    
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSFunctA"
int EPSFunctA(void *matvec_data, double alpha, void *A, void *x, double beta, void *y)
{ 
  /*
    computes y <- alpha * A * x + beta * y 
    only works if alpha=1 and beta=0
  */
  PetscErrorCode  ierr;
  EPS_LOBPCG      *lobpcg = (EPS_LOBPCG *)globaleps->data;
  HYPRE_ParVector px,py;

  PetscFunctionBegin;
  
  HYPRE_IJVectorGetObject(lobpcg->ijx,(void**)&px);
  HYPRE_ParVectorCopy((HYPRE_ParVector)x,px);
  ierr = VecHYPRE_IJVectorCopyFrom(lobpcg->ijx,lobpcg->x);CHKERRQ(ierr);
  
  ierr = STApply(globaleps->OP,lobpcg->x,lobpcg->b);CHKERRQ(ierr);
  
  ierr = VecHYPRE_IJVectorCopy(lobpcg->b,lobpcg->ijb);CHKERRQ(ierr);
  HYPRE_IJVectorGetObject(lobpcg->ijb,(void**)&py);
  HYPRE_ParVectorCopy(py,(HYPRE_ParVector)y);
    
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_LOBPCG"
PetscErrorCode EPSSetUp_LOBPCG(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_LOBPCG      *lobpcg = (EPS_LOBPCG *)eps->data;
  Mat             A;
  HYPRE_ParVector px;
  int             i,N;
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
  ierr = MatGetVecs(A,&lobpcg->x,&lobpcg->b);
  ierr = VecHYPRE_IJVectorCreate(lobpcg->x,&lobpcg->ijx);
  ierr = VecHYPRE_IJVectorCreate(lobpcg->b,&lobpcg->ijb);
  ierr = KSPSetOperators(lobpcg->ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetUp(lobpcg->ksp);CHKERRQ(ierr);
  ierr = VecGetSize(lobpcg->x,&N);CHKERRQ(ierr);

  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  HYPRE_LOBPCGSetMaxIter(lobpcg->solver,eps->max_it);
  if (!eps->tol) eps->tol = 1.e-7;
  HYPRE_LOBPCGSetTol(lobpcg->solver,eps->tol);
  HYPRE_LOBPCGSetPrintLevel(lobpcg->solver,0);

  HYPRE_LOBPCGSetPrecond(lobpcg->solver,EPSFunctSolver,PETSC_NULL,PETSC_NULL);
  HYPRE_LOBPCGSetup(lobpcg->solver,PETSC_NULL,PETSC_NULL,PETSC_NULL);

  lobpcg->bsize = eps->ncv = eps->nev;
  ierr = PetscMalloc(lobpcg->bsize*sizeof(PetscScalar),&lobpcg->eigval);CHKERRQ(ierr);
  ierr = PetscMemzero(lobpcg->eigval,lobpcg->bsize*sizeof(PetscScalar));CHKERRQ(ierr);

  ierr = HYPRE_IJVectorGetObject(lobpcg->ijx,(void**)&px);
  lobpcg->eigenvectors = hypre_MultiVectorCreateFromSampleVector(&lobpcg->interpreter,lobpcg->bsize,px);
  for (i=0;i<lobpcg->bsize;i++) {
    if (i==0) { 
      ierr = VecHYPRE_IJVectorCopy(eps->vec_initial,lobpcg->ijx);CHKERRQ(ierr);
    } else {
      ierr = SlepcVecSetRandom(lobpcg->x);CHKERRQ(ierr);
      ierr = VecHYPRE_IJVectorCopy(lobpcg->x,lobpcg->ijx);CHKERRQ(ierr);
    }
    HYPRE_ParVectorCopy(px,((hypre_TempMultiVector *) ((hypre_MultiVector *) lobpcg->eigenvectors)->data)->vector[i]);
  }
  
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_LOBPCG"
PetscErrorCode EPSSolve_LOBPCG(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_LOBPCG      *lobpcg = (EPS_LOBPCG *)eps->data;
  int             i;
  HYPRE_ParVector px;

  PetscFunctionBegin;
  globaleps = eps;
  
  HYPRE_LOBPCGSolve(lobpcg->solver,PETSC_NULL,lobpcg->eigenvectors,lobpcg->eigval);
  HYPRE_IJVectorGetObject(lobpcg->ijx,(void**)&px);
  for (i=0;i<lobpcg->bsize;i++) { 
    eps->eigr[i] = lobpcg->eigval[i]; 
    eps->eigi[i] = 0;
    HYPRE_ParVectorCopy(((hypre_TempMultiVector *) ((hypre_MultiVector *) lobpcg->eigenvectors)->data)->vector[i],px);
    ierr = VecHYPRE_IJVectorCopyFrom(lobpcg->ijx,eps->V[i]);CHKERRQ(ierr);    
  }
  eps->nconv = lobpcg->bsize;
  eps->reason = EPS_CONVERGED_TOL;
  eps->its = HYPRE_LOBPCGIterations(lobpcg->solver);
  
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
  HYPRE_LOBPCGDestroy(lobpcg->solver);
  hypre_MultiVectorDestroy(lobpcg->eigenvectors);
  ierr = PetscFree(lobpcg->eigval);CHKERRQ(ierr);
  HYPRE_IJVectorDestroy(lobpcg->ijx);
  HYPRE_IJVectorDestroy(lobpcg->ijb);
  
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
  const char*    prefix;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_LOBPCG,&lobpcg);CHKERRQ(ierr);
  PetscMemzero(lobpcg,sizeof(EPS_LOBPCG));
  PetscLogObjectMemory(eps,sizeof(EPS_LOBPCG));
  HYPRE_ParCSRSetupInterpreter(&lobpcg->interpreter);
  lobpcg->interpreter.Matvec = EPSFunctA;
  HYPRE_LOBPCGCreate(&lobpcg->interpreter,&lobpcg->solver);
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
