/*
       This file implements a wrapper to the BLOPEX solver

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "private/stimpl.h"
#include "private/epsimpl.h"
#include "slepc-interface.h"
#include "lobpcg.h"
#include "interpreter.h"
#include "multivector.h"
#include "temp_multivector.h"

typedef struct {
  lobpcg_Tolerance           tol;
  lobpcg_BLASLAPACKFunctions blap_fn;
  mv_MultiVectorPtr          eigenvectors;
  mv_MultiVectorPtr          Y;
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
  PetscInt        lits;
      
  PetscFunctionBegin;
  ierr = KSPSolve(blopex->ksp,(Vec)x,(Vec)y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  ierr = KSPGetIterationNumber(blopex->ksp, &lits); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  eps->OP->lineariterations+= lits;

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
  Mat             A;
 
  PetscFunctionBegin;
  ierr = STGetOperators(eps->OP,&A,PETSC_NULL); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  ierr = MatMult(A,(Vec)x,(Vec)y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
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
#define __FUNCT__ "OperatorBSingleVector"
static void OperatorBSingleVector(void *data,void *x,void *y)
{
  PetscErrorCode  ierr;
  EPS		  eps = (EPS)data;
  Mat             B;
  
  PetscFunctionBegin;
  ierr = STGetOperators(eps->OP,PETSC_NULL,&B); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  ierr = MatMult(B,(Vec)x,(Vec)y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "OperatorBMultiVector"
static void OperatorBMultiVector(void *data,void *x,void *y)
{
  EPS		  eps = (EPS)data;
  EPS_BLOPEX	  *blopex = (EPS_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(OperatorBSingleVector,data,x,y);
  PetscFunctionReturnVoid();
}


#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_BLOPEX"
PetscErrorCode EPSSetUp_BLOPEX(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_BLOPEX      *blopex = (EPS_BLOPEX *)eps->data;
  Mat             A,B;
  PetscInt        N;
  PetscTruth      isShift;

  PetscFunctionBegin;
  if (!eps->ishermitian) { 
    SETERRQ(PETSC_ERR_SUP,"blopex only works for hermitian problems"); 
  }
  ierr = PetscTypeCompare((PetscObject)eps->OP,STSHIFT,&isShift);CHKERRQ(ierr);
  if (!isShift) {
    SETERRQ(PETSC_ERR_SUP,"blopex only works with shift spectral transformation"); 
  }
  if (eps->which!=EPS_SMALLEST_REAL) {
    SETERRQ(1,"Wrong value of eps->which");
  }

  ierr = KSPSetType(blopex->ksp, KSPPREONLY); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(blopex->ksp);CHKERRQ(ierr);
  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  ierr = KSPSetOperators(blopex->ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  eps->ncv = eps->nev = PetscMin(eps->nev,N);
  if (!eps->max_it) eps->max_it = PetscMax(100,2*N/eps->ncv);

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);
  
  blopex->tol.absolute = eps->tol;
  blopex->tol.relative = 1e-50;
  
  LOBPCG_InitRandomContext();
  SLEPCSetupInterpreter(&blopex->ii);
  blopex->eigenvectors = mv_MultiVectorCreateFromSampleVector(&blopex->ii,eps->ncv,eps->V);
  mv_MultiVectorSetRandom(blopex->eigenvectors,1234);

  if (eps->nds > 0) {
    blopex->Y = mv_MultiVectorCreateFromSampleVector(&blopex->ii, eps->nds, eps->DS);
  } else
    blopex->Y = PETSC_NULL;
  
  blopex->blap_fn.dpotrf = PETSC_dpotrf_interface;
  blopex->blap_fn.dsygv = PETSC_dsygv_interface;

  if (eps->projection) {
     ierr = PetscInfo(eps,"Warning: projection type ignored\n");CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_BLOPEX"
PetscErrorCode EPSSolve_BLOPEX(EPS eps)
{
  EPS_BLOPEX      *blopex = (EPS_BLOPEX *)eps->data;
  int             info,its;
  
  PetscFunctionBegin;
  
  info = lobpcg_solve(blopex->eigenvectors,eps,OperatorAMultiVector,
  	eps->isgeneralized?eps:PETSC_NULL,eps->isgeneralized?OperatorBMultiVector:PETSC_NULL,
        eps,Precond_FnMultiVector,blopex->Y,
        blopex->blap_fn,blopex->tol,eps->max_it,1,&its,
        eps->eigr,PETSC_NULL,0,eps->errest,PETSC_NULL,0);
  if (info>0) SETERRQ1(PETSC_ERR_LIB,"Error in blopex (code=%d)",info); 

  eps->its = its;
  eps->nconv = eps->ncv;
  if (info==-1) eps->reason = EPS_DIVERGED_ITS;
  else eps->reason = EPS_CONVERGED_TOL;
  
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
  SLEPCSetupInterpreterForDignifiedDeath(&blopex->ii);
  mv_MultiVectorDestroy(blopex->eigenvectors);
  mv_MultiVectorDestroy(blopex->Y);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = EPSFreeSolution(eps);CHKERRQ(ierr);
  ierr = EPSDestroy_Default(eps); CHKERRQ(ierr);
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
  ierr = KSPCreate(((PetscObject)eps)->comm,&blopex->ksp);CHKERRQ(ierr);
  ierr = EPSGetOptionsPrefix(eps,&prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(blopex->ksp,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(blopex->ksp,"eps_blopex_");CHKERRQ(ierr);
  eps->data                      = (void *) blopex;
  eps->ops->solve                = EPSSolve_BLOPEX;
  eps->ops->setup                = EPSSetUp_BLOPEX;
  eps->ops->setfromoptions       = PETSC_NULL;
  eps->ops->destroy              = EPSDestroy_BLOPEX;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  eps->which = EPS_SMALLEST_REAL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
