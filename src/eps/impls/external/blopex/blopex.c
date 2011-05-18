/*
       This file implements a wrapper to the BLOPEX solver

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

#include <private/stimpl.h>       /*I "slepcst.h" I*/
#include <private/epsimpl.h>      /*I "slepceps.h" I*/
#define BlopexInt PetscInt
#include "slepc-interface.h"
#include <lobpcg.h>
#include <interpreter.h>
#include <multivector.h>
#include <temp_multivector.h>

PetscErrorCode EPSSolve_BLOPEX(EPS);

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
  PetscErrorCode ierr;
  EPS            eps = (EPS)data;
  EPS_BLOPEX     *blopex = (EPS_BLOPEX*)eps->data;
  PetscInt       lits;
      
  PetscFunctionBegin;
  ierr = KSPSolve(blopex->ksp,(Vec)x,(Vec)y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  ierr = KSPGetIterationNumber(blopex->ksp,&lits); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  eps->OP->lineariterations+= lits;
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "Precond_FnMultiVector"
static void Precond_FnMultiVector(void *data,void *x,void *y)
{
  EPS        eps = (EPS)data;
  EPS_BLOPEX *blopex = (EPS_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(Precond_FnSingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "OperatorASingleVector"
static void OperatorASingleVector(void *data,void *x,void *y)
{
  PetscErrorCode ierr;
  EPS            eps = (EPS)data;
  Mat            A;
 
  PetscFunctionBegin;
  ierr = STGetOperators(eps->OP,&A,PETSC_NULL); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  ierr = MatMult(A,(Vec)x,(Vec)y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "OperatorAMultiVector"
static void OperatorAMultiVector(void *data,void *x,void *y)
{
  EPS        eps = (EPS)data;
  EPS_BLOPEX *blopex = (EPS_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(OperatorASingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "OperatorBSingleVector"
static void OperatorBSingleVector(void *data,void *x,void *y)
{
  PetscErrorCode ierr;
  EPS            eps = (EPS)data;
  Mat            B;
  
  PetscFunctionBegin;
  ierr = STGetOperators(eps->OP,PETSC_NULL,&B); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  ierr = MatMult(B,(Vec)x,(Vec)y); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "OperatorBMultiVector"
static void OperatorBMultiVector(void *data,void *x,void *y)
{
  EPS        eps = (EPS)data;
  EPS_BLOPEX *blopex = (EPS_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(OperatorBSingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_BLOPEX"
PetscErrorCode EPSSetUp_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i;
  EPS_BLOPEX     *blopex = (EPS_BLOPEX *)eps->data;
  PetscBool      isPrecond,isPreonly;

  PetscFunctionBegin;
  if (!eps->ishermitian) { 
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"blopex only works for hermitian problems"); 
  }
  if (!eps->which) eps->which = EPS_SMALLEST_REAL;
  if (eps->which!=EPS_SMALLEST_REAL) {
    SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of eps->which");
  }

  /* Change the default sigma to inf if necessary */
  if (eps->which == EPS_LARGEST_MAGNITUDE || eps->which == EPS_LARGEST_REAL ||
      eps->which == EPS_LARGEST_IMAGINARY) {
    ierr = STSetDefaultShift(eps->OP,3e300);CHKERRQ(ierr);
  }

  ierr = STSetUp(eps->OP);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)eps->OP,STPRECOND,&isPrecond);CHKERRQ(ierr);
  if (!isPrecond) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"blopex only works with STPRECOND");
  ierr = STGetKSP(eps->OP,&blopex->ksp);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)blopex->ksp,KSPPREONLY,&isPreonly);CHKERRQ(ierr);
  if (!isPreonly)
    SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"blopex only works with KSPPREONLY");

  eps->ncv = eps->nev = PetscMin(eps->nev,eps->n);
  if (eps->mpd) PetscInfo(eps,"Warning: parameter mpd ignored\n");
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);
  
  blopex->tol.absolute = eps->tol;
  blopex->tol.relative = 1e-50;
  
  SLEPCSetupInterpreter(&blopex->ii);
  blopex->eigenvectors = mv_MultiVectorCreateFromSampleVector(&blopex->ii,eps->ncv,eps->V);
  for (i=0;i<eps->ncv;i++) { ierr = PetscObjectReference((PetscObject)eps->V[i]);CHKERRQ(ierr); }
  mv_MultiVectorSetRandom(blopex->eigenvectors,1234);

  if (eps->nds > 0) {
    blopex->Y = mv_MultiVectorCreateFromSampleVector(&blopex->ii,eps->nds,eps->DS);
    for (i=0;i<eps->nds;i++) { ierr = PetscObjectReference((PetscObject)eps->DS[i]);CHKERRQ(ierr); }
  } else
    blopex->Y = PETSC_NULL;

#if defined(PETSC_USE_COMPLEX)
  blopex->blap_fn.zpotrf = PETSC_zpotrf_interface;
  blopex->blap_fn.zhegv = PETSC_zsygv_interface;
#else
  blopex->blap_fn.dpotrf = PETSC_dpotrf_interface;
  blopex->blap_fn.dsygv = PETSC_dsygv_interface;
#endif

  if (eps->extraction) {
     ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr);
  }

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Left vectors not supported in this solver");
  eps->ops->solve = EPSSolve_BLOPEX;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_BLOPEX"
PetscErrorCode EPSSolve_BLOPEX(EPS eps)
{
  EPS_BLOPEX *blopex = (EPS_BLOPEX *)eps->data;
  int        info,its;
  
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  info = lobpcg_solve_complex(blopex->eigenvectors,eps,OperatorAMultiVector,
        eps->isgeneralized?eps:PETSC_NULL,eps->isgeneralized?OperatorBMultiVector:PETSC_NULL,
        eps,Precond_FnMultiVector,blopex->Y,
        blopex->blap_fn,blopex->tol,eps->max_it,0,&its,
        (komplex*)eps->eigr,PETSC_NULL,0,eps->errest,PETSC_NULL,0);
#else
  info = lobpcg_solve_double(blopex->eigenvectors,eps,OperatorAMultiVector,
        eps->isgeneralized?eps:PETSC_NULL,eps->isgeneralized?OperatorBMultiVector:PETSC_NULL,
        eps,Precond_FnMultiVector,blopex->Y,
        blopex->blap_fn,blopex->tol,eps->max_it,0,&its,
        eps->eigr,PETSC_NULL,0,eps->errest,PETSC_NULL,0);
#endif
  if (info>0) SETERRQ1(((PetscObject)eps)->comm,PETSC_ERR_LIB,"Error in blopex (code=%d)",info); 

  eps->its = its;
  eps->nconv = eps->ncv;
  if (info==-1) eps->reason = EPS_DIVERGED_ITS;
  else eps->reason = EPS_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSReset_BLOPEX"
PetscErrorCode EPSReset_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;
  EPS_BLOPEX     *blopex = (EPS_BLOPEX *)eps->data;

  PetscFunctionBegin;
  mv_MultiVectorDestroy(blopex->eigenvectors);
  mv_MultiVectorDestroy(blopex->Y);
  ierr = EPSReset_Default(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_BLOPEX"
PetscErrorCode EPSDestroy_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  LOBPCG_DestroyRandomContext();
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_BLOPEX"
PetscErrorCode EPSCreate_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,EPS_BLOPEX,&eps->data);CHKERRQ(ierr);
  eps->ops->setup                = EPSSetUp_BLOPEX;
  eps->ops->destroy              = EPSDestroy_BLOPEX;
  eps->ops->reset                = EPSReset_BLOPEX;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  ierr = STSetType(eps->OP,STPRECOND);CHKERRQ(ierr);
  ierr = STPrecondSetKSPHasMat(eps->OP,PETSC_TRUE);CHKERRQ(ierr);
  LOBPCG_InitRandomContext();
  PetscFunctionReturn(0);
}
EXTERN_C_END
