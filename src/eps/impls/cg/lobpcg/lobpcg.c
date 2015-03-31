/*

   SLEPc eigensolver: "lobpcg"

   Method: Locally Optimal Block Preconditioned Conjugate Gradient

   Algorithm:

       LOBPCG with soft and hard locking.

   References:

       [1] A. V. Knyazev, "Toward the optimal preconditioned eigensolver:
           locally optimal block preconditioned conjugate gradient method",
           SIAM J. Sci. Comput. 23(2):517-541, 2001.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>                /*I "slepceps.h" I*/

PetscErrorCode EPSSolve_LOBPCG(EPS);

typedef struct {
  PetscInt bs;
} EPS_LOBPCG;

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_LOBPCG"
PetscErrorCode EPSSetUp_LOBPCG(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      precond;

  PetscFunctionBegin;
  if (!eps->ishermitian) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"LOBPCG only works for Hermitian problems");
  ierr = EPSSetDimensions_Default(eps,eps->nev,&eps->ncv,&eps->mpd);CHKERRQ(ierr);
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (!eps->which) eps->which = EPS_SMALLEST_REAL;
  if (eps->which!=EPS_SMALLEST_REAL) SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");
  if (!eps->extraction) {
    ierr = EPSSetExtraction(eps,EPS_RITZ);CHKERRQ(ierr);
  } else if (eps->extraction!=EPS_RITZ) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Unsupported extraction type");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");
  /* Set STPrecond as the default ST */
  if (!((PetscObject)eps->st)->type_name) {
    ierr = STSetType(eps->st,STPRECOND);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STPRECOND,&precond);CHKERRQ(ierr);
  if (!precond) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"LOBPCG only works with precond ST");

  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = EPS_SetInnerProduct(eps);CHKERRQ(ierr);
  ierr = DSSetType(eps->ds,DSHEP);CHKERRQ(ierr);
  ierr = DSAllocate(eps->ds,eps->ncv);CHKERRQ(ierr);
  ierr = EPSSetWorkVecs(eps,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_LOBPCG"
PetscErrorCode EPSSolve_LOBPCG(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       k,ld,nv,ncv = eps->ncv,kini,nmat;
  PetscReal      norm;
  PetscBool      breakdown;
  Mat            A,B;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = STGetNumMatrices(eps->st,&nmat);CHKERRQ(ierr);
  ierr = STGetOperators(eps->st,0,&A);CHKERRQ(ierr);
  if (nmat>1) { ierr = STGetOperators(eps->st,1,&B);CHKERRQ(ierr); }
  else B = NULL;

  kini = eps->nini;
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++;
    nv = PetscMin(eps->nconv+eps->mpd,ncv);
    ierr = DSSetDimensions(eps->ds,nv,0,eps->nconv,0);CHKERRQ(ierr);
    /* Generate more initial vectors if necessary */
    while (kini<nv) {
      ierr = BVSetRandomColumn(eps->V,kini,eps->rand);CHKERRQ(ierr);
      ierr = BVOrthogonalizeColumn(eps->V,kini,NULL,&norm,&breakdown);CHKERRQ(ierr);
      if (norm>0.0 && !breakdown) {
        ierr = BVScaleColumn(eps->V,kini,1.0/norm);CHKERRQ(ierr);
        kini++;
      }
    }

    
    ierr = EPSMonitor(eps,eps->its,k,eps->eigr,eps->eigi,eps->errest,nv);CHKERRQ(ierr);
    eps->nconv = k;
  }

  /* truncate Schur decomposition and change the state to raw so that
     PSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(eps->ds,eps->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(eps->ds,DS_STATE_RAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGSetBlockSize_LOBPCG"
static PetscErrorCode EPSLOBPCGSetBlockSize_LOBPCG(EPS eps,PetscInt bs)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  ctx->bs = bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGSetBlockSize"
/*@
   EPSLOBPCGSetBlockSize - Sets the block size of the LOBPCG method.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  bs  - the block size

   Options Database Key:
.  -eps_lobpcg_blocksize - Sets the block size

   Level: advanced

.seealso: EPSLOBPCGGetBlockSize()
@*/
PetscErrorCode EPSLOBPCGSetBlockSize(EPS eps,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,bs,2);
  ierr = PetscTryMethod(eps,"EPSLOBPCGSetBlockSize_C",(EPS,PetscInt),(eps,bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGGetBlockSize_LOBPCG"
static PetscErrorCode EPSLOBPCGGetBlockSize_LOBPCG(EPS eps,PetscInt *bs)
{
  EPS_LOBPCG *ctx = (EPS_LOBPCG*)eps->data;

  PetscFunctionBegin;
  *bs = ctx->bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLOBPCGGetBlockSize"
/*@
   EPSLOBPCGGetBlockSize - Gets the block size used in the LOBPCG method.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  bs - the block size

   Level: advanced

.seealso: EPSLOBPCGSetBlockSize()
@*/
PetscErrorCode EPSLOBPCGGetBlockSize(EPS eps,PetscInt *bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(bs,2);
  ierr = PetscTryMethod(eps,"EPSLOBPCGGetBlockSize_C",(EPS,PetscInt*),(eps,bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSView_LOBPCG"
PetscErrorCode EPSView_LOBPCG(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_LOBPCG     *ctx = (EPS_LOBPCG*)eps->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  LOBPCG: block size %D\n",ctx->bs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetFromOptions_LOBPCG"
PetscErrorCode EPSSetFromOptions_LOBPCG(PetscOptions *PetscOptionsObject,EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscInt       bs;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS LOBPCG Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_lobpcg_blocksize","LOBPCG block size","EPSLOBPCGSetBlockSize",20,&bs,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = EPSLOBPCGSetBlockSize(eps,bs);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_LOBPCG"
PetscErrorCode EPSDestroy_LOBPCG(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetBlockSize_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_LOBPCG"
PETSC_EXTERN PetscErrorCode EPSCreate_LOBPCG(EPS eps)
{
  EPS_LOBPCG     *lobpcg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&lobpcg);CHKERRQ(ierr);
  eps->data = (void*)lobpcg;
  lobpcg->bs = 1;

  eps->ops->setup          = EPSSetUp_LOBPCG;
  eps->ops->solve          = EPSSolve_LOBPCG;
  eps->ops->setfromoptions = EPSSetFromOptions_LOBPCG;
  eps->ops->destroy        = EPSDestroy_LOBPCG;
  eps->ops->view           = EPSView_LOBPCG;
  eps->ops->backtransform  = EPSBackTransform_Default;
  ierr = STSetType(eps->st,STPRECOND);CHKERRQ(ierr);
  ierr = STPrecondSetKSPHasMat(eps->st,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGSetBlockSize_C",EPSLOBPCGSetBlockSize_LOBPCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSLOBPCGGetBlockSize_C",EPSLOBPCGGetBlockSize_LOBPCG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

