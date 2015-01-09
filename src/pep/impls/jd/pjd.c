/*

   SLEPc polynomial eigensolver: "jd"

   Method: Jacobi-Davidson

   Algorithm:

       Jacobi-Davidson for polynomial eigenvalue problems.
       Based on code contributed by the authors of [2] below.

   References:

       [1] G.L.G. Sleijpen et al., "Jacobi-Davidson type methods for
           generalized eigenproblems and polynomial eigenproblems", BIT
           36(3):595-633, 1996.

       [2] Feng-Nan Hwang, Zih-Hao Wei, Tsung-Ming Huang, Weichung Wang,
           "A Parallel Additive Schwarz Preconditioned Jacobi-Davidson
           Algorithm for Polynomial Eigenvalue Problems in Quantum Dot
           Simulation", J. Comput. Phys. 229(8):2932-2947, 2010.

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

#include <slepc-private/pepimpl.h>    /*I "slepcpep.h" I*/

typedef struct {
  PetscReal keep;         /* restart parameter */
} PEP_JD;

#undef __FUNCT__
#define __FUNCT__ "PEPSetUp_JD"
PetscErrorCode PEPSetUp_JD(PEP pep)
{
  PetscErrorCode ierr;
  PEP_JD         *ctx = (PEP_JD*)pep->data;
  PetscBool      precond,flg;

  PetscFunctionBegin;
  ierr = PEPSetDimensions_Default(pep,pep->nev,&pep->ncv,&pep->mpd);CHKERRQ(ierr);
  if (!pep->max_it) pep->max_it = PetscMax(100,2*pep->n/pep->ncv);
  if (!pep->which) pep->which = PEP_LARGEST_MAGNITUDE;

  /* Set STPrecond as the default ST */
  if (!((PetscObject)pep->st)->type_name) {
    ierr = STSetType(pep->st,STPRECOND);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)pep->st,STPRECOND,&precond);CHKERRQ(ierr);
  if (!precond) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"JD only works with precond spectral transformation");

  if (pep->basis!=PEP_BASIS_MONOMIAL) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver not implemented for non-monomial bases");
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_SUP,"Solver requires the ST transformation flag set, see STSetTransform()");

  if (!ctx->keep) ctx->keep = 0.5;

  ierr = PEPAllocateSolution(pep,0);CHKERRQ(ierr);
  ierr = PEPSetWorkVecs(pep,4);CHKERRQ(ierr);

  ierr = DSSetType(pep->ds,DSGNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(pep->ds,pep->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSolve_JD"
PetscErrorCode PEPSolve_JD(PEP pep)
{
  PetscErrorCode ierr;
  /*PEP_JD         *ctx = (PEP_JD*)pep->data;*/
  PetscInt       j,k=0,nv,ld;
  /*Vec            v=pep->work[0],w=pep->work[1];*/
  PetscReal      norm;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(pep->ds,&ld);CHKERRQ(ierr);
  nv = pep->ncv;

  /* Modify matrix norms so that the scaling affects the convergence test */
  norm = pep->dsfactor;
  for (j=0;j<pep->nmat;j++) {
    pep->nrma[j] *= norm;
    norm*=pep->sfactor;
  }

   /* Restart loop */
  while (pep->reason == PEP_CONVERGED_ITERATING) {
    pep->its++;

    /* Solve projected problem */
    ierr = DSSolve(pep->ds,pep->eigr,pep->eigi);CHKERRQ(ierr);
    ierr = DSSort(pep->ds,pep->eigr,pep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);

    /* Check convergence */
    if (pep->its >= pep->max_it) pep->reason = PEP_DIVERGED_ITS;
    if (k >= pep->nev) pep->reason = PEP_CONVERGED_TOL;

    pep->nconv = k;
    ierr = PEPMonitor(pep,pep->its,pep->nconv,pep->eigr,pep->eigi,pep->errest,nv);CHKERRQ(ierr);
  }

  for (j=0;j<pep->nconv;j++) {
    pep->eigr[j] *= pep->sfactor;
    pep->eigi[j] *= pep->sfactor;
  }

  /* Restore matrix norms */
  norm = pep->dsfactor;
  for (j=0;j<pep->nmat;j++) {
    pep->nrma[j] /= norm;
    norm*=pep->sfactor;
  }

  /* truncate Schur decomposition and change the state to raw so that
     DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(pep->ds,pep->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(pep->ds,DS_STATE_RAW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDSetRestart_JD"
static PetscErrorCode PEPJDSetRestart_JD(PEP pep,PetscReal keep)
{
  PEP_JD *ctx = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) ctx->keep = 0.5;
  else {
    if (keep<0.1 || keep>0.9) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument must be in the range [0.1,0.9]");
    ctx->keep = keep;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDSetRestart"
/*@
   PEPJDSetRestart - Sets the restart parameter for the Jacobi-Davidson
   method, in particular the proportion of basis vectors that must be kept
   after restart.

   Logically Collective on PEP

   Input Parameters:
+  pep  - the eigenproblem solver context
-  keep - the number of vectors to be kept at restart

   Options Database Key:
.  -pep_jd_restart - Sets the restart parameter

   Notes:
   Allowed values are in the range [0.1,0.9]. The default is 0.5.

   Level: advanced

.seealso: PEPJDGetRestart()
@*/
PetscErrorCode PEPJDSetRestart(PEP pep,PetscReal keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,keep,2);
  ierr = PetscTryMethod(pep,"PEPJDSetRestart_C",(PEP,PetscReal),(pep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDGetRestart_JD"
static PetscErrorCode PEPJDGetRestart_JD(PEP pep,PetscReal *keep)
{
  PEP_JD *ctx = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  *keep = ctx->keep;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDGetRestart"
/*@
   PEPJDGetRestart - Gets the restart parameter used in the Jacobi-Davidson method.

   Not Collective

   Input Parameter:
.  pep - the eigenproblem solver context

   Output Parameter:
.  keep - the restart parameter

   Level: advanced

.seealso: PEPJDSetRestart()
@*/
PetscErrorCode PEPJDGetRestart(PEP pep,PetscReal *keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(keep,2);
  ierr = PetscTryMethod(pep,"PEPJDGetRestart_C",(PEP,PetscReal*),(pep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetFromOptions_JD"
PetscErrorCode PEPSetFromOptions_JD(PEP pep)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscReal      keep;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("PEP JD Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pep_jd_restart","Proportion of vectors kept after restart","PEPJDSetRestart",0.5,&keep,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PEPJDSetRestart(pep,keep);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPView_JD"
PetscErrorCode PEPView_JD(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PEP_JD         *ctx = (PEP_JD*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  JD: %d%% of basis vectors kept after restart\n",(int)(100*ctx->keep));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPDestroy_JD"
PetscErrorCode PEPDestroy_JD(PEP pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetRestart_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPCreate_JD"
PETSC_EXTERN PetscErrorCode PEPCreate_JD(PEP pep)
{
  PEP_JD         *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pep,&ctx);CHKERRQ(ierr);
  pep->data = (void*)ctx;

  pep->ops->solve          = PEPSolve_JD;
  pep->ops->setup          = PEPSetUp_JD;
  pep->ops->setfromoptions = PEPSetFromOptions_JD;
  pep->ops->destroy        = PEPDestroy_JD;
  pep->ops->view           = PEPView_JD;
  pep->ops->computevectors = PEPComputeVectors_Schur;
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDSetRestart_C",PEPJDSetRestart_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pep,"PEPJDGetRestart_C",PEPJDGetRestart_JD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

