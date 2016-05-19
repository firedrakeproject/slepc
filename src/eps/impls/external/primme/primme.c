/*
   This file implements a wrapper to the PRIMME package

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/epsimpl.h>    /*I "slepceps.h" I*/

PetscErrorCode EPSSolve_PRIMME(EPS);

EXTERN_C_BEGIN
#include <primme.h>
EXTERN_C_END

typedef struct {
  primme_params primme;           /* param struc */
  primme_preset_method method;    /* primme method */
  Mat       A;                    /* problem matrix */
  EPS       eps;                  /* EPS current context */
  KSP       ksp;                  /* linear solver and preconditioner */
  Vec       x,y;                  /* auxiliary vectors */
  PetscReal target;               /* a copy of eps's target */
} EPS_PRIMME;

static void multMatvec_PRIMME(void *in,void *out,int *blockSize,primme_params *primme);
static void applyPreconditioner_PRIMME(void *in,void *out,int *blockSize,struct primme_params *primme);

static void par_GlobalSumDouble(void *sendBuf,void *recvBuf,int *count,primme_params *primme)
{
  PetscErrorCode ierr;
  ierr = MPI_Allreduce((double*)sendBuf,(double*)recvBuf,*count,MPI_DOUBLE,MPI_SUM,PetscObjectComm((PetscObject)primme->commInfo));CHKERRABORT(PetscObjectComm((PetscObject)primme->commInfo),ierr);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_PRIMME"
PetscErrorCode EPSSetUp_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  PetscMPIInt    numProcs,procID;
  EPS_PRIMME     *ops = (EPS_PRIMME*)eps->data;
  primme_params  *primme = &ops->primme;
  PetscBool      istrivial,flg;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)eps),&numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)eps),&procID);CHKERRQ(ierr);

  /* Check some constraints and set some default values */
  if (!eps->max_it) eps->max_it = PetscMax(1000,eps->n);
  ierr = STGetOperators(eps->st,0,&ops->A);CHKERRQ(ierr);
  if (!eps->ishermitian) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"PRIMME is only available for Hermitian problems");
  if (eps->isgeneralized) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"PRIMME is not available for generalized problems");
  if (eps->arbitrary) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");
  if (eps->stopping!=EPSStoppingBasic) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"External packages do not support user-defined stopping test");
  if (!eps->which) eps->which = EPS_LARGEST_REAL;
  if (eps->converged != EPSConvergedAbsolute) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"PRIMME only supports absolute convergence test");
  ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support region filtering");

  ierr = STSetUp(eps->st);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STPRECOND,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"PRIMME only works with STPRECOND");

  /* Transfer SLEPc options to PRIMME options */
  primme->n          = eps->n;
  primme->nLocal     = eps->nloc;
  primme->numEvals   = eps->nev;
  primme->matrix     = ops;
  primme->commInfo   = eps;
  primme->maxMatvecs = eps->max_it;
  primme->eps        = eps->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:eps->tol;
  primme->numProcs   = numProcs;
  primme->procID     = procID;
  primme->printLevel = 0;
  primme->correctionParams.precondition = 1;

  switch (eps->which) {
    case EPS_LARGEST_REAL:
      primme->target = primme_largest;
      break;
    case EPS_SMALLEST_REAL:
      primme->target = primme_smallest;
      break;
    case EPS_TARGET_MAGNITUDE:
    case EPS_TARGET_REAL:
      primme->target = primme_closest_abs;
      primme->numTargetShifts = 1;
      ops->target = PetscRealPart(eps->target);
      primme->targetShifts = &ops->target;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"'which' value not supported by PRIMME");
      break;
  }

  if (primme_set_method(ops->method,primme) < 0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"PRIMME method not valid");

  /* If user sets ncv, maxBasisSize is modified. If not, ncv is set as maxBasisSize */
  if (eps->ncv) primme->maxBasisSize = eps->ncv;
  else eps->ncv = primme->maxBasisSize;
  if (eps->ncv < eps->nev+primme->maxBlockSize)  SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"PRIMME needs ncv >= nev+maxBlockSize");
  if (eps->mpd) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }

  if (eps->extraction) { ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr); }

  /* Set workspace */
  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->V,BVVECS,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver requires a BV with contiguous storage");

  /* Setup the preconditioner */
  ops->eps = eps;
  if (primme->correctionParams.precondition) {
    ierr = STGetKSP(eps->st,&ops->ksp);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)ops->ksp,KSPPREONLY,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"PRIMME only works with KSPPREONLY");
    primme->preconditioner = NULL;
    primme->applyPreconditioner = applyPreconditioner_PRIMME;
  }

  /* Prepare auxiliary vectors */
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)eps),1,eps->nloc,eps->n,NULL,&ops->x);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)eps),1,eps->nloc,eps->n,NULL,&ops->y);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ops->x);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ops->y);CHKERRQ(ierr);

  /* dispatch solve method */
  eps->ops->solve = EPSSolve_PRIMME;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_PRIMME"
PetscErrorCode EPSSolve_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  EPS_PRIMME     *ops = (EPS_PRIMME*)eps->data;
  PetscScalar    *a;
  Vec            v0;
#if defined(PETSC_USE_COMPLEX)
  PetscInt       i;
  PetscReal      *evals;
#endif

  PetscFunctionBegin;
  /* Reset some parameters left from previous runs */
  ops->primme.aNorm    = 1.0;
  ops->primme.initSize = eps->nini;
  ops->primme.iseed[0] = -1;

  /* Call PRIMME solver */
  ierr = BVGetColumn(eps->V,0,&v0);CHKERRQ(ierr);
  ierr = VecGetArray(v0,&a);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = dprimme(eps->eigr,a,eps->errest,&ops->primme);
  if (ierr) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"PRIMME library failed with error code=%d",ierr);
#else
  /* PRIMME returns real eigenvalues, but SLEPc works with complex ones */
  ierr = PetscMalloc1(eps->ncv,&evals);CHKERRQ(ierr);
  ierr = zprimme(evals,(Complex_Z*)a,eps->errest,&ops->primme);
  if (ierr) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"PRIMME library failed with error code=%d",ierr);
  for (i=0;i<eps->ncv;i++) eps->eigr[i] = evals[i];
  ierr = PetscFree(evals);CHKERRQ(ierr);
#endif
  ierr = VecRestoreArray(v0,&a);CHKERRQ(ierr);
  ierr = BVRestoreColumn(eps->V,0,&v0);CHKERRQ(ierr);

  eps->nconv  = ops->primme.initSize >= 0 ? ops->primme.initSize : 0;
  eps->reason = eps->ncv >= eps->nev ? EPS_CONVERGED_TOL: EPS_DIVERGED_ITS;
  eps->its    = ops->primme.stats.numOuterIterations;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "multMatvec_PRIMME"
static void multMatvec_PRIMME(void *in,void *out,int *blockSize,primme_params *primme)
{
  PetscErrorCode ierr;
  PetscInt       i,N = primme->n;
  EPS_PRIMME     *ops = (EPS_PRIMME*)primme->matrix;
  Vec            x = ops->x,y = ops->y;
  Mat            A = ops->A;

  PetscFunctionBegin;
  for (i=0;i<*blockSize;i++) {
    /* build vectors using 'in' an 'out' workspace */
    ierr = VecPlaceArray(x,(PetscScalar*)in+N*i);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
    ierr = VecPlaceArray(y,(PetscScalar*)out+N*i);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);

    ierr = MatMult(A,x,y);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);

    ierr = VecResetArray(x);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
    ierr = VecResetArray(y);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
  }
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "applyPreconditioner_PRIMME"
static void applyPreconditioner_PRIMME(void *in,void *out,int *blockSize,struct primme_params *primme)
{
  PetscErrorCode ierr;
  PetscInt       i,N = primme->n;
  EPS_PRIMME     *ops = (EPS_PRIMME*)primme->matrix;
  Vec            x = ops->x,y = ops->y;

  PetscFunctionBegin;
  for (i=0;i<*blockSize;i++) {
    /* build vectors using 'in' an 'out' workspace */
    ierr = VecPlaceArray(x,(PetscScalar*)in+N*i);CHKERRABORT(PetscObjectComm((PetscObject)ops->ksp),ierr);
    ierr = VecPlaceArray(y,(PetscScalar*)out+N*i);CHKERRABORT(PetscObjectComm((PetscObject)ops->ksp),ierr);
    ierr = KSPSolve(ops->ksp,x,y);CHKERRABORT(PetscObjectComm((PetscObject)ops->ksp),ierr);
    ierr = VecResetArray(x);CHKERRABORT(PetscObjectComm((PetscObject)ops->ksp),ierr);
    ierr = VecResetArray(y);CHKERRABORT(PetscObjectComm((PetscObject)ops->ksp),ierr);
  }
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "EPSReset_PRIMME"
PetscErrorCode EPSReset_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  EPS_PRIMME     *ops = (EPS_PRIMME*)eps->data;

  PetscFunctionBegin;
  primme_Free(&ops->primme);
  ierr = VecDestroy(&ops->x);CHKERRQ(ierr);
  ierr = VecDestroy(&ops->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_PRIMME"
PetscErrorCode EPSDestroy_PRIMME(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPRIMMESetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPRIMMESetMethod_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPRIMMEGetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPRIMMEGetMethod_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSView_PRIMME"
PetscErrorCode EPSView_PRIMME(EPS eps,PetscViewer viewer)
{
  PetscErrorCode  ierr;
  PetscBool       isascii;
  primme_params   *primme = &((EPS_PRIMME*)eps->data)->primme;
  EPSPRIMMEMethod methodn;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  PRIMME: block size=%D\n",primme->maxBlockSize);CHKERRQ(ierr);
    ierr = EPSPRIMMEGetMethod(eps,&methodn);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  PRIMME: solver method: %s\n",EPSPRIMMEMethods[methodn]);CHKERRQ(ierr);

    /* Display PRIMME params */
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)eps),&rank);CHKERRQ(ierr);
    if (!rank) primme_display_params(*primme);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetFromOptions_PRIMME"
PetscErrorCode EPSSetFromOptions_PRIMME(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscErrorCode  ierr;
  EPS_PRIMME      *ctx = (EPS_PRIMME*)eps->data;
  PetscInt        bs;
  EPSPRIMMEMethod meth;
  PetscBool       flg;
  KSP             ksp;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS PRIMME Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_primme_block_size","Maximum block size","EPSPRIMMESetBlockSize",ctx->primme.maxBlockSize,&bs,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = EPSPRIMMESetBlockSize(eps,bs);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnum("-eps_primme_method","Method for solving the eigenproblem","EPSPRIMMESetMethod",EPSPRIMMEMethods,(PetscEnum)ctx->method,(PetscEnum*)&meth,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = EPSPRIMMESetMethod(eps,meth);CHKERRQ(ierr);
  }

  /* Set STPrecond as the default ST */
  if (!((PetscObject)eps->st)->type_name) {
    ierr = STSetType(eps->st,STPRECOND);CHKERRQ(ierr);
  }
  ierr = STPrecondSetKSPHasMat(eps->st,PETSC_TRUE);CHKERRQ(ierr);

  /* Set the default options of the KSP */
  ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSPRIMMESetBlockSize_PRIMME"
static PetscErrorCode EPSPRIMMESetBlockSize_PRIMME(EPS eps,PetscInt bs)
{
  EPS_PRIMME *ops = (EPS_PRIMME*)eps->data;

  PetscFunctionBegin;
  if (bs == PETSC_DEFAULT) ops->primme.maxBlockSize = 1;
  else if (bs <= 0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"PRIMME: block size must be positive");
  else ops->primme.maxBlockSize = bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSPRIMMESetBlockSize"
/*@
   EPSPRIMMESetBlockSize - The maximum block size that PRIMME will try to use.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  bs - block size

   Options Database Key:
.  -eps_primme_block_size - Sets the max allowed block size value

   Notes:
   If the block size is not set, the value established by primme_initialize
   is used.

   The user should set the block size based on the architecture specifics
   of the target computer, as well as any a priori knowledge of multiplicities.
   The code does NOT require bs > 1 to find multiple eigenvalues. For some
   methods, keeping bs = 1 yields the best overall performance.

   Level: advanced

.seealso: EPSPRIMMEGetBlockSize()
@*/
PetscErrorCode EPSPRIMMESetBlockSize(EPS eps,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,bs,2);
  ierr = PetscTryMethod(eps,"EPSPRIMMESetBlockSize_C",(EPS,PetscInt),(eps,bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSPRIMMEGetBlockSize_PRIMME"
static PetscErrorCode EPSPRIMMEGetBlockSize_PRIMME(EPS eps,PetscInt *bs)
{
  EPS_PRIMME *ops = (EPS_PRIMME*)eps->data;

  PetscFunctionBegin;
  *bs = ops->primme.maxBlockSize;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSPRIMMEGetBlockSize"
/*@
   EPSPRIMMEGetBlockSize - Get the maximum block size the code will try to use.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  bs - returned block size

   Level: advanced

.seealso: EPSPRIMMESetBlockSize()
@*/
PetscErrorCode EPSPRIMMEGetBlockSize(EPS eps,PetscInt *bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(bs,2);
  ierr = PetscUseMethod(eps,"EPSPRIMMEGetBlockSize_C",(EPS,PetscInt*),(eps,bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSPRIMMESetMethod_PRIMME"
static PetscErrorCode EPSPRIMMESetMethod_PRIMME(EPS eps,EPSPRIMMEMethod method)
{
  EPS_PRIMME *ops = (EPS_PRIMME*)eps->data;

  PetscFunctionBegin;
  ops->method = (primme_preset_method)method;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSPRIMMESetMethod"
/*@
   EPSPRIMMESetMethod - Sets the method for the PRIMME library.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  method - method that will be used by PRIMME

   Options Database Key:
.  -eps_primme_method - Sets the method for the PRIMME library

   Note:
   If not set, the method defaults to EPS_PRIMME_DEFAULT_MIN_TIME.

   Level: advanced

.seealso: EPSPRIMMEGetMethod(), EPSPRIMMEMethod
@*/
PetscErrorCode EPSPRIMMESetMethod(EPS eps,EPSPRIMMEMethod method)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(eps,method,2);
  ierr = PetscTryMethod(eps,"EPSPRIMMESetMethod_C",(EPS,EPSPRIMMEMethod),(eps,method));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSPRIMMEGetMethod_PRIMME"
static PetscErrorCode EPSPRIMMEGetMethod_PRIMME(EPS eps,EPSPRIMMEMethod *method)
{
  EPS_PRIMME *ops = (EPS_PRIMME*)eps->data;

  PetscFunctionBegin;
  *method = (EPSPRIMMEMethod)ops->method;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSPRIMMEGetMethod"
/*@
   EPSPRIMMEGetMethod - Gets the method for the PRIMME library.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  method - method that will be used by PRIMME

   Level: advanced

.seealso: EPSPRIMMESetMethod(), EPSPRIMMEMethod
@*/
PetscErrorCode EPSPRIMMEGetMethod(EPS eps,EPSPRIMMEMethod *method)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(method,2);
  ierr = PetscUseMethod(eps,"EPSPRIMMEGetMethod_C",(EPS,EPSPRIMMEMethod*),(eps,method));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_PRIMME"
PETSC_EXTERN PetscErrorCode EPSCreate_PRIMME(EPS eps)
{
  PetscErrorCode ierr;
  EPS_PRIMME     *primme;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&primme);CHKERRQ(ierr);
  eps->data = (void*)primme;

  eps->ops->setup          = EPSSetUp_PRIMME;
  eps->ops->setfromoptions = EPSSetFromOptions_PRIMME;
  eps->ops->destroy        = EPSDestroy_PRIMME;
  eps->ops->reset          = EPSReset_PRIMME;
  eps->ops->view           = EPSView_PRIMME;
  eps->ops->backtransform  = EPSBackTransform_Default;

  primme_initialize(&primme->primme);
  primme->primme.matrixMatvec = multMatvec_PRIMME;
  primme->primme.globalSumDouble = par_GlobalSumDouble;
  primme->method = (primme_preset_method)EPS_PRIMME_DEFAULT_MIN_TIME;

  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPRIMMESetBlockSize_C",EPSPRIMMESetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPRIMMESetMethod_C",EPSPRIMMESetMethod_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPRIMMEGetBlockSize_C",EPSPRIMMEGetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSPRIMMEGetMethod_C",EPSPRIMMEGetMethod_PRIMME);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

