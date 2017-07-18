/*
   This file implements a wrapper to the PRIMME SVD solver

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

#include <slepc/private/svdimpl.h>    /*I "slepcsvd.h" I*/

#include <primme.h>

#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define PRIMME_DRIVER cprimme_svds
#else
#define PRIMME_DRIVER zprimme_svds
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
#define PRIMME_DRIVER sprimme_svds
#else
#define PRIMME_DRIVER dprimme_svds
#endif
#endif

typedef struct {
  primme_svds_params primme;         /* param struct */
  primme_svds_preset_method method;  /* primme method */
  SVD       svd;                     /* reference to the solver */
  Vec       x,y;                     /* auxiliary vectors */
} SVD_PRIMME;

static void multMatvec_PRIMME(void*,PRIMME_INT*,void*,PRIMME_INT*,int*,int*,struct primme_svds_params*,int*);

static void par_GlobalSumReal(void *sendBuf,void *recvBuf,int *count,primme_svds_params *primme,int *ierr)
{
  *ierr = MPI_Allreduce(sendBuf,recvBuf,*count,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)primme->commInfo));CHKERRABORT(PetscObjectComm((PetscObject)primme->commInfo),*ierr);
}

PetscErrorCode SVDSetUp_PRIMME(SVD svd)
{
  PetscErrorCode     ierr;
  PetscMPIInt        numProcs,procID;
  PetscInt           n,m,nloc,mloc;
  SVD_PRIMME         *ops = (SVD_PRIMME*)svd->data;
  primme_svds_params *primme = &ops->primme;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)svd),&numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)svd),&procID);CHKERRQ(ierr);

  /* Check some constraints and set some default values */
  ierr = SVDMatGetSize(svd,&m,&n);CHKERRQ(ierr);
  ierr = SVDMatGetLocalSize(svd,&mloc,&nloc);CHKERRQ(ierr);
  ierr = SVDSetDimensions_Default(svd);CHKERRQ(ierr);
  if (!svd->max_it) svd->max_it = PetscMax(n/svd->ncv,1000);
  svd->leftbasis = PETSC_TRUE;
  if (svd->stopping!=SVDStoppingBasic) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"External packages do not support user-defined stopping test");
  if (svd->converged != SVDConvergedAbsolute) { ierr = PetscInfo(svd,"Warning: using absolute convergence test\n");CHKERRQ(ierr); }

  /* Transfer SLEPc options to PRIMME options */
  primme->m          = m;
  primme->n          = n;
  primme->mLocal     = mloc;
  primme->nLocal     = nloc;
  primme->numSvals   = svd->nsv;
  primme->matrix     = ops;
  primme->commInfo   = svd;
  primme->maxMatvecs = svd->max_it;
  primme->eps        = svd->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:svd->tol;
  primme->numProcs   = numProcs;
  primme->procID     = procID;
  primme->locking    = 1;
  primme->printLevel = 0;

  switch (svd->which) {
    case SVD_LARGEST:
      primme->target = primme_svds_largest;
      break;
    case SVD_SMALLEST:
      primme->target = primme_svds_smallest;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"'which' value not supported by PRIMME");
      break;
  }

  /* If user sets mpd or ncv, maxBasisSize is modified */
  if (svd->mpd) primme->maxBasisSize = svd->mpd;
  else if (svd->ncv) primme->maxBasisSize = svd->ncv;

  if (primme_svds_set_method(ops->method,(primme_preset_method)EPS_PRIMME_DEFAULT_MIN_TIME,(primme_preset_method)EPS_PRIMME_DEFAULT_MIN_TIME,primme) < 0) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"PRIMME method not valid");

  svd->mpd = primme->maxBasisSize;
  svd->ncv = svd->nsv;

  /* Set workspace */
  ierr = SVDAllocateSolution(svd,0);CHKERRQ(ierr);

  /* Prepare auxiliary vectors */
  if (!ops->x) {
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)svd),1,nloc,n,NULL,&ops->x);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)svd),1,mloc,m,NULL,&ops->y);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ops->x);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)svd,(PetscObject)ops->y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_PRIMME(SVD svd)
{
  PetscErrorCode ierr;
  SVD_PRIMME     *ops = (SVD_PRIMME*)svd->data;
  PetscScalar    *svecs, *a;

  PetscFunctionBegin;
  /* Reset some parameters left from previous runs */
  ops->primme.aNorm    = 0.0;
  ops->primme.initSize = svd->nini;
  ops->primme.iseed[0] = -1;
  ops->primme.iseed[1] = -1;
  ops->primme.iseed[2] = -1;
  ops->primme.iseed[3] = -1;

  /* Allocating left and right singular vectors contiguously */
  ierr = PetscMalloc1(ops->primme.numSvals*(ops->primme.mLocal+ops->primme.nLocal),&svecs);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)svd,sizeof(PetscReal)*ops->primme.numSvals*(ops->primme.mLocal+ops->primme.nLocal));CHKERRQ(ierr);
  
  /* Call PRIMME solver */
  ierr = PRIMME_DRIVER(svd->sigma,svecs,svd->errest,&ops->primme);
  if (ierr) SETERRQ1(PetscObjectComm((PetscObject)svd),PETSC_ERR_LIB,"PRIMME library failed with error code=%d",ierr);

  /* Copy left and right singular vectors into svd */
  ierr = BVGetArray(svd->U,&a);CHKERRQ(ierr);
  ierr = PetscMemcpy(a,svecs,sizeof(PetscScalar)*ops->primme.mLocal*ops->primme.initSize);CHKERRQ(ierr);
  ierr = BVRestoreArray(svd->U,&a);CHKERRQ(ierr);

  ierr = BVGetArray(svd->V,&a);CHKERRQ(ierr);
  ierr = PetscMemcpy(a,svecs+ops->primme.mLocal*ops->primme.initSize,sizeof(PetscScalar)*ops->primme.nLocal*ops->primme.initSize);CHKERRQ(ierr);
  ierr = BVRestoreArray(svd->V,&a);CHKERRQ(ierr);

  ierr = PetscFree(svecs);CHKERRQ(ierr);

  svd->nconv  = ops->primme.initSize >= 0 ? ops->primme.initSize : 0;
  svd->reason = svd->nconv >= svd->nsv ? SVD_CONVERGED_TOL: SVD_DIVERGED_ITS;
  svd->its    = ops->primme.stats.numOuterIterations;
  PetscFunctionReturn(0);
}

static void multMatvec_PRIMME(void *xa,PRIMME_INT *ldx,void *ya,PRIMME_INT *ldy,int *blockSize,int *transpose,struct primme_svds_params *primme,int *ierr)
{
  PetscInt   i;
  SVD_PRIMME *ops = (SVD_PRIMME*)primme->matrix;
  Vec        x = ops->x,y = ops->y;
  SVD        svd = ops->svd;

  PetscFunctionBegin;
  for (i=0;i<*blockSize;i++) {
    if (*transpose) {
      *ierr = VecPlaceArray(y,(PetscScalar*)xa+(*ldx)*i);CHKERRABORT(PetscObjectComm((PetscObject)svd),*ierr);
      *ierr = VecPlaceArray(x,(PetscScalar*)ya+(*ldy)*i);CHKERRABORT(PetscObjectComm((PetscObject)svd),*ierr);
      *ierr = SVDMatMult(svd,PETSC_TRUE,y,x);CHKERRABORT(PetscObjectComm((PetscObject)svd),*ierr);
    } else {
      *ierr = VecPlaceArray(x,(PetscScalar*)xa+(*ldx)*i);CHKERRABORT(PetscObjectComm((PetscObject)svd),*ierr);
      *ierr = VecPlaceArray(y,(PetscScalar*)ya+(*ldy)*i);CHKERRABORT(PetscObjectComm((PetscObject)svd),*ierr);
      *ierr = SVDMatMult(svd,PETSC_FALSE,x,y);CHKERRABORT(PetscObjectComm((PetscObject)svd),*ierr);
    }
    *ierr = VecResetArray(x);CHKERRABORT(PetscObjectComm((PetscObject)svd),*ierr);
    *ierr = VecResetArray(y);CHKERRABORT(PetscObjectComm((PetscObject)svd),*ierr);
  }
  PetscFunctionReturnVoid();
}

PetscErrorCode SVDReset_PRIMME(SVD svd)
{
  PetscErrorCode ierr;
  SVD_PRIMME     *ops = (SVD_PRIMME*)svd->data;

  PetscFunctionBegin;
  primme_svds_free(&ops->primme);
  ierr = VecDestroy(&ops->x);CHKERRQ(ierr);
  ierr = VecDestroy(&ops->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDDestroy_PRIMME(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMESetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMEGetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMESetMethod_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMEGetMethod_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDView_PRIMME(SVD svd,PetscViewer viewer)
{
  PetscErrorCode     ierr;
  PetscBool          isascii;
  primme_svds_params *primme = &((SVD_PRIMME*)svd->data)->primme;
  SVDPRIMMEMethod    methodn;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  block size=%D\n",primme->maxBlockSize);CHKERRQ(ierr);
    ierr = SVDPRIMMEGetMethod(svd,&methodn);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  solver method: %s\n",SVDPRIMMEMethods[methodn]);CHKERRQ(ierr);

    /* Display PRIMME params */
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)svd),&rank);CHKERRQ(ierr);
    if (!rank) primme_svds_display_params(*primme);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetFromOptions_PRIMME(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  PetscErrorCode  ierr;
  SVD_PRIMME      *ctx = (SVD_PRIMME*)svd->data;
  PetscInt        bs;
  SVDPRIMMEMethod meth;
  PetscBool       flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SVD PRIMME Options");CHKERRQ(ierr);

    ierr = PetscOptionsInt("-svd_primme_blocksize","Maximum block size","SVDPRIMMESetBlockSize",ctx->primme.maxBlockSize,&bs,&flg);CHKERRQ(ierr);
    if (flg) { ierr = SVDPRIMMESetBlockSize(svd,bs);CHKERRQ(ierr); }

    ierr = PetscOptionsEnum("-svd_primme_method","Method for solving the singular value problem","SVDPRIMMESetMethod",SVDPRIMMEMethods,(PetscEnum)ctx->method,(PetscEnum*)&meth,&flg);CHKERRQ(ierr);
    if (flg) { ierr = SVDPRIMMESetMethod(svd,meth);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDPRIMMESetBlockSize_PRIMME(SVD svd,PetscInt bs)
{
  SVD_PRIMME *ops = (SVD_PRIMME*)svd->data;

  PetscFunctionBegin;
  if (bs == PETSC_DEFAULT) ops->primme.maxBlockSize = 1;
  else if (bs <= 0) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"PRIMME: block size must be positive");
  else ops->primme.maxBlockSize = bs;
  PetscFunctionReturn(0);
}

/*@
   SVDPRIMMESetBlockSize - The maximum block size that PRIMME will try to use.

   Logically Collective on SVD

   Input Parameters:
+  svd - the singular value solver context
-  bs - block size

   Options Database Key:
.  -svd_primme_blocksize - Sets the max allowed block size value

   Notes:
   If the block size is not set, the value established by primme_svds_initialize
   is used.

   The user should set the block size based on the architecture specifics
   of the target computer, as well as any a priori knowledge of multiplicities.
   The code does NOT require bs > 1 to find multiple eigenvalues. For some
   methods, keeping bs = 1 yields the best overall performance.

   Level: advanced

.seealso: SVDPRIMMEGetBlockSize()
@*/
PetscErrorCode SVDPRIMMESetBlockSize(SVD svd,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveInt(svd,bs,2);
  ierr = PetscTryMethod(svd,"SVDPRIMMESetBlockSize_C",(SVD,PetscInt),(svd,bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDPRIMMEGetBlockSize_PRIMME(SVD svd,PetscInt *bs)
{
  SVD_PRIMME *ops = (SVD_PRIMME*)svd->data;

  PetscFunctionBegin;
  *bs = ops->primme.maxBlockSize;
  PetscFunctionReturn(0);
}

/*@
   SVDPRIMMEGetBlockSize - Get the maximum block size the code will try to use.

   Not Collective

   Input Parameter:
.  svd - the singular value solver context

   Output Parameter:
.  bs - returned block size

   Level: advanced

.seealso: SVDPRIMMESetBlockSize()
@*/
PetscErrorCode SVDPRIMMEGetBlockSize(SVD svd,PetscInt *bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(bs,2);
  ierr = PetscUseMethod(svd,"SVDPRIMMEGetBlockSize_C",(SVD,PetscInt*),(svd,bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDPRIMMESetMethod_PRIMME(SVD svd,SVDPRIMMEMethod method)
{
  SVD_PRIMME *ops = (SVD_PRIMME*)svd->data;

  PetscFunctionBegin;
  ops->method = (primme_svds_preset_method)method;
  PetscFunctionReturn(0);
}

/*@
   SVDPRIMMESetMethod - Sets the method for the PRIMME SVD solver.

   Logically Collective on SVD

   Input Parameters:
+  svd - the singular value solver context
-  method - method that will be used by PRIMME

   Options Database Key:
.  -svd_primme_method - Sets the method for the PRIMME SVD solver

   Note:
   If not set, the method defaults to SVD_PRIMME_HYBRID.

   Level: advanced

.seealso: SVDPRIMMEGetMethod(), SVDPRIMMEMethod
@*/
PetscErrorCode SVDPRIMMESetMethod(SVD svd,SVDPRIMMEMethod method)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveEnum(svd,method,2);
  ierr = PetscTryMethod(svd,"SVDPRIMMESetMethod_C",(SVD,SVDPRIMMEMethod),(svd,method));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDPRIMMEGetMethod_PRIMME(SVD svd,SVDPRIMMEMethod *method)
{
  SVD_PRIMME *ops = (SVD_PRIMME*)svd->data;

  PetscFunctionBegin;
  *method = (SVDPRIMMEMethod)ops->method;
  PetscFunctionReturn(0);
}

/*@
   SVDPRIMMEGetMethod - Gets the method for the PRIMME SVD solver.

   Not Collective

   Input Parameter:
.  svd - the singular value solver context

   Output Parameter:
.  method - method that will be used by PRIMME

   Level: advanced

.seealso: SVDPRIMMESetMethod(), SVDPRIMMEMethod
@*/
PetscErrorCode SVDPRIMMEGetMethod(SVD svd,SVDPRIMMEMethod *method)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(method,2);
  ierr = PetscUseMethod(svd,"SVDPRIMMEGetMethod_C",(SVD,SVDPRIMMEMethod*),(svd,method));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SVDCreate_PRIMME(SVD svd)
{
  PetscErrorCode ierr;
  SVD_PRIMME     *primme;

  PetscFunctionBegin;
  ierr = PetscNewLog(svd,&primme);CHKERRQ(ierr);
  svd->data = (void*)primme;

  primme_svds_initialize(&primme->primme);
  primme->primme.matrixMatvec = multMatvec_PRIMME;
  primme->primme.globalSumReal = par_GlobalSumReal;
  primme->method = (primme_svds_preset_method)SVD_PRIMME_HYBRID;
  primme->svd = svd;

  svd->ops->solve          = SVDSolve_PRIMME;
  svd->ops->setup          = SVDSetUp_PRIMME;
  svd->ops->setfromoptions = SVDSetFromOptions_PRIMME;
  svd->ops->destroy        = SVDDestroy_PRIMME;
  svd->ops->reset          = SVDReset_PRIMME;
  svd->ops->view           = SVDView_PRIMME;

  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMESetBlockSize_C",SVDPRIMMESetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMEGetBlockSize_C",SVDPRIMMEGetBlockSize_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMESetMethod_C",SVDPRIMMESetMethod_PRIMME);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDPRIMMEGetMethod_C",SVDPRIMMEGetMethod_PRIMME);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

