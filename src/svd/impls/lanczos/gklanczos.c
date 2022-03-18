/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc singular value solver: "lanczos"

   Method: Explicitly restarted Lanczos

   Algorithm:

       Golub-Kahan-Lanczos bidiagonalization with explicit restart.

   References:

       [1] G.H. Golub and W. Kahan, "Calculating the singular values
           and pseudo-inverse of a matrix", SIAM J. Numer. Anal. Ser.
           B 2:205-224, 1965.

       [2] V. Hernandez, J.E. Roman, and A. Tomas, "A robust and
           efficient parallel SVD solver based on restarted Lanczos
           bidiagonalization", Elec. Trans. Numer. Anal. 31:68-85,
           2008.
*/

#include <slepc/private/svdimpl.h>                /*I "slepcsvd.h" I*/

typedef struct {
  PetscBool oneside;
} SVD_LANCZOS;

PetscErrorCode SVDSetUp_Lanczos(SVD svd)
{
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;
  PetscInt       N;

  PetscFunctionBegin;
  SVDCheckStandard(svd);
  CHKERRQ(MatGetSize(svd->A,NULL,&N));
  CHKERRQ(SVDSetDimensions_Default(svd));
  PetscCheck(svd->ncv<=svd->nsv+svd->mpd,PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The value of ncv must not be larger than nev+mpd");
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = PetscMax(N/svd->ncv,100);
  svd->leftbasis = PetscNot(lanczos->oneside);
  CHKERRQ(SVDAllocateSolution(svd,1));
  CHKERRQ(DSSetType(svd->ds,DSSVD));
  CHKERRQ(DSSetCompact(svd->ds,PETSC_TRUE));
  CHKERRQ(DSSetExtraRow(svd->ds,PETSC_TRUE));
  CHKERRQ(DSAllocate(svd->ds,svd->ncv+1));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDTwoSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,BV V,BV U,PetscInt k,PetscInt *n,PetscBool *breakdown)
{
  PetscInt       i;
  Vec            u,v;
  PetscBool      lindep=PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(BVGetColumn(svd->V,k,&v));
  CHKERRQ(BVGetColumn(svd->U,k,&u));
  CHKERRQ(MatMult(svd->A,v,u));
  CHKERRQ(BVRestoreColumn(svd->V,k,&v));
  CHKERRQ(BVRestoreColumn(svd->U,k,&u));
  CHKERRQ(BVOrthonormalizeColumn(svd->U,k,PETSC_FALSE,alpha+k,&lindep));
  if (PetscUnlikely(lindep)) {
    *n = k;
    if (breakdown) *breakdown = lindep;
    PetscFunctionReturn(0);
  }

  for (i=k+1;i<*n;i++) {
    CHKERRQ(BVGetColumn(svd->V,i,&v));
    CHKERRQ(BVGetColumn(svd->U,i-1,&u));
    CHKERRQ(MatMult(svd->AT,u,v));
    CHKERRQ(BVRestoreColumn(svd->V,i,&v));
    CHKERRQ(BVRestoreColumn(svd->U,i-1,&u));
    CHKERRQ(BVOrthonormalizeColumn(svd->V,i,PETSC_FALSE,beta+i-1,&lindep));
    if (PetscUnlikely(lindep)) {
      *n = i;
      break;
    }
    CHKERRQ(BVGetColumn(svd->V,i,&v));
    CHKERRQ(BVGetColumn(svd->U,i,&u));
    CHKERRQ(MatMult(svd->A,v,u));
    CHKERRQ(BVRestoreColumn(svd->V,i,&v));
    CHKERRQ(BVRestoreColumn(svd->U,i,&u));
    CHKERRQ(BVOrthonormalizeColumn(svd->U,i,PETSC_FALSE,alpha+i,&lindep));
    if (PetscUnlikely(lindep)) {
      *n = i;
      break;
    }
  }

  if (!lindep) {
    CHKERRQ(BVGetColumn(svd->V,*n,&v));
    CHKERRQ(BVGetColumn(svd->U,*n-1,&u));
    CHKERRQ(MatMult(svd->AT,u,v));
    CHKERRQ(BVRestoreColumn(svd->V,*n,&v));
    CHKERRQ(BVRestoreColumn(svd->U,*n-1,&u));
    CHKERRQ(BVOrthogonalizeColumn(svd->V,*n,NULL,beta+*n-1,&lindep));
  }
  if (PetscUnlikely(breakdown)) *breakdown = lindep;
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDOneSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,BV V,Vec u,Vec u_1,PetscInt k,PetscInt n,PetscScalar* work)
{
  PetscInt       i,bvl,bvk;
  PetscReal      a,b;
  Vec            z,temp;

  PetscFunctionBegin;
  CHKERRQ(BVGetActiveColumns(V,&bvl,&bvk));
  CHKERRQ(BVGetColumn(V,k,&z));
  CHKERRQ(MatMult(svd->A,z,u));
  CHKERRQ(BVRestoreColumn(V,k,&z));

  for (i=k+1;i<n;i++) {
    CHKERRQ(BVGetColumn(V,i,&z));
    CHKERRQ(MatMult(svd->AT,u,z));
    CHKERRQ(BVRestoreColumn(V,i,&z));
    CHKERRQ(VecNormBegin(u,NORM_2,&a));
    CHKERRQ(BVSetActiveColumns(V,0,i));
    CHKERRQ(BVDotColumnBegin(V,i,work));
    CHKERRQ(VecNormEnd(u,NORM_2,&a));
    CHKERRQ(BVDotColumnEnd(V,i,work));
    CHKERRQ(VecScale(u,1.0/a));
    CHKERRQ(BVMultColumn(V,-1.0/a,1.0/a,i,work));

    /* h = V^* z, z = z - V h  */
    CHKERRQ(BVDotColumn(V,i,work));
    CHKERRQ(BVMultColumn(V,-1.0,1.0,i,work));
    CHKERRQ(BVNormColumn(V,i,NORM_2,&b));
    PetscCheck(PetscAbsReal(b)>10*PETSC_MACHINE_EPSILON,PetscObjectComm((PetscObject)svd),PETSC_ERR_PLIB,"Recurrence generated a zero vector; use a two-sided variant");
    CHKERRQ(BVScaleColumn(V,i,1.0/b));

    CHKERRQ(BVGetColumn(V,i,&z));
    CHKERRQ(MatMult(svd->A,z,u_1));
    CHKERRQ(BVRestoreColumn(V,i,&z));
    CHKERRQ(VecAXPY(u_1,-b,u));
    alpha[i-1] = a;
    beta[i-1] = b;
    temp = u;
    u = u_1;
    u_1 = temp;
  }

  CHKERRQ(BVGetColumn(V,n,&z));
  CHKERRQ(MatMult(svd->AT,u,z));
  CHKERRQ(BVRestoreColumn(V,n,&z));
  CHKERRQ(VecNormBegin(u,NORM_2,&a));
  CHKERRQ(BVDotColumnBegin(V,n,work));
  CHKERRQ(VecNormEnd(u,NORM_2,&a));
  CHKERRQ(BVDotColumnEnd(V,n,work));
  CHKERRQ(VecScale(u,1.0/a));
  CHKERRQ(BVMultColumn(V,-1.0/a,1.0/a,n,work));

  /* h = V^* z, z = z - V h  */
  CHKERRQ(BVDotColumn(V,n,work));
  CHKERRQ(BVMultColumn(V,-1.0,1.0,n,work));
  CHKERRQ(BVNormColumn(V,i,NORM_2,&b));

  alpha[n-1] = a;
  beta[n-1] = b;
  CHKERRQ(BVSetActiveColumns(V,bvl,bvk));
  PetscFunctionReturn(0);
}

/*
   SVDKrylovConvergence - Implements the loop that checks for convergence
   in Krylov methods.

   Input Parameters:
     svd     - the solver; some error estimates are updated in svd->errest
     getall  - whether all residuals must be computed
     kini    - initial value of k (the loop variable)
     nits    - number of iterations of the loop
     normr   - norm of triangular factor of qr([A;B]), used only in GSVD

   Output Parameter:
     kout  - the first index where the convergence test failed
*/
PetscErrorCode SVDKrylovConvergence(SVD svd,PetscBool getall,PetscInt kini,PetscInt nits,PetscReal normr,PetscInt *kout)
{
  PetscInt       k,marker,ld;
  PetscReal      *alpha,*beta,*betah,resnorm;
  PetscBool      extra;

  PetscFunctionBegin;
  if (PetscUnlikely(svd->conv == SVD_CONV_MAXIT && svd->its >= svd->max_it)) *kout = svd->nsv;
  else {
    CHKERRQ(DSGetLeadingDimension(svd->ds,&ld));
    CHKERRQ(DSGetExtraRow(svd->ds,&extra));
    PetscCheck(extra,PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Only implemented for DS with extra row");
    marker = -1;
    if (svd->trackall) getall = PETSC_TRUE;
    CHKERRQ(DSGetArrayReal(svd->ds,DS_MAT_T,&alpha));
    beta = alpha + ld;
    betah = alpha + 2*ld;
    for (k=kini;k<kini+nits;k++) {
      if (svd->isgeneralized) resnorm = SlepcAbs(beta[k],betah[k])*normr;
      else resnorm = PetscAbsReal(beta[k]);
      CHKERRQ((*svd->converged)(svd,svd->sigma[k],resnorm,&svd->errest[k],svd->convergedctx));
      if (marker==-1 && svd->errest[k] >= svd->tol) marker = k;
      if (marker!=-1 && !getall) break;
    }
    CHKERRQ(DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha));
    if (marker!=-1) k = marker;
    *kout = k;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_Lanczos(SVD svd)
{
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;
  PetscReal      *alpha,*beta;
  PetscScalar    *swork,*w,*P;
  PetscInt       i,k,j,nv,ld;
  Vec            u=0,u_1=0;
  Mat            U,V;

  PetscFunctionBegin;
  /* allocate working space */
  CHKERRQ(DSGetLeadingDimension(svd->ds,&ld));
  CHKERRQ(PetscMalloc2(ld,&w,svd->ncv,&swork));

  if (lanczos->oneside) {
    CHKERRQ(MatCreateVecs(svd->A,NULL,&u));
    CHKERRQ(MatCreateVecs(svd->A,NULL,&u_1));
  }

  /* normalize start vector */
  if (!svd->nini) {
    CHKERRQ(BVSetRandomColumn(svd->V,0));
    CHKERRQ(BVOrthonormalizeColumn(svd->V,0,PETSC_TRUE,NULL,NULL));
  }

  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->ncv);
    CHKERRQ(BVSetActiveColumns(svd->V,svd->nconv,nv));
    CHKERRQ(DSGetArrayReal(svd->ds,DS_MAT_T,&alpha));
    beta = alpha + ld;
    if (lanczos->oneside) {
      CHKERRQ(SVDOneSideLanczos(svd,alpha,beta,svd->V,u,u_1,svd->nconv,nv,swork));
    } else {
      CHKERRQ(BVSetActiveColumns(svd->U,svd->nconv,nv));
      CHKERRQ(SVDTwoSideLanczos(svd,alpha,beta,svd->V,svd->U,svd->nconv,&nv,NULL));
    }
    CHKERRQ(DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha));

    /* compute SVD of bidiagonal matrix */
    CHKERRQ(DSSetDimensions(svd->ds,nv,svd->nconv,0));
    CHKERRQ(DSSVDSetDimensions(svd->ds,nv));
    CHKERRQ(DSSetState(svd->ds,DS_STATE_INTERMEDIATE));
    CHKERRQ(DSSolve(svd->ds,w,NULL));
    CHKERRQ(DSSort(svd->ds,w,NULL,NULL,NULL,NULL));
    CHKERRQ(DSUpdateExtraRow(svd->ds));
    CHKERRQ(DSSynchronize(svd->ds,w,NULL));
    for (i=svd->nconv;i<nv;i++) svd->sigma[i] = PetscRealPart(w[i]);

    /* check convergence */
    CHKERRQ(SVDKrylovConvergence(svd,PETSC_FALSE,svd->nconv,nv-svd->nconv,1.0,&k));
    CHKERRQ((*svd->stopping)(svd,svd->its,svd->max_it,k,svd->nsv,&svd->reason,svd->stoppingctx));

    /* compute restart vector */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (k<nv) {
        CHKERRQ(DSGetArray(svd->ds,DS_MAT_V,&P));
        for (j=svd->nconv;j<nv;j++) swork[j-svd->nconv] = PetscConj(P[j+k*ld]);
        CHKERRQ(DSRestoreArray(svd->ds,DS_MAT_V,&P));
        CHKERRQ(BVMultColumn(svd->V,1.0,0.0,nv,swork));
      } else {
        /* all approximations have converged, generate a new initial vector */
        CHKERRQ(BVSetRandomColumn(svd->V,nv));
        CHKERRQ(BVOrthonormalizeColumn(svd->V,nv,PETSC_FALSE,NULL,NULL));
      }
    }

    /* compute converged singular vectors */
    CHKERRQ(DSGetMat(svd->ds,DS_MAT_V,&V));
    CHKERRQ(BVMultInPlace(svd->V,V,svd->nconv,k));
    CHKERRQ(MatDestroy(&V));
    if (!lanczos->oneside) {
      CHKERRQ(DSGetMat(svd->ds,DS_MAT_U,&U));
      CHKERRQ(BVMultInPlace(svd->U,U,svd->nconv,k));
      CHKERRQ(MatDestroy(&U));
    }

    /* copy restart vector from the last column */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      CHKERRQ(BVCopyColumn(svd->V,nv,k));
    }

    svd->nconv = k;
    CHKERRQ(SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv));
  }

  /* free working space */
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&u_1));
  CHKERRQ(PetscFree2(w,swork));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetFromOptions_Lanczos(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  PetscBool      set,val;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"SVD Lanczos Options"));

    CHKERRQ(PetscOptionsBool("-svd_lanczos_oneside","Use one-side reorthogonalization","SVDLanczosSetOneSide",lanczos->oneside,&val,&set));
    if (set) CHKERRQ(SVDLanczosSetOneSide(svd,val));

  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDLanczosSetOneSide_Lanczos(SVD svd,PetscBool oneside)
{
  SVD_LANCZOS *lanczos = (SVD_LANCZOS*)svd->data;

  PetscFunctionBegin;
  if (lanczos->oneside != oneside) {
    lanczos->oneside = oneside;
    svd->state = SVD_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   SVDLanczosSetOneSide - Indicate if the variant of the Lanczos method
   to be used is one-sided or two-sided.

   Logically Collective on svd

   Input Parameters:
+  svd     - singular value solver
-  oneside - boolean flag indicating if the method is one-sided or not

   Options Database Key:
.  -svd_lanczos_oneside <boolean> - Indicates the boolean flag

   Note:
   By default, a two-sided variant is selected, which is sometimes slightly
   more robust. However, the one-sided variant is faster because it avoids
   the orthogonalization associated to left singular vectors. It also saves
   the memory required for storing such vectors.

   Level: advanced

.seealso: SVDTRLanczosSetOneSide()
@*/
PetscErrorCode SVDLanczosSetOneSide(SVD svd,PetscBool oneside)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,oneside,2);
  CHKERRQ(PetscTryMethod(svd,"SVDLanczosSetOneSide_C",(SVD,PetscBool),(svd,oneside)));
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDLanczosGetOneSide_Lanczos(SVD svd,PetscBool *oneside)
{
  SVD_LANCZOS *lanczos = (SVD_LANCZOS*)svd->data;

  PetscFunctionBegin;
  *oneside = lanczos->oneside;
  PetscFunctionReturn(0);
}

/*@
   SVDLanczosGetOneSide - Gets if the variant of the Lanczos method
   to be used is one-sided or two-sided.

   Not Collective

   Input Parameters:
.  svd     - singular value solver

   Output Parameters:
.  oneside - boolean flag indicating if the method is one-sided or not

   Level: advanced

.seealso: SVDLanczosSetOneSide()
@*/
PetscErrorCode SVDLanczosGetOneSide(SVD svd,PetscBool *oneside)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(oneside,2);
  CHKERRQ(PetscUseMethod(svd,"SVDLanczosGetOneSide_C",(SVD,PetscBool*),(svd,oneside)));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDDestroy_Lanczos(SVD svd)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(svd->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosSetOneSide_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosGetOneSide_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode SVDView_Lanczos(SVD svd,PetscViewer viewer)
{
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  %s-sided reorthogonalization\n",lanczos->oneside? "one": "two"));
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_Lanczos(SVD svd)
{
  SVD_LANCZOS    *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(svd,&ctx));
  svd->data = (void*)ctx;

  svd->ops->setup          = SVDSetUp_Lanczos;
  svd->ops->solve          = SVDSolve_Lanczos;
  svd->ops->destroy        = SVDDestroy_Lanczos;
  svd->ops->setfromoptions = SVDSetFromOptions_Lanczos;
  svd->ops->view           = SVDView_Lanczos;
  svd->ops->computevectors = SVDComputeVectors_Left;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosSetOneSide_C",SVDLanczosSetOneSide_Lanczos));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosGetOneSide_C",SVDLanczosGetOneSide_Lanczos));
  PetscFunctionReturn(0);
}
