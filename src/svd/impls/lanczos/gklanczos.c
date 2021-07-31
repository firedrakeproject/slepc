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
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;
  PetscInt       N;

  PetscFunctionBegin;
  SVDCheckStandard(svd);
  ierr = MatGetSize(svd->A,NULL,&N);CHKERRQ(ierr);
  ierr = SVDSetDimensions_Default(svd);CHKERRQ(ierr);
  if (svd->ncv>svd->nsv+svd->mpd) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_USER_INPUT,"The value of ncv must not be larger than nev+mpd");
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = PetscMax(N/svd->ncv,100);
  svd->leftbasis = PetscNot(lanczos->oneside);
  ierr = SVDAllocateSolution(svd,1);CHKERRQ(ierr);
  ierr = DSSetType(svd->ds,DSSVD);CHKERRQ(ierr);
  ierr = DSSetCompact(svd->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSSetExtraRow(svd->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(svd->ds,svd->ncv+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDTwoSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,BV V,BV U,PetscInt k,PetscInt *n,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       i;
  Vec            u,v;
  PetscBool      lindep=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = BVGetColumn(svd->V,k,&v);CHKERRQ(ierr);
  ierr = BVGetColumn(svd->U,k,&u);CHKERRQ(ierr);
  ierr = MatMult(svd->A,v,u);CHKERRQ(ierr);
  ierr = BVRestoreColumn(svd->V,k,&v);CHKERRQ(ierr);
  ierr = BVRestoreColumn(svd->U,k,&u);CHKERRQ(ierr);
  ierr = BVOrthonormalizeColumn(svd->U,k,PETSC_FALSE,alpha+k,&lindep);CHKERRQ(ierr);
  if (lindep) {
    *n = k;
    if (breakdown) *breakdown = lindep;
    PetscFunctionReturn(0);
  }

  for (i=k+1;i<*n;i++) {
    ierr = BVGetColumn(svd->V,i,&v);CHKERRQ(ierr);
    ierr = BVGetColumn(svd->U,i-1,&u);CHKERRQ(ierr);
    ierr = MatMult(svd->AT,u,v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->V,i,&v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,i-1,&u);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(svd->V,i,PETSC_FALSE,beta+i-1,&lindep);CHKERRQ(ierr);
    if (lindep) {
      *n = i;
      break;
    }
    ierr = BVGetColumn(svd->V,i,&v);CHKERRQ(ierr);
    ierr = BVGetColumn(svd->U,i,&u);CHKERRQ(ierr);
    ierr = MatMult(svd->A,v,u);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->V,i,&v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,i,&u);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(svd->U,i,PETSC_FALSE,alpha+i,&lindep);CHKERRQ(ierr);
    if (lindep) {
      *n = i;
      break;
    }
  }

  if (!lindep) {
    ierr = BVGetColumn(svd->V,*n,&v);CHKERRQ(ierr);
    ierr = BVGetColumn(svd->U,*n-1,&u);CHKERRQ(ierr);
    ierr = MatMult(svd->AT,u,v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->V,*n,&v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,*n-1,&u);CHKERRQ(ierr);
    ierr = BVOrthogonalizeColumn(svd->V,*n,NULL,beta+*n-1,&lindep);CHKERRQ(ierr);
  }
  if (breakdown) *breakdown = lindep;
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDOneSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,BV V,Vec u,Vec u_1,PetscInt k,PetscInt n,PetscScalar* work)
{
  PetscErrorCode ierr;
  PetscInt       i,bvl,bvk;
  PetscReal      a,b;
  Vec            z,temp;

  PetscFunctionBegin;
  ierr = BVGetActiveColumns(V,&bvl,&bvk);CHKERRQ(ierr);
  ierr = BVGetColumn(V,k,&z);CHKERRQ(ierr);
  ierr = MatMult(svd->A,z,u);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,k,&z);CHKERRQ(ierr);

  for (i=k+1;i<n;i++) {
    ierr = BVGetColumn(V,i,&z);CHKERRQ(ierr);
    ierr = MatMult(svd->AT,u,z);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i,&z);CHKERRQ(ierr);
    ierr = VecNormBegin(u,NORM_2,&a);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(V,0,i);CHKERRQ(ierr);
    ierr = BVDotColumnBegin(V,i,work);CHKERRQ(ierr);
    ierr = VecNormEnd(u,NORM_2,&a);CHKERRQ(ierr);
    ierr = BVDotColumnEnd(V,i,work);CHKERRQ(ierr);
    ierr = VecScale(u,1.0/a);CHKERRQ(ierr);
    ierr = BVMultColumn(V,-1.0/a,1.0/a,i,work);CHKERRQ(ierr);

    /* h = V^* z, z = z - V h  */
    ierr = BVDotColumn(V,i,work);CHKERRQ(ierr);
    ierr = BVMultColumn(V,-1.0,1.0,i,work);CHKERRQ(ierr);
    ierr = BVNormColumn(V,i,NORM_2,&b);CHKERRQ(ierr);
    if (PetscAbsReal(b)<10*PETSC_MACHINE_EPSILON) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_PLIB,"Recurrence generated a zero vector; use a two-sided variant");
    ierr = BVScaleColumn(V,i,1.0/b);CHKERRQ(ierr);

    ierr = BVGetColumn(V,i,&z);CHKERRQ(ierr);
    ierr = MatMult(svd->A,z,u_1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(V,i,&z);CHKERRQ(ierr);
    ierr = VecAXPY(u_1,-b,u);CHKERRQ(ierr);
    alpha[i-1] = a;
    beta[i-1] = b;
    temp = u;
    u = u_1;
    u_1 = temp;
  }

  ierr = BVGetColumn(V,n,&z);CHKERRQ(ierr);
  ierr = MatMult(svd->AT,u,z);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,n,&z);CHKERRQ(ierr);
  ierr = VecNormBegin(u,NORM_2,&a);CHKERRQ(ierr);
  ierr = BVDotColumnBegin(V,n,work);CHKERRQ(ierr);
  ierr = VecNormEnd(u,NORM_2,&a);CHKERRQ(ierr);
  ierr = BVDotColumnEnd(V,n,work);CHKERRQ(ierr);
  ierr = VecScale(u,1.0/a);CHKERRQ(ierr);
  ierr = BVMultColumn(V,-1.0/a,1.0/a,n,work);CHKERRQ(ierr);

  /* h = V^* z, z = z - V h  */
  ierr = BVDotColumn(V,n,work);CHKERRQ(ierr);
  ierr = BVMultColumn(V,-1.0,1.0,n,work);CHKERRQ(ierr);
  ierr = BVNormColumn(V,i,NORM_2,&b);CHKERRQ(ierr);

  alpha[n-1] = a;
  beta[n-1] = b;
  ierr = BVSetActiveColumns(V,bvl,bvk);CHKERRQ(ierr);
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

   Output Parameter:
     kout  - the first index where the convergence test failed
*/
PetscErrorCode SVDKrylovConvergence(SVD svd,PetscBool getall,PetscInt kini,PetscInt nits,PetscInt *kout)
{
  PetscErrorCode ierr;
  PetscInt       k,marker,ld;
  PetscReal      *alpha,*beta,*betah,resnorm;
  PetscBool      extra;

  PetscFunctionBegin;
  if (svd->conv == SVD_CONV_MAXIT && svd->its >= svd->max_it) *kout = svd->nsv;
  else {
    ierr = DSGetLeadingDimension(svd->ds,&ld);CHKERRQ(ierr);
    ierr = DSGetExtraRow(svd->ds,&extra);CHKERRQ(ierr);
    if (!extra) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"Only implemented for DS with extra row");
    marker = -1;
    if (svd->trackall) getall = PETSC_TRUE;
    ierr = DSGetArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    beta = alpha + ld;
    betah = alpha + 2*ld;
    for (k=kini;k<kini+nits;k++) {
      if (svd->isgeneralized) resnorm = SlepcAbs(beta[k],betah[k]);
      else resnorm = PetscAbsReal(beta[k]);
      ierr = (*svd->converged)(svd,svd->sigma[k],resnorm,&svd->errest[k],svd->convergedctx);CHKERRQ(ierr);
      if (marker==-1 && svd->errest[k] >= svd->tol) marker = k;
      if (marker!=-1 && !getall) break;
    }
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    if (marker!=-1) k = marker;
    *kout = k;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_Lanczos(SVD svd)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;
  PetscReal      *alpha,*beta;
  PetscScalar    *swork,*w,*P;
  PetscInt       i,k,j,nv,ld;
  Vec            u=0,u_1=0;
  Mat            U,V;

  PetscFunctionBegin;
  /* allocate working space */
  ierr = DSGetLeadingDimension(svd->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc2(ld,&w,svd->ncv,&swork);CHKERRQ(ierr);

  if (lanczos->oneside) {
    ierr = MatCreateVecs(svd->A,NULL,&u);CHKERRQ(ierr);
    ierr = MatCreateVecs(svd->A,NULL,&u_1);CHKERRQ(ierr);
  }

  /* normalize start vector */
  if (!svd->nini) {
    ierr = BVSetRandomColumn(svd->V,0);CHKERRQ(ierr);
    ierr = BVOrthonormalizeColumn(svd->V,0,PETSC_TRUE,NULL,NULL);CHKERRQ(ierr);
  }

  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->ncv);
    ierr = BVSetActiveColumns(svd->V,svd->nconv,nv);CHKERRQ(ierr);
    ierr = DSGetArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    beta = alpha + ld;
    if (lanczos->oneside) {
      ierr = SVDOneSideLanczos(svd,alpha,beta,svd->V,u,u_1,svd->nconv,nv,swork);CHKERRQ(ierr);
    } else {
      ierr = BVSetActiveColumns(svd->U,svd->nconv,nv);CHKERRQ(ierr);
      ierr = SVDTwoSideLanczos(svd,alpha,beta,svd->V,svd->U,svd->nconv,&nv,NULL);CHKERRQ(ierr);
    }
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);

    /* compute SVD of bidiagonal matrix */
    ierr = DSSetDimensions(svd->ds,nv,svd->nconv,0);CHKERRQ(ierr);
    ierr = DSSVDSetDimensions(svd->ds,nv);CHKERRQ(ierr);
    ierr = DSSetState(svd->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
    ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSUpdateExtraRow(svd->ds);CHKERRQ(ierr);
    ierr = DSSynchronize(svd->ds,w,NULL);CHKERRQ(ierr);
    for (i=svd->nconv;i<nv;i++) svd->sigma[i] = PetscRealPart(w[i]);

    /* check convergence */
    ierr = SVDKrylovConvergence(svd,PETSC_FALSE,svd->nconv,nv-svd->nconv,&k);CHKERRQ(ierr);
    ierr = (*svd->stopping)(svd,svd->its,svd->max_it,k,svd->nsv,&svd->reason,svd->stoppingctx);CHKERRQ(ierr);

    /* compute restart vector */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (k<nv) {
        ierr = DSGetArray(svd->ds,DS_MAT_V,&P);CHKERRQ(ierr);
        for (j=svd->nconv;j<nv;j++) swork[j-svd->nconv] = PetscConj(P[j+k*ld]);
        ierr = DSRestoreArray(svd->ds,DS_MAT_V,&P);CHKERRQ(ierr);
        ierr = BVMultColumn(svd->V,1.0,0.0,nv,swork);CHKERRQ(ierr);
      } else {
        /* all approximations have converged, generate a new initial vector */
        ierr = BVSetRandomColumn(svd->V,nv);CHKERRQ(ierr);
        ierr = BVOrthonormalizeColumn(svd->V,nv,PETSC_FALSE,NULL,NULL);CHKERRQ(ierr);
      }
    }

    /* compute converged singular vectors */
    ierr = DSGetMat(svd->ds,DS_MAT_V,&V);CHKERRQ(ierr);
    ierr = BVMultInPlace(svd->V,V,svd->nconv,k);CHKERRQ(ierr);
    ierr = MatDestroy(&V);CHKERRQ(ierr);
    if (!lanczos->oneside) {
      ierr = DSGetMat(svd->ds,DS_MAT_U,&U);CHKERRQ(ierr);
      ierr = BVMultInPlace(svd->U,U,svd->nconv,k);CHKERRQ(ierr);
      ierr = MatDestroy(&U);CHKERRQ(ierr);
    }

    /* copy restart vector from the last column */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      ierr = BVCopyColumn(svd->V,nv,k);CHKERRQ(ierr);
    }

    svd->nconv = k;
    ierr = SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv);CHKERRQ(ierr);
  }

  /* free working space */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&u_1);CHKERRQ(ierr);
  ierr = PetscFree2(w,swork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSetFromOptions_Lanczos(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SVD Lanczos Options");CHKERRQ(ierr);

    ierr = PetscOptionsBool("-svd_lanczos_oneside","Use one-side reorthogonalization","SVDLanczosSetOneSide",lanczos->oneside,&val,&set);CHKERRQ(ierr);
    if (set) { ierr = SVDLanczosSetOneSide(svd,val);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,oneside,2);
  ierr = PetscTryMethod(svd,"SVDLanczosSetOneSide_C",(SVD,PetscBool),(svd,oneside));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidBoolPointer(oneside,2);
  ierr = PetscUseMethod(svd,"SVDLanczosGetOneSide_C",(SVD,PetscBool*),(svd,oneside));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDDestroy_Lanczos(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosSetOneSide_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosGetOneSide_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDView_Lanczos(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  %s-sided reorthogonalization\n",lanczos->oneside? "one": "two");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_Lanczos(SVD svd)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(svd,&ctx);CHKERRQ(ierr);
  svd->data = (void*)ctx;

  svd->ops->setup          = SVDSetUp_Lanczos;
  svd->ops->solve          = SVDSolve_Lanczos;
  svd->ops->destroy        = SVDDestroy_Lanczos;
  svd->ops->setfromoptions = SVDSetFromOptions_Lanczos;
  svd->ops->view           = SVDView_Lanczos;
  svd->ops->computevectors = SVDComputeVectors_Left;
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosSetOneSide_C",SVDLanczosSetOneSide_Lanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosGetOneSide_C",SVDLanczosGetOneSide_Lanczos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

