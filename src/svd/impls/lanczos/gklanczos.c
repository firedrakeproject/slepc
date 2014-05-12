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

   Last update: Jun 2007

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/svdimpl.h>                /*I "slepcsvd.h" I*/
#include <slepc-private/ipimpl.h>
#include <slepcblaslapack.h>

typedef struct {
  PetscBool oneside;
} SVD_LANCZOS;

#undef __FUNCT__
#define __FUNCT__ "SVDSetUp_Lanczos"
PetscErrorCode SVDSetUp_Lanczos(SVD svd)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;
  PetscInt       N;

  PetscFunctionBegin;
  ierr = SVDMatGetSize(svd,NULL,&N);CHKERRQ(ierr);
  if (svd->ncv) { /* ncv set */
    if (svd->ncv<svd->nsv) SETERRQ(PetscObjectComm((PetscObject)svd),1,"The value of ncv must be at least nsv");
  } else if (svd->mpd) { /* mpd set */
    svd->ncv = PetscMin(N,svd->nsv+svd->mpd);
  } else { /* neither set: defaults depend on nsv being small or large */
    if (svd->nsv<500) svd->ncv = PetscMin(N,PetscMax(2*svd->nsv,10));
    else {
      svd->mpd = 500;
      svd->ncv = PetscMin(N,svd->nsv+svd->mpd);
    }
  }
  if (!svd->mpd) svd->mpd = svd->ncv;
  if (svd->ncv>svd->nsv+svd->mpd) SETERRQ(PetscObjectComm((PetscObject)svd),1,"The value of ncv must not be larger than nev+mpd");
  if (!svd->max_it) svd->max_it = PetscMax(N/svd->ncv,100);
  svd->leftbasis = (lanczos->oneside)? PETSC_FALSE: PETSC_TRUE;
  ierr = DSSetType(svd->ds,DSSVD);CHKERRQ(ierr);
  ierr = DSSetCompact(svd->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(svd->ds,svd->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDTwoSideLanczos"
PetscErrorCode SVDTwoSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,BV V,BV U,PetscInt k,PetscInt n,PetscScalar* work)
{
  PetscErrorCode ierr;
  PetscInt       i;
  Vec            u,v;

  PetscFunctionBegin;
  ierr = BVGetColumn(svd->V,k,&v);CHKERRQ(ierr);
  ierr = BVGetColumn(svd->U,k,&u);CHKERRQ(ierr);
  ierr = SVDMatMult(svd,PETSC_FALSE,v,u);CHKERRQ(ierr);
  ierr = BVRestoreColumn(svd->V,k,&v);CHKERRQ(ierr);
  ierr = BVRestoreColumn(svd->U,k,&u);CHKERRQ(ierr);
  ierr = BVOrthogonalize(svd->U,k,work,alpha+k,NULL);CHKERRQ(ierr);
  ierr = BVScale(U,k,1.0/alpha[k]);CHKERRQ(ierr);
  for (i=k+1;i<n;i++) {
    ierr = BVGetColumn(svd->V,i,&v);CHKERRQ(ierr);
    ierr = BVGetColumn(svd->U,i-1,&u);CHKERRQ(ierr);
    ierr = SVDMatMult(svd,PETSC_TRUE,u,v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,i-1,&u);CHKERRQ(ierr);
    ierr = BVOrthogonalize(svd->V,i,work,beta+i-1,NULL);CHKERRQ(ierr);
    ierr = BVScale(V,i,1.0/beta[i-1]);CHKERRQ(ierr);

    ierr = BVGetColumn(svd->U,i,&u);CHKERRQ(ierr);
    ierr = SVDMatMult(svd,PETSC_FALSE,v,u);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,i,&u);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->V,i,&v);CHKERRQ(ierr);
    ierr = BVOrthogonalize(svd->U,i,work,alpha+i,NULL);CHKERRQ(ierr);
    ierr = BVScale(U,i,1.0/alpha[i]);CHKERRQ(ierr);
  }
  ierr = BVGetColumn(svd->U,n,&v);CHKERRQ(ierr);
  ierr = BVGetColumn(svd->U,n-1,&u);CHKERRQ(ierr);
  ierr = SVDMatMult(svd,PETSC_TRUE,u,v);CHKERRQ(ierr);
  ierr = BVOrthogonalize(svd->V,n,work,beta+n-1,NULL);CHKERRQ(ierr);
  ierr = BVRestoreColumn(svd->U,n,&v);CHKERRQ(ierr);
  ierr = BVRestoreColumn(svd->U,n-1,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*#undef __FUNCT__
#define __FUNCT__ "SVDOneSideLanczos"
static PetscErrorCode SVDOneSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,Vec *V,Vec v,Vec u,Vec u_1,PetscInt k,PetscInt n,PetscScalar* work)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      a,b;
  Vec            temp;

  PetscFunctionBegin;
  ierr = SVDMatMult(svd,PETSC_FALSE,V[k],u);CHKERRQ(ierr);
  for (i=k+1;i<n;i++) {
    ierr = SVDMatMult(svd,PETSC_TRUE,u,V[i]);CHKERRQ(ierr);
    ierr = IPNormBegin(svd->ip,u,&a);CHKERRQ(ierr);
    ierr = IPMInnerProductBegin(svd->ip,V[i],i,V,work);CHKERRQ(ierr);
    ierr = IPNormEnd(svd->ip,u,&a);CHKERRQ(ierr);
    ierr = IPMInnerProductEnd(svd->ip,V[i],i,V,work);CHKERRQ(ierr);

    ierr = VecScale(u,1.0/a);CHKERRQ(ierr);
    ierr = SlepcVecMAXPBY(V[i],1.0/a,-1.0/a,i,work,V);CHKERRQ(ierr);

    ierr = IPOrthogonalizeCGS1(svd->ip,0,NULL,i,NULL,V,V[i],work,NULL,&b);CHKERRQ(ierr);
    ierr = VecScale(V[i],1.0/b);CHKERRQ(ierr);

    ierr = SVDMatMult(svd,PETSC_FALSE,V[i],u_1);CHKERRQ(ierr);
    ierr = VecAXPY(u_1,-b,u);CHKERRQ(ierr);

    alpha[i-1] = a;
    beta[i-1] = b;
    temp = u;
    u = u_1;
    u_1 = temp;
  }
  ierr = SVDMatMult(svd,PETSC_TRUE,u,v);CHKERRQ(ierr);
  ierr = IPNormBegin(svd->ip,u,&a);CHKERRQ(ierr);
  ierr = IPMInnerProductBegin(svd->ip,v,n,V,work);CHKERRQ(ierr);
  ierr = IPNormEnd(svd->ip,u,&a);CHKERRQ(ierr);
  ierr = IPMInnerProductEnd(svd->ip,v,n,V,work);CHKERRQ(ierr);

  ierr = VecScale(u,1.0/a);CHKERRQ(ierr);
  ierr = SlepcVecMAXPBY(v,1.0/a,-1.0/a,n,work,V);CHKERRQ(ierr);

  ierr = IPOrthogonalizeCGS1(svd->ip,0,NULL,n,NULL,V,v,work,NULL,&b);CHKERRQ(ierr);

  alpha[n-1] = a;
  beta[n-1] = b;
  PetscFunctionReturn(0);
}*/

#undef __FUNCT__
#define __FUNCT__ "SVDSolve_Lanczos"
PetscErrorCode SVDSolve_Lanczos(SVD svd)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;
  PetscReal      *alpha,*beta,lastbeta,norm;
  PetscScalar    *swork,*w,*Q,*PT;
  PetscInt       i,k,j,nv,ld,off;
  Vec            v,u=0,u_1=0;
  PetscBool      conv;

  PetscFunctionBegin;
  /* allocate working space */
  ierr = DSGetLeadingDimension(svd->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc2(ld,&w,svd->ncv,&swork);CHKERRQ(ierr);

//  ierr = VecDuplicate(svd->V[0],&v);CHKERRQ(ierr);
  if (lanczos->oneside) {
    ierr = SVDMatGetVecs(svd,NULL,&u);CHKERRQ(ierr);
    ierr = SVDMatGetVecs(svd,NULL,&u_1);CHKERRQ(ierr);
  }

  /* normalize start vector */
  if (!svd->nini) {
    ierr = BVSetRandom(svd->V,0,svd->rand);CHKERRQ(ierr);
    ierr = BVNorm(svd->V,0,NORM_2,&norm);CHKERRQ(ierr);
    ierr = BVScale(svd->V,0,1.0/norm);CHKERRQ(ierr);
  }

  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->ncv);
    ierr = DSGetArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    beta = alpha + ld;
    if (lanczos->oneside) {
      //ierr = SVDOneSideLanczos(svd,alpha,beta,svd->V,v,u,u_1,svd->nconv,nv,swork);CHKERRQ(ierr);
    } else {
      ierr = SVDTwoSideLanczos(svd,alpha,beta,svd->V,svd->U,svd->nconv,nv,swork);CHKERRQ(ierr);
    }
    lastbeta = beta[nv-1];
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);

    /* compute SVD of bidiagonal matrix */
    ierr = DSSetDimensions(svd->ds,nv,nv,svd->nconv,0);CHKERRQ(ierr);
    ierr = DSSetState(svd->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
    ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

    /* compute error estimates */
    k = 0;
    conv = PETSC_TRUE;
    ierr = DSGetArray(svd->ds,DS_MAT_U,&Q);CHKERRQ(ierr);
    for (i=svd->nconv;i<nv;i++) {
      svd->sigma[i] = PetscRealPart(w[i]);
      svd->errest[i] = PetscAbsScalar(Q[nv-1+i*ld])*lastbeta;
      if (svd->sigma[i] > svd->tol) svd->errest[i] /= svd->sigma[i];
      if (conv) {
        if (svd->errest[i] < svd->tol) k++;
        else conv = PETSC_FALSE;
      }
    }

    /* check convergence */
    if (svd->its >= svd->max_it) svd->reason = SVD_DIVERGED_ITS;
    if (svd->nconv+k >= svd->nsv) svd->reason = SVD_CONVERGED_TOL;

    /* compute restart vector */
    ierr = DSGetArray(svd->ds,DS_MAT_VT,&PT);CHKERRQ(ierr);
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      for (j=svd->nconv;j<nv;j++) swork[j-svd->nconv] = PT[k+svd->nconv+j*ld];
//      ierr = SlepcVecMAXPBY(v,0.0,1.0,nv-svd->nconv,swork,svd->V+svd->nconv);CHKERRQ(ierr);
    }

    /* compute converged singular vectors */
    off = svd->nconv+svd->nconv*ld;
//    ierr = SlepcUpdateVectors(nv-svd->nconv,svd->V+svd->nconv,0,k,PT+off,ld,PETSC_TRUE);CHKERRQ(ierr);
    if (!lanczos->oneside) {
//      ierr = SlepcUpdateVectors(nv-svd->nconv,svd->U+svd->nconv,0,k,Q+off,ld,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = DSRestoreArray(svd->ds,DS_MAT_VT,&PT);CHKERRQ(ierr);
    ierr = DSRestoreArray(svd->ds,DS_MAT_U,&Q);CHKERRQ(ierr);

    /* copy restart vector from temporary space */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
//      ierr = VecCopy(v,svd->V[svd->nconv+k]);CHKERRQ(ierr);
    }

    svd->nconv += k;
    ierr = SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv);CHKERRQ(ierr);
  }

  /* free working space */
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&u_1);CHKERRQ(ierr);
  ierr = PetscFree2(w,swork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSetFromOptions_Lanczos"
PetscErrorCode SVDSetFromOptions_Lanczos(SVD svd)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SVD Lanczos Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-svd_lanczos_oneside","Lanczos one-side reorthogonalization","SVDLanczosSetOneSide",lanczos->oneside,&val,&set);CHKERRQ(ierr);
  if (set) {
    ierr = SVDLanczosSetOneSide(svd,val);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDLanczosSetOneSide_Lanczos"
static PetscErrorCode SVDLanczosSetOneSide_Lanczos(SVD svd,PetscBool oneside)
{
  SVD_LANCZOS *lanczos = (SVD_LANCZOS*)svd->data;

  PetscFunctionBegin;
  if (lanczos->oneside != oneside) {
    lanczos->oneside = oneside;
    svd->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDLanczosSetOneSide"
/*@
   SVDLanczosSetOneSide - Indicate if the variant of the Lanczos method
   to be used is one-sided or two-sided.

   Logically Collective on SVD

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

#undef __FUNCT__
#define __FUNCT__ "SVDLanczosGetOneSide"
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
  PetscValidPointer(oneside,2);
  ierr = PetscTryMethod(svd,"SVDLanczosGetOneSide_C",(SVD,PetscBool*),(svd,oneside));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDLanczosGetOneSide_Lanczos"
static PetscErrorCode SVDLanczosGetOneSide_Lanczos(SVD svd,PetscBool *oneside)
{
  SVD_LANCZOS *lanczos = (SVD_LANCZOS*)svd->data;

  PetscFunctionBegin;
  *oneside = lanczos->oneside;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDDestroy_Lanczos"
PetscErrorCode SVDDestroy_Lanczos(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosSetOneSide_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosGetOneSide_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDView_Lanczos"
PetscErrorCode SVDView_Lanczos(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS*)svd->data;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"  Lanczos: %s-sided reorthogonalization\n",lanczos->oneside? "one": "two");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCreate_Lanczos"
PETSC_EXTERN PetscErrorCode SVDCreate_Lanczos(SVD svd)
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
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosSetOneSide_C",SVDLanczosSetOneSide_Lanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDLanczosGetOneSide_C",SVDLanczosGetOneSide_Lanczos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

