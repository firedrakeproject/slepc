/*                       

   SLEPc singular value solver: "trlanczos"

   Method: Thick-restart Lanczos

   Algorithm:

       Golub-Kahan-Lanczos bidiagonalization with thick-restart.

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
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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
} SVD_TRLANCZOS;

#undef __FUNCT__
#define __FUNCT__ "SVDSetUp_TRLanczos"
PetscErrorCode SVDSetUp_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;
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
  if (svd->ncv!=svd->n) {  
    ierr = VecDuplicateVecs(svd->tl,svd->ncv,&svd->U);CHKERRQ(ierr);
  }
  ierr = DSSetType(svd->ds,DSSVD);CHKERRQ(ierr);
  ierr = DSSetCompact(svd->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(svd->ds,svd->ncv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDOneSideTRLanczosMGS"
static PetscErrorCode SVDOneSideTRLanczosMGS(SVD svd,PetscReal *alpha,PetscReal *beta,Vec *V,Vec v,Vec* U,PetscInt nconv,PetscInt l,PetscInt n,PetscScalar* work)
{
  PetscErrorCode ierr;
  PetscReal      a,b;
  PetscInt       i,k=nconv+l;

  PetscFunctionBegin;
  ierr = SVDMatMult(svd,PETSC_FALSE,V[k],U[k]);CHKERRQ(ierr);
  if (l>0) {
    for (i=0;i<l;i++) work[i]=beta[i+nconv];
    ierr = SlepcVecMAXPBY(U[k],1.0,-1.0,l,work,U+nconv);CHKERRQ(ierr);
  }
  ierr = IPNorm(svd->ip,U[k],&a);CHKERRQ(ierr);
  ierr = VecScale(U[k],1.0/a);CHKERRQ(ierr);
  alpha[k] = a;
  for (i=k+1;i<n;i++) {
    ierr = SVDMatMult(svd,PETSC_TRUE,U[i-1],V[i]);CHKERRQ(ierr);
    ierr = IPOrthogonalize(svd->ip,0,NULL,i,NULL,V,V[i],work,&b,NULL);CHKERRQ(ierr);  
    ierr = VecScale(V[i],1.0/b);CHKERRQ(ierr);
    beta[i-1] = b;
    
    ierr = SVDMatMult(svd,PETSC_FALSE,V[i],U[i]);CHKERRQ(ierr);
    ierr = VecAXPY(U[i],-b,U[i-1]);CHKERRQ(ierr);
    ierr = IPNorm(svd->ip,U[i],&a);CHKERRQ(ierr);
    ierr = VecScale(U[i],1.0/a);CHKERRQ(ierr);
    alpha[i] = a;
  }
  ierr = SVDMatMult(svd,PETSC_TRUE,U[n-1],v);CHKERRQ(ierr);
  ierr = IPOrthogonalize(svd->ip,0,NULL,n,NULL,V,v,work,&b,NULL);CHKERRQ(ierr);      
  beta[n-1] = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDOneSideTRLanczosCGS"
static PetscErrorCode SVDOneSideTRLanczosCGS(SVD svd,PetscReal *alpha,PetscReal *beta,Vec *V,Vec v,Vec* U,PetscInt nconv,PetscInt l,PetscInt n,PetscScalar* work)
{
  PetscErrorCode     ierr;
  PetscReal          a,b,sum,onorm,eta;
  PetscScalar        dot;
  PetscInt           i,j,k=nconv+l;
  IPOrthogRefineType refine;

  PetscFunctionBegin;
  ierr = SVDMatMult(svd,PETSC_FALSE,V[k],U[k]);CHKERRQ(ierr);
  if (l>0) {
    for (i=0;i<l;i++) work[i]=beta[i+nconv];
    ierr = SlepcVecMAXPBY(U[k],1.0,-1.0,l,work,U+nconv);CHKERRQ(ierr);
  }
  ierr = IPGetOrthogonalization(svd->ip,NULL,&refine,&eta);CHKERRQ(ierr);
  for (i=k+1;i<n;i++) {
    ierr = SVDMatMult(svd,PETSC_TRUE,U[i-1],V[i]);CHKERRQ(ierr);
    ierr = IPNormBegin(svd->ip,U[i-1],&a);CHKERRQ(ierr);
    if (refine == IP_ORTHOG_REFINE_IFNEEDED) {
      ierr = IPInnerProductBegin(svd->ip,V[i],V[i],&dot);CHKERRQ(ierr);
    }
    ierr = IPMInnerProductBegin(svd->ip,V[i],i,V,work);CHKERRQ(ierr);
    ierr = IPNormEnd(svd->ip,U[i-1],&a);CHKERRQ(ierr);
    if (refine == IP_ORTHOG_REFINE_IFNEEDED) {
      ierr = IPInnerProductEnd(svd->ip,V[i],V[i],&dot);CHKERRQ(ierr);
    }
    ierr = IPMInnerProductEnd(svd->ip,V[i],i,V,work);CHKERRQ(ierr);
    
    ierr = VecScale(U[i-1],1.0/a);CHKERRQ(ierr);
    for (j=0;j<i;j++) work[j] = work[j] / a;
    ierr = SlepcVecMAXPBY(V[i],1.0/a,-1.0,i,work,V);CHKERRQ(ierr);

    switch (refine) {
    case IP_ORTHOG_REFINE_NEVER:
      ierr = IPNorm(svd->ip,V[i],&b);CHKERRQ(ierr);
      break;      
    case IP_ORTHOG_REFINE_ALWAYS:
      ierr = IPOrthogonalizeCGS1(svd->ip,0,NULL,i,NULL,V,V[i],work,NULL,&b);CHKERRQ(ierr);
      break;
    case IP_ORTHOG_REFINE_IFNEEDED:
      onorm = PetscSqrtReal(PetscRealPart(dot)) / a;
      sum = 0.0;
      for (j=0;j<i;j++) {
        sum += PetscRealPart(work[j] * PetscConj(work[j]));
      }
      b = PetscRealPart(dot)/(a*a) - sum;
      if (b>0.0) b = PetscSqrtReal(b);
      else {
        ierr = IPNorm(svd->ip,V[i],&b);CHKERRQ(ierr);
      }
      if (b < eta*onorm) {
        ierr = IPOrthogonalizeCGS1(svd->ip,0,NULL,i,NULL,V,V[i],work,NULL,&b);CHKERRQ(ierr);
      }
      break;
    }
    
    ierr = VecScale(V[i],1.0/b);CHKERRQ(ierr);
  
    ierr = SVDMatMult(svd,PETSC_FALSE,V[i],U[i]);CHKERRQ(ierr);
    ierr = VecAXPY(U[i],-b,U[i-1]);CHKERRQ(ierr);

    alpha[i-1] = a;
    beta[i-1] = b;
  }
  ierr = SVDMatMult(svd,PETSC_TRUE,U[n-1],v);CHKERRQ(ierr);
  ierr = IPNormBegin(svd->ip,U[n-1],&a);CHKERRQ(ierr);
  if (refine == IP_ORTHOG_REFINE_IFNEEDED) {
    ierr = IPInnerProductBegin(svd->ip,v,v,&dot);CHKERRQ(ierr);
  }
  ierr = IPMInnerProductBegin(svd->ip,v,n,V,work);CHKERRQ(ierr);
  ierr = IPNormEnd(svd->ip,U[n-1],&a);CHKERRQ(ierr);
  if (refine == IP_ORTHOG_REFINE_IFNEEDED) {
    ierr = IPInnerProductEnd(svd->ip,v,v,&dot);CHKERRQ(ierr);
  }
  ierr = IPMInnerProductEnd(svd->ip,v,n,V,work);CHKERRQ(ierr);
    
  ierr = VecScale(U[n-1],1.0/a);CHKERRQ(ierr);
  for (j=0;j<n;j++) work[j] = work[j] / a;
  ierr = SlepcVecMAXPBY(v,1.0/a,-1.0,n,work,V);CHKERRQ(ierr);

  switch (refine) {
  case IP_ORTHOG_REFINE_NEVER:
    ierr = IPNorm(svd->ip,v,&b);CHKERRQ(ierr);
    break;      
  case IP_ORTHOG_REFINE_ALWAYS:
    ierr = IPOrthogonalizeCGS1(svd->ip,0,NULL,n,NULL,V,v,work,NULL,&b);CHKERRQ(ierr);
    break;
  case IP_ORTHOG_REFINE_IFNEEDED:
    onorm = PetscSqrtReal(PetscRealPart(dot)) / a;
    sum = 0.0;
    for (j=0;j<i;j++) {
      sum += PetscRealPart(work[j] * PetscConj(work[j]));
    }
    b = PetscRealPart(dot)/(a*a) - sum;
    if (b>0.0) b = PetscSqrtReal(b);
    else {
      ierr = IPNorm(svd->ip,v,&b);CHKERRQ(ierr);
    }
    if (b < eta*onorm) {
      ierr = IPOrthogonalizeCGS1(svd->ip,0,NULL,n,NULL,V,v,work,NULL,&b);CHKERRQ(ierr);
    }
    break;
  }
      
  alpha[n-1] = a;
  beta[n-1] = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSolve_TRLanczos"
PetscErrorCode SVDSolve_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;
  PetscReal      *alpha,*beta,lastbeta,norm;
  PetscScalar    *Q,*PT,*swork,*w;
  PetscInt       i,k,l,nv,ld,off;
  Vec            v;
  PetscBool      conv;
  IPOrthogType   orthog;
  
  PetscFunctionBegin;
  /* allocate working space */
  ierr = DSGetLeadingDimension(svd->ds,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*ld,&w);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar)*svd->n,&swork);CHKERRQ(ierr);
  ierr = VecDuplicate(svd->V[0],&v);CHKERRQ(ierr);
  ierr = IPGetOrthogonalization(svd->ip,&orthog,NULL,NULL);CHKERRQ(ierr);
  
  /* normalize start vector */
  if (!svd->nini) {
    ierr = SlepcVecSetRandom(svd->V[0],svd->rand);CHKERRQ(ierr);
  }
  ierr = VecNormalize(svd->V[0],&norm);CHKERRQ(ierr);
  
  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->n);
    ierr = DSGetArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    beta = alpha + ld;
    if (lanczos->oneside) {
      if (orthog == IP_ORTHOG_MGS) {
        ierr = SVDOneSideTRLanczosMGS(svd,alpha,beta,svd->V,v,svd->U,svd->nconv,l,nv,swork);CHKERRQ(ierr);
      } else {
        ierr = SVDOneSideTRLanczosCGS(svd,alpha,beta,svd->V,v,svd->U,svd->nconv,l,nv,swork);CHKERRQ(ierr);
      }
    } else {
      ierr = SVDTwoSideLanczos(svd,alpha,beta,svd->V,v,svd->U,svd->nconv+l,nv,swork);CHKERRQ(ierr);
    }
    lastbeta = beta[nv-1];
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    ierr = VecScale(v,1.0/lastbeta);CHKERRQ(ierr);
   
    /* compute SVD of general matrix */
    ierr = DSSetDimensions(svd->ds,nv,nv,svd->nconv,svd->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(svd->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(svd->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }
    ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
    ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

    /* compute error estimates */
    k = 0;
    conv = PETSC_TRUE;
    ierr = DSGetArray(svd->ds,DS_MAT_U,&Q);CHKERRQ(ierr);
    ierr = DSGetArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    beta = alpha + ld;
    for (i=svd->nconv;i<nv;i++) {
      svd->sigma[i] = PetscRealPart(w[i]);
      beta[i] = PetscRealPart(Q[nv-1+i*ld])*lastbeta;
      svd->errest[i] = PetscAbsScalar(beta[i]);
      if (svd->sigma[i] > svd->tol) svd->errest[i] /= svd->sigma[i];
      if (conv) {
        if (svd->errest[i] < svd->tol) k++;
        else conv = PETSC_FALSE;
      }
    }
    ierr = DSRestoreArrayReal(svd->ds,DS_MAT_T,&alpha);CHKERRQ(ierr);
    
    /* check convergence and update l */
    if (svd->its >= svd->max_it) svd->reason = SVD_DIVERGED_ITS;
    if (svd->nconv+k >= svd->nsv) svd->reason = SVD_CONVERGED_TOL;
    if (svd->reason != SVD_CONVERGED_ITERATING) l = 0;
    else l = PetscMax((nv-svd->nconv-k)/2,0);
    
    /* compute converged singular vectors and restart vectors */
    off = svd->nconv+svd->nconv*ld;
    ierr = DSGetArray(svd->ds,DS_MAT_VT,&PT);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv-svd->nconv,svd->V+svd->nconv,0,k+l,PT+off,ld,PETSC_TRUE);CHKERRQ(ierr);
    ierr = SlepcUpdateVectors(nv-svd->nconv,svd->U+svd->nconv,0,k+l,Q+off,ld,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DSRestoreArray(svd->ds,DS_MAT_VT,&PT);CHKERRQ(ierr);
    ierr = DSRestoreArray(svd->ds,DS_MAT_U,&Q);CHKERRQ(ierr);
    
    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      ierr = VecCopy(v,svd->V[svd->nconv+k+l]);CHKERRQ(ierr);
    }
    
    svd->nconv += k;
    ierr = SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,nv);CHKERRQ(ierr);
  }
  
  /* orthonormalize U columns in one side method */
  if (lanczos->oneside) {
    for (i=0;i<svd->nconv;i++) {
      ierr = IPOrthogonalize(svd->ip,0,NULL,i,NULL,svd->U,svd->U[i],NULL,&norm,NULL);CHKERRQ(ierr);
      ierr = VecScale(svd->U[i],1.0/norm);CHKERRQ(ierr);
    }
  }
  
  /* free working space */
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = PetscFree(swork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDSetFromOptions_TRLanczos"
PetscErrorCode SVDSetFromOptions_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SVD TRLanczos Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-svd_trlanczos_oneside","Lanczos one-side reorthogonalization","SVDTRLanczosSetOneSide",lanczos->oneside,&val,&set);CHKERRQ(ierr);
  if (set) {
    ierr = SVDTRLanczosSetOneSide(svd,val);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDTRLanczosSetOneSide_TRLanczos"
static PetscErrorCode SVDTRLanczosSetOneSide_TRLanczos(SVD svd,PetscBool oneside)
{
  SVD_TRLANCZOS *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  lanczos->oneside = oneside;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDTRLanczosSetOneSide"
/*@
   SVDTRLanczosSetOneSide - Indicate if the variant of the Lanczos method 
   to be used is one-sided or two-sided.

   Logically Collective on SVD

   Input Parameters:
+  svd     - singular value solver
-  oneside - boolean flag indicating if the method is one-sided or not

   Options Database Key:
.  -svd_trlanczos_oneside <boolean> - Indicates the boolean flag

   Note:
   By default, a two-sided variant is selected, which is sometimes slightly
   more robust. However, the one-sided variant is faster because it avoids 
   the orthogonalization associated to left singular vectors. 

   Level: advanced

.seealso: SVDLanczosSetOneSide()
@*/
PetscErrorCode SVDTRLanczosSetOneSide(SVD svd,PetscBool oneside)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidLogicalCollectiveBool(svd,oneside,2);
  ierr = PetscTryMethod(svd,"SVDTRLanczosSetOneSide_C",(SVD,PetscBool),(svd,oneside));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDTRLanczosGetOneSide"
/*@
   SVDTRLanczosGetOneSide - Gets if the variant of the Lanczos method 
   to be used is one-sided or two-sided.

   Not Collective

   Input Parameters:
.  svd     - singular value solver

   Output Parameters:
.  oneside - boolean flag indicating if the method is one-sided or not

   Level: advanced

.seealso: SVDTRLanczosSetOneSide()
@*/
PetscErrorCode SVDTRLanczosGetOneSide(SVD svd,PetscBool *oneside)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(oneside,2);
  ierr = PetscTryMethod(svd,"SVDTRLanczosGetOneSide_C",(SVD,PetscBool*),(svd,oneside));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDTRLanczosGetOneSide_TRLanczos"
static PetscErrorCode SVDTRLanczosGetOneSide_TRLanczos(SVD svd,PetscBool *oneside)
{
  SVD_TRLANCZOS    *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  *oneside = lanczos->oneside;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDReset_TRLanczos"
PetscErrorCode SVDReset_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(svd->n,&svd->U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDDestroy_TRLanczos"
PetscErrorCode SVDDestroy_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetOneSide_C","",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetOneSide_C","",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDView_TRLanczos"
PetscErrorCode SVDView_TRLanczos(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS*)svd->data;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"  TRLanczos: %s-sided reorthogonalization\n",lanczos->oneside? "one": "two");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDCreate_TRLanczos"
PETSC_EXTERN PetscErrorCode SVDCreate_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(svd,SVD_TRLANCZOS,&svd->data);CHKERRQ(ierr);
  svd->ops->setup          = SVDSetUp_TRLanczos;
  svd->ops->solve          = SVDSolve_TRLanczos;
  svd->ops->destroy        = SVDDestroy_TRLanczos;
  svd->ops->reset          = SVDReset_TRLanczos;
  svd->ops->setfromoptions = SVDSetFromOptions_TRLanczos;
  svd->ops->view           = SVDView_TRLanczos;
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosSetOneSide_C","SVDTRLanczosSetOneSide_TRLanczos",SVDTRLanczosSetOneSide_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)svd,"SVDTRLanczosGetOneSide_C","SVDTRLanczosGetOneSide_TRLanczos",SVDTRLanczosGetOneSide_TRLanczos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

