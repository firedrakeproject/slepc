/*                       

   SLEPc singular value solver: "trlanczos"

   Method: Golub-Kahan-Lanczos bidiagonalization with thick-restart

   Last update: Jun 2007

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

#include <private/svdimpl.h>                /*I "slepcsvd.h" I*/
#include <private/ipimpl.h>                 /*I "slepcip.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  PetscBool oneside;
} SVD_TRLANCZOS;

#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_TRLanczos"
PetscErrorCode SVDSetUp_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       i,N,nloc;
  Vec            t;

  PetscFunctionBegin;
  ierr = SVDMatGetSize(svd,PETSC_NULL,&N);CHKERRQ(ierr);
  if (svd->ncv) { /* ncv set */
    if (svd->ncv<svd->nsv) SETERRQ(((PetscObject)svd)->comm,1,"The value of ncv must be at least nsv"); 
  }
  else if (svd->mpd) { /* mpd set */
    svd->ncv = PetscMin(N,svd->nsv+svd->mpd);
  }
  else { /* neither set: defaults depend on nsv being small or large */
    if (svd->nsv<500) svd->ncv = PetscMin(N,PetscMax(2*svd->nsv,10));
    else { svd->mpd = 500; svd->ncv = PetscMin(N,svd->nsv+svd->mpd); }
  }
  if (!svd->mpd) svd->mpd = svd->ncv;
  if (svd->ncv>svd->nsv+svd->mpd) SETERRQ(((PetscObject)svd)->comm,1,"The value of ncv must not be larger than nev+mpd"); 
  if (!svd->max_it) svd->max_it = PetscMax(N/svd->ncv,100);
  if (svd->ncv!=svd->n) {  
    ierr = SlepcVecDestroyVecs(svd->n,&svd->U);CHKERRQ(ierr);
    ierr = SVDMatGetLocalSize(svd,&nloc,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(((PetscObject)svd)->comm,nloc,PETSC_DECIDE,PETSC_NULL,&t);CHKERRQ(ierr);
    ierr = SlepcVecDuplicateVecs(t,svd->ncv,&svd->U);CHKERRQ(ierr);
    ierr = VecDestroy(&t);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDOneSideTRLanczosMGS"
static PetscErrorCode SVDOneSideTRLanczosMGS(SVD svd,PetscReal *alpha,PetscReal *beta,PetscScalar* bb,Vec *V,Vec v,Vec* U,PetscInt nconv,PetscInt l,PetscInt n,PetscScalar* work)
{
  PetscErrorCode ierr;
  PetscReal      a,b;
  PetscInt       i,k=nconv+l;

  PetscFunctionBegin;
  ierr = SVDMatMult(svd,PETSC_FALSE,V[k],U[k]);CHKERRQ(ierr);
  if (l>0) {
    ierr = SlepcVecMAXPBY(U[k],1.0,-1.0,l,bb,U+nconv);CHKERRQ(ierr);
  }
  ierr = IPNorm(svd->ip,U[k],&a);CHKERRQ(ierr);
  ierr = VecScale(U[k],1.0/a);CHKERRQ(ierr);
  alpha[0] = a;
  for (i=k+1;i<n;i++) {
    ierr = SVDMatMult(svd,PETSC_TRUE,U[i-1],V[i]);CHKERRQ(ierr);
    ierr = IPOrthogonalize(svd->ip,0,PETSC_NULL,i,PETSC_NULL,V,V[i],work,&b,PETSC_NULL);CHKERRQ(ierr);  
    ierr = VecScale(V[i],1.0/b);CHKERRQ(ierr);
    beta[i-k-1] = b;
    
    ierr = SVDMatMult(svd,PETSC_FALSE,V[i],U[i]);CHKERRQ(ierr);
    ierr = VecAXPY(U[i],-b,U[i-1]);CHKERRQ(ierr);
    ierr = IPNorm(svd->ip,U[i],&a);CHKERRQ(ierr);
    ierr = VecScale(U[i],1.0/a);CHKERRQ(ierr);
    alpha[i-k] = a;
  }
  ierr = SVDMatMult(svd,PETSC_TRUE,U[n-1],v);CHKERRQ(ierr);
  ierr = IPOrthogonalize(svd->ip,0,PETSC_NULL,n,PETSC_NULL,V,v,work,&b,PETSC_NULL);CHKERRQ(ierr);      
  beta[n-k-1] = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDOneSideTRLanczosCGS"
static PetscErrorCode SVDOneSideTRLanczosCGS(SVD svd,PetscReal *alpha,PetscReal *beta,PetscScalar* bb,Vec *V,Vec v,Vec* U,PetscInt nconv,PetscInt l,PetscInt n,PetscScalar* work)
{
  PetscErrorCode ierr;
  PetscReal      a,b,sum,onorm;
  PetscScalar    dot;
  PetscInt       i,j,k=nconv+l;

  PetscFunctionBegin;
  ierr = SVDMatMult(svd,PETSC_FALSE,V[k],U[k]);CHKERRQ(ierr);
  if (l>0) {
    ierr = SlepcVecMAXPBY(U[k],1.0,-1.0,l,bb,U+nconv);CHKERRQ(ierr);
  }
  for (i=k+1;i<n;i++) {
    ierr = SVDMatMult(svd,PETSC_TRUE,U[i-1],V[i]);CHKERRQ(ierr);
    ierr = IPNormBegin(svd->ip,U[i-1],&a);CHKERRQ(ierr);
    if (svd->ip->orthog_ref == IP_ORTH_REFINE_IFNEEDED) {
      ierr = IPInnerProductBegin(svd->ip,V[i],V[i],&dot);CHKERRQ(ierr);
    }
    ierr = IPMInnerProductBegin(svd->ip,V[i],i,V,work);CHKERRQ(ierr);
    ierr = IPNormEnd(svd->ip,U[i-1],&a);CHKERRQ(ierr);
    if (svd->ip->orthog_ref == IP_ORTH_REFINE_IFNEEDED) {
      ierr = IPInnerProductEnd(svd->ip,V[i],V[i],&dot);CHKERRQ(ierr);
    }
    ierr = IPMInnerProductEnd(svd->ip,V[i],i,V,work);CHKERRQ(ierr);
    
    ierr = VecScale(U[i-1],1.0/a);CHKERRQ(ierr);
    for (j=0;j<i;j++) work[j] = work[j] / a;
    ierr = SlepcVecMAXPBY(V[i],1.0/a,-1.0,i,work,V);CHKERRQ(ierr);

    switch (svd->ip->orthog_ref) {
    case IP_ORTH_REFINE_NEVER:
      ierr = IPNorm(svd->ip,V[i],&b);CHKERRQ(ierr);
      break;      
    case IP_ORTH_REFINE_ALWAYS:
      ierr = IPOrthogonalizeCGS1(svd->ip,0,PETSC_NULL,i,PETSC_NULL,V,V[i],work,PETSC_NULL,&b);CHKERRQ(ierr);
      break;
    case IP_ORTH_REFINE_IFNEEDED:
      onorm = sqrt(PetscRealPart(dot)) / a;
      sum = 0.0;
      for (j=0;j<i;j++) {
        sum += PetscRealPart(work[j] * PetscConj(work[j]));
      }
      b = PetscRealPart(dot)/(a*a) - sum;
      if (b>0.0) b = sqrt(b);
      else {
        ierr = IPNorm(svd->ip,V[i],&b);CHKERRQ(ierr);
      }
      if (b < svd->ip->orthog_eta * onorm) {
        ierr = IPOrthogonalizeCGS1(svd->ip,0,PETSC_NULL,i,PETSC_NULL,V,V[i],work,PETSC_NULL,&b);CHKERRQ(ierr);
      }
      break;
    }
    
    ierr = VecScale(V[i],1.0/b);CHKERRQ(ierr);
  
    ierr = SVDMatMult(svd,PETSC_FALSE,V[i],U[i]);CHKERRQ(ierr);
    ierr = VecAXPY(U[i],-b,U[i-1]);CHKERRQ(ierr);

    alpha[i-k-1] = a;
    beta[i-k-1] = b;
  }
  ierr = SVDMatMult(svd,PETSC_TRUE,U[n-1],v);CHKERRQ(ierr);
  ierr = IPNormBegin(svd->ip,U[n-1],&a);CHKERRQ(ierr);
  if (svd->ip->orthog_ref == IP_ORTH_REFINE_IFNEEDED) {
    ierr = IPInnerProductBegin(svd->ip,v,v,&dot);CHKERRQ(ierr);
  }
  ierr = IPMInnerProductBegin(svd->ip,v,n,V,work);CHKERRQ(ierr);
  ierr = IPNormEnd(svd->ip,U[n-1],&a);CHKERRQ(ierr);
  if (svd->ip->orthog_ref == IP_ORTH_REFINE_IFNEEDED) {
    ierr = IPInnerProductEnd(svd->ip,v,v,&dot);CHKERRQ(ierr);
  }
  ierr = IPMInnerProductEnd(svd->ip,v,n,V,work);CHKERRQ(ierr);
    
  ierr = VecScale(U[n-1],1.0/a);CHKERRQ(ierr);
  for (j=0;j<n;j++) work[j] = work[j] / a;
  ierr = SlepcVecMAXPBY(v,1.0/a,-1.0,n,work,V);CHKERRQ(ierr);

  switch (svd->ip->orthog_ref) {
  case IP_ORTH_REFINE_NEVER:
    ierr = IPNorm(svd->ip,v,&b);CHKERRQ(ierr);
    break;      
  case IP_ORTH_REFINE_ALWAYS:
    ierr = IPOrthogonalizeCGS1(svd->ip,0,PETSC_NULL,n,PETSC_NULL,V,v,work,PETSC_NULL,&b);CHKERRQ(ierr);
    break;
  case IP_ORTH_REFINE_IFNEEDED:
    onorm = sqrt(PetscRealPart(dot)) / a;
    sum = 0.0;
    for (j=0;j<i;j++) {
      sum += PetscRealPart(work[j] * PetscConj(work[j]));
    }
    b = PetscRealPart(dot)/(a*a) - sum;
    if (b>0.0) b = sqrt(b);
    else {
      ierr = IPNorm(svd->ip,v,&b);CHKERRQ(ierr);
    }
    if (b < svd->ip->orthog_eta * onorm) {
      ierr = IPOrthogonalizeCGS1(svd->ip,0,PETSC_NULL,n,PETSC_NULL,V,v,work,PETSC_NULL,&b);CHKERRQ(ierr);
    }
    break;
  }
      
  alpha[n-k-1] = a;
  beta[n-k-1] = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_TRLanczos"
PetscErrorCode SVDSolve_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS *)svd->data;
  PetscReal      *alpha,*beta,norm;
  PetscScalar    *b,*Q,*PT,*swork;
  PetscInt       i,j,k,l,m,n,nv;
  Vec            v;
  PetscBool      conv;
  IPOrthogonalizationType orthog;
  
  PetscFunctionBegin;
  /* allocate working space */
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&alpha);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&beta);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n,&b);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&PT);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (svd->which == SVD_SMALLEST) { 
#endif
    ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&swork);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  } else {
    ierr = PetscMalloc(sizeof(PetscScalar)*svd->n,&swork);CHKERRQ(ierr);
  }
#endif
  ierr = VecDuplicate(svd->V[0],&v);CHKERRQ(ierr);
  ierr = IPGetOrthogonalization(svd->ip,&orthog,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  
  /* normalize start vector */
  if (svd->nini==0) {
    ierr = SlepcVecSetRandom(svd->V[0],svd->rand);CHKERRQ(ierr);
  }
  ierr = VecNormalize(svd->V[0],&norm);CHKERRQ(ierr);
  
  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    nv = PetscMin(svd->nconv+svd->mpd,svd->n);
    if (lanczos->oneside) {
      if (orthog == IP_ORTH_MGS) {
        ierr = SVDOneSideTRLanczosMGS(svd,alpha,beta,b+svd->nconv,svd->V,v,svd->U,svd->nconv,l,nv,swork);CHKERRQ(ierr);
      } else {
        ierr = SVDOneSideTRLanczosCGS(svd,alpha,beta,b+svd->nconv,svd->V,v,svd->U,svd->nconv,l,nv,swork);CHKERRQ(ierr);
      }
    } else {
      ierr = SVDTwoSideLanczos(svd,alpha,beta,svd->V,v,svd->U,svd->nconv+l,nv,swork);CHKERRQ(ierr);
    }
    ierr = VecScale(v,1.0/beta[nv-svd->nconv-l-1]);CHKERRQ(ierr);
   
    /* compute SVD of general matrix */
    n = nv - svd->nconv;
    /* first l columns */
    for (j=0;j<l;j++) {
      for (i=0;i<j;i++) Q[j*n+i] = 0.0;    
      Q[j*n+j] = svd->sigma[svd->nconv+j];
      for (i=j+1;i<n;i++) Q[j*n+i] = 0.0;
    }
    /* l+1 column */
    for (i=0;i<l;i++) Q[l*n+i] = b[i+svd->nconv];
    Q[l*n+l] = alpha[0];
    for (i=l+1;i<n;i++) Q[l*n+i] = 0.0;
    /* rest of matrix */
    for (j=l+1;j<n;j++) {
      for (i=0;i<j-1;i++) Q[j*n+i] = 0.0;
      Q[j*n+j-1] = beta[j-l-1];
      Q[j*n+j] = alpha[j-l];
      for (i=j+1;i<n;i++) Q[j*n+i] = 0.0;
    }
    ierr = SVDDense(n,n,Q,alpha,PETSC_NULL,PT);CHKERRQ(ierr);

    /* compute error estimates */
    k = 0;
    conv = PETSC_TRUE;
    for (i=svd->nconv;i<nv;i++) {
      if (svd->which == SVD_SMALLEST) j = n-i+svd->nconv-1;
      else j = i-svd->nconv;
      svd->sigma[i] = alpha[j];
      b[i] = Q[j*n+n-1]*beta[n-l-1];
      svd->errest[i] = PetscAbsScalar(b[i]);
      if (alpha[j] > svd->tol) svd->errest[i] /= alpha[j];
      if (conv) {
        if (svd->errest[i] < svd->tol) k++;
        else conv = PETSC_FALSE;
      }
    }
    
    /* check convergence and update l */
    if (svd->its >= svd->max_it) svd->reason = SVD_DIVERGED_ITS;
    if (svd->nconv+k >= svd->nsv) svd->reason = SVD_CONVERGED_TOL;
    if (svd->reason != SVD_CONVERGED_ITERATING) l = 0;
    else l = PetscMax((nv - svd->nconv - k) / 2,0);
    
    /* compute converged singular vectors and restart vectors*/
#if !defined(PETSC_USE_COMPLEX)
    if (svd->which == SVD_SMALLEST) {
#endif
    for (i=0;i<k+l;i++) {
      if (svd->which == SVD_SMALLEST) j = n-i-1;
      else j = i;
      for (m=0;m<n;m++) swork[j*n+m] = PT[m*n+j];
    }
    ierr = SlepcUpdateVectors(n,svd->V+svd->nconv,0,k+l,swork,n,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0;i<k+l;i++) {
      if (svd->which == SVD_SMALLEST) j = n-i-1;
      else j = i;
      for (m=0;m<n;m++) swork[j*n+m] = Q[j*n+m];
    }
    ierr = SlepcUpdateVectors(n,svd->U+svd->nconv,0,k+l,swork,n,PETSC_FALSE);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    } else {
      ierr = SlepcUpdateVectors(n,svd->V+svd->nconv,0,k+l,PT,n,PETSC_TRUE);CHKERRQ(ierr);
      ierr = SlepcUpdateVectors(n,svd->U+svd->nconv,0,k+l,Q,n,PETSC_FALSE);CHKERRQ(ierr);
    }
#endif
    
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
      ierr = IPOrthogonalize(svd->ip,0,PETSC_NULL,i,PETSC_NULL,svd->U,svd->U[i],PETSC_NULL,&norm,PETSC_NULL);CHKERRQ(ierr);
      ierr = VecScale(svd->U[i],1.0/norm);CHKERRQ(ierr);
    }
  }
  
  /* free working space */
  ierr = VecDestroy(&v);CHKERRQ(ierr);

  ierr = PetscFree(alpha);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(PT);CHKERRQ(ierr);
  ierr = PetscFree(swork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions_TRLanczos"
PetscErrorCode SVDSetFromOptions_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;
  PetscBool      set,val;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS *)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)svd)->comm,((PetscObject)svd)->prefix,"TRLanczos Singular Value Solver Options","SVD");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-svd_trlanczos_oneside","Lanczos one-side reorthogonalization","SVDTRLanczosSetOneSide",lanczos->oneside,&val,&set);CHKERRQ(ierr);
  if (set) {
    ierr = SVDTRLanczosSetOneSide(svd,val);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDTRLanczosSetOneSide_TRLanczos"
PetscErrorCode SVDTRLanczosSetOneSide_TRLanczos(SVD svd,PetscBool oneside)
{
  SVD_TRLANCZOS *lanczos = (SVD_TRLANCZOS *)svd->data;

  PetscFunctionBegin;
  lanczos->oneside = oneside;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDTRLanczosGetOneSide_TRLanczos"
PetscErrorCode SVDTRLanczosGetOneSide_TRLanczos(SVD svd,PetscBool *oneside)
{
  SVD_TRLANCZOS    *lanczos = (SVD_TRLANCZOS *)svd->data;

  PetscFunctionBegin;
  *oneside = lanczos->oneside;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "SVDDestroy_TRLanczos"
PetscErrorCode SVDDestroy_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  ierr = SVDDestroy_Default(svd);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDTRLanczosSetOneSide_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDTRLanczosGetOneSide_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDView_TRLanczos"
PetscErrorCode SVDView_TRLanczos(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS *)svd->data;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"Lanczos reorthogonalization: %s\n",lanczos->oneside ? "one-side" : "two-side");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCreate_TRLanczos"
PetscErrorCode SVDCreate_TRLanczos(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(svd,SVD_TRLANCZOS,&svd->data);CHKERRQ(ierr);
  svd->ops->setup          = SVDSetUp_TRLanczos;
  svd->ops->solve          = SVDSolve_TRLanczos;
  svd->ops->destroy        = SVDDestroy_TRLanczos;
  svd->ops->setfromoptions = SVDSetFromOptions_TRLanczos;
  svd->ops->view           = SVDView_TRLanczos;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDTRLanczosSetOneSide_C","SVDTRLanczosSetOneSide_TRLanczos",SVDTRLanczosSetOneSide_TRLanczos);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDTRLanczosGetOneSide_C","SVDTRLanczosGetOneSide_TRLanczos",SVDTRLanczosGetOneSide_TRLanczos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
