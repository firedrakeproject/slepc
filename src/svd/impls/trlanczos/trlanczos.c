/*                       

   SLEPc singular value solver: "trlanczos"

   Method: Golub-Kahan-Lanczos bidiagonalization with thick-restart

   Last update: Jun 2007

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/svd/svdimpl.h"                /*I "slepcsvd.h" I*/
#include "src/ip/ipimpl.h"
#include "slepcblaslapack.h"

typedef struct {
  PetscTruth oneside;
} SVD_TRLANCZOS;

#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_TRLANCZOS"
PetscErrorCode SVDSetUp_TRLANCZOS(SVD svd)
{
  PetscErrorCode  ierr;
  PetscInt        N;
  int             i;

  PetscFunctionBegin;
  ierr = SVDMatGetSize(svd,PETSC_NULL,&N);CHKERRQ(ierr);
  if (svd->ncv == PETSC_DECIDE)
    svd->ncv = PetscMin(N,PetscMax(2*svd->nsv,10));
  if (svd->max_it == PETSC_DECIDE)
    svd->max_it = PetscMax(N/svd->ncv,100);
  if (svd->ncv!=svd->n) {  
    if (svd->U) {
      for (i=0;i<svd->n;i++) { ierr = VecDestroy(svd->U[i]); CHKERRQ(ierr); }
      ierr = PetscFree(svd->U);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(sizeof(Vec)*svd->ncv,&svd->U);CHKERRQ(ierr);
    for (i=0;i<svd->ncv;i++) { ierr = SVDMatGetVecs(svd,PETSC_NULL,svd->U+i);CHKERRQ(ierr); }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDOneSideTRLanczos"
static PetscErrorCode SVDOneSideTRLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,PetscScalar* bb,Vec *V,Vec v,Vec* U,int nconv,int l,int n,PetscScalar* work,Vec wv,Vec wu)
{
  PetscErrorCode ierr;
  PetscReal      a,b,sum,onorm;
  PetscScalar    dot;
  int            i,j,k=nconv+l;

  PetscFunctionBegin;
  ierr = SVDMatMult(svd,PETSC_FALSE,V[k],U[k]);CHKERRQ(ierr);
  if (l>0) {
    ierr = VecSet(wu,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(wu,l,bb,U+nconv);CHKERRQ(ierr);
    ierr = VecAXPY(U[k],-1.0,wu);CHKERRQ(ierr);
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
    ierr = VecScale(V[i],1.0/a);CHKERRQ(ierr);
    for (j=0;j<i;j++) work[j] = - work[j] / a;
    ierr = VecMAXPY(V[i],i,work,V);CHKERRQ(ierr);

    switch (svd->ip->orthog_ref) {
    case IP_ORTH_REFINE_NEVER:
      ierr = IPNorm(svd->ip,V[i],&b);CHKERRQ(ierr);
      break;      
    case IP_ORTH_REFINE_ALWAYS:
      ierr = IPOrthogonalizeCGS(svd->ip,i,PETSC_NULL,V,V[i],work,PETSC_NULL,&b,wv);CHKERRQ(ierr);
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
        ierr = IPOrthogonalizeCGS(svd->ip,i,PETSC_NULL,V,V[i],work,PETSC_NULL,&b,wv);CHKERRQ(ierr);
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
  ierr = VecScale(v,1.0/a);CHKERRQ(ierr);
  for (j=0;j<n;j++) work[j] = - work[j] / a;
  ierr = VecMAXPY(v,n,work,V);CHKERRQ(ierr);

  switch (svd->ip->orthog_ref) {
  case IP_ORTH_REFINE_NEVER:
    ierr = IPNorm(svd->ip,v,&b);CHKERRQ(ierr);
    break;      
  case IP_ORTH_REFINE_ALWAYS:
    ierr = IPOrthogonalizeCGS(svd->ip,n,PETSC_NULL,V,v,work,PETSC_NULL,&b,wv);CHKERRQ(ierr);
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
      ierr = IPOrthogonalizeCGS(svd->ip,n,PETSC_NULL,V,v,work,PETSC_NULL,&b,wv);CHKERRQ(ierr);
    }
    break;
  }
      
  alpha[n-k-1] = a;
  beta[n-k-1] = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_TRLANCZOS"
PetscErrorCode SVDSolve_TRLANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS *)svd->data;
  PetscReal      *alpha,*beta,norm;
  PetscScalar    *b,*Q,*PT,*swork;
  PetscInt       *perm;
  int            i,j,k,l,m,n,nwork=0;
  Vec            v,wv,wu,*workV,*workU,*permV,*permU;
  PetscTruth     conv;
  
  PetscFunctionBegin;
  /* allocate working space */
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&alpha);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&beta);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n,&b);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&PT);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n,&swork);CHKERRQ(ierr);
  ierr = VecDuplicate(svd->V[0],&v);CHKERRQ(ierr);
  ierr = VecDuplicate(svd->V[0],&wv);CHKERRQ(ierr);
  ierr = VecDuplicate(svd->U[0],&wu);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(Vec)*svd->n,&workV);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(Vec)*svd->n,&workU);CHKERRQ(ierr);
  
  /* normalize start vector */
  ierr = VecCopy(svd->vec_initial,svd->V[0]);CHKERRQ(ierr);
  ierr = VecNormalize(svd->V[0],&norm);CHKERRQ(ierr);
  
  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    if (lanczos->oneside) {
      ierr = SVDOneSideTRLanczos(svd,alpha,beta,b+svd->nconv,svd->V,v,svd->U,svd->nconv,l,svd->n,swork,wv,wu);CHKERRQ(ierr);
    } else {
      ierr = SVDTwoSideLanczos(svd,alpha,beta,svd->V,v,svd->U,svd->nconv+l,svd->n,swork,wv,wu);CHKERRQ(ierr);
    }
    ierr = VecScale(v,1.0/beta[svd->n-svd->nconv-l-1]);CHKERRQ(ierr);
   
    /* compute SVD of general matrix */
    n = svd->n - svd->nconv;
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
    for (i=svd->nconv;i<svd->n;i++) {
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
    else l = PetscMax((svd->n - svd->nconv - k) / 2,1);
    
    /* allocate work space for converged singular and restart vectors */
    if (nwork<k+l) {
      for (i=nwork;i<k+l;i++) { 
        ierr = SVDMatGetVecs(svd,workV+i,workU+i);CHKERRQ(ierr);
      }
      nwork = k+l;
    }
    
    /* compute converged singular vectors and restart vectors*/
    for (i=0;i<k+l;i++) {
      if (svd->which == SVD_SMALLEST) j = n-i-1;
      else j = i;
      ierr = VecSet(workV[i],0.0);CHKERRQ(ierr);
      for (m=0;m<n;m++) swork[m] = PT[m*n+j];
      ierr = VecMAXPY(workV[i],n,swork,svd->V+svd->nconv);CHKERRQ(ierr);
      ierr = VecSet(workU[i],0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(workU[i],n,Q+j*n,svd->U+svd->nconv);CHKERRQ(ierr);
    }
    
    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      ierr = VecCopy(v,svd->V[svd->nconv+k+l]);CHKERRQ(ierr);
    }
    
    /* copy converged singular vectors and restart vectors from temporary space */ 
    for (i=0;i<k+l;i++) {
      ierr = VecCopy(workV[i],svd->V[i+svd->nconv]);CHKERRQ(ierr);
      ierr = VecCopy(workU[i],svd->U[i+svd->nconv]);CHKERRQ(ierr);
    }
    
    svd->nconv += k;
    SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,svd->n);
  }
  
  /* sort singular triplets */
  ierr = PetscMalloc(sizeof(PetscInt)*svd->nconv,&perm);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(Vec)*svd->nconv,&permV);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(Vec)*svd->nconv,&permU);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    alpha[i] = svd->sigma[i];
    beta[i] = svd->errest[i];
    permV[i] = svd->V[i];
    permU[i] = svd->U[i];
    perm[i] = i;
  }
  ierr = PetscSortRealWithPermutation(svd->nconv,svd->sigma,perm);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    if (svd->which == SVD_SMALLEST) j = perm[i]; 
    else j = perm[svd->nconv-i-1];
    svd->sigma[i] = alpha[j];
    svd->errest[i] = beta[j];
    svd->V[i] = permV[j];
    svd->U[i] = permU[j];
  }
  
  /* free working space */
  ierr = VecDestroy(v);CHKERRQ(ierr);
  ierr = VecDestroy(wv);CHKERRQ(ierr);
  ierr = VecDestroy(wu);CHKERRQ(ierr);
  for (i=0;i<nwork;i++) { ierr = VecDestroy(workV[i]);CHKERRQ(ierr); }
  ierr = PetscFree(workV);CHKERRQ(ierr);
  for (i=0;i<nwork;i++) { ierr = VecDestroy(workU[i]);CHKERRQ(ierr); }
  ierr = PetscFree(workU);CHKERRQ(ierr);

  ierr = PetscFree(alpha);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(PT);CHKERRQ(ierr);
  ierr = PetscFree(swork);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  ierr = PetscFree(permV);CHKERRQ(ierr);
  ierr = PetscFree(permU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions_TRLANCZOS"
PetscErrorCode SVDSetFromOptions_TRLANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS *)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)svd)->comm,((PetscObject)svd)->prefix,"TRLANCZOS Singular Value Solver Options","SVD");CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-svd_trlanczos_oneside","Lanczos one-side reorthogonalization","SVDTRLanczosSetOneSide",PETSC_FALSE,&lanczos->oneside,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDTRLanczosSetOneSide_TRLANCZOS"
PetscErrorCode SVDTRLanczosSetOneSide_TRLANCZOS(SVD svd,PetscTruth oneside)
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

   Collective on SVD

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
PetscErrorCode SVDTRLanczosSetOneSide(SVD svd,PetscTruth oneside)
{
  PetscErrorCode ierr, (*f)(SVD,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDTRLanczosSetOneSide_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,oneside);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDView_TRLANCZOS"
PetscErrorCode SVDView_TRLANCZOS(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS *)svd->data;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"Lanczos reorthogonalization: %s\n",lanczos->oneside ? "one-side" : "two-side");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCreate_TRLANCZOS"
PetscErrorCode SVDCreate_TRLANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos;

  PetscFunctionBegin;
  ierr = PetscNew(SVD_TRLANCZOS,&lanczos);CHKERRQ(ierr);
  PetscLogObjectMemory(svd,sizeof(SVD_TRLANCZOS));
  svd->data                = (void *)lanczos;
  svd->ops->setup          = SVDSetUp_TRLANCZOS;
  svd->ops->solve          = SVDSolve_TRLANCZOS;
  svd->ops->destroy        = SVDDestroy_Default;
  svd->ops->setfromoptions = SVDSetFromOptions_TRLANCZOS;
  svd->ops->view           = SVDView_TRLANCZOS;
  lanczos->oneside         = PETSC_FALSE;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDTRLanczosSetOneSide_C","SVDTRLanczosSetOneSide_TRLANCZOS",SVDTRLanczosSetOneSide_TRLANCZOS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
