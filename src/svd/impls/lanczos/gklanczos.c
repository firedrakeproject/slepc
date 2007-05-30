/*                       

   SLEPc singular value solver: "lanczos"

   Method: Golub-Kahan-Lanczos bidiagonalization

   Last update: Mar 2007

*/
#include "src/svd/svdimpl.h"                /*I "slepcsvd.h" I*/
#include "slepcblaslapack.h"

typedef struct {
  PetscTruth oneside;
} SVD_LANCZOS;

#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_LANCZOS"
PetscErrorCode SVDSetUp_LANCZOS(SVD svd)
{
  PetscErrorCode  ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;
  PetscInt        N;
  int             i;

  PetscFunctionBegin;
  ierr = SVDMatGetSize(svd,PETSC_NULL,&N);CHKERRQ(ierr);
  if (svd->ncv == PETSC_DECIDE)
    svd->ncv = PetscMin(N,PetscMax(2*svd->nsv,10));
  if (svd->max_it == PETSC_DECIDE)
    svd->max_it = PetscMax(N/svd->ncv,100);
  if (svd->U) {
    for (i=0;i<svd->n;i++) { ierr = VecDestroy(svd->U[i]); CHKERRQ(ierr); }
    ierr = PetscFree(svd->U);CHKERRQ(ierr);
  }
  if (!lanczos->oneside) {
    ierr = PetscMalloc(sizeof(Vec)*svd->ncv,&svd->U);CHKERRQ(ierr);
    for (i=0;i<svd->ncv;i++) { ierr = SVDMatGetVecs(svd,PETSC_NULL,svd->U+i);CHKERRQ(ierr); }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDTwoSideLanczos"
PetscErrorCode SVDTwoSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,Vec *V,Vec v,Vec *U,int k,int n,PetscScalar* work,Vec wv,Vec wu)
{
  PetscErrorCode ierr;
  int            i;
  
  PetscFunctionBegin;
  ierr = SVDMatMult(svd,PETSC_FALSE,V[k],U[k]);CHKERRQ(ierr);
  ierr = IPOrthogonalize(svd->ip,k,PETSC_NULL,U,U[k],work,alpha,PETSC_NULL,wu);CHKERRQ(ierr);
  ierr = VecScale(U[k],1.0/alpha[0]);CHKERRQ(ierr);
  for (i=k+1;i<n;i++) {
    ierr = SVDMatMult(svd,PETSC_TRUE,U[i-1],V[i]);CHKERRQ(ierr);
    ierr = IPOrthogonalize(svd->ip,i,PETSC_NULL,V,V[i],work,beta+i-k-1,PETSC_NULL,wv);CHKERRQ(ierr);
    ierr = VecScale(V[i],1.0/beta[i-k-1]);CHKERRQ(ierr);

    ierr = SVDMatMult(svd,PETSC_FALSE,V[i],U[i]);CHKERRQ(ierr);
    ierr = IPOrthogonalize(svd->ip,i,PETSC_NULL,U,U[i],work,alpha+i-k,PETSC_NULL,wu);CHKERRQ(ierr);
    ierr = VecScale(U[i],1.0/alpha[i-k]);CHKERRQ(ierr);
  }
  ierr = SVDMatMult(svd,PETSC_TRUE,U[n-1],v);CHKERRQ(ierr);
  ierr = IPOrthogonalize(svd->ip,n,PETSC_NULL,V,v,work,beta+n-k-1,PETSC_NULL,wv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDOneSideLanczos"
static PetscErrorCode SVDOneSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,Vec *V,Vec v,Vec u,Vec u_1,int k,int n,PetscScalar* work,Vec wv)
{
  PetscErrorCode ierr;
  int            i,j;
  PetscReal      a,b;
  Vec            temp;
  
  PetscFunctionBegin;
  ierr = SVDMatMult(svd,PETSC_FALSE,V[k],u);CHKERRQ(ierr);
  for (i=k+1;i<n;i++) {
    ierr = SVDMatMult(svd,PETSC_TRUE,u,V[i]);CHKERRQ(ierr);
    ierr = IPNormBegin(svd->ip,u,&a);CHKERRQ(ierr);
    ierr = IPMInnerProductBegin(svd->ip,i,V[i],V,work);CHKERRQ(ierr);
    ierr = IPNormEnd(svd->ip,u,&a);CHKERRQ(ierr);
    ierr = IPMInnerProductEnd(svd->ip,i,V[i],V,work);CHKERRQ(ierr);
    
    ierr = VecScale(u,1.0/a);CHKERRQ(ierr);
    ierr = VecScale(V[i],1.0/a);CHKERRQ(ierr);
    for (j=0;j<i;j++) work[j] = - work[j] / a;
    ierr = VecMAXPY(V[i],i,work,V);CHKERRQ(ierr);

    ierr = IPOrthogonalizeCGS(svd->ip,i,PETSC_NULL,V,V[i],work,PETSC_NULL,&b,wv);CHKERRQ(ierr);
    ierr = VecScale(V[i],1.0/b);CHKERRQ(ierr);
  
    ierr = SVDMatMult(svd,PETSC_FALSE,V[i],u_1);CHKERRQ(ierr);
    ierr = VecAXPY(u_1,-b,u);CHKERRQ(ierr);

    alpha[i-k-1] = a;
    beta[i-k-1] = b;
    temp = u;
    u = u_1;
    u_1 = temp;
  }
  ierr = SVDMatMult(svd,PETSC_TRUE,u,v);CHKERRQ(ierr);
  ierr = IPNormBegin(svd->ip,u,&a);CHKERRQ(ierr);
  ierr = IPMInnerProductBegin(svd->ip,n,v,V,work);CHKERRQ(ierr);
  ierr = IPNormEnd(svd->ip,u,&a);CHKERRQ(ierr);
  ierr = IPMInnerProductEnd(svd->ip,n,v,V,work);CHKERRQ(ierr);
    
  ierr = VecScale(u,1.0/a);CHKERRQ(ierr);
  ierr = VecScale(v,1.0/a);CHKERRQ(ierr);
  for (j=0;j<n;j++) work[j] = - work[j] / a;
  ierr = VecMAXPY(v,n,work,V);CHKERRQ(ierr);

  ierr = IPOrthogonalizeCGS(svd->ip,n,PETSC_NULL,V,v,work,PETSC_NULL,&b,wv);CHKERRQ(ierr);
  
  alpha[n-k-1] = a;
  beta[n-k-1] = b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_LANCZOS"
PetscErrorCode SVDSolve_LANCZOS(SVD svd)
{
#if defined(SLEPC_MISSING_LAPACK_BDSDC)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"BDSDC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;
  PetscReal      *alpha,*beta,norm,*work,*Q,*PT;
  PetscScalar    *swork;
  PetscInt       *perm;
  int            i,j,k,m,n,info,nwork=0,*iwork;
  Vec            v,u,u_1,wv,wu,*workV,*workU,*permV,*permU;
  PetscTruth     conv;
  
  PetscFunctionBegin;
  /* allocate working space */
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&alpha);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&beta);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n*svd->n,&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n*svd->n,&PT);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*(3*svd->n+4)*svd->n,&work);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int)*8*svd->n,&iwork);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n,&swork);CHKERRQ(ierr);
  ierr = VecDuplicate(svd->V[0],&v);CHKERRQ(ierr);
  ierr = VecDuplicate(svd->V[0],&wv);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(Vec)*svd->n,&workV);CHKERRQ(ierr);
  if (lanczos->oneside) {
    ierr = SVDMatGetVecs(svd,PETSC_NULL,&u);CHKERRQ(ierr);
    ierr = SVDMatGetVecs(svd,PETSC_NULL,&u_1);CHKERRQ(ierr);
  } else {
    ierr = VecDuplicate(svd->U[0],&wu);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(Vec)*svd->n,&workU);CHKERRQ(ierr);
  }
  
  /* normalize start vector */
  ierr = VecCopy(svd->vec_initial,svd->V[0]);CHKERRQ(ierr);
  ierr = VecNormalize(svd->V[0],&norm);CHKERRQ(ierr);
  
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    if (lanczos->oneside) {
      ierr = SVDOneSideLanczos(svd,alpha,beta,svd->V,v,u,u_1,svd->nconv,svd->n,swork,wv);CHKERRQ(ierr);
    } else {
      ierr = SVDTwoSideLanczos(svd,alpha,beta,svd->V,v,svd->U,svd->nconv,svd->n,swork,wv,wu);CHKERRQ(ierr);
    }

    /* compute SVD of bidiagonal matrix */
    n = svd->n - svd->nconv;
    ierr = PetscMemzero(PT,sizeof(PetscReal)*n*n);CHKERRQ(ierr);
    ierr = PetscMemzero(Q,sizeof(PetscReal)*n*n);CHKERRQ(ierr);
    for (i=0;i<n;i++)
      PT[i*n+i] = Q[i*n+i] = 1.0;
    ierr = PetscLogEventBegin(SVD_Dense,0,0,0,0);CHKERRQ(ierr);
    LAPACKbdsdc_("U","I",&n,alpha,beta,Q,&n,PT,&n,PETSC_NULL,PETSC_NULL,work,iwork,&info,1,1);
    ierr = PetscLogEventEnd(SVD_Dense,0,0,0,0);CHKERRQ(ierr);

    /* compute error estimates */
    k = 0;
    conv = PETSC_TRUE;
    for (i=svd->nconv;i<svd->n;i++) {
      if (svd->which == SVD_SMALLEST) j = n-i+svd->nconv-1;
      else j = i-svd->nconv;
      svd->sigma[i] = alpha[j];
      svd->errest[i] = PetscAbsScalar(Q[j*n+n-1])*beta[n-1];
      if (alpha[j] > svd->tol) svd->errest[i] /= alpha[j];
      if (conv) {
        if (svd->errest[i] < svd->tol) k++;
        else conv = PETSC_FALSE;
      }
    }
    
    /* check convergence */
    if (svd->its >= svd->max_it) svd->reason = SVD_DIVERGED_ITS;
    if (svd->nconv+k >= svd->nsv) svd->reason = SVD_CONVERGED_TOL;
    
    /* allocate work space for converged singular vectors */
    if (nwork<k) {
      for (i=nwork;i<k;i++) 
        if (lanczos->oneside) { ierr = SVDMatGetVecs(svd,workV+i,PETSC_NULL);CHKERRQ(ierr); }
        else { ierr = SVDMatGetVecs(svd,workV+i,workU+i);CHKERRQ(ierr); }
      nwork = k;
    }
    
    /* compute converged singular vectors */
    for (i=0;i<k;i++) {
      if (svd->which == SVD_SMALLEST) j = n-i-1;
      else j = i;
      ierr = VecSet(workV[i],0.0);CHKERRQ(ierr);
      for (m=0;m<n;m++) swork[m] = PT[m*n+j];
      ierr = VecMAXPY(workV[i],n,swork,svd->V+svd->nconv);CHKERRQ(ierr);
      if (!lanczos->oneside) {
        ierr = VecSet(workU[i],0.0);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        ierr = VecMAXPY(workU[i],n,Q+j*n,svd->U+svd->nconv);CHKERRQ(ierr);
#else
        for (m=0;m<n;m++) swork[m] = Q[j*n+m];
        ierr = VecMAXPY(workU[i],n,swork,svd->U+svd->nconv);CHKERRQ(ierr);        
#endif
      }
    }
    
    /* compute restart vector */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      if (svd->which == SVD_SMALLEST) j = n-k-1;
      else j = k;
      ierr = VecSet(v,0.0);CHKERRQ(ierr);
      for (m=0;m<n;m++) swork[m] = PT[m*n+j];
      ierr = VecMAXPY(v,m,swork,svd->V+svd->nconv);CHKERRQ(ierr);
      ierr = VecCopy(v,svd->V[svd->nconv+k]);CHKERRQ(ierr);
    }
    
    /* copy converged singular vectors from temporary space */ 
    for (i=0;i<k;i++) {
      ierr = VecCopy(workV[i],svd->V[i+svd->nconv]);CHKERRQ(ierr);
      if (!lanczos->oneside) {
        ierr = VecCopy(workU[i],svd->U[i+svd->nconv]);CHKERRQ(ierr);
      }
    }
    
    svd->nconv += k;
    SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,svd->n);
  }
  
  /* sort singular triplets */
  ierr = PetscMalloc(sizeof(PetscInt)*svd->nconv,&perm);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(Vec)*svd->nconv,&permV);CHKERRQ(ierr);
  if (!lanczos->oneside) { ierr = PetscMalloc(sizeof(Vec)*svd->nconv,&permU);CHKERRQ(ierr); }
  for (i=0;i<svd->nconv;i++) {
    alpha[i] = svd->sigma[i];
    beta[i] = svd->errest[i];
    permV[i] = svd->V[i];
    if (!lanczos->oneside) permU[i] = svd->U[i];
    perm[i] = i;
  }
  ierr = PetscSortRealWithPermutation(svd->nconv,svd->sigma,perm);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    if (svd->which == SVD_SMALLEST) j = perm[i]; 
    else j = perm[svd->nconv-i-1];
    svd->sigma[i] = alpha[j];
    svd->errest[i] = beta[j];
    svd->V[i] = permV[j];
    if (!lanczos->oneside) svd->U[i] = permU[j];
  }
  
  /* free working space */
  ierr = VecDestroy(v);CHKERRQ(ierr);
  ierr = VecDestroy(wv);CHKERRQ(ierr);
  for (i=0;i<nwork;i++) { ierr = VecDestroy(workV[i]);CHKERRQ(ierr); }
  ierr = PetscFree(workV);CHKERRQ(ierr);
  if (lanczos->oneside) {
    ierr = VecDestroy(u);CHKERRQ(ierr);
    ierr = VecDestroy(u_1);CHKERRQ(ierr);
  } else {
    for (i=0;i<nwork;i++) { ierr = VecDestroy(workU[i]);CHKERRQ(ierr); }
    ierr = PetscFree(workU);CHKERRQ(ierr);
    ierr = PetscFree(permU);CHKERRQ(ierr);
    ierr = VecDestroy(wu);CHKERRQ(ierr);
  }
  ierr = PetscFree(alpha);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(PT);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscFree(swork);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  ierr = PetscFree(permV);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions_LANCZOS"
PetscErrorCode SVDSetFromOptions_LANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(svd->comm,svd->prefix,"LANCZOS Singular Value Solver Options","SVD");CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-svd_lanczos_oneside","Lanczos one-side reorthogonalization","SVDLanczosSetOneSide",PETSC_FALSE,&lanczos->oneside,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDLanczosSetOneSide_LANCZOS"
PetscErrorCode SVDLanczosSetOneSide_LANCZOS(SVD svd,PetscTruth oneside)
{
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;

  PetscFunctionBegin;
  if (lanczos->oneside != oneside) {
    lanczos->oneside = oneside;
    svd->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "SVDLanczosSetOneSide"
PetscErrorCode SVDLanczosSetOneSide(SVD svd,PetscTruth oneside)
{
  PetscErrorCode ierr, (*f)(SVD,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDLanczosSetOneSide_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(svd,oneside);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDView_LANCZOS"
PetscErrorCode SVDView_LANCZOS(SVD svd,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"Lanczos reorthogonalization: %s\n",lanczos->oneside ? "one-side" : "two-side");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCreate_LANCZOS"
PetscErrorCode SVDCreate_LANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos;

  PetscFunctionBegin;
  ierr = PetscNew(SVD_LANCZOS,&lanczos);CHKERRQ(ierr);
  PetscLogObjectMemory(svd,sizeof(SVD_LANCZOS));
  svd->data                = (void *)lanczos;
  svd->ops->setup          = SVDSetUp_LANCZOS;
  svd->ops->solve          = SVDSolve_LANCZOS;
  svd->ops->setfromoptions = SVDSetFromOptions_LANCZOS;
  svd->ops->view           = SVDView_LANCZOS;
  lanczos->oneside         = PETSC_FALSE;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDLanczosSetOneSide_C","SVDLanczosSetOneSide_LANCZOS",SVDLanczosSetOneSide_LANCZOS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
