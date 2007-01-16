/*                       

   SLEPc singular value solver: "lanczos"

   Method: Golub-Kahan-Lanczos bidiagonalization

   Last update: Nov 2006

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
static PetscErrorCode SVDTwoSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,Vec *V,Vec v,Vec *U,int k,int n,PetscScalar* work)
{
  PetscErrorCode ierr;
  int            i;
  
  PetscFunctionBegin;
  for (i=k;i<n;i++) {
    ierr = SVDMatMult(svd,PETSC_FALSE,V[i],U[i]);CHKERRQ(ierr);
    svd->dots += i;
    ierr = IPOrthogonalize(svd->ip,i,PETSC_NULL,U,U[i],work,alpha+i-k,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecScale(U[i],1.0/alpha[i-k]);CHKERRQ(ierr);
    ierr = SVDMatMult(svd,PETSC_TRUE,U[i],v);CHKERRQ(ierr);
    svd->dots += i+1;
    ierr = IPOrthogonalize(svd->ip,i+1,PETSC_NULL,V,v,work,beta+i-k,PETSC_NULL);CHKERRQ(ierr);
    if (i<n-1) {
      ierr = VecCopy(v,V[i+1]);CHKERRQ(ierr);
      ierr = VecScale(V[i+1],1.0/beta[i-k]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDOneSideLanczos"
static PetscErrorCode SVDOneSideLanczos(SVD svd,PetscReal *alpha,PetscReal *beta,Vec *V,Vec v,Vec u,Vec u_1,int k,int n,PetscScalar* work)
{
  PetscErrorCode ierr;
  int            i,j;
  
  PetscFunctionBegin;
  for (i=k;i<n;i++) {
    ierr = SVDMatMult(svd,PETSC_FALSE,V[i],u);CHKERRQ(ierr);
    if (i>k) { ierr = VecAXPY(u,-beta[i-k-1],u_1);CHKERRQ(ierr); }
    
    ierr = SVDMatMult(svd,PETSC_TRUE,u,v);CHKERRQ(ierr);
    
    svd->dots += i+1;
    ierr = VecNormBegin(u,NORM_2,alpha+i-k);CHKERRQ(ierr);
    ierr = VecMDotBegin(v,i+1,V,work);CHKERRQ(ierr);
    ierr = VecNormEnd(u,NORM_2,alpha+i-k);CHKERRQ(ierr);
    ierr = VecMDotEnd(v,i+1,V,work);CHKERRQ(ierr);
    
    ierr = VecScale(u,1.0/alpha[i-k]);CHKERRQ(ierr);
    ierr = VecCopy(u,u_1);CHKERRQ(ierr);
    ierr = VecScale(v,1.0/alpha[i-k]);CHKERRQ(ierr);
    for (j=0;j<=i;j++) work[j] = - work[j] / alpha[i-k];
    ierr = VecMAXPY(v,i+1,work,V);CHKERRQ(ierr);

    ierr = IPOrthogonalizeGS(svd->ip,i+1,PETSC_NULL,V,v,work,PETSC_NULL,beta+i-k);CHKERRQ(ierr);
    if (i<n-1) {
      ierr = VecCopy(v,V[i+1]);CHKERRQ(ierr);
      ierr = VecScale(V[i+1],1.0/beta[i-k]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_LANCZOS"
PetscErrorCode SVDSolve_LANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;
  PetscReal      *alpha,*beta,norm,*work;
  PetscScalar    *Q,*PT;
  PetscInt       *perm;
  int            i,j,k,l,n,zero=0,info,nwork=0;
  Vec            v,u,u_1,*workV,*workU,*permV,*permU;
  PetscTruth     conv;
  
  PetscFunctionBegin;
  /* allocate working space */
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&alpha);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&beta);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&PT);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*4*svd->n,&work);CHKERRQ(ierr);
  ierr = VecDuplicate(svd->V[0],&v);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(Vec)*svd->n,&workV);CHKERRQ(ierr);
  if (lanczos->oneside) {
    ierr = SVDMatGetVecs(svd,PETSC_NULL,&u);CHKERRQ(ierr);
    ierr = SVDMatGetVecs(svd,PETSC_NULL,&u_1);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc(sizeof(Vec)*svd->n,&workU);CHKERRQ(ierr);
  }
  
  /* normalize start vector */
  ierr = VecCopy(svd->vec_initial,svd->V[0]);CHKERRQ(ierr);
  ierr = VecNormalize(svd->V[0],&norm);CHKERRQ(ierr);
  
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    if (lanczos->oneside) {
      ierr = SVDOneSideLanczos(svd,alpha,beta,svd->V,v,u,u_1,svd->nconv,svd->n,PT);CHKERRQ(ierr);
    } else {
      ierr = SVDTwoSideLanczos(svd,alpha,beta,svd->V,v,svd->U,svd->nconv,svd->n,PT);CHKERRQ(ierr);
    }

    /* compute SVD of bidiagonal matrix */
    n = svd->n - svd->nconv;
    ierr = PetscMemzero(PT,sizeof(PetscScalar)*n*n);CHKERRQ(ierr);
    ierr = PetscMemzero(Q,sizeof(PetscScalar)*n*n);CHKERRQ(ierr);
    for (i=0;i<n;i++)
      PT[i*n+i] = Q[i*n+i] = 1.0;
    LAPACKbdsqr_("U",&n,&n,&n,&zero,alpha,beta,PT,&n,Q,&n,PETSC_NULL,&n,work,&info,1);

    /* compute error estimates and converged singular vectors */
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
    
    /* allocate work space */
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
      for (l=0;l<n;l++) {
        ierr = VecAXPY(workV[i],PT[l*n+j],svd->V[l+svd->nconv]);CHKERRQ(ierr);
      }      
      if (!lanczos->oneside) {
        ierr = VecSet(workU[i],0.0);CHKERRQ(ierr);
        ierr = VecMAXPY(workU[i],n,Q+j*n,svd->U+svd->nconv);CHKERRQ(ierr);
      }
    }

    if (svd->its > svd->max_it) svd->reason = SVD_DIVERGED_ITS;
    if (svd->nconv+k >= svd->nsv) svd->reason = SVD_CONVERGED_TOL;
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      /* compute restart vector */
      if (svd->which == SVD_SMALLEST) j = n-k-1;
      else j = k;
      ierr = VecSet(v,0.0);CHKERRQ(ierr);
      for (l=0;l<n;l++) {
	ierr = VecAXPY(v,PT[l*n+j],svd->V[l+svd->nconv]);CHKERRQ(ierr);
      }      
      ierr = VecCopy(v,svd->V[k+svd->nconv]);CHKERRQ(ierr);
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
  for (i=0;i<nwork;i++) { ierr = VecDestroy(workV[i]);CHKERRQ(ierr); }
  ierr = PetscFree(workV);CHKERRQ(ierr);
  if (lanczos->oneside) {
    ierr = VecDestroy(u);CHKERRQ(ierr);
    ierr = VecDestroy(u_1);CHKERRQ(ierr);
  } else {
    for (i=0;i<nwork;i++) { ierr = VecDestroy(workU[i]);CHKERRQ(ierr); }
    ierr = PetscFree(workU);CHKERRQ(ierr);
  }
  ierr = PetscFree(alpha);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(PT);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  ierr = PetscFree(permV);CHKERRQ(ierr);
  if (!lanczos->oneside) { ierr = PetscFree(permU);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions_LANCZOS"
PetscErrorCode SVDSetFromOptions_LANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(svd->comm,svd->prefix,"LANCZOS Singular Value Solver Options","SVD");CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-svd_lanczos_oneside","Lanczos one-side reorthogonalization","SVDLanczosSetOneSideReorthogonalization",PETSC_FALSE,&lanczos->oneside,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_BEGIN

#undef __FUNCT__  
#define __FUNCT__ "SVDLanczosSetOneSideReorthogonalization_LANCZOS"
PetscErrorCode SVDLanczosSetOneSideReorthogonalization_LANCZOS(SVD svd,PetscTruth oneside)
{
  SVD_LANCZOS    *lanczos = (SVD_LANCZOS *)svd->data;

  PetscFunctionBegin;
  if (lanczos->oneside != oneside) {
    lanczos->oneside = oneside;
    svd->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_BEGIN

#undef __FUNCT__
#define __FUNCT__ "SVDLanczosSetOneSideReorthogonalization"
PetscErrorCode SVDLanczosSetOneSideReorthogonalization(SVD svd,PetscTruth oneside)
{
  PetscErrorCode ierr, (*f)(SVD,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDLanczosSetOneSideReorthogonalization_C",(void (**)())&f);CHKERRQ(ierr);
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
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDLanczosSetOneSideReorthogonalization_C","SVDLanczosSetOneSideReorthogonalization_LANCZOS",SVDLanczosSetOneSideReorthogonalization_LANCZOS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
