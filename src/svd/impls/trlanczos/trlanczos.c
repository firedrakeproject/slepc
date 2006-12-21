/*                       

   SLEPc singular value solver: "trlanczos"

   Method: Golub-Kahan-Lanczos bidiagonalization with thick-restart

   Last update: Nov 2006

*/
#include "src/svd/svdimpl.h"                /*I "slepcsvd.h" I*/
#include "slepcblaslapack.h"

typedef struct {
  PetscTruth oneside;
} SVD_TRLANCZOS;

#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_TRLANCZOS"
PetscErrorCode SVDSetUp_TRLANCZOS(SVD svd)
{
  PetscErrorCode  ierr;
  PetscInt        M,N;

  PetscFunctionBegin;
  ierr = MatGetSize(svd->A,&M,&N);CHKERRQ(ierr);
  if (svd->ncv == PETSC_DECIDE)
    svd->ncv = PetscMin(PetscMin(M,N),PetscMax(2*svd->nsv,10));
  if (svd->max_it == PETSC_DECIDE)
    svd->max_it = PetscMax(PetscMin(M,N)/svd->ncv,100);
  if (svd->nsv >= svd->ncv)
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"nsv bigger or equal than ncv");
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode cgs(Vec,int,Vec*,PetscReal*);

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_TRLANCZOS"
PetscErrorCode SVDSolve_TRLANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS *)svd->data;
  PetscReal      *alpha,*beta,norm;
  PetscScalar    *b,*Q,*PT;
  PetscInt       *perm;
  int            i,j,k,l,m,n;
  Vec            *V,*U;
  
  PetscFunctionBegin;
  /* allocate working space */
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&alpha);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&beta);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n,&b);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&PT);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(svd->V[0],svd->n+1,&V);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(svd->U[0],svd->n,&U);CHKERRQ(ierr);
  
  /* normalize start vector */
  ierr = VecCopy(svd->vec_initial,V[0]);CHKERRQ(ierr);
  ierr = VecNormalize(V[0],&norm);CHKERRQ(ierr);
  
  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    for (i=svd->nconv+l;i<svd->n;i++) {
      ierr = MatMult(svd->A,V[i],U[i]);CHKERRQ(ierr);
      if (lanczos->oneside) {
        if (i==svd->nconv+l) {
          ierr = VecSet(svd->U[i],0.0);CHKERRQ(ierr);
          ierr = VecMAXPY(svd->U[i],l,b+svd->nconv,U+svd->nconv);CHKERRQ(ierr);
          ierr = VecAXPY(U[i],-1.0,svd->U[i]);CHKERRQ(ierr);
        } else {
          ierr = VecAXPY(U[i],-beta[i-svd->nconv-1],U[i-1]);CHKERRQ(ierr);
        }
      } else {
        ierr = cgs(U[i],i,U,PETSC_NULL);CHKERRQ(ierr);
        ierr = cgs(U[i],i,U,alpha+i-svd->nconv);CHKERRQ(ierr);
        ierr = VecScale(U[i],1.0/alpha[i-svd->nconv]);CHKERRQ(ierr);
      }

      if (svd->AT) {
	ierr = MatMult(svd->AT,U[i],V[i+1]);CHKERRQ(ierr);
      } else {
	ierr = MatMultTranspose(svd->A,U[i],V[i+1]);CHKERRQ(ierr);
      }
      if (lanczos->oneside) {
        ierr = VecNormBegin(U[i],NORM_2,alpha+i-svd->nconv);CHKERRQ(ierr);
        ierr = VecMDotBegin(V[i+1],i+1,V,PT);CHKERRQ(ierr);
        ierr = VecNormEnd(U[i],NORM_2,alpha+i-svd->nconv);CHKERRQ(ierr);
        ierr = VecMDotEnd(V[i+1],i+1,V,PT);CHKERRQ(ierr);
        
        ierr = VecScale(U[i],1.0/alpha[i-svd->nconv]);CHKERRQ(ierr);
        ierr = VecScale(V[i+1],1.0/alpha[i-svd->nconv]);CHKERRQ(ierr);
        for (j=0;j<=i;j++) PT[j] = - PT[j] / alpha[i-svd->nconv];
        ierr = VecMAXPY(V[i+1],i+1,PT,V);CHKERRQ(ierr);

        ierr = cgs(V[i+1],i+1,V,beta+i-svd->nconv);CHKERRQ(ierr);
        ierr = VecScale(V[i+1],1.0/beta[i-svd->nconv]);CHKERRQ(ierr);
      } else {
        ierr = cgs(V[i+1],i+1,V,PETSC_NULL);CHKERRQ(ierr);
        ierr = cgs(V[i+1],i+1,V,beta+i-svd->nconv);CHKERRQ(ierr);
        ierr = VecScale(V[i+1],1.0/beta[i-svd->nconv]);CHKERRQ(ierr);
      }
    }

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
    Q[l*n+l] = alpha[l];
    for (i=l+1;i<n;i++) Q[l*n+i] = 0.0;
    /* rest of matrix */
    for (j=l+1;j<n;j++) {
      for (i=0;i<j-1;i++) Q[j*n+i] = 0.0;
      Q[j*n+j-1] = beta[j-1];
      Q[j*n+j] = alpha[j];
      for (i=j+1;i<n;i++) Q[j*n+i] = 0.0;
    }
    ierr = SVDDense(n,n,Q,alpha,PETSC_NULL,PT);CHKERRQ(ierr);

    /* compute error estimates */
    for (i=svd->nconv;i<svd->n;i++) {
      if (svd->which == SVD_SMALLEST) j = n-i+svd->nconv-1;
      else j = i-svd->nconv;
      svd->sigma[i] = alpha[j];
      b[i] = Q[j*n+n-1]*beta[n-1];
      svd->errest[i] = PetscAbsReal(b[i]);
    }
    
    /* check convergence and update l */
    k = svd->nconv;
    while (svd->errest[k] < svd->tol && k<svd->n) k++;
    if (svd->its > svd->max_it) svd->reason = SVD_DIVERGED_ITS;
    if (k >= svd->nsv) svd->reason = SVD_CONVERGED_TOL;
    if (svd->reason != SVD_CONVERGED_ITERATING) l = 0;
    else l = PetscMax((svd->n - k) / 2,1);
    
    /* compute converged singular vectors and restart vectors*/
    for (i=svd->nconv;i<k+l;i++) {
      if (svd->which == SVD_SMALLEST) j = n-i+svd->nconv-1;
      else j = i-svd->nconv;
      ierr = VecSet(svd->V[i],0.0);CHKERRQ(ierr);
      for (m=0;m<n;m++) {
        ierr = VecAXPY(svd->V[i],PT[m*n+j],V[m+svd->nconv]);CHKERRQ(ierr);
      }      
      ierr = VecSet(svd->U[i],0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(svd->U[i],n,Q+j*n,U+svd->nconv);CHKERRQ(ierr);
    }
       
    /* copy converged singular vectors and restart vectors from temporary space */ 
    for (i=svd->nconv;i<k+l;i++) {
      ierr = VecCopy(svd->V[i],V[i]);CHKERRQ(ierr);
      ierr = VecCopy(svd->U[i],U[i]);CHKERRQ(ierr);
    }
    
    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      ierr = VecCopy(V[svd->n],V[k+l]);CHKERRQ(ierr);
    }
    
    svd->nconv = k;
    SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,svd->n);
  }
  
  /* sort singular triplets */
  ierr = PetscMalloc(sizeof(PetscInt)*svd->nconv,&perm);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    alpha[i] = svd->sigma[i];
    beta[i] = svd->errest[i];
    perm[i] = i;
  }
  ierr = PetscSortRealWithPermutation(svd->nconv,svd->sigma,perm);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    if (svd->which == SVD_SMALLEST) j = perm[i]; 
    else j = perm[svd->nconv-i-1]; 
    svd->sigma[i] = alpha[j];
    svd->errest[i] = beta[j];
    ierr = VecCopy(V[j],svd->V[i]);CHKERRQ(ierr);
    ierr = VecCopy(U[j],svd->U[i]);CHKERRQ(ierr);
  }
  
  /* free working space */
  ierr = VecDestroyVecs(V,n+1);CHKERRQ(ierr);
  ierr = VecDestroyVecs(U,n);CHKERRQ(ierr);

  ierr = PetscFree(alpha);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(PT);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSetFromOptions_TRLANCZOS"
PetscErrorCode SVDSetFromOptions_TRLANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS *)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(svd->comm,svd->prefix,"TRLANCZOS Singular Value Solver Options","SVD");CHKERRQ(ierr);
  ierr = PetscOptionsName("-svd_trlanczos_oneside","Lanczos one-side reorthogonalization","SVDLanczosSetOneSideReorthogonalization",&lanczos->oneside);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_BEGIN

#undef __FUNCT__  
#define __FUNCT__ "SVDTRLanczosSetOneSideReorthogonalization_TRLANCZOS"
PetscErrorCode SVDTRLanczosSetOneSideReorthogonalization_TRLANCZOS(SVD svd,PetscTruth oneside)
{
  SVD_TRLANCZOS *lanczos = (SVD_TRLANCZOS *)svd->data;

  PetscFunctionBegin;
  lanczos->oneside = oneside;
  PetscFunctionReturn(0);
}
EXTERN_C_BEGIN

#undef __FUNCT__
#define __FUNCT__ "SVDTRLanczosSetOneSideReorthogonalization"
PetscErrorCode SVDTRLanczosSetOneSideReorthogonalization(SVD svd,PetscTruth oneside)
{
  PetscErrorCode ierr, (*f)(SVD,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)svd,"SVDTRLanczosSetOneSideReorthogonalization_C",(void (**)())&f);CHKERRQ(ierr);
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
  svd->ops->setfromoptions = SVDSetFromOptions_TRLANCZOS;
  svd->ops->view           = SVDView_TRLANCZOS;
  lanczos->oneside         = PETSC_FALSE;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)svd,"SVDTRLanczosSetOneSideReorthogonalization_C","SVDTRLanczosSetOneSideReorthogonalization_TRLANCZOS",SVDTRLanczosSetOneSideReorthogonalization_TRLANCZOS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
