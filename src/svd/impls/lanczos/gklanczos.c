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
  PetscInt        M,N;

  PetscFunctionBegin;
  ierr = MatGetSize(svd->A,&M,&N);CHKERRQ(ierr);
  if (svd->ncv == PETSC_DECIDE)
    svd->ncv = PetscMin(PetscMin(M,N),PetscMax(2*svd->nsv,10));
  if (svd->max_it == PETSC_DECIDE)
    svd->max_it = PetscMax(PetscMin(M,N)/svd->ncv,100);
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
  int            i,j,k,l,n,zero=0,info;
  Vec            *V,*U;
  PetscTruth     conv;
  
  PetscFunctionBegin;
  /* allocate working space */
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&alpha);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&beta);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&PT);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*4*svd->n,&work);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(svd->V[0],svd->n+1,&V);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(svd->U[0],svd->n,&U);CHKERRQ(ierr);
  
  /* normalize start vector */
  ierr = VecCopy(svd->vec_initial,V[0]);CHKERRQ(ierr);
  ierr = VecNormalize(V[0],&norm);CHKERRQ(ierr);
  
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    for (i=svd->nconv;i<svd->n;i++) {
      svd->matvecs++;
      ierr = MatMult(svd->A,V[i],U[i]);CHKERRQ(ierr);
      if (lanczos->oneside) {
        if (i>svd->nconv) { ierr = VecAXPY(U[i],-beta[i-svd->nconv-1],U[i-1]);CHKERRQ(ierr); }
      } else {
        svd->dots += i;
        ierr = IPOrthogonalize(svd->ip,i,PETSC_NULL,U,U[i],PT,alpha+i-svd->nconv,PETSC_NULL);CHKERRQ(ierr);
        ierr = VecScale(U[i],1.0/alpha[i-svd->nconv]);CHKERRQ(ierr);
      }

      svd->matvecs++;
      if (svd->AT) {
	ierr = MatMult(svd->AT,U[i],V[i+1]);CHKERRQ(ierr);
      } else {
	ierr = MatMultTranspose(svd->A,U[i],V[i+1]);CHKERRQ(ierr);
      }
      if (lanczos->oneside) {
        svd->dots += i+1;
        ierr = VecNormBegin(U[i],NORM_2,alpha+i-svd->nconv);CHKERRQ(ierr);
        ierr = VecMDotBegin(V[i+1],i+1,V,PT);CHKERRQ(ierr);
        ierr = VecNormEnd(U[i],NORM_2,alpha+i-svd->nconv);CHKERRQ(ierr);
        ierr = VecMDotEnd(V[i+1],i+1,V,PT);CHKERRQ(ierr);
        
        ierr = VecScale(U[i],1.0/alpha[i-svd->nconv]);CHKERRQ(ierr);
        ierr = VecScale(V[i+1],1.0/alpha[i-svd->nconv]);CHKERRQ(ierr);
        for (j=0;j<=i;j++) PT[j] = - PT[j] / alpha[i-svd->nconv];
        ierr = VecMAXPY(V[i+1],i+1,PT,V);CHKERRQ(ierr);

        ierr = IPOrthogonalizeGS(svd->ip,i+1,PETSC_NULL,V,V[i+1],PT,PETSC_NULL,beta+i-svd->nconv);CHKERRQ(ierr);
        ierr = VecScale(V[i+1],1.0/beta[i-svd->nconv]);CHKERRQ(ierr);
      } else {
        svd->dots += i+1;
        ierr = IPOrthogonalize(svd->ip,i+1,PETSC_NULL,V,V[i+1],PT,beta+i-svd->nconv,PETSC_NULL);CHKERRQ(ierr);
        ierr = VecScale(V[i+1],1.0/beta[i-svd->nconv]);CHKERRQ(ierr);
      }
    }

    /* compute SVD of bidiagonal matrix */
    n = svd->n - svd->nconv;
    ierr = PetscMemzero(PT,sizeof(PetscScalar)*n*n);CHKERRQ(ierr);
    ierr = PetscMemzero(Q,sizeof(PetscScalar)*n*n);CHKERRQ(ierr);
    for (i=0;i<n;i++)
      PT[i*n+i] = Q[i*n+i] = 1.0;
    LAPACKbdsqr_("U",&n,&n,&n,&zero,alpha,beta,PT,&n,Q,&n,PETSC_NULL,&n,work,&info,1);

    /* compute error estimates and converged singular vectors */
    k = svd->nconv;
    conv = PETSC_TRUE;
    for (i=svd->nconv;i<svd->n;i++) {
      if (svd->which == SVD_SMALLEST) j = n-i+svd->nconv-1;
      else j = i-svd->nconv;
      svd->sigma[i] = alpha[j];
      svd->errest[i] = PetscAbsReal(Q[j*n+n-1])*beta[n-1] / alpha[j];
      if (conv) {
        if (svd->errest[i] < svd->tol) {
          ierr = VecSet(svd->V[i],0.0);CHKERRQ(ierr);
          for (l=0;l<n;l++) {
            ierr = VecAXPY(svd->V[i],PT[l*n+j],V[l+svd->nconv]);CHKERRQ(ierr);
          }      
          ierr = VecSet(svd->U[i],0.0);CHKERRQ(ierr);
          ierr = VecMAXPY(svd->U[i],n,Q+j*n,U+svd->nconv);CHKERRQ(ierr);
          k++;
        } else conv = PETSC_FALSE;
      }
    }
    
    if (svd->its > svd->max_it) svd->reason = SVD_DIVERGED_ITS;
    if (k >= svd->nsv) svd->reason = SVD_CONVERGED_TOL;
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      /* compute restart vector */
      if (svd->which == SVD_SMALLEST) j = n-k+svd->nconv-1;
      else j = k-svd->nconv;
      ierr = VecSet(svd->V[k],0.0);CHKERRQ(ierr);
      for (l=0;l<n;l++) {
	ierr = VecAXPY(svd->V[k],PT[l*n+j],V[l+svd->nconv]);CHKERRQ(ierr);
      }      
      ierr = VecCopy(svd->V[k],V[k]);CHKERRQ(ierr);
    }
    
    /* copy converged singular vectors from temporary space */ 
    for (i=svd->nconv;i<k;i++) {
      ierr = VecCopy(svd->V[i],V[i]);CHKERRQ(ierr);
      ierr = VecCopy(svd->U[i],U[i]);CHKERRQ(ierr);
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
  ierr = VecDestroyVecs(V,svd->n+1);CHKERRQ(ierr);
  ierr = VecDestroyVecs(U,svd->n);CHKERRQ(ierr);

  ierr = PetscFree(alpha);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(PT);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
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
  lanczos->oneside = oneside;
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
