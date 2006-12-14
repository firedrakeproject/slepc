/*                       

   SLEPc singular value solver: "lanczos"

   Method: Golub-Kahan-Lanczos bidiagonalization

   Last update: Nov 2006

*/
#include "src/svd/svdimpl.h"                /*I "slepcsvd.h" I*/
#include "slepcblaslapack.h"

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
#define __FUNCT__ "cgs"
PetscErrorCode cgs(Vec v, int n, Vec *V)
{
  PetscErrorCode ierr;
  Vec            w;
  PetscScalar    *h;
  
  PetscFunctionBegin;
  if (n>0) {
    ierr = VecDuplicate(v,&w);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar)*n,&h);CHKERRQ(ierr);

    ierr = VecMDot(v,n,V,h);CHKERRQ(ierr);
    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    ierr = VecMAXPY(w,n,h,V);CHKERRQ(ierr);
    ierr = VecAXPY(v,-1.0,w);CHKERRQ(ierr);

    ierr = VecDestroy(w);CHKERRQ(ierr);
    ierr = PetscFree(h);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_LANCZOS"
PetscErrorCode SVDSolve_LANCZOS(SVD svd)
{
  PetscErrorCode ierr;
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
      ierr = MatMult(svd->A,V[i],U[i]);CHKERRQ(ierr);
      ierr = cgs(U[i],i,U);CHKERRQ(ierr);
      ierr = cgs(U[i],i,U);CHKERRQ(ierr);
      ierr = VecNormalize(U[i],alpha+i-svd->nconv);CHKERRQ(ierr);

      if (svd->AT) {
	ierr = MatMult(svd->AT,U[i],V[i+1]);CHKERRQ(ierr);
      } else {
	ierr = MatMultTranspose(svd->A,U[i],V[i+1]);CHKERRQ(ierr);
      }
      ierr = cgs(V[i+1],i+1,V);CHKERRQ(ierr);
      ierr = cgs(V[i+1],i+1,V);CHKERRQ(ierr);
      ierr = VecNormalize(V[i+1],beta+i-svd->nconv);CHKERRQ(ierr);    
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
      svd->errest[i] = PetscAbsReal(Q[j*n+n-1])*beta[n-1];
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
    if (svd->nconv >= svd->nsv) svd->reason = SVD_CONVERGED_TOL;
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
  ierr = VecDestroyVecs(V,n+1);CHKERRQ(ierr);
  ierr = VecDestroyVecs(U,n);CHKERRQ(ierr);

  ierr = PetscFree(alpha);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(PT);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SVDCreate_LANCZOS"
PetscErrorCode SVDCreate_LANCZOS(SVD svd)
{
  PetscFunctionBegin;
  svd->ops->setup = SVDSetUp_LANCZOS;
  svd->ops->solve = SVDSolve_LANCZOS;
  PetscFunctionReturn(0);
}
EXTERN_C_END
