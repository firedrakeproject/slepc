/*                       

   SLEPc singular value solver: "lanczos"

   Method: Golub-Kaham-Lanczos bidiagonalization

   Last update: Nov 2006

*/
#include "src/svd/svdimpl.h"                /*I "slepcsvd.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "SVDSetUp_LANCZOS"
PetscErrorCode SVDSetUp_LANCZOS(SVD svd)
{
  PetscErrorCode  ierr;
  PetscInt        M,N;

  PetscFunctionBegin;
  ierr = MatGetSize(svd->A,&M,&N);CHKERRQ(ierr);
  if (svd->ncv == PETSC_DECIDE)
    svd->ncv = PetscMin(PetscMin(M,N),10);
  if (svd->max_it == PETSC_DEFAULT)
    svd->max_it = PetscMax(PetscMax(M,N)/svd->ncv,10);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "cgs2"
PetscErrorCode cgs2(Vec v, int n, Vec *V)
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
#define __FUNCT__ "computeres"
PetscErrorCode computeres(SVD svd,PetscScalar sigma,Vec u,Vec v,PetscReal *norm1,PetscReal *norm2)
{
  PetscErrorCode ierr;
  Vec            x,y;
  
  PetscFunctionBegin;
  ierr = VecDuplicate(u,&x);CHKERRQ(ierr);
  ierr = MatMult(svd->A,v,x);CHKERRQ(ierr);
  ierr = VecAXPY(x,-sigma,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,norm1);CHKERRQ(ierr);

  ierr = VecDuplicate(v,&y);CHKERRQ(ierr);
  if (svd->AT) {
    ierr = MatMult(svd->AT,u,y);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(svd->A,u,y);CHKERRQ(ierr);
  }
  ierr = VecAXPY(y,-sigma,v);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,norm2);CHKERRQ(ierr);

  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_LANCZOS"
PetscErrorCode SVDSolve_LANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  PetscReal      *alpha,*beta,norm1,norm2,*work;
  PetscScalar    *Q,*PT;
  int            i,j,k,n,zero=0,info;
  Vec            *V,*U;
  
  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&alpha);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&beta);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&PT);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*4*svd->n,&work);CHKERRQ(ierr);
  
  ierr = VecDuplicateVecs(svd->V[0],svd->n+1,&V);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(svd->U[0],svd->n,&U);CHKERRQ(ierr);
  
  ierr = VecCopy(svd->vec_initial,V[0]);CHKERRQ(ierr);
  
  svd->nconv = 0;
  for (svd->its=1;svd->its<svd->max_it;svd->its++) {
    n = svd->n - svd->nconv;
    for (i=svd->nconv;i<svd->n;i++) {
      ierr = MatMult(svd->A,V[i],U[i]);CHKERRQ(ierr);
      ierr = cgs2(U[i],i,U);CHKERRQ(ierr);
      ierr = VecNormalize(U[i],alpha+i-svd->nconv);CHKERRQ(ierr);

      if (svd->AT) {
	ierr = MatMult(svd->AT,U[i],V[i+1]);CHKERRQ(ierr);
      } else {
	ierr = MatMultTranspose(svd->A,U[i],V[i+1]);CHKERRQ(ierr);
      }
      ierr = cgs2(V[i+1],i+1,V);CHKERRQ(ierr);
      ierr = VecNormalize(V[i+1],beta+i-svd->nconv);CHKERRQ(ierr);    
    }

    ierr = PetscMemzero(PT,sizeof(PetscScalar)*n*n);CHKERRQ(ierr);
    ierr = PetscMemzero(Q,sizeof(PetscScalar)*n*n);CHKERRQ(ierr);
    for (i=0;i<n;i++)
      PT[i*n+i] = Q[i*n+i] = 1.0;
    dbdsqr_("U",&n,&n,&n,&zero,alpha,beta,PT,&n,Q,&n,PETSC_NULL,&n,work,&info,1);

    k = svd->nconv;
    for (i=svd->nconv;i<svd->n;i++) {
      svd->sigma[i] = alpha[i-svd->nconv];
      
      ierr = VecSet(svd->V[i],0.0);CHKERRQ(ierr);
      for (j=0;j<n;j++) {
	ierr = VecAXPY(svd->V[i],PT[j*n+i-svd->nconv],V[j+svd->nconv]);CHKERRQ(ierr);
      }
      
      ierr = VecSet(svd->U[i],0.0);CHKERRQ(ierr);
      ierr = VecMAXPY(svd->U[i],n,Q+(i-svd->nconv)*n,U+svd->nconv);CHKERRQ(ierr);
      
      ierr = computeres(svd,svd->sigma[i],svd->U[i],svd->V[i],&norm1,&norm2);CHKERRQ(ierr);
      printf("[%i] sigma[%i] = %g error = %g,%g\n",svd->its,i,svd->sigma[i],norm1,norm2);
      if (norm1+norm2 < svd->tol) {
        k++;
      } else break;
    }
    svd->nconv = k;
    if (svd->nconv >= svd->nsv) break;
    ierr = VecCopy(svd->V[svd->nconv],V[svd->nconv]);CHKERRQ(ierr);
  }
  
  ierr = VecDestroyVecs(V,n+1);CHKERRQ(ierr);
  ierr = VecDestroyVecs(U,n);CHKERRQ(ierr);

  ierr = PetscFree(alpha);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(PT);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
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
