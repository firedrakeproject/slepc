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

#undef __FUNCT__  
#define __FUNCT__ "SVDSolve_TRLANCZOS"
PetscErrorCode SVDSolve_TRLANCZOS(SVD svd)
{
  PetscErrorCode ierr;
  SVD_TRLANCZOS  *lanczos = (SVD_TRLANCZOS *)svd->data;
  PetscReal      *alpha,*beta,norm,*sigma,*errest;
  PetscScalar    *b,*Q,*PT;
  PetscInt       *perm,nrv,strategy=1;
  int            i,j,k,l,n=svd->n;
  Vec            *V,*U;
  char           *conv;
  PetscTruth     *wanted;
  
  PetscFunctionBegin;
  /* allocate working space */
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&alpha);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&beta);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&sigma);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*svd->n,&errest);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(char)*svd->n,&conv);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscTruth)*svd->n,&wanted);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n,&b);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*svd->n*svd->n,&PT);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(svd->V[0],svd->n+1,&V);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(svd->U[0],svd->n,&U);CHKERRQ(ierr);
  
  /* normalize start vector */
  ierr = VecCopy(svd->vec_initial,V[0]);CHKERRQ(ierr);
  ierr = VecNormalize(V[0],&norm);CHKERRQ(ierr);
  
  nrv = n / 2;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nrv",&nrv,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-strategy",&strategy,PETSC_NULL);CHKERRQ(ierr);
  
  l = 0;
  while (svd->reason == SVD_CONVERGED_ITERATING) {
    svd->its++;

    /* inner loop */
    for (i=l;i<n;i++) {
      svd->matvecs++;
      ierr = MatMult(svd->A,V[i],U[i]);CHKERRQ(ierr);
      if (lanczos->oneside) {
        if (i==l) {
          ierr = VecSet(svd->U[i],0.0);CHKERRQ(ierr);
          ierr = VecMAXPY(svd->U[i],l,b,U);CHKERRQ(ierr);
          ierr = VecAXPY(U[i],-1.0,svd->U[i]);CHKERRQ(ierr);
        } else {
          ierr = VecAXPY(U[i],-beta[i-1],U[i-1]);CHKERRQ(ierr);
        }
      } else {
        svd->dots += svd->nconv + i;
        ierr = IPOrthogonalize(svd->ip,svd->nconv,PETSC_NULL,svd->U,U[i],PT,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
        ierr = IPOrthogonalize(svd->ip,i,PETSC_NULL,U,U[i],PT,alpha+i,PETSC_NULL);CHKERRQ(ierr);
        ierr = VecScale(U[i],1.0/alpha[i]);CHKERRQ(ierr);
      }

      svd->matvecs++;
      if (svd->AT) {
	ierr = MatMult(svd->AT,U[i],V[i+1]);CHKERRQ(ierr);
      } else {
	ierr = MatMultTranspose(svd->A,U[i],V[i+1]);CHKERRQ(ierr);
      }
      svd->dots += svd->nconv + i + 1;
      ierr = IPOrthogonalize(svd->ip,svd->nconv,PETSC_NULL,svd->V,V[i+1],PT,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      if (lanczos->oneside) {
        ierr = VecNormBegin(U[i],NORM_2,alpha+i);CHKERRQ(ierr);
        ierr = VecMDotBegin(V[i+1],i+1,V,PT);CHKERRQ(ierr);
        ierr = VecNormEnd(U[i],NORM_2,alpha+i);CHKERRQ(ierr);
        ierr = VecMDotEnd(V[i+1],i+1,V,PT);CHKERRQ(ierr);
        
        ierr = VecScale(U[i],1.0/alpha[i]);CHKERRQ(ierr);
        ierr = VecScale(V[i+1],1.0/alpha[i]);CHKERRQ(ierr);
        for (j=0;j<=i;j++) PT[j] = - PT[j] / alpha[i];
        ierr = VecMAXPY(V[i+1],i+1,PT,V);CHKERRQ(ierr);

        ierr = IPOrthogonalizeGS(svd->ip,i+1,PETSC_NULL,V,V[i+1],PT,beta+i,PETSC_NULL);CHKERRQ(ierr);
        ierr = VecScale(V[i+1],1.0/beta[i]);CHKERRQ(ierr);
      } else {
        ierr = IPOrthogonalize(svd->ip,i+1,PETSC_NULL,V,V[i+1],PT,beta+i,PETSC_NULL);CHKERRQ(ierr);
        ierr = VecScale(V[i+1],1.0/beta[i]);CHKERRQ(ierr);
      }
    }

    /* compute SVD of general matrix */
    /* first l columns */
    for (j=0;j<l;j++) {
      for (i=0;i<j;i++) Q[j*n+i] = 0.0;    
      Q[j*n+j] = alpha[j];
      for (i=j+1;i<n;i++) Q[j*n+i] = 0.0;
    }
    /* l+1 column */
    for (i=0;i<l;i++) Q[l*n+i] = b[i];
    Q[l*n+l] = alpha[l];
    for (i=l+1;i<n;i++) Q[l*n+i] = 0.0;
    /* rest of matrix */
    for (j=l+1;j<n;j++) {
      for (i=0;i<j-1;i++) Q[j*n+i] = 0.0;
      Q[j*n+j-1] = beta[j-1];
      Q[j*n+j] = alpha[j];
      for (i=j+1;i<n;i++) Q[j*n+i] = 0.0;
    }
    ierr = SVDDense(n,n,Q,sigma,PETSC_NULL,PT);CHKERRQ(ierr);

    /* compute error estimates */
    for (i=0;i<n;i++) {
      errest[i] = PetscAbsScalar(Q[i*n+n-1]) * beta[n-1];
      if (sigma[i] > svd->tol) errest[i] /= sigma[i];
    }
    
    /* flag converged values and restart vectors */
    if (svd->which == SVD_SMALLEST) {
      k=0;
      switch (strategy) {
      case 1: 
        // aceptar los convergidos al final de la descomposicion
        // ignorar el resto de convergidos
        for (j=n-1;errest[j]<svd->tol && j>=0;j--,k++) conv[j] = 'C';
        for (;j>=0;j--) conv[j] = 'N';
        break;
      case 2: 
        // aceptar cualquier valor convergido
        for (i=0;i<n;i++)
          if (errest[i]<svd->tol) {
            conv[i] = 'C';
            k++;
          } else conv[i] ='N';
        break;
      case 3:
        // aceptar los convergidos al final de la descomposicion
        // hacer deflacion con los convergidos al principio de la descomposicion
        // ignorar el resto de convergidos      
        for (j=n-1;errest[j]<svd->tol && j>0;j--,k++) conv[j] = 'C';
        for (i=0;errest[i]<svd->tol && i<=j;i++,k++) conv[i] = 'U';
        for (l=i;l<=j;l++) conv[l] = 'N';
        break;
      case 4:
        // aceptar los convergidos al final de la descomposicion
        // hacer deflacion con el resto de convergidos      
        for (j=n-1;errest[j]<svd->tol && j>=0;j--,k++) conv[j] = 'C';
        for (i=0;i<=j;i++)
          if (errest[i]<svd->tol) {
            conv[i] = 'U';
            k++;
          } else conv[i] ='N';
        break;        
      }
      // reiniciar con los ultimos l valores no convergidos 
      l = PetscMin(nrv,svd->n-k-svd->nconv);
      for (i=n-1;i>=0 && l>0;i--)
        if (conv[i] == 'N') {
	  conv[i] = 'R';
	  l--;
	}
    } else {
      for (i=0,k=0;errest[i]<svd->tol && i<n;i++,k++) conv[i] = 'C';
      for (j=n-1;errest[j]<svd->tol && j>i;j--,k++) conv[j] = 'U';
      l = PetscMin(svd->n/2,svd->n-k-svd->nconv-1);
      for (k=i;k<i+l && k<=j;k++) conv[k] = 'R';
      for (k=i+l;k<=j;k++) conv[k] = 'N';
    }

    /* compute converged singular vectors */
    k = svd->nconv;
    for (i=0;i<n;i++)
      if (conv[i] == 'C' || conv[i] == 'U') {
        svd->sigma[k] = sigma[i];
        svd->errest[k] = errest[i];
        wanted[k] = (conv[i] == 'C') ? PETSC_TRUE : PETSC_FALSE;
        ierr = VecSet(svd->V[k],0.0);CHKERRQ(ierr);
        for (j=0;j<n;j++) {
          ierr = VecAXPY(svd->V[k],PT[j*n+i],V[j]);CHKERRQ(ierr);
        }      
        ierr = VecSet(svd->U[k],0.0);CHKERRQ(ierr);
        ierr = VecMAXPY(svd->U[k],n,Q+i*n,U);CHKERRQ(ierr);
        k++;
      }
    
    /* compute restart vectors */
    l = 0;
    for (i=0;i<n;i++)
      if (conv[i] == 'R') {
        svd->sigma[k+l] = sigma[i];
        svd->errest[k+l] = errest[i];
        alpha[l] = sigma[i];
        b[l] = Q[i*n+n-1]*beta[n-1];
        ierr = VecSet(svd->V[k+l],0.0);CHKERRQ(ierr);
        for (j=0;j<n;j++) {
          ierr = VecAXPY(svd->V[k+l],PT[j*n+i],V[j]);CHKERRQ(ierr);
        }
        ierr = VecSet(svd->U[k+l],0.0);CHKERRQ(ierr);
        ierr = VecMAXPY(svd->U[k+l],n,Q+i*n,U);CHKERRQ(ierr);
        l++;
      }
    
    j = k+l;
    for (i=0;i<n && j<n;i++)
      if (conv[i] == 'N') {
        svd->sigma[j] = alpha[i];
        svd->errest[j] = errest[i];
        j++;
      }

    /* copy restart vectors from temporary space */ 
    for (i=0;i<l;i++) {
      ierr = VecCopy(svd->V[k+i],V[i]);CHKERRQ(ierr);
      ierr = VecCopy(svd->U[k+i],U[i]);CHKERRQ(ierr);
    }
    
    /* copy the last vector to be the next initial vector */
    if (svd->reason == SVD_CONVERGED_ITERATING) {
      ierr = VecCopy(V[n],V[l]);CHKERRQ(ierr);
    }
    
    svd->nconv = k;
    SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,n);
  
    /* check stopping conditions */
    if (svd->its > svd->max_it) svd->reason = SVD_DIVERGED_ITS;
    for (i=0,k=0;i<svd->nconv;i++) if (wanted[i]) k++;
    if (k >= svd->nsv) svd->reason = SVD_CONVERGED_TOL;
  }
  
  /* sort singular triplets */
  ierr = PetscMalloc(sizeof(PetscInt)*svd->nconv,&perm);CHKERRQ(ierr);
  for (i=0;i<svd->nconv;i++) {
    alpha[i] = svd->sigma[i];
    beta[i] = svd->errest[i];
    perm[i] = i;
  }
  ierr = PetscSortRealWithPermutation(svd->nconv,alpha,perm);CHKERRQ(ierr);
  for (i=0,k=0;i<svd->nconv;i++) {
    if (svd->which == SVD_SMALLEST) j = perm[i]; 
    else j = perm[svd->nconv-i-1];
    if (wanted[j]) {
      svd->sigma[k] = alpha[j];
      svd->errest[k] = beta[j];
      ierr = VecCopy(svd->V[j],V[k]);CHKERRQ(ierr);
      ierr = VecCopy(svd->U[j],U[k]);CHKERRQ(ierr);
      k++;
    }
  }
  svd->nconv = k;
  for (i=0;i<svd->nconv;i++) {
    ierr = VecCopy(V[i],svd->V[i]);CHKERRQ(ierr);
    ierr = VecCopy(U[i],svd->U[i]);CHKERRQ(ierr);
  }
  
  /* free working space */
  ierr = VecDestroyVecs(V,svd->n+1);CHKERRQ(ierr);
  ierr = VecDestroyVecs(U,svd->n);CHKERRQ(ierr);

  ierr = PetscFree(alpha);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  ierr = PetscFree(sigma);CHKERRQ(ierr);
  ierr = PetscFree(errest);CHKERRQ(ierr);
  ierr = PetscFree(conv);CHKERRQ(ierr);
  ierr = PetscFree(wanted);CHKERRQ(ierr);
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
