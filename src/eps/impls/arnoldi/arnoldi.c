/*                       

   SLEPc eigensolver: "arnoldi"

   Method: Explicitly Restarted Arnoldi

   Description:

       This solver implements the Arnoldi method with explicit restart
       and deflation.

   Algorithm:

       The implemented algorithm builds an Arnoldi factorization of order
       ncv. Converged eigenpairs are locked and the iteration is restarted
       with the rest of the columns being the active columns for the next
       Arnoldi factorization. Currently, no filtering is applied to the
       vector used for restarting.

   References:

       [1] Z. Bai et al. (eds.), "Templates for the Solution of Algebraic
       Eigenvalue Problems: A Practical Guide", SIAM (2000), pp 161-165.

       [2] Y. Saad, "Numerical Methods for Large Eigenvalue Problems",
       John Wiley (1992), pp 172-183.

   Last update: June 2004

*/
#include "src/eps/epsimpl.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_ARNOLDI"
PetscErrorCode EPSSetUp_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  int            N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMax(2*eps->nev,eps->nev+10);
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  if (eps->T) { ierr = PetscFree(eps->T);CHKERRQ(ierr); }  
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,eps->ncv+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi"
/*
   EPSBasicArnoldi - Computes an m-step Arnoldi factorization. The first k
   columns are assumed to be locked and therefore they are not modified. On
   exit, the following relation is satisfied:

                    OP * V - V * H = f * e_m^T

   where the columns of V are the Arnoldi vectors (which are B-orthonormal),
   H is an upper Hessenberg matrix, f is the next vector (v_{m+1}) and e_m
   is the m-th vector of the canonical basis. On exit, beta contains the
   B-norm of f.
*/
static PetscErrorCode EPSBasicArnoldi(EPS eps,PetscScalar *H,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            j;
  PetscReal      norm;
  PetscScalar    t;
  PetscTruth     breakdown;

  PetscFunctionBegin;
  for (j=k;j<m-1;j++) {
    ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,V[j+1],PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,j+1,V,V[j+1],H+m*j,&norm,&breakdown);CHKERRQ(ierr);
    H[(m+1)*j+1] = norm;
    if (breakdown) {
      PetscLogInfo(eps,"Breakdown in Arnoldi method (norm=%g)",norm);
      ierr = SlepcVecSetRandom(V[j+1]);CHKERRQ(ierr);
      ierr = STNorm(eps->OP,V[j+1],&norm);CHKERRQ(ierr);
    }
    t = 1 / norm;
    ierr = VecScale(&t,V[j+1]);CHKERRQ(ierr);
  }
  ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr);
  ierr = EPSOrthogonalize(eps,m,V,f,H+m*(m-1),beta,PETSC_NULL);CHKERRQ(ierr);
  t = 1 / *beta;
  ierr = VecScale(&t,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_ARNOLDI"
PetscErrorCode EPSSolve_ARNOLDI(EPS eps)
{
  PetscErrorCode ierr;
  int            i,k,mout,info,ifst,ilst,ncv=eps->ncv;
  Vec            f=eps->work[ncv];
  PetscScalar    *H=eps->T,*U,*Y,*work,t;
  PetscReal      norm,beta;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif

  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY)
  SETERRQ(PETSC_ERR_SUP,"TREVC - Lapack routine is unavailable.");
#endif 
  ierr = PetscMemzero(H,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&U);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(3*ncv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif

  /* The first Arnoldi vector is the normalized initial vector */
  ierr = VecCopy(eps->vec_initial,eps->V[0]);CHKERRQ(ierr);
  ierr = STNorm(eps->OP,eps->V[0],&norm);CHKERRQ(ierr);
  t = 1 / norm;
  ierr = VecScale(&t,eps->V[0]);CHKERRQ(ierr);
  
  eps->nconv = 0;
  eps->its = 0;
  /* Restart loop */
  while (eps->its<eps->max_it) {
    /* Compute an ncv-step Arnoldi factorization */
    ierr = EPSBasicArnoldi(eps,H,eps->V,eps->nconv,ncv,f,&beta);CHKERRQ(ierr);
    eps->its = eps->its + ncv - eps->nconv;

    /* At this point, H has the following structure

              | *   * | *   *   *   * |
              |     * | *   *   *   * |
              | ------|-------------- |
          H = |       | *   *   *   * |
              |       | *   *   *   * |
              |       |     *   *   * |
              |       |         *   * |

       that is, an upper Hessenberg matrix of order ncv whose principal 
       submatrix of order nconv is (quasi-)triangular.  */

    /* Reduce H to (quasi-)triangular form, H <- U H U' */
    ierr = PetscMemzero(U,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<ncv;i++) { U[i*(ncv+1)] = 1.0; }
    ierr = EPSDenseSchur(ncv,eps->nconv,H,U,eps->eigr,eps->eigi);CHKERRQ(ierr);

    /* Compute eigenvectors Y of H */
    ierr = PetscMemcpy(Y,U,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    LAtrevc_("R","B",PETSC_NULL,&ncv,H,&ncv,PETSC_NULL,&ncv,Y,&ncv,&ncv,&mout,work,&info,1,1);
#else
    LAtrevc_("R","B",PETSC_NULL,&ncv,H,&ncv,PETSC_NULL,&ncv,Y,&ncv,&ncv,&mout,work,rwork,&info,1,1);
#endif
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREVC %i",info);

    /* Compute residual norm estimates as beta*abs(Y(m,:)) */
    for (i=eps->nconv;i<ncv;i++) { 
      eps->errest[i] = beta*PetscAbsScalar(Y[i*ncv+ncv-1]);
    }  

    /* Look for converged eigenpairs. If necessary, reorder the Arnoldi 
       factorization so that all converged eigenvalues are first */
    k = eps->nconv;
    while (k<ncv && eps->errest[k]<eps->tol) { k++; }
    for (i=k;i<ncv;i++) {
      if (eps->errest[i]<eps->tol) {
        ifst = i + 1;
        ilst = k + 1;
#if !defined(PETSC_USE_COMPLEX)
        LAtrexc_("V",&ncv,H,&ncv,U,&ncv,&ifst,&ilst,work,&info,1);
        eps->eigr[k] = eps->eigr[i];
        eps->eigi[k] = eps->eigi[i];
        if (eps->eigi[i] != 0) {
          eps->eigr[k+1] = eps->eigr[i+1];
          eps->eigi[k+1] = eps->eigi[i+1];
          k++;       
        }
#else
        LAtrexc_("V",&ncv,H,&ncv,U,&ncv,&ifst,&ilst,&info,1);
        eps->eigr[k] = eps->eigr[i];
#endif
        if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREXC %d",info);
        k++;
      }
#if !defined(PETSC_USE_COMPLEX)
      if (eps->eigi[i] != 0) i++;
#endif
    } 

    /* Sort the remaining columns of the Schur form  */
    ierr = EPSSortDenseSchur(ncv,k,H,U,eps->eigr,eps->eigi);CHKERRQ(ierr);

    /* Update V(:,idx) = V*U(:,idx) */
    ierr = EPSReverseProjection(eps,eps->V,U,eps->nconv,ncv,eps->work);CHKERRQ(ierr);
    eps->nconv = k;
    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,ncv);
    if (eps->nconv >= eps->nev) break;
  }
  
  if( eps->nconv >= eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
#if defined(PETSC_USE_COMPLEX)
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
#endif

  ierr = PetscFree(U);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_ARNOLDI"
PetscErrorCode EPSCreate_ARNOLDI(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = (void *) 0;
  eps->ops->solve                = EPSSolve_ARNOLDI;
  eps->ops->setup                = EPSSetUp_ARNOLDI;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Schur;
  PetscFunctionReturn(0);
}
EXTERN_C_END

