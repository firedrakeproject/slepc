/*                       

   SLEPc eigensolver: "lanczos"

   Method: Explicitly Restarted Symmetric/Hermitian Lanczos

   Description:

       This solver implements the Lanczos method for symmetric (Hermitian)
       problems, with explicit restart and deflation. When building the 
       Lanczos factorization, several reorthogonalization strategies can
       be selected.

   Algorithm:

       The implemented algorithm builds a Lanczos factorization of order
       ncv. Converged eigenpairs are locked and the iteration is restarted
       with the rest of the columns being the active columns for the next
       Lanczos factorization. Currently, no filtering is applied to the
       vector used for restarting.

       The following reorthogonalization schemes are currently implemented:

       - Full reorthogonalization: at each Lanczos step, the corresponding
       Lanczos vector is orthogonalized with respect to all the previous
       vectors.

   References:

       [1] B.N. Parlett, "The Symmetric Eigenvalue Problem", SIAM Classics in 
       Applied Mathematics (1998), ch. 13.

       [2] L. Komzsik, "The Lanczos Method. Evolution and Application", SIAM
       (2003).

       [3] B.N. Parlett and D.S. Scott, The Lanczos algorithm with selective
       orthogonalization, Math. Comp., 33 (1979), no. 145, 217-238.

       [4] H.D. Simon, The Lanczos algorithm with partial reorthogonalization,
       Math. Comp., 42 (1984), no. 165, 115-142.

   Last update: October 2004

*/
#include "src/eps/epsimpl.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_LANCZOS"
PetscErrorCode EPSSetUp_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  int            N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->nev > N) eps->nev = N;
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMin(N,PetscMax(2*eps->nev,eps->nev+15));
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");
  if (!eps->ishermitian)
    SETERRQ(PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  if (eps->T) { ierr = PetscFree(eps->T);CHKERRQ(ierr); }  
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,eps->ncv+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSFullLanczos"
/*
   EPSFullLanczos - Full reorthogonalization.

   In this variant, at each Lanczos step, the corresponding Lanczos vector 
   is orthogonalized with respect to all the previous Lanczos vectors.
*/
static PetscErrorCode EPSFullLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int m,Vec f,PetscReal *beta)
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
    ierr = EPSOrthogonalize(eps,j+1,V,V[j+1],T+m*j,&norm,&breakdown);CHKERRQ(ierr);
    T[(m+1)*j+1] = norm;
    if (breakdown) {
      PetscLogInfo(eps,"Breakdown in Lanczos method (norm=%g)\n",norm);
      ierr = SlepcVecSetRandom(V[j+1]);CHKERRQ(ierr);
      ierr = STNorm(eps->OP,V[j+1],&norm);CHKERRQ(ierr);
    }
    t = 1 / norm;
    ierr = VecScale(&t,V[j+1]);CHKERRQ(ierr);
  }
  ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr);
  ierr = EPSOrthogonalize(eps,m,V,f,T+m*(m-1),beta,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicLanczos"
/*
   EPSBasicLanczos - Computes an m-step Lanczos factorization. The first k
   columns are assumed to be locked and therefore they are not modified. On
   exit, the following relation is satisfied:

                    OP * V - V * T = f * e_m^T

   where the columns of V are the Lanczos vectors, T is a tridiagonal matrix, 
   f is the residual vector and e_m is the m-th vector of the canonical basis. 
   The Lanczos vectors (together with vector f) are B-orthogonal (to working
   accuracy) if full reorthogonalization is being used, otherwise they are
   (B-)semi-orthogonal. On exit, beta contains the B-norm of f and the next 
   Lanczos vector can be computed as v_{m+1} = f / beta. 

   This function simply calls another function which depends on the selected
   reorthogonalization strategy.
*/
static PetscErrorCode EPSBasicLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = EPSFullLanczos(eps,T,V,k,m,f,beta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "RefineBounds"
PetscErrorCode RefineBounds(int n,PetscReal *ritz,PetscReal *bnd,PetscReal eps,PetscReal tol)
{
  int       i,mid;
  PetscReal gapl,gapr,max,eps34 = sqrt(eps*sqrt(eps));
  
  PetscFunctionBegin;
  
  max = bnd[0];
  mid = 0;
  for (i=1;i<n;i++) {
    if (bnd[i] > max) {
      max = bnd[i];
      mid = i;
    }
  }
  
  for (i=n-1;i>mid;i--) 
    if (PetscAbsReal(ritz[i-1]-ritz[i]) < eps34*PetscAbsReal(ritz[i]))
      if (bnd[i-1] > tol && bnd[i] > tol) {
        bnd[i-1] = sqrt(bnd[i-1]*bnd[i-1]+bnd[i]*bnd[i]);
        bnd[i] = 0;
      }
      
  for (i=0;i<mid;i++)
    if (PetscAbsReal(ritz[i+1]-ritz[i]) < eps34*PetscAbsReal(ritz[i]))
      if (bnd[i+1] > tol && bnd[i] > tol) {
        bnd[i+1] = sqrt(bnd[i+1]*bnd[i+1]+bnd[i]*bnd[i]);
        bnd[i] = 0;
      }
  
  gapl = ritz[n-1] - ritz[0]; 
  for (i=0;i<n-1;i++) {
    gapr = ritz[i+1] - ritz[i];
    if (gapl < gapr) gapl = gapr;
    if (bnd[i] < gapl) bnd[i] = bnd[i]*(bnd[i]/gapl);
    gapl = gapr;
  }
  if (bnd[n-1] < gapl) bnd[n-1] = bnd[n-1]*bnd[n-1]/gapl;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_LANCZOS"
PetscErrorCode EPSSolve_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  int            nconv,i,j,n,m,*isuppz,*iwork,info,N,
                 ncv=eps->ncv,
                 lwork=18*ncv,
                 liwork=10*ncv;
  Vec            f=eps->work[ncv];
  PetscScalar    *T=eps->T,*Y,ts;
  PetscReal      *ritz,*bnd,*D,*E,*work,norm,anorm,beta,abstol;

  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY)
  SETERRQ(PETSC_ERR_SUP,"xSTEGR - Lapack routine is unavailable.");
#endif 
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&ritz);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&bnd);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&D);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&E);CHKERRQ(ierr);
  ierr = PetscMalloc(2*ncv*sizeof(int),&isuppz);CHKERRQ(ierr);
  ierr = PetscMalloc(lwork*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(liwork*sizeof(int),&iwork);CHKERRQ(ierr);

  /* The first Lanczos vector is the normalized initial vector */
  ierr = VecCopy(eps->vec_initial,eps->V[0]);CHKERRQ(ierr);
  ierr = STNorm(eps->OP,eps->V[0],&norm);CHKERRQ(ierr);
  ts = 1 / norm;
  ierr = VecScale(&ts,eps->V[0]);CHKERRQ(ierr);
  
  nconv = 0;
  eps->its = 0;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  /* Restart loop */
  while (eps->its<eps->max_it) {
    /* Compute an ncv-step Lanczos factorization */
    ierr = EPSBasicLanczos(eps,T,eps->V,nconv,ncv,f,&beta);CHKERRQ(ierr);

    /* At this point, T has the following structure

              | *     |               |
              |     * |               |
              | ------|-------------- |
          T = |       | *   *         |
              |       | *   *   *     |
              |       |     *   *   * |
              |       |         *   * |

       that is, a real symmetric tridiagonal matrix of order ncv whose 
       principal submatrix of order nconv is diagonal.  */

    /* Extract the tridiagonal block from T. Store the diagonal elements in D 
       and the off-diagonal elements in E  */
    n = ncv - nconv;
    for (i=0;i<n;i++) {
      D[i] = PetscRealPart(T[(i+nconv)*(ncv+1)]);
      E[i] = PetscRealPart(T[(i+nconv)*(ncv+1)+1]);
    }

    /* Compute eigenvalues W and eigenvectors Y of the tridiagonal block */
    abstol = 0.0;
    LAstegr_("V","A",&n,D,E,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,
             &abstol,&m,ritz,Y,&n,isuppz,work,&lwork,iwork,&liwork,&info,1,1);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xSTEGR %i",info);
    
    /* Compute residual norm estimates as beta*abs(Y(m,:)) */
    anorm = 0;
    for (i=0;i<n;i++) {
      if (ritz[i] > anorm) anorm = ritz[i];
      bnd[i] = beta*PetscAbsScalar(Y[i*n+n-1]);
    }
    ierr = RefineBounds(n,ritz,bnd,eps->tol,eps->tol*anorm*N);CHKERRQ(ierr);
    
    /* reverse order of ritz values and caculate relateive error bounds */
    for (i=0;i<n;i++) {
      eps->eigr[ncv-i-1] = ritz[i];
      eps->errest[ncv-i-1] = bnd[i] / ritz[i];
    }

    /* Update V(:,idx) = V*Y(:,idx) */
    ierr = PetscMemzero(T,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<n;i++) 
      for (j=0;j<n;j++) 
          T[i+nconv+(j+nconv)*ncv] = Y[i+(n-j-1)*n];
    ierr = EPSReverseProjection(eps,eps->V,T,nconv,ncv,eps->work);CHKERRQ(ierr);

    /* Look for converged eigenpairs */
    while (nconv<ncv && eps->errest[nconv]<eps->tol) nconv++;

    EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,ncv);
    eps->its = eps->its + ncv - nconv;
    if (nconv >= eps->nev) break;
  }
  
  eps->nconv = nconv;
  if( eps->nconv >= eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;

  ierr = PetscFree(ritz);CHKERRQ(ierr);
  ierr = PetscFree(bnd);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  ierr = PetscFree(D);CHKERRQ(ierr);
  ierr = PetscFree(E);CHKERRQ(ierr);
  ierr = PetscFree(isuppz);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_LANCZOS"
PetscErrorCode EPSCreate_LANCZOS(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = (void *) 0;
  eps->ops->solve                = EPSSolve_LANCZOS;
  eps->ops->setup                = EPSSetUp_LANCZOS;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END

