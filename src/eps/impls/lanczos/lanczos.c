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
#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

typedef struct {
  EPSLanczosOrthogType reorthog;
} EPS_LANCZOS;

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
/*  if (eps->which!=EPS_LARGEST_MAGNITUDE)
    SETERRQ(1,"Wrong value of eps->which");*/
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
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,V[j+1],PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,j+1,V,V[j+1],T+m*j,&norm,&breakdown);CHKERRQ(ierr);
    T[(m+1)*j+1] = norm;
    if (breakdown) {
      PetscLogInfo(eps,"Breakdown in Lanczos method (norm=%g)\n",norm);
      ierr = EPSGetStartVector(eps,j+1,V[j+1]);CHKERRQ(ierr);
    } else {
      t = 1 / norm;
      ierr = VecScale(&t,V[j+1]);CHKERRQ(ierr);
    }
  }
  ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr);
  eps->its++;
  ierr = EPSOrthogonalize(eps,m,V,f,T+m*(m-1),beta,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSPlainLanczos"
/*
   EPSPlainLanczos - Local reorthogonalization.

   This is the simplest variant. At each Lanczos step, the corresponding Lanczos vector 
   is orthogonalized with respect to the two previous Lanczos vectors, according to
   the three term Lanczos recurrence. WARNING: This variant does not track the loss of 
   orthogonality that occurs in finite-precision arithmetic and, therefore, the 
   generated vectors are not guaranteed to be (semi-)orthogonal.
*/
static PetscErrorCode EPSPlainLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            j,i;
  PetscReal      norm;
  PetscScalar    t;
  PetscTruth     breakdown = PETSC_FALSE;

  PetscFunctionBegin;
  for (j=k;j<m-1;j++) {
    ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,V[j+1],PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,k,V,V[j+1],PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    if (j == k || breakdown) {
      for (i=0;i<j;i++) T[m*j+i] = 0;
      ierr = EPSOrthogonalize(eps,1,V+j,V[j+1],T+m*j+j,&norm,&breakdown);CHKERRQ(ierr);
    } else {
      for (i=0;i<j-1;i++) T[m*j+i] = 0;
      ierr = EPSOrthogonalize(eps,2,V+j-1,V[j+1],T+m*j+j-1,&norm,&breakdown);CHKERRQ(ierr);
    }
    if (breakdown) {
      T[m*j+j+1] = 0;
      ierr = VecDot(V[j],V[j+1],&t);CHKERRQ(ierr);
      printf("Breakdown in Lanczos method (its=%i,k=%i,j=%i,norm=%g,dot=%g)\n",eps->its,k,j,norm,t);
      PetscLogInfo(eps,"Breakdown in Lanczos method (norm=%g)\n",norm);
      ierr = EPSGetStartVector(eps,j+1,V[j+1]);CHKERRQ(ierr);
    } else {
      T[m*j+j+1] = norm;
      t = 1 / norm;
      ierr = VecScale(&t,V[j+1]);CHKERRQ(ierr);
    }
  }
  ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr);
  eps->its++;
  ierr = EPSOrthogonalize(eps,2,V+m-2,f,T+m*m-2,beta,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSUpdateOmega"
static PetscErrorCode EPSUpdateOmega(int j,PetscReal *alpha,PetscReal* beta,PetscReal psi,PetscReal theta,PetscReal *omega,PetscReal *oldomega)
{
  int k;
  PetscReal t;
  PetscFunctionBegin;
  alpha--; beta-=2; omega--; oldomega--;
  if (j>1) {
    t = beta[2]*omega[2]+(alpha[1]-alpha[j])*omega[1]-beta[j]*oldomega[1];
    if (t>0) oldomega[1] = (t + theta)/beta[j+1];
    else oldomega[1] = (t - theta)/beta[j+1];
    for (k=2;k<j;k++) {
      t = beta[k+1]*omega[k+1]+(alpha[k]-alpha[j])*omega[k]+
          beta[k]*omega[k-1]-beta[j]*oldomega[k];
      if (t>0) oldomega[k] = (t + theta)/beta[j+1];
      else oldomega[k] = (t - theta)/beta[j+1];
    }
  }
  oldomega[j] = psi;
  /* SWAP(oldomega,omega) */
  for (k=0;k<=j;k++) {
    t = oldomega[k];
    oldomega[k] = omega[k];
    omega[k] = t;
  }
  PetscFunctionReturn(0);
}

int orthog_count = 0;

#undef __FUNCT__  
#define __FUNCT__ "EPSPeriodicLanczos"
static PetscErrorCode EPSPeriodicLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  PetscErrorCode ierr;
  int            i,j,l,n;
  PetscReal      norm,*omega,*oldomega,*a,*b,delta,eps1;
  PetscScalar    *H,t;
  PetscTruth     breakdown,reorthog;

  PetscFunctionBegin;
  ierr = PetscMalloc(m*sizeof(PetscScalar),&H);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscReal),&a);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscReal),&b);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscReal),&omega);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscReal),&oldomega);CHKERRQ(ierr);
  ierr = VecGetSize(eps->vec_initial,&n);CHKERRQ(ierr);
  eps1 = PETSC_MACHINE_EPSILON * sqrt(n);
  delta = sqrt(PETSC_MACHINE_EPSILON / eps->ncv);
  reorthog = PETSC_FALSE;
  
  for (j=k;j<m-1;j++) {
    ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,V[j+1],PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSOrthogonalize(eps,k,V,V[j+1],PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    if (j==k) {
      ierr = EPSOrthogonalize(eps,1,V+j,V[j+1],H+j,&norm,&breakdown);CHKERRQ(ierr);
    } else {
      ierr = EPSOrthogonalize(eps,2,V+j-1,V[j+1],H+j-1,&norm,&breakdown);CHKERRQ(ierr);
    }
    a[j-k] = T[j*m+j] = H[j];
    b[j-k] = T[j*m+j+1] = norm;
    ierr = EPSUpdateOmega(j-k+1,a,b,eps1,eps1,omega,oldomega);CHKERRQ(ierr);
    for (i=k;i<=j;i++) {
      if (PetscAbsReal(omega[i-k]) > delta || reorthog) {
        ierr = EPSOrthogonalize(eps,j+1,V,V[j+1],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr);
        for (l=0;l<=j;l++) omega[l] = eps1;
	if (reorthog) reorthog = PETSC_FALSE;
	else reorthog = PETSC_TRUE;
        break;
      }
    }
    if (breakdown || norm < PETSC_MACHINE_EPSILON) {
      T[m*j+j+1] = 0;
      printf("Breakdown in Lanczos method (norm=%g)\n",norm);
      PetscLogInfo(eps,"Breakdown in Lanczos method (norm=%g)\n",norm);
      ierr = SlepcVecSetRandom(V[j+1]);CHKERRQ(ierr);
      ierr = STNorm(eps->OP,V[j+1],&norm);CHKERRQ(ierr);
    } else {
      T[m*j+j+1] = norm;
      t = 1 / norm;
      ierr = VecScale(&t,V[j+1]);CHKERRQ(ierr);
    }
  }
  
  ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr);
  eps->its++;
  ierr = EPSOrthogonalize(eps,2,V+m-2,f,T+m*m-2,beta,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscFree(H);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(omega);CHKERRQ(ierr);
  ierr = PetscFree(oldomega);CHKERRQ(ierr);
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
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (lanczos->reorthog) {
    case EPSLANCZOS_ORTHOG_NONE:
      ierr = EPSPlainLanczos(eps,T,V,k,m,f,beta);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_ORTHOG_FULL:
      ierr = EPSFullLanczos(eps,T,V,k,m,f,beta);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_ORTHOG_PERIODIC:
      ierr = EPSPeriodicLanczos(eps,T,V,k,m,f,beta);CHKERRQ(ierr);
      break;
  }
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
  int            nconv,i,j,n,info,N,*perm,
                 ncv=eps->ncv;
  Vec            f=eps->work[ncv];
  PetscScalar    *T=eps->T,*Y,*W,ts;
  PetscReal      *ritz,*bnd,*E,*work,norm,anorm,beta;

  PetscFunctionBegin;
#if defined(PETSC_BLASLAPACK_ESSL_ONLY)
  SETERRQ(PETSC_ERR_SUP,"xSTEGR - Lapack routine is unavailable.");
#endif 
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&ritz);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&bnd);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&W);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&E);CHKERRQ(ierr);
  ierr = PetscMalloc(2*ncv*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(int),&perm);CHKERRQ(ierr);

  /* The first Lanczos vector is the normalized initial vector */
  ierr = VecCopy(eps->vec_initial,eps->V[0]);CHKERRQ(ierr);
  ierr = STNorm(eps->OP,eps->V[0],&norm);CHKERRQ(ierr);
  ts = 1 / norm;
  ierr = VecScale(&ts,eps->V[0]);CHKERRQ(ierr);
  
  nconv = 0;
  eps->its = 0;
  for (i=0;i<eps->ncv;i++) eps->eigi[i]=0.0;
  EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,ncv);
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
      ritz[i] = PetscRealPart(T[(i+nconv)*(ncv+1)]);
      E[i] = PetscRealPart(T[(i+nconv)*(ncv+1)+1]);
    }

    /* Compute eigenvalues and eigenvectors Y of the tridiagonal block */
    LAsteqr_("I",&n,ritz,E,Y,&n,work,&info,1);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xSTEQR %i",info);
    
    /* Compute residual norm estimates as beta*abs(Y(m,:)) */
    anorm = 0;
    for (i=0;i<n;i++) {
      if (PetscAbsReal(ritz[i]) > anorm) anorm = PetscAbsReal(ritz[i]);
      bnd[i] = beta*PetscAbsScalar(Y[i*n+n-1]);
    }
/*    ierr = RefineBounds(n,ritz,bnd,eps->tol,eps->tol*anorm*N);CHKERRQ(ierr);*/
    
    ierr = EPSSortEigenvalues(n,ritz,eps->eigi,eps->which,n,perm);CHKERRQ(ierr);
    /* Reverse order of Ritz values and calculate relative error bounds */
    for (i=0;i<n;i++) {
      eps->eigr[nconv+i] = ritz[perm[i]];
      eps->errest[nconv+i] = bnd[perm[i]] / PetscAbsReal(ritz[perm[i]]);
    }

    /* Update V(:,idx) = V*Y */
    for (j=0;j<n;j++) 
      for (i=0;i<n;i++) 
        W[i+j*n] = Y[i+perm[j]*n];
    ierr = EPSReverseProjection(eps,eps->V+nconv,W,0,n,eps->work);CHKERRQ(ierr);

    /* Look for converged eigenpairs */
    while (nconv<ncv && eps->errest[nconv]<eps->tol) {
      ierr = STNorm(eps->OP,eps->V[nconv],&norm);CHKERRQ(ierr);
      eps->errest[nconv] = eps->errest[nconv] / norm;
      if (eps->errest[nconv]<eps->tol) {
        ts = 1 / norm;
        ierr = VecScale(&ts,eps->V[nconv]);CHKERRQ(ierr);
        nconv++;
      }
    }

    EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,ncv);
    if (nconv >= eps->nev) break;
  }
  
  eps->nconv = nconv;
  if( eps->nconv >= eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;

  ierr = PetscFree(ritz);CHKERRQ(ierr);
  ierr = PetscFree(bnd);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  ierr = PetscFree(W);CHKERRQ(ierr);
  ierr = PetscFree(E);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_LANCZOS"
PetscErrorCode EPSSetFromOptions_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS *)eps->data;
  PetscTruth     flg;
  const char     *list[4] = { "none", "full" , "periodic", "partial" };

  PetscFunctionBegin;
  ierr = PetscOptionsHead("LANCZOS options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-eps_lanczos_orthog","Reorthogonalization type","EPSLanczosSetOrthog",list,4,list[lanczos->reorthog],(int*)&lanczos->reorthog,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosSetOrthog_LANCZOS"
PetscErrorCode EPSLanczosSetOrthog_LANCZOS(EPS eps,EPSLanczosOrthogType reorthog)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;

  PetscFunctionBegin;
  switch (reorthog) {
    case EPSLANCZOS_ORTHOG_NONE:
    case EPSLANCZOS_ORTHOG_FULL:
    case EPSLANCZOS_ORTHOG_PERIODIC:
    case EPSLANCZOS_ORTHOG_PARTIAL:
      lanczos->reorthog = reorthog;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid reorthogonalization type");
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosSetOrthog"
/*@
   EPSLanczosSetOrthog - Sets the type of reorthogonalization used during the Lanczos
   iteration. 

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  reorthog - the type of reorthogonalization

   Options Database Key:
.  -eps_lanczos_orthog - Sets the reorthogonalization type (either 'none' or 
                           'full')
   
   Level: advanced

.seealso: EPSLanczosGetOrthog()
@*/
PetscErrorCode EPSLanczosSetOrthog(EPS eps,EPSLanczosOrthogType reorthog)
{
  PetscErrorCode ierr, (*f)(EPS,EPSLanczosOrthogType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSLanczosSetOrthog_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,reorthog);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosGetOrthog_LANCZOS"
PetscErrorCode EPSLanczosGetOrthog_LANCZOS(EPS eps,EPSLanczosOrthogType *reorthog)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscFunctionBegin;
  *reorthog = lanczos->reorthog;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosGetOrthog"
/*@C
   EPSLanczosGetOrthog - Gets the type of reorthogonalization used during the Lanczos
   iteration. 

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Input Parameter:
.  reorthog - the type of reorthogonalization

   Level: advanced

.seealso: EPSLanczosSetOrthog()
@*/
PetscErrorCode EPSLanczosGetOrthog(EPS eps,EPSLanczosOrthogType *reorthog)
{
  PetscErrorCode ierr, (*f)(EPS,EPSLanczosOrthogType*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSLanczosGetOrthog_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,reorthog);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSView_LANCZOS"
PetscErrorCode EPSView_LANCZOS(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS *)eps->data;
  PetscTruth     isascii;
  const char     *list[4] = { "none", "full" , "periodic", "partial" };

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSLANCZOS",((PetscObject)viewer)->type_name);
  }  
  ierr = PetscViewerASCIIPrintf(viewer,"reorthogonalization: %s\n",list[lanczos->reorthog]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_LANCZOS"
PetscErrorCode EPSCreate_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_LANCZOS,&lanczos);CHKERRQ(ierr);
  PetscMemzero(lanczos,sizeof(EPS_LANCZOS));
  PetscLogObjectMemory(eps,sizeof(EPS_LANCZOS));
  eps->data                      = (void *) lanczos;
  eps->ops->solve                = EPSSolve_LANCZOS;
  eps->ops->setup                = EPSSetUp_LANCZOS;
  eps->ops->setfromoptions       = EPSSetFromOptions_LANCZOS;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->view                 = EPSView_LANCZOS;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  lanczos->reorthog              = EPSLANCZOS_ORTHOG_FULL;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosSetOrthog_C","EPSLanczosSetOrthog_LANCZOS",EPSLanczosSetOrthog_LANCZOS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosGetOrthog_C","EPSLanczosGetOrthog_LANCZOS",EPSLanczosGetOrthog_LANCZOS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

