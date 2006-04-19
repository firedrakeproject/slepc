/*                       

   SLEPc eigensolver: "lanczos"

   Method: Explicitly Restarted Symmetric/Hermitian Lanczos

   Algorithm:

       Lanczos method for symmetric (Hermitian) problems, with explicit 
       restart and deflation. Several reorthogonalization strategies can
       be selected.

   References:

       [1] "Lanczos Methods in SLEPc", SLEPc Technical Report STR-5, 
           available at http://www.grycap.upv.es/slepc.

   Last update: June 2005

*/
#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/
#include "src/st/stimpl.h"
#include "slepcblaslapack.h"

typedef struct {
  EPSLanczosReorthogType reorthog;
} EPS_LANCZOS;

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_LANCZOS"
PetscErrorCode EPSSetUp_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->nev > N) eps->nev = N;
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMin(N,PetscMax(2*eps->nev,eps->nev+15));
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;
  if (eps->solverclass==EPS_ONE_SIDE) {
    if (eps->which == EPS_LARGEST_IMAGINARY || eps->which == EPS_SMALLEST_IMAGINARY)
      SETERRQ(1,"Wrong value of eps->which");
    if (!eps->ishermitian)
      SETERRQ(PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");
  } else {
    if (eps->which != EPS_LARGEST_MAGNITUDE)
      SETERRQ(1,"Wrong value of eps->which");
  }
  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = PetscFree(eps->T);CHKERRQ(ierr);
  ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->T);CHKERRQ(ierr);
  if (eps->solverclass==EPS_TWO_SIDE) {
    ierr = PetscFree(eps->Tl);CHKERRQ(ierr);
    ierr = PetscMalloc(eps->ncv*eps->ncv*sizeof(PetscScalar),&eps->Tl);CHKERRQ(ierr);
    ierr = EPSDefaultGetWork(eps,2);CHKERRQ(ierr);
  }
  else { ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSFullLanczos"
/*
   EPSFullLanczos - Full reorthogonalization.

   In this variant, at each Lanczos step, the corresponding Lanczos vector 
   is orthogonalized with respect to all the previous Lanczos vectors.
*/
PetscErrorCode EPSFullLanczos(EPS eps,PetscScalar *H,Vec *V,int k,int *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            j,m = *M;
  PetscReal      norm;
  PetscScalar    *c,lc[100];

  PetscFunctionBegin;
  if (m>100 && eps->orthog_ref !=EPS_ORTH_REFINE_NEVER) {
    ierr = PetscMalloc(sizeof(PetscScalar)*m,&c);CHKERRQ(ierr);
  } else {
    c = lc;
  }

  for (j=k;j<m-1;j++) {
    ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr);
    eps->its++;
    ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,V[j+1],PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = EPSOrthogonalize2(eps,j+1,V,V[j+1],H+m*j,c,&norm,breakdown);CHKERRQ(ierr);
    H[(m+1)*j+1] = norm;
    if (*breakdown) {
      *M = j+1;
      *beta = norm;
      PetscFunctionReturn(0);
    } else {
      ierr = VecScale(V[j+1],1/norm);CHKERRQ(ierr);
    }
  }
  ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr);
  eps->its++;
  ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSOrthogonalize2(eps,m,V,f,H+m*(m-1),c,beta,PETSC_NULL);CHKERRQ(ierr);

  if (m>100 && eps->orthog_ref !=EPS_ORTH_REFINE_NEVER) {
    ierr = PetscFree(c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSLocalLanczos"
/*
   EPSLocalLanczos - Local reorthogonalization.

   This is the simplest variant. At each Lanczos step, the corresponding Lanczos vector 
   is orthogonalized with respect to the two previous Lanczos vectors, according to
   the three term Lanczos recurrence. WARNING: This variant does not track the loss of 
   orthogonality that occurs in finite-precision arithmetic and, therefore, the 
   generated vectors are not guaranteed to be (semi-)orthogonal.
*/
static PetscErrorCode EPSLocalLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *M,Vec f,PetscReal *beta,PetscTruth *breakdown)
{
  PetscErrorCode ierr;
  int            i,j,nv,m = *M;
  PetscReal      norm;
  Vec            *VV,lVV[100];
  PetscScalar    *w,lw[100],*c,lc[100];
  
  PetscFunctionBegin;
  nv = eps->nds+k+2;
  if (nv>100) {
    ierr = PetscMalloc((nv)*sizeof(Vec),&VV);CHKERRQ(ierr);
  } else VV = lVV;
  for (i=2;i<nv;i++)
    VV[i] = eps->DSV[i-2];
  if (m>100) {
    ierr = PetscMalloc(sizeof(PetscScalar)*m,&w);CHKERRQ(ierr);
    if (eps->orthog_ref !=EPS_ORTH_REFINE_NEVER) {
      ierr = PetscMalloc(sizeof(PetscScalar)*m,&c);CHKERRQ(ierr);
    }
  } else {
    w = lw;
    c = lc;
  }

  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    if (j == k) {
      VV[1] = V[j];
      ierr = PetscLogEventBegin(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
      ierr = EPSOrthogonalize2(eps,nv-1,VV+1,f,w,c,&norm,breakdown);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
      T[m*j+j] = w[0];
    } else {
      VV[0] = V[j-1];
      VV[1] = V[j];
      ierr = PetscLogEventBegin(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
      ierr = EPSOrthogonalize2(eps,nv,VV,f,w,c,&norm,breakdown);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
      T[m*j+j] = w[1];
    }    
    if (*breakdown) {
      *M = j+1;
      break;
    }
    if (j<m-1) {
      T[m*j+j+1] = norm;
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;

  if (nv>100) { ierr = PetscFree(VV);CHKERRQ(ierr); }
  if (m>100) { 
    ierr = PetscFree(w);CHKERRQ(ierr);
    if (eps->orthog_ref !=EPS_ORTH_REFINE_NEVER) {
      ierr = PetscFree(c);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSelectiveLanczos"
/*
   EPSSelectiveLanczos - Selective reorthogonalization.
*/
static PetscErrorCode EPSSelectiveLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *M,Vec f,PetscReal *beta,PetscTruth *breakdown,PetscReal anorm)
{
#if defined(SLEPC_MISSING_LAPACK_DSTEVR) || defined(SLEPC_MISSING_LAPACK_DLAMCH)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"DSTEVR - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  int            i,j,m = *M,n,nv,il,iu,mout,*isuppz,*iwork,lwork,liwork,info;
  PetscReal      *D,*E,*ritz,*Y,*work,abstol,vl,vu,norm;
  PetscTruth     conv;
  Vec            *VV,lVV[100];
  PetscScalar    *w,lw[100],*c,lc[100];

  PetscFunctionBegin;
  ierr = PetscMalloc(m*sizeof(PetscReal),&D);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscReal),&E);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscReal),&ritz);CHKERRQ(ierr);
  ierr = PetscMalloc(m*m*sizeof(PetscReal),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(2*m*sizeof(int),&isuppz);CHKERRQ(ierr);
  lwork = 20*m;
  ierr = PetscMalloc(lwork*sizeof(PetscReal),&work);CHKERRQ(ierr);
  liwork = 10*m;
  ierr = PetscMalloc(liwork*sizeof(int),&iwork);CHKERRQ(ierr);
  abstol = 2.0*LAPACKlamch_("S",1);

  nv = eps->nds+k+2;
  if (nv>100) {
    ierr = PetscMalloc((nv)*sizeof(Vec),&VV);CHKERRQ(ierr);
  } else VV = lVV;
  for (i=2;i<nv;i++)
    VV[i] = eps->DSV[i-2];
  if (m>100) {
    ierr = PetscMalloc(sizeof(PetscScalar)*m,&w);CHKERRQ(ierr);
    if (eps->orthog_ref !=EPS_ORTH_REFINE_NEVER) {
      ierr = PetscMalloc(sizeof(PetscScalar)*m,&c);CHKERRQ(ierr);
    }
  } else {
    w = lw;
    c = lc;
  }

  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    if (j == k) {
      VV[1] = V[j];
      ierr = PetscLogEventBegin(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
      ierr = EPSOrthogonalize2(eps,nv-1,VV+1,f,w,c,&norm,breakdown);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
      T[m*j+j] = w[0];
    } else {
      VV[0] = V[j-1];
      VV[1] = V[j];
      ierr = PetscLogEventBegin(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
      ierr = EPSOrthogonalize2(eps,nv,VV,f,w,c,&norm,breakdown);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(EPS_Orthogonalize,eps,0,0,0);CHKERRQ(ierr);
      T[m*j+j] = w[1];
    }    
    
    if (*breakdown) {
      *M = j+1;
      break;
    }

    n = j-k+1;
    for (i=0;i<n-1;i++) {
      ritz[i] = PetscRealPart(T[(i+k)*(m+1)]);
      E[i] = PetscRealPart(T[(i+k)*(m+1)+1]);
    }
    ritz[n-1] = PetscRealPart(T[(n-1+k)*(m+1)]);

    /* Compute eigenvalues and eigenvectors Y of the tridiagonal block */
    LAPACKstevr_("V","A",&n,D,E,&vl,&vu,&il,&iu,&abstol,&mout,ritz,Y,&n,isuppz,work,&lwork,iwork,&liwork,&info,1,1);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xSTEQR %i",info);
    
    /* Estimate ||A|| */
    for (i=0;i<n;i++) 
      if (PetscAbsReal(ritz[i]) > anorm) anorm = PetscAbsReal(ritz[i]);
    
    /* Exit if residual norm is small [Parlett, page 300] */
    conv = PETSC_FALSE;
    for (i=0;i<n && !conv;i++)
      if (norm*PetscAbsScalar(Y[i*n+n-1]) < PETSC_SQRT_MACHINE_EPSILON*anorm)
        conv = PETSC_TRUE;
    
    if (conv) {
      *M = j+1;
      break;
    }

    if (j<m-1) {
      T[m*j+j+1] = norm;
      ierr = VecScale(f,1.0 / norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;
  
  ierr = PetscFree(D);CHKERRQ(ierr);
  ierr = PetscFree(E);CHKERRQ(ierr);
  ierr = PetscFree(ritz);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  ierr = PetscFree(isuppz);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  if (nv>100) { ierr = PetscFree(VV);CHKERRQ(ierr); }
  if (m>100) { 
    ierr = PetscFree(w);CHKERRQ(ierr);
    if (eps->orthog_ref !=EPS_ORTH_REFINE_NEVER) {
      ierr = PetscFree(c);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "update_omega"
static void update_omega(PetscReal *omega,PetscReal *omega_old,int j,PetscScalar *alpha,PetscScalar *beta,PetscReal eps1,PetscReal anorm)
{
  int            k;
  PetscReal      T,binv,temp;

  PetscFunctionBegin;
  /* Estimate of contribution to roundoff errors from A*v 
       fl(A*v) = A*v + f, 
     where ||f|| \approx eps1*||A||.
     For a full matrix A, a rule-of-thumb estimate is eps1 = sqrt(n)*eps. */
  T = eps1*anorm;
  binv = 1.0/beta[j+1];

  for (k=0;k<j;k++)
    omega_old[k] = omega[k];
  /* Update omega(1) using omega(0)==0. */
  omega_old[0]= beta[1]*omega[1] + (alpha[0]-alpha[j])*omega[0] - 
                beta[j]*omega_old[0];
  if (omega_old[0] > 0) 
    omega_old[0] = binv*(omega_old[0] + T);
  else
    omega_old[0] = binv*(omega_old[0] - T);  
  
  /* Update remaining components. */
  for (k=1;k<j-1;k++) {
    omega_old[k] = beta[k+1]*omega[k+1] + (alpha[k]-alpha[j])*omega[k] +
                   beta[k]*omega[k-1] - beta[j]*omega_old[k];
    if (omega_old[k] > 0) 
      omega_old[k] = binv*(omega_old[k] + T);       
    else
      omega_old[k] = binv*(omega_old[k] - T);       
  }
  omega_old[j-1] = binv*T;
  
  /* Swap omega and omega_old. */
  for (k=0;k<j;k++) {
    temp = omega[k];
    omega[k] = omega_old[k];
    omega_old[k] = omega[k];
  }
  omega[j] = eps1;  
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "compute_int"
static void compute_int(PetscTruth *which,PetscReal *mu,int j,PetscReal delta,PetscReal eta)
{
  int        i,k,maxpos;
  PetscReal  max;
  PetscTruth found;
  
  PetscFunctionBegin;  
  /* initialize which */
  found = PETSC_FALSE;
  max = 0.0;
  for (i=0;i<j;i++) {
    if (PetscAbsScalar(mu[i]) >= delta) {
      which[i] = PETSC_TRUE;
      found = PETSC_TRUE;
    } else which[i] = PETSC_FALSE;
    if (PetscAbsScalar(mu[i]) > max) {
      maxpos = i;
      max = PetscAbsScalar(mu[i]);
    }
  }
  if (!found) which[maxpos] = PETSC_TRUE;    
  
  for (i=0;i<j;i++)
    if (which[i]) {
      /* find left interval */
      for (k=i;k>=0;k--) {
        if (PetscAbsScalar(mu[k])<eta || which[k]) break;
	else which[k] = PETSC_TRUE;
      }
      /* find right interval */
      for (k=i+1;k<j;k++) {
        if (PetscAbsScalar(mu[k])<eta || which[k]) break;
	else which[k] = PETSC_TRUE;
      }
    }
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "EPSPartialLanczos"
/*
   EPSPartialLanczos - Partial reorthogonalization.
*/
static PetscErrorCode EPSPartialLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *M,Vec f,PetscReal *beta, PetscTruth *breakdown,PetscReal anorm)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscErrorCode ierr;
  Mat            A;
  int            i,j,n,l,nv,m = *M;
  PetscReal      norm,*omega,lomega[100],*omega_old,lomega_old[100],eps1,delta,eta;
  PetscScalar    alpha,*a,la[100],*b,lb[101];
  PetscTruth     *which,lwhich[100],reorth,force_reorth = PETSC_FALSE,fro = PETSC_FALSE;;
  Vec            *VV,lVV[100];

  PetscFunctionBegin;
  l = m - k;
  if (l>100) {
    ierr = PetscMalloc(l*sizeof(PetscScalar),&a);CHKERRQ(ierr);
    ierr = PetscMalloc((l+1)*sizeof(PetscScalar),&b);CHKERRQ(ierr);
    ierr = PetscMalloc(l*sizeof(PetscScalar),&omega);CHKERRQ(ierr);
    ierr = PetscMalloc(l*sizeof(PetscScalar),&omega_old);CHKERRQ(ierr);
    ierr = PetscMalloc(l*sizeof(PetscTruth),&which);CHKERRQ(ierr);
  } else {
    a = la;
    b = lb;
    omega = lomega;
    omega_old = lomega_old;
    which = lwhich;
  }
  if (eps->nds+eps->ncv > 100) {
    ierr = PetscMalloc((eps->nds+eps->ncv)*sizeof(Vec),&VV);CHKERRQ(ierr);
  } else VV = lVV;
  
  ierr = STGetOperators(eps->OP,&A,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,&n,PETSC_NULL);CHKERRQ(ierr);
  eps1 = sqrt((PetscReal)n)*PETSC_MACHINE_EPSILON/2;
  delta = PETSC_SQRT_MACHINE_EPSILON/sqrt((PetscReal)eps->ncv);
  eta = pow(PETSC_MACHINE_EPSILON,3.0/4.0)/sqrt((PetscReal)eps->ncv);
  
  for (j=k;j<m;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    eps->its++;
    if (fro) {
      /* Lanczos step with full reorthogonalization */
      ierr = EPSOrthogonalize(eps,eps->nds,eps->DS,f,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      ierr = EPSOrthogonalize(eps,j+1,V,f,T+m*j,&norm,breakdown);CHKERRQ(ierr);      
    } else {
      /* Lanczos step */
      if (j>k) {
	ierr = VecAXPY(f,-norm,V[j-1]);CHKERRQ(ierr);
      }
      ierr = VecDot(V[j],f,&alpha);CHKERRQ(ierr);
      ierr = VecAXPY(f,-alpha,V[j]);CHKERRQ(ierr);
      T[m*j+j] = a[j-k] = alpha;

      /* Check if reorthogonalization is needed */
      reorth = PETSC_FALSE;
      if (j>k) {
	ierr = STNorm(eps->OP,f,&b[j-k+1]);CHKERRQ(ierr);
	update_omega(omega,omega_old,j-k,a,b,eps1,anorm);
	for (i=0;i<j-k;i++)
	  if (PetscAbsScalar(omega[i]) > delta) reorth = PETSC_TRUE;
      }

      if (reorth || force_reorth) {
	eps->OP->lineariterations++;
	if (lanczos->reorthog == EPSLANCZOS_REORTHOG_PERIODIC) {
	  /* Periodic reorthogonalization */
	  if (force_reorth) force_reorth = PETSC_FALSE;
	  else force_reorth = PETSC_TRUE;
	  ierr = EPSOrthogonalize(eps,eps->nds+j+1,eps->DSV,f,PETSC_NULL,&norm,breakdown);CHKERRQ(ierr);
	  for (i=0;i<j-k;i++)
            omega[i] = eps1;
	} else {
	  /* Partial reorthogonalization */
	  if (force_reorth) force_reorth = PETSC_FALSE;
	  else {
	    force_reorth = PETSC_TRUE;
	    compute_int(which,omega,j-k,delta,eta);
	    nv = 0;
	    for (i=0;i<eps->nds;i++) {
	      VV[nv] = eps->DS[i];
	      nv++;
	    }
	    for (i=0;i<k;i++) {
	      VV[nv] = eps->V[i];
	      nv++;
	    }
	    for (i=0;i<j-k;i++) 
	      if (which[i]) {
		omega[i] = eps1;
		VV[nv] = eps->V[i+k];
		nv++;
	      }
	  } 
	  ierr = EPSOrthogonalize(eps,nv,VV,f,PETSC_NULL,&norm,breakdown);CHKERRQ(ierr);	
	}	
      } else {
        /* Deflation with locked vectors */
	ierr = EPSOrthogonalize(eps,eps->nds+k,eps->DSV,f,PETSC_NULL,&norm,breakdown);CHKERRQ(ierr);
      }
    }
    if (*breakdown || norm < n*anorm*PETSC_MACHINE_EPSILON) {
      *M = j+1;
      break;
    }
    if (!fro && norm*delta < anorm*eps1) {
      fro = PETSC_TRUE;
      PetscInfo1(eps,"Switching to full reorthogonalization at iteration %i\n",eps->its);	
    }
    if (j<m-1) {
      T[m*j+j+1] = b[j-k+1] = norm;
      ierr = VecScale(f,1.0/norm);CHKERRQ(ierr);
      ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
    }
  }
  *beta = norm;

  if (l>100) {
    ierr = PetscFree(a);CHKERRQ(ierr);
    ierr = PetscFree(b);CHKERRQ(ierr);
    ierr = PetscFree(omega);CHKERRQ(ierr);
    ierr = PetscFree(omega_old);CHKERRQ(ierr);
    ierr = PetscFree(which);CHKERRQ(ierr);
  }
  if (eps->nds+eps->ncv > 100) {
    ierr = PetscFree(VV);CHKERRQ(ierr);
  }
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
static PetscErrorCode EPSBasicLanczos(EPS eps,PetscScalar *T,Vec *V,int k,int *m,Vec f,PetscReal *beta,PetscTruth *breakdown,PetscReal anorm)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscErrorCode ierr;
  const char     *prefix;
  PetscInt       type = 1;

  PetscFunctionBegin;
  ierr = PetscObjectGetOptionsPrefix((PetscObject)eps,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(prefix,"-lanczos",&type,PETSC_NULL);CHKERRQ(ierr);
  switch (lanczos->reorthog) {
    case EPSLANCZOS_REORTHOG_LOCAL:
      ierr = EPSLocalLanczos(eps,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_REORTHOG_SELECTIVE:
      ierr = EPSSelectiveLanczos(eps,T,V,k,m,f,beta,breakdown,anorm);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_REORTHOG_PARTIAL:
    case EPSLANCZOS_REORTHOG_PERIODIC:
      ierr = EPSPartialLanczos(eps,T,V,k,m,f,beta,breakdown,anorm);CHKERRQ(ierr);
      break;
    case EPSLANCZOS_REORTHOG_FULL:
      switch (type) {
      case 1:
        ierr = EPSFullLanczos(eps,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);
        break;
      default:
        ierr = EPSBasicArnoldi(eps,PETSC_FALSE,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);
      }
      break;
    case EPSLANCZOS_REORTHOG_DELAYED:
      if (eps->orthog_ref == EPS_ORTH_REFINE_NEVER) {
        ierr = EPSDelayedArnoldi1(eps,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);       
      } else {
        ierr = EPSDelayedArnoldi(eps,T,V,k,m,f,beta,breakdown);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid reorthogonalization type"); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_LANCZOS"
PetscErrorCode EPSSolve_LANCZOS(EPS eps)
{
#if defined(SLEPC_MISSING_LAPACK_DSTEVR) || defined(SLEPC_MISSING_LAPACK_DLAMCH)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"DSTEVR - Lapack routine is unavailable.");
#else
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscErrorCode ierr;
  int            nconv,i,j,k,n,m,*perm,restart,
                 *isuppz,*iwork,mout,lwork,liwork,il,iu,info,
                 ncv=eps->ncv;
  Vec            f=eps->work[0];
  PetscScalar    *T=eps->T;
  PetscReal      lev,*ritz,*Y,*bnd,*D,*E,*work,anorm,beta,norm,abstol,vl,vu,restart_ritz;
  PetscTruth     breakdown,orthog,reorthog,normalize;
  char           *conv;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PETSC_NULL,"-reorthog",&reorthog);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-normalize",&normalize);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-orthog",&orthog);CHKERRQ(ierr);
  eps->level_orthog = 0.0;
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&D);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&E);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&ritz);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscReal),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(2*ncv*sizeof(int),&isuppz);CHKERRQ(ierr);
  lwork = 20*ncv;
  ierr = PetscMalloc(lwork*sizeof(PetscReal),&work);CHKERRQ(ierr);
  liwork = 10*ncv;
  ierr = PetscMalloc(liwork*sizeof(int),&iwork);CHKERRQ(ierr);

  ierr = PetscMalloc(ncv*sizeof(PetscReal),&bnd);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(int),&perm);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(char),&conv);CHKERRQ(ierr);

  /* The first Lanczos vector is the normalized initial vector */
  ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
  
  abstol = 2.0*LAPACKlamch_("S",1);
  anorm = 1.0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-anorm",&anorm,PETSC_NULL);CHKERRQ(ierr);
  nconv = 0;
  eps->its = 0;
  restart_ritz = 0.0;
  for (i=0;i<eps->ncv;i++) eps->eigi[i]=0.0;
  EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,ncv);
  
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    /* Compute an ncv-step Lanczos factorization */
    m = ncv;
    ierr = EPSBasicLanczos(eps,T,eps->V,nconv,&m,f,&beta,&breakdown,anorm);CHKERRQ(ierr);

    if (orthog) {
      ierr = SlepcCheckOrthogonality(eps->V+nconv,m - nconv,eps->V+nconv,m - nconv,PETSC_NULL,&lev);CHKERRQ(ierr);
      if (lev > eps->level_orthog) eps->level_orthog = lev;
    }

    /* Extract the tridiagonal block from T. Store the diagonal elements in D 
       and the off-diagonal elements in E  */
    n = m - nconv;
    for (i=0;i<n-1;i++) {
      D[i] = PetscRealPart(T[(i+nconv)*(ncv+1)]);
      E[i] = PetscRealPart(T[(i+nconv)*(ncv+1)+1]);
    }
    D[n-1] = PetscRealPart(T[(n-1+nconv)*(ncv+1)]);

    /* Compute eigenvalues and eigenvectors Y of the tridiagonal block */
    LAPACKstevr_("V","A",&n,D,E,&vl,&vu,&il,&iu,&abstol,&mout,ritz,Y,&n,isuppz,work,&lwork,iwork,&liwork,&info,1,1);
    if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xSTEVR %i",info);

    /* Estimate ||A|| */
    for (i=0;i<n;i++) 
      if (PetscAbsReal(ritz[i]) > anorm) anorm = PetscAbsReal(ritz[i]);
    
    /* Compute residual norm estimates as beta*abs(Y(m,:)) + eps*||A|| */
    for (i=0;i<n;i++)
      bnd[i] = beta*PetscAbsReal(Y[i*n+n-1]) + PETSC_MACHINE_EPSILON*anorm;

#ifdef PETSC_USE_COMPLEX
    for (i=0;i<n;i++) {
      eps->eigr[i+nconv] = ritz[i];
    }
    ierr = EPSSortEigenvalues(n,eps->eigr+nconv,eps->eigi,eps->which,n,perm);CHKERRQ(ierr);
#else
    ierr = EPSSortEigenvalues(n,ritz,eps->eigi,eps->which,n,perm);CHKERRQ(ierr);
#endif

    /* Look for converged eigenpairs */
    k = nconv;
    for (i=0;i<n;i++) {
      eps->eigr[k] = ritz[perm[i]];
      eps->errest[k] = bnd[perm[i]] / PetscAbsScalar(eps->eigr[k]);    
      if (eps->errest[k] < eps->tol) {
	      
	if (lanczos->reorthog == EPSLANCZOS_REORTHOG_LOCAL) {
          if (i>0 && PetscAbsScalar((eps->eigr[k]-ritz[perm[i-1]])/eps->eigr[k]) < eps->tol) {
  	    /* Discard repeated eigenvalues */
            conv[i] = 'R';
	    continue;
 	  }
	}
	  
	ierr = VecSet(eps->AV[k],0.0);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
	ierr = VecMAXPY(eps->AV[k],n,Y+perm[i]*n,eps->V+nconv);CHKERRQ(ierr);
#else
        for (j=0;j<n;j++) {
	  ierr = VecAXPY(eps->AV[k],Y[perm[i]*n+j],eps->V[nconv+j]);CHKERRQ(ierr);
	}
#endif	  

	if (lanczos->reorthog == EPSLANCZOS_REORTHOG_LOCAL) {
	  /* normalize locked vector and compute residual norm */
	  ierr = VecNorm(eps->AV[k],NORM_2,&norm);CHKERRQ(ierr);
          ierr = VecScale(eps->AV[k],1.0/norm);CHKERRQ(ierr);
	  ierr = STApply(eps->OP,eps->AV[k],f);CHKERRQ(ierr);
	  ierr = VecAXPY(f,-eps->eigr[k],eps->AV[k]);CHKERRQ(ierr);
	  ierr = VecNorm(f,NORM_2,&norm);CHKERRQ(ierr);
	  eps->errest[k] = norm / PetscAbsScalar(eps->eigr[k]);
          if (eps->errest[k] >= eps->tol) {
	    conv[i] = 'S';
	    continue;
	  }
	}
	  
        conv[i] = 'C';
        k++;
      } else conv[i] = 'N';
    }

    /* Look for non-converged eigenpairs */
    j = k;
    restart = -1;
    for (i=0;i<n;i++) {
      if (conv[i] != 'C') {
        if (restart == -1 && conv[i] == 'N') restart = i;
        eps->eigr[j] = ritz[perm[i]];
        eps->errest[j] = bnd[perm[i]] / ritz[perm[i]];
        j++;
      } 
    }

    if (breakdown) {
      restart = -1;
      PetscInfo2(eps,"Breakdown in Lanczos method (it=%i norm=%g)\n",eps->its,beta);
      eps->count_breakdown++;
    }
    
    if (k<eps->nev) {
      if (lanczos->reorthog == EPSLANCZOS_REORTHOG_SELECTIVE && restart != -1) {
        /* Avoid stagnation in selective reorthogonalization */
	if (PetscAbsScalar(restart_ritz - ritz[perm[restart]]) < PETSC_MACHINE_EPSILON) {
	  restart = -1;
	  restart_ritz = 0;
        } else restart_ritz = ritz[perm[restart]];
      }
      if (restart != -1) {
	/* Use first non-converged vector for restarting */
	ierr = VecSet(eps->AV[k],0.0);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
	ierr = VecMAXPY(eps->AV[k],n,Y+perm[restart]*n,eps->V+nconv);CHKERRQ(ierr);
#else
        for (j=0;j<n;j++) {
	  ierr = VecAXPY(eps->AV[k],Y[perm[restart]*n+j],eps->V[nconv+j]);CHKERRQ(ierr);
	}
#endif
	ierr = VecCopy(eps->AV[k],eps->V[k]);CHKERRQ(ierr);
      }
    }
    
    /* Copy converged vectors to V */
    for (i=nconv;i<k;i++) {
      ierr = VecCopy(eps->AV[i],eps->V[i]);CHKERRQ(ierr);
    }

    if (k<eps->nev) {
      if (restart == -1) {
	/* Use random vector for restarting */
	PetscInfo(eps,"Using random vector for restart\n");
	ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
      } else switch (lanczos->reorthog) {
      case EPSLANCZOS_REORTHOG_LOCAL:
      case EPSLANCZOS_REORTHOG_SELECTIVE:
      case EPSLANCZOS_REORTHOG_PERIODIC:
      case EPSLANCZOS_REORTHOG_PARTIAL:
        /* Reorthonormalize restart vector */
	ierr = EPSOrthogonalize(eps,eps->nds+k,eps->DSV,eps->V[k],PETSC_NULL,&norm,&breakdown);CHKERRQ(ierr);
	ierr = VecScale(eps->V[k],1.0/norm);CHKERRQ(ierr);
	break;
      default:
         breakdown = PETSC_FALSE;
      }
      if (breakdown) {
	eps->reason = EPS_DIVERGED_BREAKDOWN;
	PetscInfo(eps,"Unable to generate more start vectors\n");
      }
    }

    nconv = k;
    EPSMonitor(eps,eps->its,nconv,eps->eigr,eps->eigi,eps->errest,nconv+n);
    if (eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
    if (nconv >= eps->nev) eps->reason = EPS_CONVERGED_TOL;
  }
  
  eps->nconv = nconv;

  ierr = PetscFree(D);CHKERRQ(ierr);
  ierr = PetscFree(E);CHKERRQ(ierr);
  ierr = PetscFree(ritz);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  ierr = PetscFree(isuppz);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);

  ierr = PetscFree(bnd);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  ierr = PetscFree(conv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif 
}

static const char *lanczoslist[6] = { "local", "full", "selective", "periodic", "partial" , "delayed" };

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_LANCZOS"
PetscErrorCode EPSSetFromOptions_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos = (EPS_LANCZOS *)eps->data;
  PetscTruth     flg;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("LANCZOS options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-eps_lanczos_reorthog","Lanczos reorthogonalization","EPSLanczosSetReorthog",lanczoslist,6,lanczoslist[lanczos->reorthog],&i,&flg);CHKERRQ(ierr);
  if (flg) lanczos->reorthog = (EPSLanczosReorthogType)i;
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosSetReorthog_LANCZOS"
PetscErrorCode EPSLanczosSetReorthog_LANCZOS(EPS eps,EPSLanczosReorthogType reorthog)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;

  PetscFunctionBegin;
  switch (reorthog) {
    case EPSLANCZOS_REORTHOG_LOCAL:
    case EPSLANCZOS_REORTHOG_FULL:
    case EPSLANCZOS_REORTHOG_DELAYED:
    case EPSLANCZOS_REORTHOG_SELECTIVE:
    case EPSLANCZOS_REORTHOG_PERIODIC:
    case EPSLANCZOS_REORTHOG_PARTIAL:
      lanczos->reorthog = reorthog;
      break;
    default:
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid reorthogonalization type");
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosSetReorthog"
/*@
   EPSLanczosSetReorthog - Sets the type of reorthogonalization used during the Lanczos
   iteration. 

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  reorthog - the type of reorthogonalization

   Options Database Key:
.  -eps_lanczos_orthog - Sets the reorthogonalization type (either 'local', 'selective',
                         'periodic', 'partial' or 'full')
   
   Level: advanced

.seealso: EPSLanczosGetReorthog()
@*/
PetscErrorCode EPSLanczosSetReorthog(EPS eps,EPSLanczosReorthogType reorthog)
{
  PetscErrorCode ierr, (*f)(EPS,EPSLanczosReorthogType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSLanczosSetReorthog_C",(void (**)())&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(eps,reorthog);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosGetReorthog_LANCZOS"
PetscErrorCode EPSLanczosGetReorthog_LANCZOS(EPS eps,EPSLanczosReorthogType *reorthog)
{
  EPS_LANCZOS *lanczos = (EPS_LANCZOS *)eps->data;
  PetscFunctionBegin;
  *reorthog = lanczos->reorthog;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "EPSLanczosGetReorthog"
/*@C
   EPSLanczosGetReorthog - Gets the type of reorthogonalization used during the Lanczos
   iteration. 

   Collective on EPS

   Input Parameter:
.  eps - the eigenproblem solver context

   Input Parameter:
.  reorthog - the type of reorthogonalization

   Level: advanced

.seealso: EPSLanczosSetReorthog()
@*/
PetscErrorCode EPSLanczosGetReorthog(EPS eps,EPSLanczosReorthogType *reorthog)
{
  PetscErrorCode ierr, (*f)(EPS,EPSLanczosReorthogType*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)eps,"EPSLanczosGetReorthog_C",(void (**)())&f);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(1,"Viewer type %s not supported for EPSLANCZOS",((PetscObject)viewer)->type_name);
  }  
  ierr = PetscViewerASCIIPrintf(viewer,"reorthogonalization: %s\n",lanczoslist[lanczos->reorthog]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
EXTERN PetscErrorCode EPSSolve_TS_LANCZOS(EPS);
*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_LANCZOS"
PetscErrorCode EPSCreate_LANCZOS(EPS eps)
{
  PetscErrorCode ierr;
  EPS_LANCZOS    *lanczos;

  PetscFunctionBegin;
  ierr = PetscNew(EPS_LANCZOS,&lanczos);CHKERRQ(ierr);
  PetscLogObjectMemory(eps,sizeof(EPS_LANCZOS));
  eps->data                      = (void *) lanczos;
  eps->ops->solve                = EPSSolve_LANCZOS;
/*  eps->ops->solvets              = EPSSolve_TS_LANCZOS;*/
  eps->ops->setup                = EPSSetUp_LANCZOS;
  eps->ops->setfromoptions       = EPSSetFromOptions_LANCZOS;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->view                 = EPSView_LANCZOS;
  eps->ops->backtransform        = EPSBackTransform_Default;
  /*if (eps->solverclass==EPS_TWO_SIDE)
       eps->ops->computevectors       = EPSComputeVectors_Schur;
  else*/ eps->ops->computevectors       = EPSComputeVectors_Default;
  lanczos->reorthog              = EPSLANCZOS_REORTHOG_LOCAL;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosSetReorthog_C","EPSLanczosSetReorthog_LANCZOS",EPSLanczosSetReorthog_LANCZOS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)eps,"EPSLanczosGetReorthog_C","EPSLanczosGetReorthog_LANCZOS",EPSLanczosGetReorthog_LANCZOS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

