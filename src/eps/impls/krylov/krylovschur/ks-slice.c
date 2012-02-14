/*                       

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur with spectrum slicing for symmetric eigenproblems

   References:

       [1] R.G. Grimes et al., "A shifted block Lanczos algorithm for solving
           sparse symmetric generalized eigenproblems", SIAM J. Matrix Analysis
           and App., 15(1), pp. 228–272, 1994.

       [2] G.W. Stewart, "A Krylov-Schur Algorithm for Large Eigenproblems",
           SIAM J. Matrix Analysis and App., 23(3), pp. 601-614, 2001. 

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <private/epsimpl.h>                /*I "slepceps.h" I*/
#include <slepcblaslapack.h>

extern PetscErrorCode EPSProjectedKSSym(EPS,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscReal*,PetscInt*);

/* Type of data characterizing a shift (place from where an eps is applied) */
typedef struct _n_shift *shift;
struct _n_shift{
  PetscReal	value;
  PetscInt	inertia;
  PetscBool	comp[2]; /* Shows completion of subintervals (left and right) */
  shift  	neighb[2];/* Adjacent shifts */
  PetscInt	index;/* Index in eig where found values are stored */
  PetscInt	neigs; /* Number of values found */
  PetscReal     ext[2];   /* Limits for accepted values */ 
  PetscInt      nsch[2];  /* Number of missing values for each subinterval */
  PetscInt      nconv[2]; /* Converged on each side (accepted or not)*/
  PetscBool     expf;
};

/* Type of data  for storing the state of spectrum slicing*/
struct _n_SR{
  PetscReal       int0,int1; /* Extremes of the interval */
  PetscInt        dir; /* Determines the order of values in eig (+1 incr, -1 decr) */
  PetscBool       hasEnd; /* Tells whether the interval has an end */
  PetscInt        inertia0,inertia1;
  Vec             *V;
  PetscScalar     *eig,*eigi,*monit,*back;
  PetscReal       *errest;
  PetscInt        *perm;/* Permutation for keeping the eigenvalues in order */
  PetscInt        numEigs; /* Number of eigenvalues in the interval */
  PetscInt        indexEig;
  shift           sPres; /* Present shift */
  shift           *pending;/* Pending shifts array */
  PetscInt        nPend;/* Number of pending shifts */
  PetscInt        maxPend;/* Size of "pending" array */
  Vec             *VDef; /* Vector for deflation */
  PetscInt        *idxDef;/* For deflation */
  PetscInt        nMAXCompl;
  PetscInt        iterCompl;
  PetscInt        itsKs; /* Krylovschur restarts */
  PetscInt        nleap;
  shift           s0;/* Initial shift */
  PetscScalar     *S;/* Matrix for projected problem */
  PetscInt        nS;
  PetscReal       beta;
  shift           sPrev;
};
typedef struct _n_SR  *SR;

/* 
   Fills the fields of a shift structure

*/
#undef __FUNCT__
#define __FUNCT__ "EPSCreateShift"
static PetscErrorCode EPSCreateShift(EPS eps,PetscReal val, shift neighb0,shift neighb1)
{
  PetscErrorCode   ierr;
  shift            s,*pending2;
  PetscInt         i;
  SR               sr;

  PetscFunctionBegin;
  sr = (SR)(eps->data);
  ierr = PetscMalloc(sizeof(struct _n_shift),&s);CHKERRQ(ierr);
  s->value = val;
  s->neighb[0] = neighb0;
  if(neighb0) neighb0->neighb[1] = s;
  s->neighb[1] = neighb1;
  if(neighb1) neighb1->neighb[0] = s;
  s->comp[0] = PETSC_FALSE;
  s->comp[1] = PETSC_FALSE;
  s->index = -1;
  s->neigs = 0;
  s->nconv[0] = s->nconv[1] = 0;
  s->nsch[0] = s->nsch[1]=0;
  /* Inserts in the stack of pending shifts */
  /* If needed, the array is resized */
  if(sr->nPend >= sr->maxPend){
    sr->maxPend *= 2;
    ierr = PetscMalloc((sr->maxPend)*sizeof(shift),&pending2);CHKERRQ(ierr);
    for(i=0;i < sr->nPend; i++)pending2[i] = sr->pending[i];
    ierr = PetscFree(sr->pending);CHKERRQ(ierr);
    sr->pending = pending2;
  }
  sr->pending[sr->nPend++]=s;
  PetscFunctionReturn(0);
}

/* Provides next shift to be computed */
#undef __FUNCT__
#define __FUNCT__ "EPSExtractShift"
static PetscErrorCode EPSExtractShift(EPS eps){
  PetscErrorCode   ierr;
  PetscInt         iner,dir,i,k;
  Mat              F;
  PC               pc;
  KSP              ksp;
  SR               sr;

  PetscFunctionBegin;  
  sr = (SR)(eps->data);
  if(sr->nPend > 0){
    sr->sPrev = sr->sPres;
    sr->sPres = sr->pending[--sr->nPend];
    ierr = STSetShift(eps->OP, sr->sPres->value);CHKERRQ(ierr);
    ierr = STGetKSP(eps->OP, &ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
    ierr = MatGetInertia(F,&iner,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    sr->sPres->inertia = iner;
    eps->target = sr->sPres->value;
    eps->reason = EPS_CONVERGED_ITERATING;
    eps->its = 0;
    sr->sPres->expf = PETSC_FALSE;
    /* For rational Krylov */
    if(sr->nS >0 && (sr->sPrev == sr->sPres->neighb[0] || sr->sPrev == sr->sPres->neighb[1]) ){
      dir = (sr->sPres->neighb[0] == sr->sPrev)?1:-1;
      dir*=sr->dir;
      k = 0;
      for(i=0;i<sr->nS;i++){
        if(dir*sr->S[i] >0){
          sr->S[k] = sr->S[i];
          sr->S[sr->nS+k] = sr->S[sr->nS+i];
          ierr = VecCopy(eps->V[eps->nconv+i],eps->V[k++]);CHKERRQ(ierr);
          if(k>=sr->nS/2)break;
        }
      }
      for(i=0;i<k;i++)sr->S[k+i] = sr->S[sr->nS+i];
      sr->nS = k;
      /* Normalize u and append it to V */      
      ierr = VecAXPBY(eps->V[sr->nS],1.0/sr->beta,0.0,eps->work[0]);CHKERRQ(ierr);
    }
    eps->nconv = 0;
  }else sr->sPres = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSUpdateShiftRKS"
static PetscErrorCode EPSUpdateShiftRKS(EPS eps,PetscInt n,PetscReal sigma1,PetscReal sigma2,PetscScalar *S)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscScalar    *L,*tau,*work2,*R,*work,alpha;
  PetscBLASInt   n1,n0,lwork,info;

  PetscFunctionBegin;
  lwork = PetscBLASIntCast(n+1);
  i = 2*n*n+4*n+2;
  ierr = PetscMalloc(i*sizeof(PetscScalar),&work);CHKERRQ(ierr);  
  ierr = PetscMemzero(work,i*sizeof(PetscScalar));CHKERRQ(ierr);
  L = work;/* size (n+1)*(n+1) */
  tau = L+(n+1)*(n+1);
  work2 = tau+n;
  R = work2+(n+1);
  
  for (i=0;i<n;i++) 
    L[i+i*(n+1)] = 1.0+(sigma1-sigma2)*S[i];
  for (i=0;i<n;i++)
    L[n+i*(n+1)] = (sigma1-sigma2)*S[n+i];
  ierr = PetscMemzero(S,(n+1)*n*sizeof(PetscScalar));CHKERRQ(ierr);
  
  /* Compute qr */
  n1 = PetscBLASIntCast(n+1);
  n0 = PetscBLASIntCast(n);
  LAPACKgeqrf_(&n1,&n0,L,&n1,tau,work2,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);
  /* Copying R from L */
  for (j=0;j<n;j++)
    for(i=0;i<=j;i++)
      R[i+j*n]=L[i+j*(n+1)];
  /* Compute the orthogonal matrix in L */
  LAPACKorgqr_(&n1,&n1,&n0,L,&n1,tau,work2,&lwork,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGQR %d",info);
  /* Compute the updated matrix of projected problem */
  for(j=0;j<n;j++){
    for(i=0;i<n+1;i++)
      S[j*(n+1)+i]=L[i*(n+1)+j];
  }
  alpha = -1.0/(sigma1-sigma2);
  BLAStrsm_("R","U","N","N",&n1,&n0,&alpha,R,&n0,S,&n1);
  for(i=0;i<n;i++)
    S[(n+1)*i+i]-=alpha;
  /* Update vectors */
  ierr = SlepcUpdateVectors(n+1,eps->V,0,n+1,L,n+1,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscFree(work);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSProjectedKS_Slice"
/*
   EPSProjectedKS_ - Solves the projected eigenproblem in the Krylov-Schur
   method (Spectrum Slicing).

   On input:
     n is the matrix dimension
     l is the number of vectors kept in previous restart
     a contains diagonal elements (length n)
     b contains offdiagonal elements (length n-1)

   On output:
     eig is the sorted list of eigenvalues
     Q is the eigenvector matrix (order n, leading dimension n)

   Workspace:
     work is workspace to store a real square matrix of order n
     perm is workspace to store 2n integers
*/
static PetscErrorCode EPSProjectedKS_Slice(EPS eps,PetscInt n_,PetscScalar *Z,PetscInt l,PetscReal *d,PetscReal *e,PetscScalar *eig,PetscScalar *Q,PetscReal *work,PetscInt *perm)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,p;
  PetscReal      rtmp,*Qreal = (PetscReal*)Q;
  PetscBLASInt   n,n1,n2,lwork,info;

#if defined(SLEPC_MISSING_LAPACK_SYTRD) || defined(SLEPC_MISSING_LAPACK_ORGTR) || defined(SLEPC_MISSING_LAPACK_STEQR)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SYTRD/ORGTR/STEQR - Lapack routine is unavailable.");
#else
  PetscFunctionBegin;
/* Compute eigendecomposition of projected matrix */
  ierr = PetscLogEventBegin(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
  /* Quick return */
  if (n == 1) {
    Q[0] = 1.0;
    PetscFunctionReturn(0);    
  }
  n1 = PetscBLASIntCast(l+1);    /* size of leading block, including residuals */
  n2 = PetscBLASIntCast(n-l-1);  /* size of trailing block */
  ierr = PetscMemzero(work,n*n*sizeof(PetscReal));CHKERRQ(ierr);
  if(l>0){
    /* Flip matrix, copying the values saved in Q */
    if(!Z){
      for (i=0;i<n;i++) 
        work[(n-1-i)+(n-1-i)*n] = d[i];
      for (i=0;i<l;i++)
        work[(n-1-i)+(n-1-l)*n] = e[i];
      for (i=l;i<n-1;i++)
        work[(n-1-i)+(n-1-i-1)*n] = e[i];
    }else{
      for(i=0;i<n_-l-1;i++){
        work[i*n_+i] = d[n_-1-i];
        work[i*n_+i+1] = e[n_-2-i];
      }
      for(j=n_-l-1;j<n_;j++){
        for(i=j;i<n_;i++){
          work[j*n_+i] = PetscRealPart(Z[(n_-i-1)*(l+1)+(n_-j-1)]);
        }
      }
      work[(n_-l-1)*(n_+1)]=d[l];
    }
    /* Reduce (2,2)-block of flipped S to tridiagonal form */
    lwork = PetscBLASIntCast(n_*n_-n_);
    LAPACKsytrd_("L",&n1,work+n2*(n+1),&n,d,e,Qreal,Qreal+n,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSYTRD %d",info);

    /* Flip back diag and subdiag, put them in d and e */
    for (i=0;i<n-1;i++) {
      d[n-i-1] = work[i+i*n];
      e[n-i-2] = work[i+1+i*n];
    }
    d[0] = work[n-1+(n-1)*n];

    /* Compute the orthogonal matrix used for tridiagonalization */
    LAPACKorgtr_("L",&n1,work+n2*(n+1),&n,Qreal,Qreal+n,&lwork,&info);
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGTR %d",info);

    /* Create full-size Q, flipped back to original order */
    for (i=0;i<n;i++) 
      for (j=0;j<n;j++) 
        Qreal[i+j*n] = 0.0;
    for (i=n1;i<n;i++) 
      Qreal[i+i*n] = 1.0;
    for (i=0;i<n1;i++) 
      for (j=0;j<n1;j++) 
        Qreal[i+j*n] = work[n-i-1+(n-j-1)*n];

    /* Solve the tridiagonal eigenproblem */
    LAPACKsteqr_("V",&n,d,e,Qreal,&n,work,&info);
  }else {
    LAPACKsteqr_("I",&n,d,e,Qreal,&n,work,&info);
  }
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSTEQR %d",info);
  /* Sort eigendecomposition according to eps->which */
  ierr = EPSSortEigenvaluesReal(eps,n,d,perm);CHKERRQ(ierr);
  for (i=0;i<n;i++)
    eig[i] = d[perm[i]];
  for (i=0;i<n;i++) {
    p = perm[i];
    if (p != i) {
      j = i + 1;
      while (perm[j] != i) j++;
      perm[j] = p; perm[i] = i;
      /* swap eigenvectors i and j */
      for (k=0;k<n;k++) {
        rtmp = Qreal[k+p*n]; Qreal[k+p*n] = Qreal[k+i*n]; Qreal[k+i*n] = rtmp;
      }
    }
  }
#if defined(PETSC_USE_COMPLEX)
  for (j=n-1;j>=0;j--)
    for (i=n-1;i>=0;i--) 
      Q[i+j*n] = Qreal[i+j*n];
#endif
  ierr = PetscLogEventEnd(EPS_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

/*
   Symmetric KrylovSchur adapted to spectrum slicing:
   Allows searching an specific amount of eigenvalues in the subintervals left and right.
   Returns whether the search has succeeded
*/
#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchur_Slice"
static PetscErrorCode EPSKrylovSchur_Slice(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,conv,k,l,lds,lt,nv,m,*iwork,p,j;
  Vec            u=eps->work[0];
  PetscScalar    *Q,nu,rtmp,alpha;
  PetscReal      *a,*b,*work,beta;
  PetscBool      breakdown;
  PetscInt       count0,count1;
  PetscReal      lambda;
  shift          sPres;
  PetscBool      complIterating,iscayley;
  PetscBool      sch0,sch1;
  PetscInt       iterCompl=0,n0,n1,aux,auxc;
  SR             sr;

  PetscFunctionBegin;
  /* Spectrum slicing data */
  sr = (SR)eps->data;
  sPres = sr->sPres;
  complIterating =PETSC_FALSE;
  sch1 = sch0 = PETSC_TRUE;
  lds = PetscMin(eps->mpd,eps->ncv);
  ierr = PetscMalloc(lds*lds*sizeof(PetscReal),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(lds*lds*sizeof(PetscScalar),&Q);CHKERRQ(ierr);
  ierr = PetscMalloc(2*lds*sizeof(PetscInt),&iwork);CHKERRQ(ierr);
  lt = PetscMin(eps->nev+eps->mpd,eps->ncv);
  ierr = PetscMalloc(lt*sizeof(PetscReal),&a);CHKERRQ(ierr);
  ierr = PetscMalloc(lt*sizeof(PetscReal),&b);CHKERRQ(ierr);
  count0=0;count1=0; /* Found on both sides */
  /* filling in values for the monitor */
  if(eps->numbermonitors >0){
    ierr = PetscTypeCompare((PetscObject)eps->OP,STCAYLEY,&iscayley);CHKERRQ(ierr);
    if(iscayley){
      ierr = STCayleyGetAntishift(eps->OP,&nu);CHKERRQ(ierr);    
      for(i=0;i<sr->indexEig;i++){
        sr->monit[i]=(nu + sr->eig[i])/(sr->eig[i] - sPres->value);
      }
    }else{
      for(i=0;i<sr->indexEig;i++){
        sr->monit[i]=1.0/(sr->eig[i] - sPres->value);
      }
    }
  }
  if(sr->nS > 0 && (sPres->neighb[0] == sr->sPrev || sPres->neighb[1] == sr->sPrev) ){
    /* Rational Krylov */
    ierr = EPSUpdateShiftRKS(eps,sr->nS,sr->sPrev->value,sPres->value,sr->S);CHKERRQ(ierr);
    l = sr->nS;
  }else{
    /* Get the starting Lanczos vector */
    ierr = EPSGetStartVector(eps,0,eps->V[0],PETSC_NULL);CHKERRQ(ierr);
    l = 0;
  }
  /* Restart loop */
  while (eps->reason == EPS_CONVERGED_ITERATING) {
    eps->its++; sr->itsKs++;
    /* Compute an nv-step Lanczos factorization */
    m = PetscMin(eps->nconv+eps->mpd,eps->ncv);
    ierr = EPSFullLanczos(eps,a+l,b+l,eps->V,eps->nconv+l,&m,u,&breakdown);CHKERRQ(ierr);
    if(breakdown){/* explicit purification*/
      sPres->expf = PETSC_TRUE;     
    }
    nv = m - eps->nconv;
    beta = b[nv-1];
    /* Solve projected problem and compute residual norm estimates */
    if(eps->its == 1 && l > 0){/* After rational update */
      ierr =  EPSProjectedKS_Slice(eps,nv,sr->S,l,a,b,eps->eigr+eps->nconv,Q,work,iwork);CHKERRQ(ierr);
    }else{/* Restart */
      ierr = EPSProjectedKS_Slice(eps,nv,PETSC_NULL,l,a,b,eps->eigr+eps->nconv,Q,work,iwork);CHKERRQ(ierr);
    }
    /* Residual */
    ierr = EPSKrylovConvergence(eps,PETSC_TRUE,PETSC_TRUE,eps->nconv,nv,PETSC_NULL,nv,Q,eps->V+eps->nconv,nv,beta,1.0,&k,PETSC_NULL);CHKERRQ(ierr);
    /* Check convergence */ 
    conv=k=j=0;
    for(i=0;i<nv;i++)if(eps->errest[eps->nconv+i] < eps->tol)conv++;
    for(i=0;i<nv;i++){
      if(eps->errest[eps->nconv+i] < eps->tol){
        iwork[j++]=i;
      }else iwork[conv+k++]=i;
    }
    for(i=0;i<nv;i++){
      a[i]=PetscRealPart(eps->eigr[eps->nconv+i]);
      b[i]=eps->errest[eps->nconv+i];
    }
    for(i=0;i<nv;i++){
      eps->eigr[eps->nconv+i] = a[iwork[i]];
      eps->errest[eps->nconv+i] = b[iwork[i]];
    }
    for( i=0;i<nv;i++){
      p=iwork[i];
        if(p!=i){
          j=i+1;
          while(iwork[j]!=i)j++;
          iwork[j]=p;iwork[i]=i;
          for(k=0;k<nv;k++){
            rtmp=Q[k+p*nv];Q[k+p*nv]=Q[k+i*nv];Q[k+i*nv]=rtmp; 
          }
        } 
      }
      k=eps->nconv+conv;
    /* Checking values obtained for completing */
    for(i=0;i<k;i++){
      sr->back[i]=eps->eigr[i];
    }
    ierr = STBackTransform(eps->OP,k,sr->back,eps->eigi);CHKERRQ(ierr);
    count0=count1=0;
    for(i=0;i<k;i++){      
      lambda = PetscRealPart(sr->back[i]);
      if( ((sr->dir)*(sPres->value - lambda) > 0) && ((sr->dir)*(lambda - sPres->ext[0]) > 0))count0++;
      if( ((sr->dir)*(lambda - sPres->value) > 0) && ((sr->dir)*(sPres->ext[1] - lambda) > 0))count1++;
    }
    
    /* Checks completion */
    if( (!sch0||count0 >= sPres->nsch[0]) && (!sch1 ||count1 >= sPres->nsch[1]) ) {
      eps->reason = EPS_CONVERGED_TOL;
    }else {
      if(!complIterating && eps->its >= eps->max_it) eps->reason = EPS_DIVERGED_ITS;
      if(complIterating){
        if(--iterCompl <= 0) eps->reason = EPS_DIVERGED_ITS;
      }else if (k >= eps->nev) {
        n0 = sPres->nsch[0]-count0;
        n1 = sPres->nsch[1]-count1;
        if( sr->iterCompl>0 && ( (n0>0 && n0<= sr->nMAXCompl)||(n1>0&&n1<=sr->nMAXCompl) )){
          /* Iterating for completion*/
          complIterating = PETSC_TRUE;
          if(n0 >sr->nMAXCompl)sch0 = PETSC_FALSE;
          if(n1 >sr->nMAXCompl)sch1 = PETSC_FALSE;
          iterCompl = sr->iterCompl;
        }else eps->reason = EPS_CONVERGED_TOL;
      }      
    }
    /* Update l */
    if(eps->reason == EPS_CONVERGED_ITERATING )l = (eps->nconv+nv-k)/2;
    else l=eps->nconv+nv-k;
    if(breakdown)l=0;

    if (eps->reason == EPS_CONVERGED_ITERATING) {
      if (breakdown) {
        /* Start a new Lanczos factorization */
        ierr = PetscInfo2(eps,"Breakdown in Krylov-Schur method (it=%D norm=%G)\n",eps->its,beta);CHKERRQ(ierr);
        ierr = EPSGetStartVector(eps,k,eps->V[k],&breakdown);CHKERRQ(ierr);
        if (breakdown) {
          eps->reason = EPS_DIVERGED_BREAKDOWN;
          ierr = PetscInfo(eps,"Unable to generate more start vectors\n");CHKERRQ(ierr);
        }
      } else {
        /* Prepare the Rayleigh quotient for restart */
        for (i=0;i<l;i++) {
          a[i] = PetscRealPart(eps->eigr[i+k]);
          b[i] = PetscRealPart(Q[nv-1+(i+k-eps->nconv)*nv]*beta);
        }
      }
    }
    /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
    ierr = SlepcUpdateVectors(nv,eps->V+eps->nconv,0,k+l-eps->nconv,Q,nv,PETSC_FALSE);CHKERRQ(ierr);
    /* Purification */
    if(!sPres->expf){/* u not saved if breakdown */
      for(i=eps->nconv; i<k;i++){
        alpha = (Q[nv-1+(i-eps->nconv)*nv])/PetscRealPart(eps->eigr[i]);
        ierr = VecAXPY(eps->V[i], alpha, u);CHKERRQ(ierr);
      }
    }
    /* Normalize u and append it to V */
    if ( eps->reason == EPS_CONVERGED_ITERATING && !breakdown) {
      ierr = VecAXPBY(eps->V[k+l],1.0/beta,0.0,u);CHKERRQ(ierr);
    }
    /* Monitor */
    if(eps->numbermonitors >0){
      aux = auxc = 0;
      for(i=0;i<nv+eps->nconv;i++){
        sr->back[i]=eps->eigr[i];
      }
      ierr = STBackTransform(eps->OP,nv+eps->nconv,sr->back,eps->eigi);CHKERRQ(ierr);
      for(i=0;i<nv+eps->nconv;i++){
        lambda = PetscRealPart(sr->back[i]);
        if( ((sr->dir)*(lambda - sPres->ext[0]) > 0)&& ((sr->dir)*(sPres->ext[1] - lambda) > 0)){ 
          sr->monit[sr->indexEig+aux]=eps->eigr[i];
          sr->errest[sr->indexEig+aux]=eps->errest[i];
          aux++;
          if(eps->errest[i] < eps->tol)auxc++;
        }
      }
      ierr = EPSMonitor(eps,eps->its,auxc+sr->indexEig,sr->monit,sr->eigi,sr->errest,sr->indexEig+aux);CHKERRQ(ierr);
    }
    conv = k - eps->nconv;
    eps->nconv = k;
    
    if(eps->reason != EPS_CONVERGED_ITERATING){
      /* Store approximated values for next shift */
      sr->nS = l;
      for (i=0;i<l;i++) {
        sr->S[i] = eps->eigr[i+k];/* Diagonal elements */
        sr->S[i+l] = Q[nv-1+(i+conv)*nv]*beta; /* Out of diagonal elements */
      }
      sr->beta = beta;
    }
  }
  /* Check for completion */
  for(i=0;i< eps->nconv; i++){
    if( (sr->dir)*PetscRealPart(eps->eigr[i])>0 )sPres->nconv[1]++;
    else sPres->nconv[0]++;
  }
  sPres->comp[0] = (count0 >= sPres->nsch[0])?PETSC_TRUE:PETSC_FALSE;
  sPres->comp[1] = (count1 >= sPres->nsch[1])?PETSC_TRUE:PETSC_FALSE;
  if(count0 > sPres->nsch[0] || count1 > sPres->nsch[1])SETERRQ(((PetscObject)eps)->comm,1,"Unexpected error in Spectrum Slicing!\nMismatch between number of values found and information from inertia");

  ierr = PetscFree(Q);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
  Obtains value of subsequent shift
*/
#undef __FUNCT__
#define __FUNCT__ "EPSGetNewShiftValue"
static PetscErrorCode EPSGetNewShiftValue(EPS eps,PetscInt side,PetscReal *newS)
{
  PetscReal   lambda,d_prev;
  PetscInt    i,idxP;
  SR          sr;
  shift       sPres,s;

  PetscFunctionBegin;  
  sr = (SR)eps->data;
  sPres = sr->sPres;
  if( sPres->neighb[side]){
  /* Completing a previous interval */
    if(!sPres->neighb[side]->neighb[side] && sPres->neighb[side]->nconv[side]==0){ /* One of the ends might be too far from eigenvalues */
      if(side) *newS = (sPres->value + PetscRealPart(sr->eig[sr->perm[sr->indexEig-1]]))/2;
      else *newS = (sPres->value + PetscRealPart(sr->eig[sr->perm[0]]))/2;
    }else *newS=(sPres->value + sPres->neighb[side]->value)/2;
  }else{ /* (Only for side=1). Creating a new interval. */
    if(sPres->neigs==0){/* No value has been accepted*/
      if(sPres->neighb[0]){
        /* Multiplying by 10 the previous distance */
        *newS = sPres->value + 10*(sr->dir)*PetscAbsReal(sPres->value - sPres->neighb[0]->value);
        sr->nleap++;
        /* Stops when the interval is open and no values are found in the last 5 shifts (there might be infinite eigenvalues) */
        if( !sr->hasEnd && sr->nleap > 5)SETERRQ(((PetscObject)eps)->comm,1,"Unable to compute the wanted eigenvalues with open interval");           
      }else {/* First shift */
        if(eps->nconv != 0){
           /* Unaccepted values give information for next shift */
           idxP=0;/* Number of values left from shift */
           for(i=0;i<eps->nconv;i++){
             lambda = PetscRealPart(eps->eigr[i]);
             if( (sr->dir)*(lambda - sPres->value) <0)idxP++;
             else break;
           }
           /* Avoiding subtraction of eigenvalues (might be the same).*/
           if(idxP>0){
             d_prev = PetscAbsReal(sPres->value - PetscRealPart(eps->eigr[0]))/(idxP+0.3);
           }else {
             d_prev = PetscAbsReal(sPres->value - PetscRealPart(eps->eigr[eps->nconv-1]))/(eps->nconv+0.3);
           }
           *newS = sPres->value + ((sr->dir)*d_prev*eps->nev)/2;
        }else{/* No values found, no information for next shift */
          SETERRQ(((PetscObject)eps)->comm,1,"First shift renders no information"); 
        }
      }
    }else{/* Accepted values found */
      sr->nleap = 0;
      /* Average distance of values in previous subinterval */
      s = sPres->neighb[0];
      while(s && PetscAbs(s->inertia - sPres->inertia)==0){
        s = s->neighb[0];/* Looking for previous shifts with eigenvalues within */
      }
      if(s){
        d_prev = PetscAbsReal( (sPres->value - s->value)/(sPres->inertia - s->inertia));
      }else{/* First shift. Average distance obtained with values in this shift */
        /* first shift might be too far from first wanted eigenvalue (no values found outside the interval)*/
        if( (sr->dir)*(PetscRealPart(sr->eig[0])-sPres->value)>0 && PetscAbsReal( (PetscRealPart(sr->eig[sr->indexEig-1]) - PetscRealPart(sr->eig[0]))/PetscRealPart(sr->eig[0])) > PetscSqrtReal(eps->tol) ){
          d_prev =  PetscAbsReal( (PetscRealPart(sr->eig[sr->indexEig-1]) - PetscRealPart(sr->eig[0])))/(sPres->neigs+0.3);
        }else{
          d_prev = PetscAbsReal( PetscRealPart(sr->eig[sr->indexEig-1]) - sPres->value)/(sPres->neigs+0.3);          
        }
      }
      /* Average distance is used for next shift by adding it to value on the right or to shift */
      if( (sr->dir)*(PetscRealPart(sr->eig[sPres->index + sPres->neigs -1]) - sPres->value) >0){
        *newS = PetscRealPart(sr->eig[sPres->index + sPres->neigs -1])+ ((sr->dir)*d_prev*(eps->nev))/2;   
      }else{/* Last accepted value is on the left of shift. Adding to shift */
        *newS = sPres->value + ((sr->dir)*d_prev*(eps->nev))/2;
      }
    }
    /* End of interval can not be surpassed */
    if((sr->dir)*( sr->int1 - *newS) < 0) *newS = sr->int1;
  }/* of neighb[side]==null */
  PetscFunctionReturn(0);
}

/* 
  Function for sorting an array of real values
*/
#undef __FUNCT__
#define __FUNCT__ "sortRealEigenvalues"
static PetscErrorCode sortRealEigenvalues(PetscScalar *r,PetscInt *perm,PetscInt nr,PetscBool prev,PetscInt dir)
{
  PetscReal      re;
  PetscInt       i,j,tmp;
  
  PetscFunctionBegin; 
  if(!prev) for (i=0; i < nr; i++) { perm[i] = i; }
  /* Insertion sort */
  for (i=1; i < nr; i++) {
    re = PetscRealPart(r[perm[i]]);
    j = i-1;
    while ( j>=0 && dir*(re - PetscRealPart(r[perm[j]])) <= 0 ) {
      tmp = perm[j]; perm[j] = perm[j+1]; perm[j+1] = tmp; j--;
    }
  }
  PetscFunctionReturn(0);
}

/* Stores the pairs obtained since the last shift in the global arrays */
#undef __FUNCT__
#define __FUNCT__ "EPSStoreEigenpairs"
PetscErrorCode EPSStoreEigenpairs(EPS eps)
{
  PetscErrorCode ierr;
  PetscReal      lambda,err,norm;
  PetscInt       i,count;
  PetscBool      iscayley;
  SR             sr;
  shift          sPres;

  PetscFunctionBegin; 
  sr = (SR)(eps->data);
  sPres = sr->sPres;
  sPres->index = sr->indexEig;
  count = sr->indexEig;
  /* Back-transform */
  ierr = EPSBackTransform_Default(eps);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)eps->OP,STCAYLEY,&iscayley);CHKERRQ(ierr);
  /* Sort eigenvalues */
  ierr = sortRealEigenvalues(eps->eigr,eps->perm,eps->nconv,PETSC_FALSE,sr->dir);
  /* Values stored in global array */
  for( i=0; i < eps->nconv ;i++ ){
    lambda = PetscRealPart(eps->eigr[eps->perm[i]]);
    err = eps->errest[eps->perm[i]];

    if(  (sr->dir)*(lambda - sPres->ext[0]) > 0 && (sr->dir)*(sPres->ext[1] - lambda) > 0  ){/* Valid value */
      if(count>=sr->numEigs){/* Error found */
         SETERRQ(((PetscObject)eps)->comm,1,"Unexpected error in Spectrum Slicing!");
      }
      sr->eig[count] = lambda;
      sr->errest[count] = err;
      /* Unlikely explicit purification */
      if (sPres->expf && eps->isgeneralized && !iscayley){
        ierr = STApply(eps->OP,eps->V[eps->perm[i]],sr->V[count]);CHKERRQ(ierr);
        ierr = IPNorm(eps->ip,sr->V[count],&norm);CHKERRQ(ierr); 
        ierr = VecScale(sr->V[count],1.0/norm);CHKERRQ(ierr);
      }else{
        ierr = VecCopy(eps->V[eps->perm[i]], sr->V[count]);CHKERRQ(ierr);
      }
      count++;
    }
  }
  sPres->neigs = count - sr->indexEig;
  sr->indexEig = count;
  /* Global ordering array updating */ 
  ierr = sortRealEigenvalues(sr->eig,sr->perm,count,PETSC_TRUE,sr->dir);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSLookForDeflation"
PetscErrorCode EPSLookForDeflation(EPS eps)
{
  PetscReal       val;
  PetscInt        i,count0=0,count1=0;
  shift           sPres;
  PetscInt        ini,fin,k,idx0,idx1;
  SR              sr;

  PetscFunctionBegin; 
  sr = (SR)(eps->data);
  sPres = sr->sPres;

  if(sPres->neighb[0]) ini = (sr->dir)*(sPres->neighb[0]->inertia - sr->inertia0);
  else ini = 0;
  fin = sr->indexEig;
  /* Selection of ends for searching new values */
  if(!sPres->neighb[0]) sPres->ext[0] = sr->int0;/* First shift */
  else sPres->ext[0] = sPres->neighb[0]->value;
  if(!sPres->neighb[1]) {
    if(sr->hasEnd) sPres->ext[1] = sr->int1;
    else sPres->ext[1] = (sr->dir > 0)?PETSC_MAX_REAL:PETSC_MIN_REAL;
  }else sPres->ext[1] = sPres->neighb[1]->value;
  /* Selection of values between right and left ends */
  for(i=ini;i<fin;i++){
    val=PetscRealPart(sr->eig[sr->perm[i]]);
    /* Values to the right of left shift */
    if( (sr->dir)*(val - sPres->ext[1]) < 0 ){
      if((sr->dir)*(val - sPres->value) < 0)count0++;
      else count1++;
    }else break;
  }
  /* The number of values on each side are found */
  if(sPres->neighb[0]){
     sPres->nsch[0] = (sr->dir)*(sPres->inertia - sPres->neighb[0]->inertia)-count0;
     if(sPres->nsch[0]<0)SETERRQ(((PetscObject)eps)->comm,1,"Unexpected error in Spectrum Slicing!\nMismatch between number of values found and information from inertia");
  }else sPres->nsch[0] = 0;

  if(sPres->neighb[1]){
    sPres->nsch[1] = (sr->dir)*(sPres->neighb[1]->inertia - sPres->inertia) - count1;
    if(sPres->nsch[1]<0)SETERRQ(((PetscObject)eps)->comm,1,"Unexpected error in Spectrum Slicing!\nMismatch between number of values found and information from inertia");
  }else sPres->nsch[1] = (sr->dir)*(sr->inertia1 - sPres->inertia);

  /* Completing vector of indexes for deflation */
  idx0 = ini;
  idx1 = ini+count0+count1;
  k=0;
  for(i=idx0;i<idx1;i++)sr->idxDef[k++]=sr->perm[i];
  for(i=0;i<k;i++)sr->VDef[i]=sr->V[sr->idxDef[i]];
  eps->DS = sr->VDef;
  eps->nds = k;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "EPSSolve_KrylovSchur_Slice"
PetscErrorCode EPSSolve_KrylovSchur_Slice(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i,lds;
  PetscReal      newS;
  KSP            ksp;
  PC             pc;
  Mat            F;  
  PetscReal      *errest_left;
  Vec            t;
  SR             sr;
  shift          s;
 
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Spectrum slicing not supported in complex scalars"); 
#endif
  ierr = PetscMalloc(sizeof(struct _n_SR),&sr);CHKERRQ(ierr);
  eps->data = sr;
  sr->itsKs = 0;
  sr->nleap = 0;
  sr->nMAXCompl = eps->nev/4;
  sr->iterCompl = eps->max_it/4;
  sr->sPres = PETSC_NULL;
  sr->nS = 0;
  lds = PetscMin(eps->mpd,eps->ncv);
  /* Checking presence of ends and finding direction */
  if( eps->inta > PETSC_MIN_REAL){
    sr->int0 = eps->inta;
    sr->int1 = eps->intb;
    sr->dir = 1;
    if(eps->intb >= PETSC_MAX_REAL){ /* Right-open interval */
      sr->hasEnd = PETSC_FALSE;
      sr->inertia1 = eps->n;
    }else sr->hasEnd = PETSC_TRUE;
  }else{ /* Left-open interval */
    sr->int0 = eps->intb;
    sr->int1 = eps->inta;
    sr->dir = -1;
    sr->hasEnd = PETSC_FALSE;
    sr->inertia1 = 0;
  }
  /* Array of pending shifts */
  sr->maxPend = 100;/* Initial size */
  ierr = PetscMalloc((sr->maxPend)*sizeof(shift),&sr->pending);CHKERRQ(ierr);
  if(sr->hasEnd){
    ierr = STGetKSP(eps->OP, &ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);
    /* Not looking for values in b (just inertia).*/
    ierr = MatGetInertia(F,&sr->inertia1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    ierr = PCReset(pc);CHKERRQ(ierr); /* avoiding memory leak */
  }
  sr->nPend = 0;
  ierr = EPSCreateShift(eps,sr->int0,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSExtractShift(eps);
  sr->s0 = sr->sPres;
  sr->inertia0 = sr->s0->inertia;
  sr->numEigs = (sr->dir)*(sr->inertia1 - sr->inertia0);
  sr->indexEig = 0;
  /* Only with eigenvalues present in the interval ...*/
  if(sr->numEigs==0){ 
    eps->reason = EPS_CONVERGED_TOL;
    ierr = PetscFree(sr->s0);CHKERRQ(ierr);
    ierr = PetscFree(sr->pending);CHKERRQ(ierr);
    ierr = PetscFree(sr);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* Memory reservation for eig, V and perm */
  ierr = PetscMalloc(lds*lds*sizeof(PetscScalar),&sr->S);CHKERRQ(ierr);
  ierr = PetscMemzero(sr->S,lds*lds*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc((sr->numEigs)*sizeof(PetscScalar),&sr->eig);CHKERRQ(ierr);
  ierr = PetscMalloc((sr->numEigs)*sizeof(PetscScalar),&sr->eigi);CHKERRQ(ierr);
  ierr = PetscMalloc((sr->numEigs+eps->ncv) *sizeof(PetscReal),&sr->errest);CHKERRQ(ierr);
  ierr = PetscMalloc((sr->numEigs+eps->ncv)*sizeof(PetscReal),&errest_left);CHKERRQ(ierr);
  ierr = PetscMalloc((sr->numEigs+eps->ncv)*sizeof(PetscScalar),&sr->monit);CHKERRQ(ierr);
  ierr = PetscMalloc((eps->ncv)*sizeof(PetscScalar),&sr->back);CHKERRQ(ierr);
  for(i=0;i<sr->numEigs;i++){sr->eigi[i]=0;sr->eig[i] = 0;}
  for(i=0;i<sr->numEigs+eps->ncv;i++){errest_left[i]=0;sr->errest[i]=0;sr->monit[i]=0;}
  ierr = VecCreateMPI(((PetscObject)eps)->comm,eps->nloc,PETSC_DECIDE,&t);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(t,sr->numEigs,&sr->V);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  /* Vector for maintaining order of eigenvalues */
  ierr = PetscMalloc((sr->numEigs)*sizeof(PetscInt),&sr->perm);CHKERRQ(ierr);
  for(i=0;i< sr->numEigs;i++)sr->perm[i]=i;
  /* Vectors for deflation */
  ierr = PetscMalloc((sr->numEigs)*sizeof(PetscInt),&sr->idxDef);CHKERRQ(ierr);
  ierr = PetscMalloc((sr->numEigs)*sizeof(Vec),&sr->VDef);CHKERRQ(ierr);
  sr->indexEig = 0;
  /* Main loop */
  while(sr->sPres){
    /* Search for deflation */
    ierr = EPSLookForDeflation(eps);CHKERRQ(ierr);
    /* KrylovSchur */
    ierr = EPSKrylovSchur_Slice(eps);CHKERRQ(ierr);
    
    ierr = EPSStoreEigenpairs(eps);CHKERRQ(ierr);
    /* Select new shift */
    if(!sr->sPres->comp[1]){
      ierr = EPSGetNewShiftValue(eps,1,&newS);CHKERRQ(ierr);
      ierr = EPSCreateShift(eps,newS,sr->sPres,sr->sPres->neighb[1]);
    }
    if(!sr->sPres->comp[0]){
      /* Completing earlier interval */
      ierr = EPSGetNewShiftValue(eps,0,&newS);CHKERRQ(ierr);
      ierr = EPSCreateShift(eps,newS,sr->sPres->neighb[0],sr->sPres);
    }
    /* Preparing for a new search of values */
    ierr = EPSExtractShift(eps);CHKERRQ(ierr);
  }

  /* Updating eps values prior to exit */
  
  ierr = VecDestroyVecs(eps->allocated_ncv,&eps->V);CHKERRQ(ierr);
  eps->V = sr->V;
  ierr = PetscFree(sr->S);CHKERRQ(ierr);
  ierr = PetscFree(eps->eigr);CHKERRQ(ierr);
  ierr = PetscFree(eps->eigi);CHKERRQ(ierr);
  ierr = PetscFree(eps->errest);CHKERRQ(ierr);
  ierr = PetscFree(eps->errest_left);CHKERRQ(ierr);
  ierr = PetscFree(eps->perm);CHKERRQ(ierr);
  eps->eigr = sr->eig;
  eps->eigi = sr->eigi;
  eps->errest = sr->errest;
  eps->errest_left = errest_left;
  eps->perm = sr->perm;
  eps->ncv = eps->allocated_ncv = sr->numEigs;
  eps->nconv = sr->indexEig;
  eps->reason = EPS_CONVERGED_TOL;
  eps->its = sr->itsKs;
  eps->nds = 0;
  eps->DS = PETSC_NULL;
  eps->evecsavailable = PETSC_TRUE; 
  ierr = PetscFree(sr->VDef);CHKERRQ(ierr);
  ierr = PetscFree(sr->idxDef);CHKERRQ(ierr);
  ierr = PetscFree(sr->pending);CHKERRQ(ierr);
  ierr = PetscFree(sr->monit);CHKERRQ(ierr);
  ierr = PetscFree(sr->back);CHKERRQ(ierr);
  /* Reviewing list of shifts to free memory */
  s = sr->s0;
  if(s){
    while(s->neighb[1]){
      s = s->neighb[1];
      ierr = PetscFree(s->neighb[0]);CHKERRQ(ierr);
    }
    ierr = PetscFree(s);CHKERRQ(ierr);
  }
  ierr = PetscFree(sr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
