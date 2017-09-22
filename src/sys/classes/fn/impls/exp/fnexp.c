/*
   Exponential function  exp(x)

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/fnimpl.h>      /*I "slepcfn.h" I*/
#include <slepcblaslapack.h>

PetscErrorCode FNEvaluateFunction_Exp(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
  *y = PetscExpScalar(x);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateDerivative_Exp(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
  *y = PetscExpScalar(x);
  PetscFunctionReturn(0);
}

#define MAX_PADE 6
#define SWAP(a,b,t) {t=a;a=b;b=t;}

PetscErrorCode FNEvaluateFunctionMat_Exp_Pade(FN fn,Mat A,Mat B)
{
#if defined(PETSC_MISSING_LAPACK_GESV) || defined(SLEPC_MISSING_LAPACK_LANGE)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESV/LANGE - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscBLASInt   n,ld,ld2,*ipiv,info,inc=1;
  PetscInt       m,j,k,sexp;
  PetscBool      odd;
  const PetscInt p=MAX_PADE;
  PetscReal      c[MAX_PADE+1],s,*rwork;
  PetscScalar    scale,mone=-1.0,one=1.0,two=2.0,zero=0.0;
  PetscScalar    *Aa,*Ba,*As,*A2,*Q,*P,*W,*aux;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(A,&Aa);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&Ba);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ld  = n;
  ld2 = ld*ld;
  P   = Ba;
  ierr = PetscMalloc6(m*m,&Q,m*m,&W,m*m,&As,m*m,&A2,ld,&rwork,ld,&ipiv);CHKERRQ(ierr);
  ierr = PetscMemcpy(As,Aa,ld2*sizeof(PetscScalar));CHKERRQ(ierr);

  /* Pade' coefficients */
  c[0] = 1.0;
  for (k=1;k<=p;k++) c[k] = c[k-1]*(p+1-k)/(k*(2*p+1-k));

  /* Scaling */
  s = LAPACKlange_("I",&n,&n,As,&ld,rwork);
  ierr = PetscLogFlops(1.0*n*n);CHKERRQ(ierr);
  if (s>0.5) {
    sexp = PetscMax(0,(int)(PetscLogReal(s)/PetscLogReal(2.0))+2);
    scale = PetscPowRealInt(2.0,-sexp);
    PetscStackCallBLAS("BLASscal",BLASscal_(&ld2,&scale,As,&inc));
    ierr = PetscLogFlops(1.0*n*n);CHKERRQ(ierr);
  } else sexp = 0;

  /* Horner evaluation */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,As,&ld,As,&ld,&zero,A2,&ld));
  ierr = PetscLogFlops(2.0*n*n*n);CHKERRQ(ierr);
  ierr = PetscMemzero(Q,ld2*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(P,ld2*sizeof(PetscScalar));CHKERRQ(ierr);
  for (j=0;j<n;j++) {
    Q[j+j*ld] = c[p];
    P[j+j*ld] = c[p-1];
  }

  odd = PETSC_TRUE;
  for (k=p-1;k>0;k--) {
    if (odd) {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,Q,&ld,A2,&ld,&zero,W,&ld));
      SWAP(Q,W,aux);
      for (j=0;j<n;j++) Q[j+j*ld] += c[k-1];
      odd = PETSC_FALSE;
    } else {
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,P,&ld,A2,&ld,&zero,W,&ld));
      SWAP(P,W,aux);
      for (j=0;j<n;j++) P[j+j*ld] += c[k-1];
      odd = PETSC_TRUE;
    }
    ierr = PetscLogFlops(2.0*n*n*n);CHKERRQ(ierr);
  }
  if (odd) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,Q,&ld,As,&ld,&zero,W,&ld));
    SWAP(Q,W,aux);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&ld2,&mone,P,&inc,Q,&inc));
    PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n,&n,Q,&ld,ipiv,P,&ld,&info));
    SlepcCheckLapackInfo("gesv",info);
    PetscStackCallBLAS("BLASscal",BLASscal_(&ld2,&two,P,&inc));
    for (j=0;j<n;j++) P[j+j*ld] += 1.0;
    PetscStackCallBLAS("BLASscal",BLASscal_(&ld2,&mone,P,&inc));
  } else {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,P,&ld,As,&ld,&zero,W,&ld));
    SWAP(P,W,aux);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&ld2,&mone,P,&inc,Q,&inc));
    PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n,&n,Q,&ld,ipiv,P,&ld,&info));
    SlepcCheckLapackInfo("gesv",info);
    PetscStackCallBLAS("BLASscal",BLASscal_(&ld2,&two,P,&inc));
    for (j=0;j<n;j++) P[j+j*ld] += 1.0;
  }
  ierr = PetscLogFlops(2.0*n*n*n+2.0*n*n*n/3.0+4.0*n*n);CHKERRQ(ierr);

  for (k=1;k<=sexp;k++) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,P,&ld,P,&ld,&zero,W,&ld));
    ierr = PetscMemcpy(P,W,ld2*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  if (P!=Ba) { ierr = PetscMemcpy(Ba,P,ld2*sizeof(PetscScalar));CHKERRQ(ierr); }
  ierr = PetscLogFlops(2.0*n*n*n*sexp);CHKERRQ(ierr);

  ierr = PetscFree6(Q,W,As,A2,rwork,ipiv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(A,&Aa);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Ba);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#define ITMAX 5

/*
 * Estimate norm(A^m,1) by block 1-norm power method (required workspace is 11*n)
 */
static PetscReal normest1(PetscBLASInt n,PetscScalar *A,PetscInt m,PetscScalar *work,PetscRandom rand)
{
  PetscScalar  *X,*Y,*Z,*S,*S_old,*aux,val,sone=1.0,szero=0.0;
  PetscReal    est=0.0,est_old,vals[2],*zvals,maxzval[2],raux;
  PetscBLASInt i,j,t=2,it=0,ind[2],est_j=0,m1;

  PetscFunctionBegin;
  X = work;
  Y = work + 2*n;
  Z = work + 4*n;
  S = work + 6*n;
  S_old = work + 8*n;
  zvals = (PetscReal*)(work + 10*n);

  for (i=0;i<n;i++) {  /* X has columns of unit 1-norm */
    X[i] = 1.0/n;
    PetscRandomGetValue(rand,&val);
    if (PetscRealPart(val) < 0.5) X[i+n] = -1.0/n;
    else X[i+n] = 1.0/n;
  }
  for (i=0;i<t*n;i++) S[i] = 0.0;
  ind[0] = 0; ind[1] = 0;
  est_old = 0;
  while (1) {
    it++;
    for (j=0;j<m;j++) {  /* Y = A^m*X */
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&t,&n,&sone,A,&n,X,&n,&szero,Y,&n));
      if (j<m-1) SWAP(X,Y,aux);
    }
    for (j=0;j<t;j++) {  /* vals[j] = norm(Y(:,j),1) */
      vals[j] = 0.0;
      for (i=0;i<n;i++) vals[j] += PetscAbsScalar(Y[i+j*n]);
    }
    if (vals[0]<vals[1]) {
      SWAP(vals[0],vals[1],raux);
      m1 = 1;
    } else m1 = 0;
    est = vals[0];
    if (est>est_old || it==2) est_j = ind[m1];
    if (it>=2 && est<=est_old) {
      est = est_old;
      break;
    }
    est_old = est;
    if (it>ITMAX) break;
    SWAP(S,S_old,aux);
    for (i=0;i<t*n;i++) {  /* S = sign(Y) */
      S[i] = (PetscRealPart(Y[i]) < 0.0)? -1.0: 1.0;
    }
    for (j=0;j<m;j++) {  /* Z = (A^T)^m*S */
      PetscStackCallBLAS("BLASgemm",BLASgemm_("C","N",&n,&t,&n,&sone,A,&n,S,&n,&szero,Z,&n));
      if (j<m-1) SWAP(S,Z,aux);
    }
    maxzval[0] = -1; maxzval[1] = -1;
    ind[0] = 0; ind[1] = 0;
    for (i=0;i<n;i++) {  /* zvals[i] = norm(Z(i,:),inf) */
      zvals[i] = PetscMax(PetscAbsScalar(Z[i+0*n]),PetscAbsScalar(Z[i+1*n]));
      if (zvals[i]>maxzval[0]) {
        maxzval[0] = zvals[i];
        ind[0] = i;
      } else if (zvals[i]>maxzval[1]) {
        maxzval[1] = zvals[i];
        ind[1] = i;
      }
    }
    if (it>=2 && maxzval[0]==zvals[est_j]) break;
    for (i=0;i<t*n;i++) X[i] = 0.0;
    for (j=0;j<t;j++) X[ind[j]+j*n] = 1.0;
  }
  /* Flop count is roughly (it * 2*m * t*gemv) = 4*its*m*t*n*n */
  PetscLogFlops(4.0*it*m*t*n*n);
  PetscFunctionReturn(est);
}

#define SMALLN 100

/*
 * Estimate norm(A^m,1) (required workspace is 2*n*n)
 */
static PetscReal normAm(PetscBLASInt n,PetscScalar *A,PetscInt m,PetscScalar *work,PetscRandom rand)
{
  PetscScalar  *v=work,*w=work+n*n,*aux,sone=1.0,szero=0.0;
  PetscReal    nrm,rwork[1],tmp;
  PetscBLASInt i,j,one=1;
  PetscBool    isrealpos=PETSC_TRUE;

  PetscFunctionBegin;
  if (n<SMALLN) {   /* compute matrix power explicitly */
    if (m==1) {
      nrm = LAPACKlange_("O",&n,&n,A,&n,rwork);
      PetscLogFlops(1.0*n*n);
    } else {  /* m>=2 */
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,A,&n,A,&n,&szero,v,&n));
      for (j=0;j<m-2;j++) {
        PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,A,&n,v,&n,&szero,w,&n));
        SWAP(v,w,aux);
      }
      nrm = LAPACKlange_("O",&n,&n,v,&n,rwork);
      PetscLogFlops(2.0*n*n*n*(m-1)+1.0*n*n);
    }
  } else {
    for (i=0;i<n;i++)
      for (j=0;j<n;j++) 
#if defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(A[i+j*n])<0.0 || PetscImaginaryPart(A[i+j*n])!=0.0) { isrealpos = PETSC_FALSE; break; }
#else
        if (A[i+j*n]<0.0) { isrealpos = PETSC_FALSE; break; }
#endif
    if (isrealpos) {   /* for positive matrices only */
      for (i=0;i<n;i++) v[i] = 1.0;
      for (j=0;j<m;j++) {  /* w = A'*v */
        PetscStackCallBLAS("BLASgemv",BLASgemv_("C",&n,&n,&sone,A,&n,v,&one,&szero,w,&one));
        SWAP(v,w,aux);
      }
      PetscLogFlops(2.0*n*n*m);
      nrm = 0.0;
      for (i=0;i<n;i++) if ((tmp = PetscAbsScalar(v[i])) > nrm) nrm = tmp;   /* norm(v,inf) */
    } else {
      nrm = normest1(n,A,m,work,rand);
    }
  }
  PetscFunctionReturn(nrm);
}

/*
 * Function needed to compute optimal parameters (required workspace is 3*n*n)
 */
static PetscInt ell(PetscBLASInt n,PetscScalar *A,PetscReal coeff,PetscInt m,PetscScalar *work,PetscRandom rand)
{
  PetscScalar  *Ascaled=work;
  PetscReal    nrm,alpha,beta,rwork[1];
  PetscInt     t;
  PetscBLASInt i,j;

  PetscFunctionBegin;
  beta = PetscPowReal(coeff,1.0/(2*m+1));
  for (i=0;i<n;i++)
    for (j=0;j<n;j++) 
      Ascaled[i+j*n] = beta*PetscAbsScalar(A[i+j*n]);
  nrm = LAPACKlange_("O",&n,&n,A,&n,rwork);
  PetscLogFlops(2.0*n*n);
  alpha = normAm(n,Ascaled,2*m+1,work+n*n,rand)/nrm;
  t = PetscMax((PetscInt)PetscCeilReal(PetscLogReal(2.0*alpha/PETSC_MACHINE_EPSILON)/PetscLogReal(2.0)/(2*m)),0);
  PetscFunctionReturn(t);
}

/*
 * Compute scaling parameter (s) and order of Pade approximant (m)  (required workspace is 4*n*n)
 */
static PetscErrorCode expm_params(PetscInt n,PetscScalar **Apowers,PetscInt *s,PetscInt *m,PetscScalar *work)
{
  PetscErrorCode  ierr;
  PetscScalar     sfactor,sone=1.0,szero=0.0,*A=Apowers[0],*Ascaled;
  PetscReal       d4,d6,d8,d10,eta1,eta3,eta4,eta5,rwork[1];
  PetscBLASInt    n_,n2,one=1;
  PetscRandom     rand;
  const PetscReal coeff[5] = { 9.92063492063492e-06, 9.94131285136576e-11,  /* backward error function */
                               2.22819456055356e-16, 1.69079293431187e-22, 8.82996160201868e-36 };
  const PetscReal theta[5] = { 1.495585217958292e-002,    /* m = 3  */
                               2.539398330063230e-001,    /* m = 5  */
                               9.504178996162932e-001,    /* m = 7  */
                               2.097847961257068e+000,    /* m = 9  */
                               5.371920351148152e+000 };  /* m = 13 */

  PetscFunctionBegin;
  *s = 0;
  *m = 13;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rand);CHKERRQ(ierr);
  d4 = PetscPowReal(LAPACKlange_("O",&n_,&n_,Apowers[2],&n_,rwork),1.0/4.0);
  if (d4==0.0) { /* safeguard for the case A = 0 */
    *m = 3;
    goto done;
  }
  d6 = PetscPowReal(LAPACKlange_("O",&n_,&n_,Apowers[3],&n_,rwork),1.0/6.0);
  ierr = PetscLogFlops(2.0*n*n);CHKERRQ(ierr);
  eta1 = PetscMax(d4,d6);
  if (eta1<=theta[0] && !ell(n_,A,coeff[0],3,work,rand)) {
    *m = 3;
    goto done;
  }
  if (eta1<=theta[1] && !ell(n_,A,coeff[1],5,work,rand)) {
    *m = 5;
    goto done;
  }
  if (n<SMALLN) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,Apowers[2],&n_,Apowers[2],&n_,&szero,work,&n_));
    d8 = PetscPowReal(LAPACKlange_("O",&n_,&n_,work,&n_,rwork),1.0/8.0);
    ierr = PetscLogFlops(2.0*n*n*n+1.0*n*n);CHKERRQ(ierr);
  } else {
    d8 = PetscPowReal(normAm(n_,Apowers[2],2,work,rand),1.0/8.0);
  }
  eta3 = PetscMax(d6,d8);
  if (eta3<=theta[2] && !ell(n_,A,coeff[2],7,work,rand)) {
    *m = 7;
    goto done;
  }
  if (eta3<=theta[3] && !ell(n_,A,coeff[3],9,work,rand)) {
    *m = 9;
    goto done;
  }
  if (n<SMALLN) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,Apowers[2],&n_,Apowers[3],&n_,&szero,work,&n_));
    d10 = PetscPowReal(LAPACKlange_("O",&n_,&n_,work,&n_,rwork),1.0/10.0);
    ierr = PetscLogFlops(2.0*n*n*n+1.0*n*n);CHKERRQ(ierr);
  } else {
    d10 = PetscPowReal(normAm(n_,Apowers[1],5,work,rand),1.0/10.0);
  }
  eta4 = PetscMax(d8,d10);
  eta5 = PetscMin(eta3,eta4);
  *s = PetscMax((PetscInt)PetscCeilReal(PetscLogReal(eta5/theta[4])/PetscLogReal(2.0)),0);
  if (*s) {
    Ascaled = work+3*n*n;
    n2 = n_*n_;
    PetscStackCallBLAS("BLAScopy",BLAScopy_(&n2,A,&one,Ascaled,&one));
    sfactor = PetscPowRealInt(2.0,-(*s));
    PetscStackCallBLAS("BLASscal",BLASscal_(&n2,&sfactor,Ascaled,&one));
    ierr = PetscLogFlops(1.0*n*n);CHKERRQ(ierr);
  } else Ascaled = A;
  *s += ell(n_,Ascaled,coeff[4],13,work,rand);
done:
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * Matrix exponential implementation based on algorithm and matlab code by N. Higham and co-authors
 *
 *     N. J. Higham, "The scaling and squaring method for the matrix exponential 
 *     revisited", SIAM J. Matrix Anal. Appl. 26(4):1179-1193, 2005.
 */
PetscErrorCode FNEvaluateFunctionMat_Exp_Higham(FN fn,Mat A,Mat B)
{
#if defined(PETSC_MISSING_LAPACK_GESV) || defined(SLEPC_MISSING_LAPACK_LANGE)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GESV/LANGE - Lapack routines are unavailable");
#else
  PetscErrorCode    ierr;
  PetscBLASInt      n_,n2,*ipiv,info,one=1;
  PetscInt          n,m,j,s;
  PetscScalar       scale,smone=-1.0,sone=1.0,stwo=2.0,szero=0.0;
  PetscScalar       *Aa,*Ba,*Apowers[5],*Q,*P,*W,*work,*aux;
  const PetscScalar *c;
  const PetscScalar c3[4]   = { 120, 60, 12, 1 };
  const PetscScalar c5[6]   = { 30240, 15120, 3360, 420, 30, 1 };
  const PetscScalar c7[8]   = { 17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1 };
  const PetscScalar c9[10]  = { 17643225600, 8821612800, 2075673600, 302702400, 30270240,
                                2162160, 110880, 3960, 90, 1 };
  const PetscScalar c13[14] = { 64764752532480000, 32382376266240000, 7771770303897600,
                                1187353796428800,  129060195264000,   10559470521600,
                                670442572800,      33522128640,       1323241920,
                                40840800,          960960,            16380,  182,  1 };

  PetscFunctionBegin;
  ierr = MatDenseGetArray(A,&Aa);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&Ba);CHKERRQ(ierr);
  ierr = MatGetSize(A,&n,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  n2 = n_*n_;
  ierr = PetscMalloc2(8*n*n,&work,n,&ipiv);CHKERRQ(ierr);

  /* Matrix powers */
  Apowers[0] = work;                  /* Apowers[0] = A   */
  Apowers[1] = Apowers[0] + n*n;      /* Apowers[1] = A^2 */
  Apowers[2] = Apowers[1] + n*n;      /* Apowers[2] = A^4 */
  Apowers[3] = Apowers[2] + n*n;      /* Apowers[3] = A^6 */
  Apowers[4] = Apowers[3] + n*n;      /* Apowers[4] = A^8 */

  ierr = PetscMemcpy(Apowers[0],Aa,n2*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,Apowers[0],&n_,Apowers[0],&n_,&szero,Apowers[1],&n_));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,Apowers[1],&n_,Apowers[1],&n_,&szero,Apowers[2],&n_));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,Apowers[1],&n_,Apowers[2],&n_,&szero,Apowers[3],&n_));
  ierr = PetscLogFlops(6.0*n*n*n);CHKERRQ(ierr);

  /* Compute scaling parameter and order of Pade approximant */
  ierr = expm_params(n,Apowers,&s,&m,Apowers[4]);CHKERRQ(ierr);

  if (s) { /* rescale */
    for (j=0;j<4;j++) {
      scale = PetscPowRealInt(2.0,-PetscMax(2*j,1)*s);
      PetscStackCallBLAS("BLASscal",BLASscal_(&n2,&scale,Apowers[j],&one));
    }
    ierr = PetscLogFlops(4.0*n*n);CHKERRQ(ierr);
  }

  /* Evaluate the Pade approximant */
  switch (m) {
    case 3:  c = c3;  break;
    case 5:  c = c5;  break;
    case 7:  c = c7;  break;
    case 9:  c = c9;  break;
    case 13: c = c13; break;
    default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong value of m %d",m);
  }
  P = Ba;
  Q = Apowers[4] + n*n;
  W = Q + n*n;
  switch (m) {
    case 3:
    case 5:
    case 7:
    case 9:
      if (m==9) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,Apowers[1],&n_,Apowers[3],&n_,&szero,Apowers[4],&n_));
      ierr = PetscMemzero(P,n2*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscMemzero(Q,n2*sizeof(PetscScalar));CHKERRQ(ierr);
      for (j=0;j<n;j++) {
        P[j+j*n] = c[1];
        Q[j+j*n] = c[0];
      }
      for (j=m;j>=3;j-=2) {
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[j],Apowers[(j+1)/2-1],&one,P,&one));
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[j-1],Apowers[(j+1)/2-1],&one,Q,&one));
        ierr = PetscLogFlops(4.0*n*n);CHKERRQ(ierr);
      }
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,Apowers[0],&n_,P,&n_,&szero,W,&n_));
      ierr = PetscLogFlops(2.0*n*n*n);CHKERRQ(ierr);
      SWAP(P,W,aux);
      break;
    case 13:
      /*  P = A*(Apowers[3]*(c[13]*Apowers[3] + c[11]*Apowers[2] + c[9]*Apowers[1]) 
              + c[7]*Apowers[3] + c[5]*Apowers[2] + c[3]*Apowers[1] + c[1]*I)       */
      PetscStackCallBLAS("BLAScopy",BLAScopy_(&n2,Apowers[3],&one,P,&one));
      PetscStackCallBLAS("BLASscal",BLASscal_(&n2,&c[13],P,&one));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[11],Apowers[2],&one,P,&one));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[9],Apowers[1],&one,P,&one));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,Apowers[3],&n_,P,&n_,&szero,W,&n_));
      ierr = PetscLogFlops(5.0*n*n+2.0*n*n*n);CHKERRQ(ierr);
      ierr = PetscMemzero(P,n2*sizeof(PetscScalar));CHKERRQ(ierr);
      for (j=0;j<n;j++) P[j+j*n] = c[1];
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[7],Apowers[3],&one,P,&one));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[5],Apowers[2],&one,P,&one));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[3],Apowers[1],&one,P,&one));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&sone,P,&one,W,&one));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,Apowers[0],&n_,W,&n_,&szero,P,&n_));
      ierr = PetscLogFlops(7.0*n*n+2.0*n*n*n);CHKERRQ(ierr);
      /*  Q = Apowers[3]*(c[12]*Apowers[3] + c[10]*Apowers[2] + c[8]*Apowers[1])
              + c[6]*Apowers[3] + c[4]*Apowers[2] + c[2]*Apowers[1] + c[0]*I        */
      PetscStackCallBLAS("BLAScopy",BLAScopy_(&n2,Apowers[3],&one,Q,&one));
      PetscStackCallBLAS("BLASscal",BLASscal_(&n2,&c[12],Q,&one));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[10],Apowers[2],&one,Q,&one));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[8],Apowers[1],&one,Q,&one));
      PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,Apowers[3],&n_,Q,&n_,&szero,W,&n_));
      ierr = PetscLogFlops(5.0*n*n+2.0*n*n*n);CHKERRQ(ierr);
      ierr = PetscMemzero(Q,n2*sizeof(PetscScalar));CHKERRQ(ierr);
      for (j=0;j<n;j++) Q[j+j*n] = c[0];
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[6],Apowers[3],&one,Q,&one));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[4],Apowers[2],&one,Q,&one));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&c[2],Apowers[1],&one,Q,&one));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&sone,W,&one,Q,&one));
      ierr = PetscLogFlops(7.0*n*n);CHKERRQ(ierr);
      break;
    default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong value of m %d",m);
  }
  PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&n2,&smone,P,&one,Q,&one));
  PetscStackCallBLAS("LAPACKgesv",LAPACKgesv_(&n_,&n_,Q,&n_,ipiv,P,&n_,&info));
  SlepcCheckLapackInfo("gesv",info);
  PetscStackCallBLAS("BLASscal",BLASscal_(&n2,&stwo,P,&one));
  for (j=0;j<n;j++) P[j+j*n] += 1.0;
  ierr = PetscLogFlops(2.0*n*n*n/3.0+4.0*n*n);CHKERRQ(ierr);

  /* Squaring */
  for (j=1;j<=s;j++) {
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n_,&n_,&n_,&sone,P,&n_,P,&n_,&szero,W,&n_));
    SWAP(P,W,aux);
  }
  if (P!=Ba) { ierr = PetscMemcpy(Ba,P,n2*sizeof(PetscScalar));CHKERRQ(ierr); }
  ierr = PetscLogFlops(2.0*n*n*n*s);CHKERRQ(ierr);

  ierr = PetscFree2(work,ipiv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(A,&Aa);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Ba);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

PetscErrorCode FNView_Exp(FN fn,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;
  char           str[50];
  const char     *methodname[] = {
                  "scaling & squaring, [m/m] Pade approximant (Higham)",
                  "scaling & squaring, [6/6] Pade approximant"
  };
  const int      nmeth=sizeof(methodname)/sizeof(methodname[0]);

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (fn->beta==(PetscScalar)1.0) {
      if (fn->alpha==(PetscScalar)1.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: exp(x)\n");CHKERRQ(ierr);
      } else {
        ierr = SlepcSNPrintfScalar(str,50,fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: exp(%s*x)\n",str);CHKERRQ(ierr);
      }
    } else {
      ierr = SlepcSNPrintfScalar(str,50,fn->beta,PETSC_TRUE);CHKERRQ(ierr);
      if (fn->alpha==(PetscScalar)1.0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: %s*exp(x)\n",str);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: %s",str);CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
        ierr = SlepcSNPrintfScalar(str,50,fn->alpha,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"*exp(%s*x)\n",str);CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
    if (fn->method<nmeth) {
      ierr = PetscViewerASCIIPrintf(viewer,"  computing matrix functions with: %s\n",methodname[fn->method]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode FNCreate_Exp(FN fn)
{
  PetscFunctionBegin;
  fn->ops->evaluatefunction       = FNEvaluateFunction_Exp;
  fn->ops->evaluatederivative     = FNEvaluateDerivative_Exp;
  fn->ops->evaluatefunctionmat[0] = FNEvaluateFunctionMat_Exp_Higham;
  fn->ops->evaluatefunctionmat[1] = FNEvaluateFunctionMat_Exp_Pade;
  fn->ops->view                   = FNView_Exp;
  PetscFunctionReturn(0);
}

