/*
   DQDS-type dense solver for generalized symmetric-indefinite eigenproblem.
   Based on Matlab code from Carla Ferreira.

   References:

       [1] C. Ferreira, B. Parlett, "Real DQDS for the nonsymmetric tridiagonal
           eigenvalue problem", preprint, 2012.

       [2] C. Ferreira. The unsymmetric tridiagonal eigenvalue problem. Ph.D
           Thesis, University of Minho, 2007.

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
#include <slepc/private/dsimpl.h>
#include <slepcblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "ScanJ"
/*
  INPUT:
    a ---- diagonal of J
    b ---- subdiagonal of J;
    the superdiagonal of J is all 1's

  OUTPUT:
    For an eigenvalue lambda of J we have:
      gl=<real(lambda)<=gr
      -sigma=<imag(lambda)<=sigma
*/
static PetscErrorCode ScanJ(PetscInt n,PetscReal *a,PetscReal *b,PetscReal *gl,PetscReal *gr,PetscReal *sigma)
{
  PetscInt  i;
  PetscReal b0,b1,rad;

  PetscFunctionBegin;
  /* For original matrix C, C_bal=T+S; T-symmetric and S=skew-symmetric
   C_bal is the balanced form of C */
  /* Bounds on the imaginary part of C (Gersgorin bound for S)*/
  *sigma = 0.0;
  b0 = 0.0;
  for (i=0;i<n-1;i++) {
    if (b[i]<0.0) b1 = PetscSqrtReal(-b[i]);
    else b1 = 0.0;
    *sigma = PetscMax(*sigma,b1+b0);
    b0 = b1;
  }
  *sigma = PetscMax(*sigma,b0);
  /* Gersgorin bounds for T (=Gersgorin bounds on the real part for C) */
  rad = (b[0]>0.0)?PetscSqrtReal(b[0]):0.0; /* rad = b1+b0, b0 = 0 */
  *gr = a[0]+rad;
  *gl = a[0]-rad;
  b0 = rad;
  for (i=1;i<n-1;i++) {
    b1 = (b[i]>0.0)?PetscSqrtReal(b[i]):0.0;
    rad = b0+b1;
    *gr = PetscMax(*gr,a[i]+rad);
    *gl = PetscMin(*gl,a[i]-rad);
    b0 = b1;
  }
  rad = b0;
  *gr = PetscMax(*gr,a[n-1]+rad);
  *gl = PetscMin(*gl,a[n-1]-rad);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Prologue"
/*
  INPUT:
    a  - vector with the diagonal elements
    b  - vector with the subdiagonal elements
    gl - Gersgorin left bound (real axis)
    gr - Gersgorin right bound (real axis)
  OUTPUT:
    eigvalue - multiple eigenvalue (if there is an eigenvalue)
    m        - its multiplicity    (m=0 if there isn't a multiple eigenvalue)
    X        - matrix of generalized eigenvectors
    shift
    dim(work)=5*n+4
*/
static PetscErrorCode Prologue(PetscInt n,PetscReal *a,PetscReal *b,PetscReal gl,PetscReal gr,PetscInt *m,PetscReal *shift,PetscReal *work)
{

  PetscErrorCode ierr;
  PetscReal      mu,tol,*a1,*y,*yp,*x,*xp;
  PetscInt       i,k;

  PetscFunctionBegin;
  *m = 0;
  mu = 0.0;
  for (i=0;i<n;i++) mu += a[i];
  mu /= n;
  tol = n*PETSC_MACHINE_EPSILON*(gr-gl);
  a1 = work; /* size n */
  y = work+n; /* size n+1 */
  yp = y+n+1; /* size n+1. yp is the derivative of y (p for "prime") */
  x = yp+n+1; /* size n+1 */
  xp = x+n+1; /* size n+1 */
  for (i=0;i<n;i++) a1[i] = mu-a[i];
  x[0] = 1;
  xp[0] = 0;
  x[1] = a1[0];
  xp[1] = 1;
  for (i=1;i<n;i++) {
    x[i+1]=a1[i]*x[i]-b[i-1]*x[i-1];
    xp[i+1]=a1[i]*xp[i]+x[i]-b[i-1]*xp[i-1];
  }
  *shift = mu;
  if (PetscAbsReal(x[n])<tol) {
    /* mu is an eigenvalue */
    *m = *m+1;
    if (PetscAbsReal(xp[n])<tol) {
      /* mu is a multiple eigenvalue; Is it the one-point spectrum case? */
      k = 0;
      while (PetscAbsReal(xp[n])<tol && k<n-1) {
        ierr = PetscMemcpy(x,y,(n+1)*sizeof(PetscReal));CHKERRQ(ierr);
        ierr = PetscMemcpy(xp,yp,(n+1)*sizeof(PetscReal));CHKERRQ(ierr);
        x[k] = 0.0;
        k++;
        x[k] = 1.0;
        xp[k] = 0.0;
        x[k+1] = a1[k] + y[k];
        xp[k+1] = 1+yp[k];
        for (i=k+1;i<n;i++) {
          x[i+1] = a1[i]*x[i]-b[i-1]*x[i-1]+y[i];
          xp[i+1]=a1[i]*xp[i]+x[i]-b[i-1]*xp[i-1]+yp[i];
        }
        *m = *m+1;
      }
    }
  }
/*
  When mu is not an eigenvalue or it it an eigenvalue but it is not the one-point spectrum case, we will always have shift=mu

  Need to check for overflow!

  After calling Prologue, eigenComplexdqds and eigen3dqds will test if m==n in which case we have the one-point spectrum case;
  If m!=0, the only output to be used is the shift returned.
*/
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LUfac"
static PetscErrorCode LUfac(PetscInt n,PetscReal *a,PetscReal *b,PetscReal shift,PetscReal tol,PetscReal norm,PetscReal *L,PetscReal *U,PetscInt *fail,PetscReal *work)
{
  PetscInt       i;
  PetscReal      *a1;

  PetscFunctionBegin;
  a1 = work;
  for (i=0;i<n;i++) a1[i] = a[i]-shift;
  *fail = 0;
  for (i=0;i<n-1;i++) {
    U[i] = a1[i];
    L[i] = b[i]/U[i];
    a1[i+1] = a1[i+1]-L[i];
  }
  U[n-1] = a1[n-1];

  /* Check if there are NaN values */
  for (i=0;i<n-1 && !*fail;i++) {
    if (PetscIsInfOrNanReal(L[i])) *fail=1;
    if (PetscIsInfOrNanReal(U[i])) *fail=1;
  }
  if (!*fail && PetscIsInfOrNanReal(U[n-1])) *fail=1;

  for (i=0;i<n-1 && !*fail;i++) {
    if (PetscAbsReal(L[i])>tol*norm) *fail = 1;  /* This demands IEEE arithmetic */
    if (PetscAbsReal(U[i])>tol*norm) *fail = 1;
  }
  if (!*fail && PetscAbsReal(U[n-1])>tol*norm) *fail = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RealDQDS"
static PetscErrorCode RealDQDS(PetscInt n,PetscReal *L,PetscReal *U,PetscReal shift,PetscReal tol,PetscReal norm,PetscReal *L1,PetscReal *U1,PetscInt *fail)
{
  PetscReal d;
  PetscInt  i;

  PetscFunctionBegin;
  *fail = 0;
  d = U[0]-shift;
  for (i=0;i<n-1;i++) {
    U1[i] = d+L[i];
    L1[i] = L[i]*(U[i+1]/U1[i]);
    d = d*(U[i+1]/U1[i])-shift;
  }
  U1[n-1]=d;

  /* The following demands IEEE arithmetic */
  for (i=0;i<n-1 && !*fail;i++) {
    if (PetscIsInfOrNanReal(L1[i])) *fail=1;
    if (PetscIsInfOrNanReal(U1[i])) *fail=1;
  }
  if (!*fail && PetscIsInfOrNanReal(U1[n-1])) *fail=1;
  for (i=0;i<n-1 && !*fail;i++) {
    if (PetscAbsReal(L1[i])>tol*norm) *fail=1;
    if (PetscAbsReal(U1[i])>tol*norm) *fail=1;
  }
  if (!*fail && PetscAbsReal(U1[n-1])>tol*norm) *fail=1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TridqdsZhuang3"
static PetscErrorCode TridqdsZhuang3(PetscInt n,PetscReal *e,PetscReal *q,PetscReal sum,PetscReal prod,PetscReal tol,PetscReal norm,PetscReal tolDef,PetscInt *fail)
{
  PetscReal xl,yl,xr,yr,zr;
  PetscInt  i;

  PetscFunctionBegin;
  *fail = 0;
  xr = 1.0;
  yr = e[0];
  zr = 0.0;
  /* Step 1 */
  /* the efect of Z1 */
  xr = xr*q[0]+yr;
  /* the inverse of L1 */
  xl = (q[0]+e[0])*(q[0]+e[0])+q[1]*e[0]-sum*(q[0]+e[0])+prod;
  yl = -(q[2]*e[1]*q[1]*e[0])/xl;
  xl = -(q[1]*e[0]*(q[0]+e[0]+q[1]+e[1]-sum))/xl;
  /* the efect of L1 */
  q[0] = xr-xl;
  xr = yr-xl;
  yr = zr-yl-xl*e[1];
  /*the inverse of Y1 */
  xr = xr/q[0];
  yr = yr/q[0];
  /*the effect of Y1 inverse */
  e[0] = xl+yr+xr*q[1];
  xl = yl+zr+yr*q[2];      /* zr=0  when n=3 */
  /*the effect of Y1 */
  xr = 1.0-xr;
  yr = e[1]-yr;

  /* STEP n-1 */

  if (PetscAbsReal(e[n-3])>tolDef*PetscAbsReal(xl) || PetscAbsReal(e[n-3])>tolDef*PetscAbsReal(q[n-3])) {
    /* the efect of Zn-1 */
    xr = xr*q[n-2]+yr;
    /* the inverse of Ln-1 */
    xl = -xl/e[n-3];
    /* the efect of Ln-1 */
    q[n-2] = xr-xl;
    xr = yr-xl;
    /*the inverse of Yn-1 */
    xr = xr/q[n-2];
    /*the effect of the inverse of Yn-1 */
    e[n-2] = xl+xr*q[n-1];
    /*the effects of Yn-1 */
    xr = 1.0-xr;
    /* STEP n */
    /*the effect of Zn */
    xr = xr*q[n-1];
    /*the inverse of Ln=I */
    /*the effect of Ln */
    q[n-1] = xr;
    /* the inverse of  Yn-1=I */

  } else { /* Free deflation */
    e[n-2] = (e[n-3]+(xr*q[n-2]+yr)+q[n-1])*0.5;       /* Sum=trace/2 */
    q[n-2] = (e[n-3]+q[n-2]*xr)*q[n-1]-xl;             /* det */
    q[n-1] = e[n-2]*e[n-2]-q[n-2];
    *fail = 2;
  }

  /* The following demands IEEE arithmetic */
  for (i=0;i<n-1 && !*fail;i++) {
    if (PetscIsInfOrNanReal(e[i])) *fail=1;
    if (PetscIsInfOrNanReal(q[i])) *fail=1;
  }
  if (!*fail && PetscIsInfOrNanReal(q[n-1])) *fail=1;
  for (i=0;i<n-1 && !*fail;i++) {
    if (PetscAbsReal(e[i])>tol*norm) *fail=1;
    if (PetscAbsReal(q[i])>tol*norm) *fail=1;
  }
  if (!*fail && PetscAbsReal(q[n-1])>tol*norm) *fail=1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TridqdsZhuang"
static PetscErrorCode TridqdsZhuang(PetscInt n,PetscReal *e,PetscReal *q,PetscReal sum,PetscReal prod,PetscReal tol,PetscReal norm,PetscReal tolDef,PetscReal *e1,PetscReal *q1,PetscInt *fail)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      xl,yl,xr,yr,zr;

  PetscFunctionBegin;
  for (i=0;i<n-1;i++) {
    e1[i] = e[i];
    q1[i] = q[i];
  }
  q1[n-1] = q[n-1];
  *fail = 0;
  if (n>3) {   /* For n>3 */
    *fail = 0;
    xr = 1;
    yr = e1[0];
    zr = 0;
    /* step 1 */
    /* the efect of Z1 */
    xr = xr*q1[0]+yr;
    /* the inverse of L1 */
    xl = (q1[0]+e1[0])*(q1[0]+e1[0])+q1[1]*e1[0]-sum*(q1[0]+e1[0])+prod;
    yl = -(q1[2]*e1[1]*q1[1]*e1[0])/xl;
    xl = -(q1[1]*e1[0]*(q1[0]+e1[0]+q1[1]+e1[1]-sum))/xl;
    /* the efect of L1 */
    q1[0] = xr-xl;
    xr = yr-xl;
    yr = zr-yl-xl*e1[1];
    zr = -yl*e1[2];
    /* the inverse of Y1 */
    xr = xr/q1[0];
    yr = yr/q1[0];
    zr = zr/q1[0];
    /* the effect of Y1 inverse */
    e1[0] = xl+yr+xr*q1[1];
    xl = yl+zr+yr*q1[2];
    yl = zr*q1[3];
    /* the effect of Y1 */
    xr = 1-xr;
    yr = e1[1]-yr;
    zr = -zr;
    /* step i=2,...,n-3 */
    for (i=1;i<n-3;i++) {
      /* the efect of Zi */
      xr = xr*q1[i]+yr;
      /* the inverse of Li */
      xl = -xl/e1[i-1];
      yl = -yl/e1[i-1];
      /* the efect of Li */
      q1[i] = xr-xl;
      xr = yr-xl;
      yr = zr-yl-xl*e1[i+1];
      zr = -yl*e1[i+2];
      /* the inverse of Yi */
      xr = xr/q1[i];
      yr = yr/q1[i];
      zr = zr/q1[i];
      /* the effect of the inverse of Yi */
      e1[i] = xl+yr+xr*q1[i+1];
      xl = yl+zr+yr*q1[i+2];
      yl = zr*q1[i+3];
      /* the effects of Yi */
      xr = 1.0-xr;
      yr = e1[i+1]-yr;
      zr = -zr;
    }

    /* STEP n-2            zr is no longer needed */

    /* the efect of Zn-2 */
    xr = xr*q1[n-3]+yr;
    /* the inverse of Ln-2 */
    xl = -xl/e1[n-4];
    yl = -yl/e1[n-4];
    /* the efect of Ln-2 */
    q1[n-3] = xr-xl;
    xr = yr-xl;
    yr = zr-yl-xl*e1[n-2];
    /* the inverse of Yn-2 */
    xr = xr/q1[n-3];
    yr = yr/q1[n-3];
    /* the effect of the inverse of Yn-2 */
    e1[n-3] = xl+yr+xr*q1[n-2];
    xl = yl+yr*q1[n-1];
    /* the effect of Yn-2 */
    xr = 1.0-xr;
    yr = e1[n-2]-yr;

    /* STEP n-1           yl and yr are no longer needed */
    /* Testing for EARLY DEFLATION */

    if (PetscAbsReal(e1[n-3])>tolDef*PetscAbsReal(xl) || PetscAbsReal(e1[n-3])>tolDef*PetscAbsReal(q1[n-3])) {
      /* the efect of Zn-1 */
      xr = xr*q1[n-2]+yr;
      /* the inverse of Ln-1 */
      xl = -xl/e1[n-3];
      /* the efect of Ln-1 */
      q1[n-2] = xr-xl;
      xr = yr-xl;
      /*the inverse of Yn-1 */
      xr = xr/q1[n-2];
      /*the effect of the inverse of Yn-1 */
      e1[n-2] = xl+xr*q1[n-1];
      /*the effects of Yn-1 */
      xr = 1.0-xr;

      /* STEP n;     xl no longer needed */
      /* the effect of Zn */
      xr = xr*q1[n-1];
      /* the inverse of Ln = I */
      /* the effect of Ln */
      q1[n-1] = xr;
      /* the inverse of  Yn-1=I */

    } else {  /* FREE DEFLATION */
      e1[n-2] = (e1[n-3]+xr*q1[n-2]+yr+q1[n-1])*0.5;     /* sum=trace/2 */
      q1[n-2] = (e1[n-3]+q1[n-2]*xr)*q1[n-1]-xl;         /* det */
      q1[n-1] = e1[n-2]*e1[n-2]-q1[n-2];
      *fail = 2;
    }

    for (i=0;i<n-1 && !*fail;i++) {
      if (PetscIsInfOrNanReal(e1[i])) *fail=1;
      if (PetscIsInfOrNanReal(q1[i])) *fail=1;
    }
    if (!*fail && PetscIsInfOrNanReal(q1[n-1])) *fail=1;
    for (i=0;i<n-1 && !*fail;i++) {
      if (PetscAbsReal(e1[i])>tol*norm) *fail = 1;  /* This demands IEEE arithmetic */
      if (PetscAbsReal(q1[i])>tol*norm) *fail = 1;
    }
    if (!*fail && PetscAbsReal(q1[n-1])>tol*norm) *fail = 1;

  } else {  /* The case n=3 */
    ierr = TridqdsZhuang3(n,e1,q1,sum,prod,tol,norm,tolDef,fail);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSGHIEP_Eigen3DQDS"
static PetscErrorCode DSGHIEP_Eigen3DQDS(PetscInt n,PetscReal *a,PetscReal *b,PetscReal *c,PetscScalar *wr,PetscScalar *wi,PetscReal *work)
{
  PetscInt       totalIt=0;       /* Total Number of Iterations  */
  PetscInt       totalFail=0;     /* Total number of failures */
  PetscInt       nFail=0;         /* Number of failures per transformation */
  PetscReal      tolZero=1.0/16;  /* Tolerance for zero shifts */
  PetscInt       maxIt=10*n;      /* Maximum number of iterations */
  PetscInt       maxFail=10*n;    /* Maximum number of failures allowed per each transformation */
  PetscReal      tolDef=PETSC_MACHINE_EPSILON;  /* Tolerance for deflation eps, 10*eps, 100*eps */
  PetscReal      tolGrowth=100000;
  PetscErrorCode ierr;
  PetscInt       i,k,nwu=0,begin,ind,flag,dim,m,*split,lastSplit;
  PetscReal      norm,gr,gl,sigma,delta,meanEig,*U,*L,*U1,*L1;
  PetscReal      acShift,initialShift,shift=0.0,sum,det,disc,prod,x1,x2;
  PetscBool      test1,test2;

  PetscFunctionBegin;
  dim = n;
  /* Test if the matrix is unreduced */
  for (i=0;i<n-1;i++) {
    if (PetscAbsReal(b[i])==0.0 || PetscAbsReal(c[i])==0.0) SETERRQ(PETSC_COMM_SELF,1,"Initial tridiagonal matrix is not unreduced");
  }
  U = work;
  L = work+n;
  U1 = work+2*n;
  L1 = work+3*n;
  nwu = 4*n;
  if (wi) {
    ierr = PetscMemzero(wi,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  /* Normalization - the J form of C */
  for (i=0;i<n-1;i++) b[i] *= c[i]; /* subdiagonal of the J form */

  /* Scan matrix J  ---- Finding a box of inclusion for the eigenvalues */
  norm = 0.0;
  for (i=0;i<n-1;i++) {
    norm = PetscMax(norm,PetscMax(PetscAbsReal(a[i]),PetscAbsReal(b[i])));
  }
  norm = PetscMax(norm,PetscMax(1,PetscAbsReal(a[n-1])));
  ierr = ScanJ(n,a,b,&gl,&gr,&sigma);CHKERRQ(ierr);
  delta = (gr-gl)/n; /* How much to add to the shift, in case of failure (element growth) */
  meanEig = 0.0;
  for (i=0;i<n;i++) meanEig += a[i];
  meanEig /= n; /* shift = initial shift = mean of eigenvalues */
  ierr = Prologue(n,a,b,gl,gr,&m,&shift,work+nwu);CHKERRQ(ierr);
  if (m==n) { /* Multiple eigenvalue, we have the one-point spectrum case */
    for (i=0;i<dim;i++) {
      wr[i] = shift;
      if (wi) wi[i] = 0.0;
    }
    PetscFunctionReturn(0);
  }
  /* Initial LU Factorization */
  if (delta==0.0) shift=0.0;  /* The case when all eigenvalues are pure imaginary */
  ierr = LUfac(n,a,b,shift,tolGrowth,norm,L,U,&flag,work+nwu);CHKERRQ(ierr); /* flag=1 failure; flag=0 successful transformation*/
  while (flag==1 && nFail<maxFail) {
    shift=shift+delta;
    if (shift>gr || shift<gl) { /* Successive failures */
      shift=meanEig;
      delta=-delta;
    }
    nFail=nFail+1;
    ierr = LUfac(n,a,b,shift,tolGrowth,norm,L,U,&flag,work+nwu);CHKERRQ(ierr); /* flag=1 failure; flag=0 successful transformation*/
  }
  if (nFail==maxFail) SETERRQ(PETSC_COMM_SELF,1,"Maximun number of failures reached in Initial LU factorization");
  /* Successful Initial transformation */
  totalFail = totalFail+nFail;
  nFail = 0;
  acShift = 0;
  initialShift = shift;
  shift = 0;
  begin = 0;
  lastSplit = 0;
  ierr = PetscMalloc1(n,&split);CHKERRQ(ierr);
  split[lastSplit] = begin;
  while (begin!=-1) {
    while (n-begin>2 && totalIt<maxIt) {
      /* Check for deflation before performing a transformation */
      test1 = (PetscAbsReal(L[n-2])<tolDef*PetscAbsReal(U[n-2])
            && PetscAbsReal(L[n-2])<tolDef*PetscAbsReal(U[n-1]+acShift)
            && PetscAbsReal(L[n-2]*U[n])<tolDef*PetscAbsReal(acShift+U[n-1])
            && PetscAbsReal(L[n-2])*(PetscAbsReal(U[n-2])+1)<tolDef*PetscAbsReal(acShift+U[n-1]))? PETSC_TRUE: PETSC_FALSE;
      if (flag==2) {  /* Early 2x2 deflation */
        test2 = PETSC_TRUE;
      } else {
        if (n-begin>4) {
          test2 = (PetscAbsReal(L[n-3])<tolDef*PetscAbsReal(U[n-3])
               && PetscAbsReal(L[n-3]*(U[n-4]+L[n-4]))< tolDef*PetscAbsReal(U[n-4]*(U[n-3]+L[n-3])+L[n-4]*L[n-3]))? PETSC_TRUE: PETSC_FALSE;
        } else { /* n-begin+1=3 */
          test2 = (PetscAbsReal(L[begin])<tolDef*PetscAbsReal(U[begin]))? PETSC_TRUE: PETSC_FALSE;
        }
      }
      while (test2 || test1) {
        /* 2x2 deflation */
        if (test2) {
          if (flag==2) { /* Early deflation */
            sum = L[n-2];
            det = U[n-2];
            disc = U[n-1];
            flag = 0;
          } else {
            sum = (L[n-2]+(U[n-2]+U[n-1]))/2;
            disc = (L[n-2]*(L[n-2]+2*(U[n-2]+U[n-1]))+(U[n-2]-U[n-1])*(U[n-2]-U[n-1]))/4;
            det = U[n-2]*U[n-1];
          }
          if (disc<=0) {
#if !defined(PETSC_USE_COMPLEX)
            wr[--n] = sum+acShift; if (wi) wi[n] = PetscSqrtReal(-disc);
            wr[--n] = sum+acShift; if (wi) wi[n] = -PetscSqrtReal(-disc);
#else
            wr[--n] = sum-PETSC_i*PetscSqrtReal(-disc)+acShift; if (wi) wi[n] = 0.0;
            wr[--n] = sum+PETSC_i*PetscSqrtReal(-disc)+acShift; if (wi) wi[n] = 0.0;
#endif
          } else {
            if (sum==0.0) {
              x1 = PetscSqrtReal(disc);
              x2 = -x1;
            } else {
              x1 = ((sum>=0.0)?1.0:-1.0)*(PetscAbsReal(sum)+PetscSqrtReal(disc));
              x2 = det/x1;
            }
            wr[--n] = x1+acShift;
            wr[--n] = x2+acShift;
          }
        } else { /* test1 -- 1x1 deflation */
          x1 = U[n-1]+acShift;
          wr[--n] = x1;
        }

        if (n<=begin+2) {
          break;
        } else {
          test1 = (PetscAbsReal(L[n-2])<tolDef*PetscAbsReal(U[n-2])
                && PetscAbsReal(L[n-2])<tolDef*PetscAbsReal(U[n-1]+acShift)
                && PetscAbsReal(L[n-2]*U[n-1])<tolDef*PetscAbsReal(acShift+U[n-1])
                && PetscAbsReal(L[n-2])*(PetscAbsReal(U[n-2])+1)< tolDef*PetscAbsReal(acShift+U[n-1]))? PETSC_TRUE: PETSC_FALSE;
          if (n-begin>4) {
            test2 = (PetscAbsReal(L[n-3])<tolDef*PetscAbsReal(U[n-3])
                  && PetscAbsReal(L[n-3]*(U[n-4]+L[n-4]))< tolDef*PetscAbsReal(U[n-4]*(U[n-3]+L[n-3])+L[n-4]*L[n-3]))? PETSC_TRUE: PETSC_FALSE;
          } else { /* n-begin+1=3 */
            test2 = (PetscAbsReal(L[begin])<tolDef*PetscAbsReal(U[begin]))? PETSC_TRUE: PETSC_FALSE;
          }
        }
      } /* end "WHILE deflations" */
      /* After deflation */
      if (n>begin+3) {
        ind = begin;
        for (k=n-4;k>=begin+1;k--) {
          if (PetscAbsReal(L[k])<tolDef*PetscAbsReal(U[k])
           && PetscAbsReal(L[k]*U[k+1]*(U[k+2]+L[k+2])*(U[k-1]+L[k-1]))<tolDef*PetscAbsReal((U[k-1]*(U[k]+L[k])+L[k-1]*L[k])*(U[k+1]*(U[k+2]+L[k+2])+L[k+1]*L[k+2]))) {
            ind=k;
            break;
          }
        }
        if (ind>begin || PetscAbsReal(L[begin]) <tolDef*PetscAbsReal(U[begin])) {
          lastSplit = lastSplit+1;
          split[lastSplit] = begin;
          L[ind] = acShift; /* Use of L[ind] to save acShift */
          begin = ind+1;
        }
      }

      if (n>begin+2) {
        disc = (L[n-2]*(L[n-2]+2*(U[n-2]+U[n-1]))+(U[n-2]-U[n-1])*(U[n-2]-U[n-1]))/4;
        if ((PetscAbsReal(L[n-2])>tolZero) && (PetscAbsReal(L[n-3])>tolZero)) { /* L's are big */
          shift = 0;
          ierr = RealDQDS(n-begin,L+begin,U+begin,0,tolGrowth,norm,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
          if (flag) {  /* Failure */
            ierr = TridqdsZhuang(n-begin,L+begin,U+begin,0.0,0.0,tolGrowth,norm,tolDef,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
            shift = 0.0;
            while (flag==1 && nFail<maxFail) {  /* Successive failures */
              shift = shift+delta;
              if (shift>gr-acShift || shift<gl-acShift) {
                shift = meanEig-acShift;
                delta = -delta;
              }
              nFail++;
              ierr = RealDQDS(n-begin,L+begin,U+begin,0,tolGrowth,norm,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
            }
          }
        } else { /* L's are small */
          if (disc<0) {  /* disc <0   Complex case; Francis shift; 3dqds */
            sum = U[n-2]+L[n-2]+U[n-1];
            prod = U[n-2]*U[n-1];
            ierr = TridqdsZhuang(n-begin,L+begin,U+begin,sum,prod,tolGrowth,norm,tolDef,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
            shift = 0.0; /* Restoring transformation */
            while (flag==1 && nFail<maxFail) { /* In case of failure */
              shift = shift+U[n-1];  /* first time shift=0 */
              ierr = RealDQDS(n-begin,L+begin,U+begin,shift,tolGrowth,norm,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
              nFail++;
            }
          } else  { /* disc >0  Real case; real Wilkinson shift; dqds */
            sum = (L[n-2]+U[n-2]+U[n-1])/2;
            if (sum==0.0) {
              x1 = PetscSqrtReal(disc);
              x2 = -x1;
            } else {
              x1 = ((sum>=0)?1.0:-1.0)*(PetscAbsReal(sum)+PetscSqrtReal(disc));
              x2 = U[n-2]*U[n-1]/x1;
            }
            /* Take the eigenvalue closest to UL(n,n) */
            if (PetscAbsReal(x1-U[n-1])<PetscAbsReal(x2-U[n-1])) {
              shift = x1;
            } else {
              shift = x2;
            }
            ierr = RealDQDS(n-begin,L+begin,U+begin,shift,tolGrowth,norm,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
            /* In case of failure */
            while (flag==1 && nFail<maxFail) {
              sum = 2*shift;
              prod = shift*shift;
              ierr = TridqdsZhuang(n-1-begin,L+begin,U+begin,sum,prod,tolGrowth,norm,tolDef,L1+begin,U1+begin,&flag);CHKERRQ(ierr);
              /* In case of successive failures */
              if (shift==0.0) {
                shift = PetscMin(PetscAbsReal(L[n-2]),PetscAbsReal(L[n-3]))*delta;
              } else {
                shift=shift+delta;
              }
              if (shift>gr-acShift || shift<gl-acShift) {
                shift = meanEig-acShift;
                delta = -delta;
              }
              if (!flag) { /* We changed from real dqds to 3dqds */
                shift=0;
              }
              nFail++;
            }
          }
        } /* end "if tolZero" */
        if (nFail==maxFail) SETERRQ(PETSC_COMM_SELF,1,"Maximun number of failures reached. No convergence in DQDS");
        /* Successful Transformation; flag==0 */
        totalIt++;
        acShift = shift+acShift;
        for (i=begin;i<n-1;i++) {
          L[i] = L1[i];
          U[i] = U1[i];
        }
        U[n-1] = U1[n-1];
        totalFail = totalFail+nFail;
        nFail = 0;
      }  /* end "if n>begin+1" */
    }  /* end WHILE 1 */
    if (totalIt>=maxIt) SETERRQ(PETSC_COMM_SELF,1,"Maximun number of iterations reached. No convergence in DQDS");
    /* END: n=2 or n=1  % n=begin+1 or n=begin */
    if (n==begin+2) {
      sum = (L[n-2]+U[n-2]+U[n-1])/2;
      disc = (L[n-2]*(L[n-2]+2*(U[n-2]+U[n-1]))+(U[n-2]-U[n-1])*(U[n-2]-U[n-1]))/4;
      if (disc<=0) {  /* Complex case */
        /* Deflation 2 */
#if !defined(PETSC_USE_COMPLEX)
        wr[--n] = sum+acShift; if (wi) wi[n] = PetscSqrtReal(-disc);
        wr[--n] = sum+acShift; if (wi) wi[n] = -PetscSqrtReal(-disc);
#else
        wr[--n] = sum-PETSC_i*PetscSqrtReal(-disc)+acShift; if (wi) wi[n] = 0.0;
        wr[--n] = sum+PETSC_i*PetscSqrtReal(-disc)+acShift; if (wi) wi[n] = 0.0;
#endif
      } else { /* Real case */
        if (sum==0.0) {
          x1 = PetscSqrtReal(disc);
          x2 = -x1;
        } else {
          x1 = ((sum>=0)?1.0:-1.0)*(PetscAbsReal(sum)+PetscSqrtReal(disc));
          x2 = U[n-2]*U[n-1]/x1;
        }
        /* Deflation 2 */
        wr[--n] = x2+acShift;
        wr[--n] = x1+acShift;
      }
    } else { /* n=1   n=begin */
      /* deflation 1 */
      x1 = U[n-1]+acShift;
      wr[--n] = x1;
    }
    switch (n) {
      case 0:
        begin = -1;
        break;
      case 1:
        acShift = L[begin-1];
        begin = split[lastSplit];
        lastSplit--;
        break;
      default : /* n>=2 */
        acShift = L[begin-1];
        begin = split[lastSplit];
        lastSplit--;
    }
  }/* While begin~=-1 */
  for (i=0;i<dim;i++) {
    wr[i] = wr[i]+initialShift;
  }
  ierr = PetscFree(split);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DSSolve_GHIEP_DQDS_II"
PetscErrorCode DSSolve_GHIEP_DQDS_II(DS ds,PetscScalar *wr,PetscScalar *wi)
{
  PetscErrorCode ierr;
  PetscInt       i,off,ld,nwall,nwu;
  PetscScalar    *A,*B,*Q,*vi;
  PetscReal      *d,*e,*s,*a,*b,*c;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  PetscValidPointer(wi,3);
#endif
  ld = ds->ld;
  off = ds->l + ds->l*ld;
  A = ds->mat[DS_MAT_A];
  B = ds->mat[DS_MAT_B];
  Q = ds->mat[DS_MAT_Q];
  d = ds->rmat[DS_MAT_T];
  e = ds->rmat[DS_MAT_T] + ld;
  s = ds->rmat[DS_MAT_D];
  /* Quick return if possible */
  if (ds->n-ds->l == 1) {
    *(Q+off) = 1;
    if (!ds->compact) {
      d[ds->l] = PetscRealPart(A[off]);
      s[ds->l] = PetscRealPart(B[off]);
    }
    wr[ds->l] = d[ds->l]/s[ds->l];
    if (wi) wi[ds->l] = 0.0;
    PetscFunctionReturn(0);
  }
  nwall = 12*ld+4;
  ierr = DSAllocateWork_Private(ds,0,nwall,0);CHKERRQ(ierr);
  /* Reduce to pseudotriadiagonal form */
  ierr = DSIntermediate_GHIEP(ds);CHKERRQ(ierr);

  /* Compute Eigenvalues (DQDS) */
  /* Form pseudosymmetric tridiagonal */
  a = ds->rwork;
  b = a+ld;
  c = b+ld;
  nwu = 3*ld;
  if (ds->compact) {
    for (i=ds->l;i<ds->n-1;i++) {
      a[i] = d[i]*s[i];
      b[i] = e[i]*s[i+1];
      c[i] = e[i]*s[i];
    }
    a[ds->n-1] = d[ds->n-1]*s[ds->n-1];
  } else {
    for (i=ds->l;i<ds->n-1;i++) {
      a[i] = PetscRealPart(A[i+i*ld]*B[i+i*ld]);
      b[i] = PetscRealPart(A[i+1+i*ld]*s[i+1]);
      c[i] = PetscRealPart(A[i+(i+1)*ld]*s[i]);
    }
    a[ds->n-1] = PetscRealPart(A[ds->n-1+(ds->n-1)*ld]*B[ds->n-1+(ds->n-1)*ld]);
  }
  vi = (wi)?wi+ds->l:NULL;
  ierr = DSGHIEP_Eigen3DQDS(ds->n-ds->l,a+ds->l,b+ds->l,c+ds->l,wr+ds->l,vi,ds->rwork+nwu);CHKERRQ(ierr);

  /* Compute Eigenvectors with Inverse Iteration */
  ierr = DSGHIEPInverseIteration(ds,wr,wi);CHKERRQ(ierr);

  /* Recover eigenvalues from diagonal */
  ierr = DSGHIEPComplexEigs(ds,0,ds->l,wr,wi);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  if (wi) {
    for (i=ds->l;i<ds->n;i++) wi[i] = 0.0;
  }
#endif
  PetscFunctionReturn(0);
}

