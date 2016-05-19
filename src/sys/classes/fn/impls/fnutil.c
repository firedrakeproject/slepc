/*
   Utility subroutines common to several impls

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

#undef __FUNCT__
#define __FUNCT__ "SlepcMatDenseSqrt"
/*
   Compute the square root of an upper quasi-triangular matrix T,
   using Higham's algorithm (LAA 88, 1987). T is overwritten with sqrtm(T).
 */
PetscErrorCode SlepcMatDenseSqrt(PetscBLASInt n,PetscScalar *T,PetscBLASInt ld)
{
#if defined(SLEPC_MISSING_LAPACK_TRSYL)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"TRSYL - Lapack routine unavailable");
#else
  PetscScalar  one=1.0,mone=-1.0;
  PetscReal    done=1.0;
  PetscBLASInt i,j,si,sj,r,ione=1,info;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal    alpha,theta,mu,mu2;
#endif

  PetscFunctionBegin;
  for (j=0;j<n;j++) {
#if defined(PETSC_USE_COMPLEX)
    sj = 1;
    T[j+j*ld] = PetscSqrtScalar(T[j+j*ld]);
#else
    sj = (j==n-1 || T[j+1+j*ld] == 0.0)? 1: 2;
    if (sj==1) {
      if (T[j+j*ld]<0.0) SETERRQ(PETSC_COMM_SELF,1,"Matrix has a real negative eigenvalue, no real primary square root exists");
      T[j+j*ld] = PetscSqrtReal(T[j+j*ld]);
    } else {
      /* square root of 2x2 block */
      theta = (T[j+j*ld]+T[j+1+(j+1)*ld])/2.0;
      mu    = (T[j+j*ld]-T[j+1+(j+1)*ld])/2.0;
      mu2   = -mu*mu-T[j+1+j*ld]*T[j+(j+1)*ld];
      mu    = PetscSqrtReal(mu2);
      if (theta>0.0) alpha = PetscSqrtReal((theta+PetscSqrtReal(theta*theta+mu2))/2.0);
      else alpha = mu/PetscSqrtReal(2.0*(-theta+PetscSqrtReal(theta*theta+mu2)));
      T[j+j*ld]       /= 2.0*alpha;
      T[j+1+(j+1)*ld] /= 2.0*alpha;
      T[j+(j+1)*ld]   /= 2.0*alpha;
      T[j+1+j*ld]     /= 2.0*alpha;
      T[j+j*ld]       += alpha-theta/(2.0*alpha);
      T[j+1+(j+1)*ld] += alpha-theta/(2.0*alpha);
    }
#endif
    for (i=j-1;i>=0;i--) {
#if defined(PETSC_USE_COMPLEX)
      si = 1;
#else
      si = (i==0 || T[i+(i-1)*ld] == 0.0)? 1: 2;
      if (si==2) i--;
#endif
      /* solve Sylvester equation of order si x sj */
      r = j-i-si;
      if (r) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&si,&sj,&r,&mone,T+i+(i+si)*ld,&ld,T+i+si+j*ld,&ld,&one,T+i+j*ld,&ld));
      PetscStackCallBLAS("LAPACKtrsyl",LAPACKtrsyl_("N","N",&ione,&si,&sj,T+i+i*ld,&ld,T+j+j*ld,&ld,T+i+j*ld,&ld,&done,&info));
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTRSYL %d",info);
    }
    if (sj==2) j++;
  }
  PetscFunctionReturn(0);
#endif
}

#define BLOCKSIZE 64

#undef __FUNCT__
#define __FUNCT__ "SlepcSchurParlettSqrt"
/*
   Simplified Schur-Parlett algorithm on an upper quasi-triangular matrix T,
   particularized for the square root function. T is overwritten with sqrtm(T).
   If firstonly then only the first column of T will contain relevant values.
 */
PetscErrorCode SlepcSchurParlettSqrt(PetscBLASInt n,PetscScalar *T,PetscBLASInt ld,PetscBool firstonly)
{
#if defined(SLEPC_MISSING_LAPACK_GEES) || defined(SLEPC_MISSING_LAPACK_TRSYL)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEES/TRSYL - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscBLASInt   i,j,k,r,ione=1,sdim,lwork,*s,*p,info,bs=BLOCKSIZE;
  PetscScalar    *wr,*W,*Q,*work,one=1.0,zero=0.0,mone=-1.0;
  PetscInt       m,nblk;
  PetscReal      done=1.0;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#else
  PetscReal      *wi;
#endif

  PetscFunctionBegin;
  m     = n;
  nblk  = (m+bs-1)/bs;
  lwork = 5*n;
  k     = firstonly? 1: n;

  /* compute Schur decomposition A*Q = Q*T */
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc7(m,&wr,m,&wi,m*k,&W,m*m,&Q,lwork,&work,nblk,&s,nblk,&p);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgees",LAPACKgees_("V","N",NULL,&n,T,&ld,&sdim,wr,wi,Q,&ld,work,&lwork,NULL,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEES %d",info);
#else
  ierr = PetscMalloc7(m,&wr,m,&rwork,m*k,&W,m*m,&Q,lwork,&work,nblk,&s,nblk,&p);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgees",LAPACKgees_("V","N",NULL,&n,T,&ld,&sdim,wr,Q,&ld,work,&lwork,rwork,NULL,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEES %d",info);
#endif

  /* determine block sizes and positions, to avoid cutting 2x2 blocks */
  j = 0;
  p[j] = 0;
  do {
    s[j] = PetscMin(bs,n-p[j]);
#if !defined(PETSC_USE_COMPLEX)
    if (p[j]+s[j]!=n && T[p[j]+s[j]+(p[j]+s[j]-1)*ld]!=0.0) s[j]++;
#endif
    if (p[j]+s[j]==n) break;
    j++;
    p[j] = p[j-1]+s[j-1];
  } while (1);
  nblk = j+1;

  for (j=0;j<nblk;j++) {
    /* evaluate f(T_jj) */
    ierr = SlepcMatDenseSqrt(s[j],T+p[j]+p[j]*ld,ld);CHKERRQ(ierr);
    for (i=j-1;i>=0;i--) {
      /* solve Sylvester equation for block (i,j) */
      r = p[j]-p[i]-s[i];
      if (r) PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",s+i,s+j,&r,&mone,T+p[i]+(p[i]+s[i])*ld,&ld,T+p[i]+s[i]+p[j]*ld,&ld,&one,T+p[i]+p[j]*ld,&ld));
      PetscStackCallBLAS("LAPACKtrsyl",LAPACKtrsyl_("N","N",&ione,s+i,s+j,T+p[i]+p[i]*ld,&ld,T+p[j]+p[j]*ld,&ld,T+p[i]+p[j]*ld,&ld,&done,&info));
      if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xTRSYL %d",info);
    }
  }

  /* backtransform B = Q*T*Q' */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","C",&n,&k,&n,&one,T,&ld,Q,&ld,&zero,W,&ld));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&k,&n,&one,Q,&ld,W,&ld,&zero,T,&ld));

#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscFree7(wr,wi,W,Q,work,s,p);CHKERRQ(ierr);
#else
  ierr = PetscFree7(wr,rwork,W,Q,work,s,p);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
#endif
}

