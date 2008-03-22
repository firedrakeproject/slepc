/*                       

   SLEPc eigensolver: "krylovschur"

   Method: Krylov-Schur

   Algorithm:

       Single-vector Krylov-Schur method for both symmetric and non-symmetric
       problems.

   References:

       [1] "Krylov-Schur Methods in SLEPc", SLEPc Technical Report STR-7, 
           available at http://www.grycap.upv.es/slepc.

       [2] G.W. Stewart, "A Krylov-Schur Algorithm for Large Eigenproblems",
           SIAM J. Matrix Analysis and App., 23(3), pp. 601-614, 2001. 

   Last update: Oct 2006

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/eps/epsimpl.h"                /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSTranslateHarmonic"
/*
   EPSTranslateHarmonic - Computes a translation of the Krylov decomposition
   in order to perform a harmonic projection.

   On input:
     S is the Rayleigh quotient (leading dimension is m)
     tau is the translation amount
     b is assumed to be beta*e_m^T

   On output:
     g = (B-sigma*eye(m))'\b
     S is updated as S + g*b'

   Workspace:
     work is workspace to store a working copy of S and the pivots (int of length m)
*/
PetscErrorCode EPSTranslateHarmonic(PetscScalar *S,int m,PetscScalar tau,PetscScalar beta,PetscScalar *g,PetscScalar *work)
{
#if defined(PETSC_MISSING_LAPACK_GETRF) || defined(PETSC_MISSING_LAPACK_GETRS) 
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"GETRF,GETRS - Lapack routines are unavailable.");
#else
  PetscErrorCode ierr;
  PetscBLASInt   info,one = 1;
  int            i;
  PetscScalar    *B = work; 
  int            *ipiv = (int*)(work+m*m);

  PetscFunctionBegin;
  /* Copy S to workspace B */
  ierr = PetscMemcpy(B,S,m*m*sizeof(PetscScalar));CHKERRQ(ierr);
  /* Vector g initialy stores b */
  ierr = PetscMemzero(g,m*sizeof(PetscScalar));CHKERRQ(ierr);
  g[m-1] = beta;
 
  /* g = (B-sigma*eye(m))'\b */
  for (i=0;i<m;i++) 
    B[i+i*m] -= tau;
  LAPACKgetrf_(&m,&m,B,&m,ipiv,&info);
  if (info<0) SETERRQ(PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");
  ierr = PetscLogFlops(2*m*m*m/3);CHKERRQ(ierr);
  LAPACKgetrs_("C",&m,&one,B,&m,ipiv,g,&m,&info);
  if (info) SETERRQ(PETSC_ERR_LIB,"GETRS - Bad solve");
  ierr = PetscLogFlops(2*m*m-m);CHKERRQ(ierr);

  /* S = S + g*b' */
  for (i=0;i<m;i++) 
    S[i+(m-1)*m] = S[i+(m-1)*m] + g[i]*beta;

  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSRecoverHarmonic"
/*
   EPSRecoverHarmonic - Computes a translation of the truncated Krylov decomposition
   in order to recover the original non-translated state

   On input:
     S is the truncated Rayleigh quotient (size n, leading dimension m)
     k and l indicate the active columns of S
     [U, u] is the basis of the Krylov subspace
     g is the vector computed in the original translation
     Q is the similarity transformation used to reduce to sorted Schur form

   On output:
     S is updated as S + g*b'
     u is re-orthonormalized with respect to U
     b is re-scaled
     g is destroyed

   Workspace:
     ghat is workspace to store a vector of length n
*/
PetscErrorCode EPSRecoverHarmonic(PetscScalar *S,int n,int k,int l,int m,PetscScalar *g,PetscScalar *Q,Vec *U,Vec u,PetscScalar *ghat)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscBLASInt   one=1,ncol=k+l;
  PetscScalar    done=1.0,dmone=-1.0,dzero=0.0;
  PetscReal      gamma,gnorm;
  int            i,j;

  PetscFunctionBegin;

  /* g^ = -Q(:,idx)'*g */
  BLASgemv_("C",&n,&ncol,&dmone,Q,&m,g,&one,&dzero,ghat,&one);

  /* S = S + g^*b' */
  for (i=0;i<l;i++) {
    for (j=k;j<k+l;j++) {
      S[i+j*m] += ghat[i]*S[k+l+j*m];
    }
  }

  /* g~ = (I-Q(:,idx)*Q(:,idx)')*g = g+Q(:,idx)*g^ */
  BLASgemv_("N",&n,&ncol,&done,Q,&m,ghat,&one,&done,g,&one);

  /* gamma u^ = u - U*g~ */
  for (i=0;i<n;i++) 
    g[i] = -g[i];
  ierr = VecMAXPY(u,m,g,U);CHKERRQ(ierr);        

  /* Renormalize u */
  gnorm = 0.0;
  for (i=0;i<n;i++)
    gnorm = gnorm + g[i]*g[i];
  gamma = sqrt(1.0+gnorm);
  ierr = VecScale(u,1.0/gamma);CHKERRQ(ierr);

  /* b = gamma*b */
  for (i=k;i<k+l;i++) {
    S[i*m+k+l] *= gamma;
  }
  PetscFunctionReturn(0);
}

