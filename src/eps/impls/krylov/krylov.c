/*                       
   Common subroutines for all Krylov-type solvers.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include "private/epsimpl.h"                /*I "slepceps.h" I*/
#include "private/slepcimpl.h"
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi"
/*
   EPSBasicArnoldi - Computes an m-step Arnoldi factorization. The first k
   columns are assumed to be locked and therefore they are not modified. On
   exit, the following relation is satisfied:

                    OP * V - V * H = f * e_m^T

   where the columns of V are the Arnoldi vectors (which are B-orthonormal),
   H is an upper Hessenberg matrix, f is the residual vector and e_m is
   the m-th vector of the canonical basis. The vector f is B-orthogonal to
   the columns of V. On exit, beta contains the B-norm of f and the next 
   Arnoldi vector can be computed as v_{m+1} = f / beta. 
*/
PetscErrorCode EPSBasicArnoldi(EPS eps,PetscBool trans,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscReal *beta,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       j,m = *M;
  PetscReal      norm;

  PetscFunctionBegin;
  
  for (j=k;j<m-1;j++) {
    if (trans) { ierr = STApplyTranspose(eps->OP,V[j],V[j+1]);CHKERRQ(ierr); }
    else { ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr); }
    ierr = IPOrthogonalize(eps->ip,eps->nds,eps->DS,j+1,PETSC_NULL,V,V[j+1],H+ldh*j,&norm,breakdown);CHKERRQ(ierr);
    H[j+1+ldh*j] = norm;
    if (*breakdown) {
      *M = j+1;
      *beta = norm;
      PetscFunctionReturn(0);
    } else {
      ierr = VecScale(V[j+1],1/norm);CHKERRQ(ierr);
    }
  }
  if (trans) { ierr = STApplyTranspose(eps->OP,V[m-1],f);CHKERRQ(ierr); }
  else { ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr); }
  ierr = IPOrthogonalize(eps->ip,eps->nds,eps->DS,m,PETSC_NULL,V,f,H+ldh*(m-1),beta,PETSC_NULL);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSKrylovConvergence"
/*
   EPSKrylovConvergence - Implements the loop that checks for convergence
   in Krylov methods.

   Input Parameters:
     eps   - the eigensolver; some error estimates are updated in eps->errest 
     issym - whether the projected problem is symmetric or not
     kini  - initial value of k (the loop variable)
     nits  - number of iterations of the loop
     S     - Schur form of projected matrix (not referenced if issym)
     lds   - leading dimension of S
     Q     - Schur vectors of projected matrix (eigenvectors if issym)
     V     - set of basis vectors (used only if trueresidual is activated)
     nv    - number of vectors to process (dimension of Q, columns of V)
     beta  - norm of f (the residual vector of the Arnoldi/Lanczos factorization)
     corrf - correction factor for residual estimates (only in harmonic KS)

   Output Parameters:
     kout  - the first index where the convergence test failed

   Workspace:
     work is workspace to store 5*nv scalars, nv booleans and nv reals (only if !issym)
*/
PetscErrorCode EPSKrylovConvergence(EPS eps,PetscBool issym,PetscInt kini,PetscInt nits,PetscScalar *S,PetscInt lds,PetscScalar *Q,Vec *V,PetscInt nv,PetscReal beta,PetscReal corrf,PetscInt *kout,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscInt       k,marker;
  PetscScalar    re,im,*Z = work,*work2 = work;
  PetscReal      resnorm;
  PetscBool      iscomplex,isshift;

  PetscFunctionBegin;
  if (!issym) { Z = work; work2 = work+2*nv; }
  ierr = PetscTypeCompare((PetscObject)eps->OP,STSHIFT,&isshift);CHKERRQ(ierr);
  marker = -1;
  for (k=kini;k<kini+nits;k++) {
    /* eigenvalue */
    re = eps->eigr[k];
    im = eps->eigi[k];
    if (eps->trueres || isshift) {
      ierr = STBackTransform(eps->OP,1,&re,&im);CHKERRQ(ierr);
    }
    iscomplex = PETSC_FALSE;
    if (!issym && k<nv-1 && S[k+1+k*lds] != 0.0) iscomplex = PETSC_TRUE;
    /* residual norm */
    if (issym) {
      resnorm = beta*PetscAbsScalar(Q[(k-kini+1)*nv-1]);
    } else {
      ierr = DenseSelectedEvec(S,lds,Q,Z,k,iscomplex,nv,work2);CHKERRQ(ierr);
      if (iscomplex) resnorm = beta*SlepcAbsEigenvalue(Z[nv-1],Z[2*nv-1]);
      else resnorm = beta*PetscAbsScalar(Z[nv-1]);
    }
    if (eps->trueres) {
      if (issym) Z = Q+(k-kini)*nv;
      ierr = EPSComputeTrueResidual(eps,re,im,Z,V,nv,&resnorm);CHKERRQ(ierr);
    }
    else resnorm *= corrf;
    /* error estimate */
    ierr = (*eps->conv_func)(eps,re,im,resnorm,&eps->errest[k],eps->conv_ctx);CHKERRQ(ierr);
    if (marker==-1 && eps->errest[k] >= eps->tol) marker = k;
    if (iscomplex) { eps->errest[k+1] = eps->errest[k]; k++; }
    if (marker!=-1 && !eps->trackall) break;
  }
  if (marker!=-1) k = marker;
  *kout = k;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSFullLanczos"
/*
   EPSFullLanczos - Computes an m-step Lanczos factorization with full
   reorthogonalization.  At each Lanczos step, the corresponding Lanczos
   vector is orthogonalized with respect to all previous Lanczos vectors.
   This is equivalent to computing an m-step Arnoldi factorization and
   exploting symmetry of the operator.

   The first k columns are assumed to be locked and therefore they are 
   not modified. On exit, the following relation is satisfied:

                    OP * V - V * T = f * e_m^T

   where the columns of V are the Lanczos vectors (which are B-orthonormal),
   T is a real symmetric tridiagonal matrix, f is the residual vector and e_m
   is the m-th vector of the canonical basis. The tridiagonal is stored as
   two arrays: alpha contains the diagonal elements, beta the off-diagonal.
   The vector f is B-orthogonal to the columns of V. On exit, the last element
   of beta contains the B-norm of f and the next Lanczos vector can be 
   computed as v_{m+1} = f / beta(end). 

*/
PetscErrorCode EPSFullLanczos(EPS eps,PetscReal *alpha,PetscReal *beta,Vec *V,PetscInt k,PetscInt *M,Vec f,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       j,m = *M;
  PetscReal      norm;
  PetscScalar    *hwork,lhwork[100];

  PetscFunctionBegin;
  if (m > 100) {
    ierr = PetscMalloc((eps->nds+m)*sizeof(PetscScalar),&hwork);CHKERRQ(ierr);
  } else {
    hwork = lhwork;
  }

  for (j=k;j<m-1;j++) {
    ierr = STApply(eps->OP,V[j],V[j+1]);CHKERRQ(ierr);
    ierr = IPOrthogonalize(eps->ip,eps->nds,eps->DS,j+1,PETSC_NULL,V,V[j+1],hwork,&norm,breakdown);CHKERRQ(ierr);
    alpha[j-k] = PetscRealPart(hwork[j]);
    beta[j-k] = norm;
    if (*breakdown) {
      *M = j+1;
      if (m > 100) {
        ierr = PetscFree(hwork);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    } else {
      ierr = VecScale(V[j+1],1.0/norm);CHKERRQ(ierr);
    }
  }
  ierr = STApply(eps->OP,V[m-1],f);CHKERRQ(ierr);
  ierr = IPOrthogonalize(eps->ip,eps->nds,eps->DS,m,PETSC_NULL,V,f,hwork,&norm,PETSC_NULL);CHKERRQ(ierr);
  alpha[m-1-k] = PetscRealPart(hwork[m-1]); 
  beta[m-1-k] = norm;
  
  if (m > 100) {
    ierr = PetscFree(hwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


