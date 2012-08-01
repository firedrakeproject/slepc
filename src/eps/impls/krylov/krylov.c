/*                       
   Common subroutines for all Krylov-type solvers.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>
#include <slepc-private/slepcimpl.h>
#include <slepcblaslapack.h>

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
    ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,j+1,PETSC_NULL,V,V[j+1],H+ldh*j,&norm,breakdown);CHKERRQ(ierr);
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
  ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,m,PETSC_NULL,V,f,H+ldh*(m-1),beta,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSKrylovConvergence"
/*
   EPSKrylovConvergence - Implements the loop that checks for convergence
   in Krylov methods.

   Input Parameters:
     eps   - the eigensolver; some error estimates are updated in eps->errest 
     getall - whether all residuals must be computed
     kini  - initial value of k (the loop variable)
     nits  - number of iterations of the loop
     V     - set of basis vectors (used only if trueresidual is activated)
     nv    - number of vectors to process (dimension of Q, columns of V)
     beta  - norm of f (the residual vector of the Arnoldi/Lanczos factorization)
     corrf - correction factor for residual estimates (only in harmonic KS)

   Output Parameters:
     kout  - the first index where the convergence test failed
*/
PetscErrorCode EPSKrylovConvergence(EPS eps,PetscBool getall,PetscInt kini,PetscInt nits,Vec *V,PetscInt nv,PetscReal beta,PetscReal corrf,PetscInt *kout)
{
  PetscErrorCode ierr;
  PetscInt       k,newk,marker,ld;
  PetscScalar    re,im,*Zr,*Zi,*X;
  PetscReal      resnorm;
  PetscBool      isshift;
  Vec            x,y;

  PetscFunctionBegin;
  if (eps->trueres) {
    ierr = VecDuplicate(eps->t,&x);CHKERRQ(ierr);
    ierr = VecDuplicate(eps->t,&y);CHKERRQ(ierr);
  }
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->OP,STSHIFT,&isshift);CHKERRQ(ierr);
  marker = -1;
  if (eps->trackall) getall = PETSC_TRUE;
  for (k=kini;k<kini+nits;k++) {
    /* eigenvalue */
    re = eps->eigr[k];
    im = eps->eigi[k];
    if (eps->trueres || isshift) {
      ierr = STBackTransform(eps->OP,1,&re,&im);CHKERRQ(ierr);
    }
    newk = k;
    ierr = DSVectors(eps->ds,DS_MAT_X,&newk,&resnorm);CHKERRQ(ierr);
    resnorm *= beta;
    if (eps->trueres) {
      ierr = DSGetArray(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      Zr = X+k*ld;
      if (newk==k+1) Zi = X+newk*ld;
      else Zi = PETSC_NULL;
      ierr = EPSComputeRitzVector(eps,Zr,Zi,V,nv,x,y);CHKERRQ(ierr);
      ierr = DSRestoreArray(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = EPSComputeResidualNorm_Private(eps,re,im,x,y,&resnorm);
    }
    else resnorm *= corrf;
    /* error estimate */
    ierr = (*eps->conv_func)(eps,re,im,resnorm,&eps->errest[k],eps->conv_ctx);CHKERRQ(ierr);
    if (marker==-1 && eps->errest[k] >= eps->tol) marker = k;
    if (newk==k+1) { eps->errest[k+1] = eps->errest[k]; k++; }
    if (marker!=-1 && !getall) break;
  }
  if (marker!=-1) k = marker;
  *kout = k;
  if (eps->trueres) {
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
  }
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
    ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,j+1,PETSC_NULL,V,V[j+1],hwork,&norm,breakdown);CHKERRQ(ierr);
    alpha[j] = PetscRealPart(hwork[j]);
    beta[j] = norm;
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
  ierr = IPOrthogonalize(eps->ip,eps->nds,eps->defl,m,PETSC_NULL,V,f,hwork,&norm,PETSC_NULL);CHKERRQ(ierr);
  alpha[m-1] = PetscRealPart(hwork[m-1]); 
  beta[m-1] = norm;
  
  if (m > 100) {
    ierr = PetscFree(hwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


