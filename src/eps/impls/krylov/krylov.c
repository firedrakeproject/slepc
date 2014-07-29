/*
   Common subroutines for all Krylov-type solvers.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

                    OP * V - V * H = beta*v_m * e_m^T

   where the columns of V are the Arnoldi vectors (which are B-orthonormal),
   H is an upper Hessenberg matrix, e_m is the m-th vector of the canonical basis.
   On exit, beta contains the B-norm of V[m] before normalization.
*/
PetscErrorCode EPSBasicArnoldi(EPS eps,PetscBool trans,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *M,PetscReal *beta,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       j,m = *M;
  Vec            vj,vj1;

  PetscFunctionBegin;
  ierr = BVSetActiveColumns(eps->V,0,m);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    ierr = BVGetColumn(eps->V,j,&vj);CHKERRQ(ierr);
    ierr = BVGetColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    if (trans) {
      ierr = STApplyTranspose(eps->st,vj,vj1);CHKERRQ(ierr);
    } else {
      ierr = STApply(eps->st,vj,vj1);CHKERRQ(ierr);
    }
    ierr = BVRestoreColumn(eps->V,j,&vj);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = BVOrthogonalizeColumn(eps->V,j+1,H+ldh*j,beta,breakdown);CHKERRQ(ierr);
    H[j+1+ldh*j] = *beta;
    if (*breakdown) {
      *M = j+1;
      break;
    } else {
      ierr = BVScaleColumn(eps->V,j+1,1.0/(*beta));CHKERRQ(ierr);
    }
  }
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
PetscErrorCode EPSKrylovConvergence(EPS eps,PetscBool getall,PetscInt kini,PetscInt nits,PetscReal beta,PetscReal corrf,PetscInt *kout)
{
  PetscErrorCode ierr;
  PetscInt       k,newk,marker,ld,inside;
  PetscScalar    re,im,*Zr,*Zi,*X;
  PetscReal      resnorm;
  PetscBool      isshift,refined,istrivial;
  Vec            x,y;

  PetscFunctionBegin;
  ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
  if (eps->trueres) {
    ierr = BVGetVec(eps->V,&x);CHKERRQ(ierr);
    ierr = BVGetVec(eps->V,&y);CHKERRQ(ierr);
  }
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = DSGetRefined(eps->ds,&refined);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isshift);CHKERRQ(ierr);
  marker = -1;
  if (eps->trackall) getall = PETSC_TRUE;
  for (k=kini;k<kini+nits;k++) {
    /* eigenvalue */
    re = eps->eigr[k];
    im = eps->eigi[k];
    if (!istrivial || eps->trueres || isshift || eps->conv==EPS_CONV_NORM) {
      ierr = STBackTransform(eps->st,1,&re,&im);CHKERRQ(ierr);
    }
    if (!istrivial) {
      ierr = RGCheckInside(eps->rg,1,&re,&im,&inside);CHKERRQ(ierr);
      if (marker==-1 && inside<=0) marker = k;
      if (!(eps->trueres || isshift || eps->conv==EPS_CONV_NORM)) {  /* make sure eps->converged below uses the right value */
        re = eps->eigr[k];
        im = eps->eigi[k];
      }
    }
    newk = k;
    ierr = DSVectors(eps->ds,DS_MAT_X,&newk,&resnorm);CHKERRQ(ierr);
    if (eps->trueres) {
      ierr = DSGetArray(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      Zr = X+k*ld;
      if (newk==k+1) Zi = X+newk*ld;
      else Zi = NULL;
      ierr = EPSComputeRitzVector(eps,Zr,Zi,eps->V,x,y);CHKERRQ(ierr);
      ierr = DSRestoreArray(eps->ds,DS_MAT_X,&X);CHKERRQ(ierr);
      ierr = EPSComputeResidualNorm_Private(eps,re,im,x,y,&resnorm);CHKERRQ(ierr);
    }
    else if (!refined) resnorm *= beta*corrf;
    /* error estimate */
    ierr = (*eps->converged)(eps,re,im,resnorm,&eps->errest[k],eps->convergedctx);CHKERRQ(ierr);
    if (marker==-1 && eps->errest[k] >= eps->tol) marker = k;
    if (newk==k+1) {
      eps->errest[k+1] = eps->errest[k];
      k++;
    }
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

                    OP * V - V * T = beta_m*v_m * e_m^T

   where the columns of V are the Lanczos vectors (which are B-orthonormal),
   T is a real symmetric tridiagonal matrix, and e_m is the m-th vector of
   the canonical basis. The tridiagonal is stored as two arrays: alpha
   contains the diagonal elements, beta the off-diagonal. On exit, the last
   element of beta contains the B-norm of V[m] before normalization.
*/
PetscErrorCode EPSFullLanczos(EPS eps,PetscReal *alpha,PetscReal *beta,PetscInt k,PetscInt *M,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       j,m = *M;
  Vec            vj,vj1;
  PetscScalar    *hwork,lhwork[100];

  PetscFunctionBegin;
  if (m > 100) {
    ierr = PetscMalloc1(m,&hwork);CHKERRQ(ierr);
  } else hwork = lhwork;

  ierr = BVSetActiveColumns(eps->V,0,m);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    ierr = BVGetColumn(eps->V,j,&vj);CHKERRQ(ierr);
    ierr = BVGetColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = STApply(eps->st,vj,vj1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j,&vj);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = BVOrthogonalizeColumn(eps->V,j+1,hwork,beta+j,breakdown);CHKERRQ(ierr);
    alpha[j] = PetscRealPart(hwork[j]);
    if (*breakdown) {
      *M = j+1;
      break;
    } else {
      ierr = BVScaleColumn(eps->V,j+1,1.0/beta[j]);CHKERRQ(ierr);
    }
  }
  if (m > 100) {
    ierr = PetscFree(hwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSPseudoLanczos"
PetscErrorCode EPSPseudoLanczos(EPS eps,PetscReal *alpha,PetscReal *beta,PetscReal *omega,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscReal *cos,Vec w)
{
  PetscErrorCode ierr;
  PetscInt       j,m = *M;
  Vec            vj,vj1;
  PetscScalar    *hwork,lhwork[100];
  PetscReal      norm,norm1,norm2,t;

  PetscFunctionBegin;
  if (cos) *cos = 1.0;
  if (m > 100) {
    ierr = PetscMalloc1(m,&hwork);CHKERRQ(ierr);
  } else hwork = lhwork;

  ierr = BVSetActiveColumns(eps->V,0,m);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    ierr = BVGetColumn(eps->V,j,&vj);CHKERRQ(ierr);
    ierr = BVGetColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = STApply(eps->st,vj,vj1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j,&vj);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = BVOrthogonalizeColumn(eps->V,j+1,hwork,&norm,breakdown);CHKERRQ(ierr);
    alpha[j] = PetscRealPart(hwork[j]);
    beta[j] = PetscAbsReal(norm);
    omega[j+1] = (norm<0.0)? -1.0: 1.0;
    ierr = BVScaleColumn(eps->V,j+1,1.0/norm);CHKERRQ(ierr);
    /* */
    ierr = BVGetColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = VecNorm(vj1,NORM_2,&norm1);CHKERRQ(ierr);
    ierr = BVApplyMatrix(eps->V,vj1,w);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&norm2);CHKERRQ(ierr);
    t = 1.0/(norm1*norm2);
    if (cos && *cos>t) *cos = t;
  }
  if (m > 100) {
    ierr = PetscFree(hwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

