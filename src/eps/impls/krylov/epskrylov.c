/*
   Common subroutines for all Krylov-type solvers.

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

#include <slepc/private/epsimpl.h>
#include <slepc/private/slepcimpl.h>
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
    if (*breakdown) {
      H[j+1+ldh*j] = 0.0;
      *M = j+1;
      break;
    } else {
      H[j+1+ldh*j] = *beta;
      ierr = BVScaleColumn(eps->V,j+1,1.0/(*beta));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDelayedArnoldi"
/*
   EPSDelayedArnoldi - This function is equivalent to EPSBasicArnoldi but
   performs the computation in a different way. The main idea is that
   reorthogonalization is delayed to the next Arnoldi step. This version is
   more scalable but in some cases convergence may stagnate.
*/
PetscErrorCode EPSDelayedArnoldi(EPS eps,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *M,PetscReal *beta,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m=*M;
  Vec            u,t;
  PetscScalar    shh[100],*lhh,dot,dot2;
  PetscReal      norm1=0.0,norm2=1.0;
  Vec            vj,vj1,vj2;

  PetscFunctionBegin;
  if (m<=100) lhh = shh;
  else {
    ierr = PetscMalloc1(m,&lhh);CHKERRQ(ierr);
  }
  ierr = BVCreateVec(eps->V,&u);CHKERRQ(ierr);
  ierr = BVCreateVec(eps->V,&t);CHKERRQ(ierr);

  ierr = BVSetActiveColumns(eps->V,0,m);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    ierr = BVGetColumn(eps->V,j,&vj);CHKERRQ(ierr);
    ierr = BVGetColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = STApply(eps->st,vj,vj1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j,&vj);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);

    ierr = BVDotColumnBegin(eps->V,j+1,H+ldh*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = BVDotColumnBegin(eps->V,j,lhh);CHKERRQ(ierr);
      ierr = BVGetColumn(eps->V,j,&vj);CHKERRQ(ierr);
      ierr = VecDotBegin(vj,vj,&dot);CHKERRQ(ierr);
    }
    if (j>k+1) {
      ierr = BVNormVecBegin(eps->V,u,NORM_2,&norm2);CHKERRQ(ierr);
      ierr = BVGetColumn(eps->V,j-2,&vj2);CHKERRQ(ierr);
      ierr = VecDotBegin(u,vj2,&dot2);CHKERRQ(ierr);
    }

    ierr = BVDotColumnEnd(eps->V,j+1,H+ldh*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = BVDotColumnEnd(eps->V,j,lhh);CHKERRQ(ierr);
      ierr = VecDotEnd(vj,vj,&dot);CHKERRQ(ierr);
      ierr = BVRestoreColumn(eps->V,j,&vj);CHKERRQ(ierr);
    }
    if (j>k+1) {
      ierr = BVNormVecEnd(eps->V,u,NORM_2,&norm2);CHKERRQ(ierr);
      ierr = VecDotEnd(u,vj2,&dot2);CHKERRQ(ierr);
      ierr = BVRestoreColumn(eps->V,j-2,&vj2);CHKERRQ(ierr);
    }

    if (j>k) {
      norm1 = PetscSqrtReal(PetscRealPart(dot));
      for (i=0;i<j;i++)
        H[ldh*j+i] = H[ldh*j+i]/norm1;
      H[ldh*j+j] = H[ldh*j+j]/dot;

      ierr = BVCopyVec(eps->V,j,t);CHKERRQ(ierr);
      ierr = BVScaleColumn(eps->V,j,1.0/norm1);CHKERRQ(ierr);
      ierr = BVScaleColumn(eps->V,j+1,1.0/norm1);CHKERRQ(ierr);
    }

    ierr = BVMultColumn(eps->V,-1.0,1.0,j+1,H+ldh*j);CHKERRQ(ierr);

    if (j>k) {
      ierr = BVSetActiveColumns(eps->V,0,j);CHKERRQ(ierr);
      ierr = BVMultVec(eps->V,-1.0,1.0,t,lhh);CHKERRQ(ierr);
      ierr = BVSetActiveColumns(eps->V,0,m);CHKERRQ(ierr);
      for (i=0;i<j;i++)
        H[ldh*(j-1)+i] += lhh[i];
    }

    if (j>k+1) {
      ierr = BVGetColumn(eps->V,j-1,&vj1);CHKERRQ(ierr);
      ierr = VecCopy(u,vj1);CHKERRQ(ierr);
      ierr = BVRestoreColumn(eps->V,j-1,&vj1);CHKERRQ(ierr);
      ierr = BVScaleColumn(eps->V,j-1,1.0/norm2);CHKERRQ(ierr);
      H[ldh*(j-2)+j-1] = norm2;
    }

    if (j<m-1) {
      ierr = VecCopy(t,u);CHKERRQ(ierr);
    }
  }

  ierr = BVNormVec(eps->V,t,NORM_2,&norm2);CHKERRQ(ierr);
  ierr = VecScale(t,1.0/norm2);CHKERRQ(ierr);
  ierr = BVGetColumn(eps->V,m-1,&vj1);CHKERRQ(ierr);
  ierr = VecCopy(t,vj1);CHKERRQ(ierr);
  ierr = BVRestoreColumn(eps->V,m-1,&vj1);CHKERRQ(ierr);
  H[ldh*(m-2)+m-1] = norm2;

  ierr = BVDotColumn(eps->V,m,lhh);CHKERRQ(ierr);

  ierr = BVMultColumn(eps->V,-1.0,1.0,m,lhh);CHKERRQ(ierr);
  for (i=0;i<m;i++)
    H[ldh*(m-1)+i] += lhh[i];

  ierr = BVNormColumn(eps->V,m,NORM_2,beta);CHKERRQ(ierr);
  ierr = BVScaleColumn(eps->V,m,1.0 / *beta);CHKERRQ(ierr);
  *breakdown = PETSC_FALSE;

  if (m>100) { ierr = PetscFree(lhh);CHKERRQ(ierr); }
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDelayedArnoldi1"
/*
   EPSDelayedArnoldi1 - This function is similar to EPSDelayedArnoldi,
   but without reorthogonalization (only delayed normalization).
*/
PetscErrorCode EPSDelayedArnoldi1(EPS eps,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *M,PetscReal *beta,PetscBool *breakdown)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m=*M;
  PetscScalar    dot;
  PetscReal      norm=0.0;
  Vec            vj,vj1;

  PetscFunctionBegin;
  ierr = BVSetActiveColumns(eps->V,0,m);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    ierr = BVGetColumn(eps->V,j,&vj);CHKERRQ(ierr);
    ierr = BVGetColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = STApply(eps->st,vj,vj1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
    ierr = BVDotColumnBegin(eps->V,j+1,H+ldh*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = VecDotBegin(vj,vj,&dot);CHKERRQ(ierr);
    }
    ierr = BVDotColumnEnd(eps->V,j+1,H+ldh*j);CHKERRQ(ierr);
    if (j>k) {
      ierr = VecDotEnd(vj,vj,&dot);CHKERRQ(ierr);
    }
    ierr = BVRestoreColumn(eps->V,j,&vj);CHKERRQ(ierr);

    if (j>k) {
      norm = PetscSqrtReal(PetscRealPart(dot));
      ierr = BVScaleColumn(eps->V,j,1.0/norm);CHKERRQ(ierr);
      H[ldh*(j-1)+j] = norm;

      for (i=0;i<j;i++)
        H[ldh*j+i] = H[ldh*j+i]/norm;
      H[ldh*j+j] = H[ldh*j+j]/dot;
      ierr = BVScaleColumn(eps->V,j+1,1.0/norm);CHKERRQ(ierr);
      *beta = norm;
    }
    ierr = BVMultColumn(eps->V,-1.0,1.0,j+1,H+ldh*j);CHKERRQ(ierr);
  }

  *breakdown = PETSC_FALSE;
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
  Vec            x,y,w[3];

  PetscFunctionBegin;
  ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
  if (eps->trueres) {
    ierr = BVCreateVec(eps->V,&x);CHKERRQ(ierr);
    ierr = BVCreateVec(eps->V,&y);CHKERRQ(ierr);
    ierr = BVCreateVec(eps->V,&w[0]);CHKERRQ(ierr);
    ierr = BVCreateVec(eps->V,&w[2]);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = BVCreateVec(eps->V,&w[1]);CHKERRQ(ierr);
#else
    w[1] = NULL;
#endif
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
      if (marker==-1 && inside<0) marker = k;
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
      ierr = EPSComputeResidualNorm_Private(eps,re,im,x,y,w,&resnorm);CHKERRQ(ierr);
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
    ierr = VecDestroy(&w[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&w[2]);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = VecDestroy(&w[1]);CHKERRQ(ierr);
#endif
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
PetscErrorCode EPSPseudoLanczos(EPS eps,PetscReal *alpha,PetscReal *beta,PetscReal *omega,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscBool *symmlost,PetscReal *cos,Vec w)
{
  PetscErrorCode ierr;
  PetscInt       j,m = *M,i,ld,l;
  Vec            vj,vj1;
  PetscScalar    *hwork,lhwork[100];
  PetscReal      norm,norm1,norm2,t,*f,sym=0.0,fro=0.0;
  PetscBLASInt   j_,one=1;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(eps->ds,&ld);CHKERRQ(ierr);
  ierr = DSGetDimensions(eps->ds,NULL,NULL,&l,NULL,NULL);CHKERRQ(ierr);
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
    ierr = DSGetArrayReal(eps->ds,DS_MAT_T,&f);CHKERRQ(ierr);
    if (j==k) { 
      for (i=l;i<j-1;i++) hwork[i]-= f[2*ld+i];
      for (i=0;i<l;i++) hwork[i] = 0.0;
    }
    ierr = DSRestoreArrayReal(eps->ds,DS_MAT_T,&f);CHKERRQ(ierr);
    hwork[j-1] -= beta[j-1];
    ierr = PetscBLASIntCast(j,&j_);CHKERRQ(ierr);
    sym = SlepcAbs(BLASnrm2_(&j_,hwork,&one),sym);
    fro = SlepcAbs(fro,SlepcAbs(alpha[j],beta[j]));
    if (j>0) fro = SlepcAbs(fro,beta[j-1]);
    if (sym/fro>PetscMax(PETSC_SQRT_MACHINE_EPSILON,10*eps->tol)) { *symmlost = PETSC_TRUE; *M=j+1; break; }
    omega[j+1] = (norm<0.0)? -1.0: 1.0;
    ierr = BVScaleColumn(eps->V,j+1,1.0/norm);CHKERRQ(ierr);
    /* */
    if (cos) {
      ierr = BVGetColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
      ierr = VecNorm(vj1,NORM_2,&norm1);CHKERRQ(ierr);
      ierr = BVApplyMatrix(eps->V,vj1,w);CHKERRQ(ierr);
      ierr = BVRestoreColumn(eps->V,j+1,&vj1);CHKERRQ(ierr);
      ierr = VecNorm(w,NORM_2,&norm2);CHKERRQ(ierr);
      t = 1.0/(norm1*norm2);
      if (*cos>t) *cos = t;
    }
  }
  if (m > 100) {
    ierr = PetscFree(hwork);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

