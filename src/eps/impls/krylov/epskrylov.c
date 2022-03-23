/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Common subroutines for all Krylov-type solvers
*/

#include <slepc/private/epsimpl.h>
#include <slepc/private/slepcimpl.h>
#include <slepcblaslapack.h>

/*
   EPSDelayedArnoldi - This function is equivalent to BVMatArnoldi but
   performs the computation in a different way. The main idea is that
   reorthogonalization is delayed to the next Arnoldi step. This version is
   more scalable but in some cases convergence may stagnate.
*/
PetscErrorCode EPSDelayedArnoldi(EPS eps,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *M,PetscReal *beta,PetscBool *breakdown)
{
  PetscInt       i,j,m=*M;
  Vec            u,t;
  PetscScalar    shh[100],*lhh,dot,dot2;
  PetscReal      norm1=0.0,norm2=1.0;
  Vec            vj,vj1,vj2=NULL;

  PetscFunctionBegin;
  if (m<=100) lhh = shh;
  else CHKERRQ(PetscMalloc1(m,&lhh));
  CHKERRQ(BVCreateVec(eps->V,&u));
  CHKERRQ(BVCreateVec(eps->V,&t));

  CHKERRQ(BVSetActiveColumns(eps->V,0,m));
  for (j=k;j<m;j++) {
    CHKERRQ(BVGetColumn(eps->V,j,&vj));
    CHKERRQ(BVGetColumn(eps->V,j+1,&vj1));
    CHKERRQ(STApply(eps->st,vj,vj1));
    CHKERRQ(BVRestoreColumn(eps->V,j,&vj));
    CHKERRQ(BVRestoreColumn(eps->V,j+1,&vj1));

    CHKERRQ(BVDotColumnBegin(eps->V,j+1,H+ldh*j));
    if (j>k) {
      CHKERRQ(BVDotColumnBegin(eps->V,j,lhh));
      CHKERRQ(BVGetColumn(eps->V,j,&vj));
      CHKERRQ(VecDotBegin(vj,vj,&dot));
      if (j>k+1) {
        CHKERRQ(BVNormVecBegin(eps->V,u,NORM_2,&norm2));
        CHKERRQ(BVGetColumn(eps->V,j-2,&vj2));
        CHKERRQ(VecDotBegin(u,vj2,&dot2));
      }
      CHKERRQ(BVDotColumnEnd(eps->V,j+1,H+ldh*j));
      CHKERRQ(BVDotColumnEnd(eps->V,j,lhh));
      CHKERRQ(VecDotEnd(vj,vj,&dot));
      CHKERRQ(BVRestoreColumn(eps->V,j,&vj));
      if (j>k+1) {
        CHKERRQ(BVNormVecEnd(eps->V,u,NORM_2,&norm2));
        CHKERRQ(VecDotEnd(u,vj2,&dot2));
        CHKERRQ(BVRestoreColumn(eps->V,j-2,&vj2));
      }
      norm1 = PetscSqrtReal(PetscRealPart(dot));
      for (i=0;i<j;i++) H[ldh*j+i] = H[ldh*j+i]/norm1;
      H[ldh*j+j] = H[ldh*j+j]/dot;
      CHKERRQ(BVCopyVec(eps->V,j,t));
      CHKERRQ(BVScaleColumn(eps->V,j,1.0/norm1));
      CHKERRQ(BVScaleColumn(eps->V,j+1,1.0/norm1));
    } else CHKERRQ(BVDotColumnEnd(eps->V,j+1,H+ldh*j)); /* j==k */

    CHKERRQ(BVMultColumn(eps->V,-1.0,1.0,j+1,H+ldh*j));

    if (j>k) {
      CHKERRQ(BVSetActiveColumns(eps->V,0,j));
      CHKERRQ(BVMultVec(eps->V,-1.0,1.0,t,lhh));
      CHKERRQ(BVSetActiveColumns(eps->V,0,m));
      for (i=0;i<j;i++) H[ldh*(j-1)+i] += lhh[i];
    }

    if (j>k+1) {
      CHKERRQ(BVGetColumn(eps->V,j-1,&vj1));
      CHKERRQ(VecCopy(u,vj1));
      CHKERRQ(BVRestoreColumn(eps->V,j-1,&vj1));
      CHKERRQ(BVScaleColumn(eps->V,j-1,1.0/norm2));
      H[ldh*(j-2)+j-1] = norm2;
    }

    if (j<m-1) CHKERRQ(VecCopy(t,u));
  }

  CHKERRQ(BVNormVec(eps->V,t,NORM_2,&norm2));
  CHKERRQ(VecScale(t,1.0/norm2));
  CHKERRQ(BVGetColumn(eps->V,m-1,&vj1));
  CHKERRQ(VecCopy(t,vj1));
  CHKERRQ(BVRestoreColumn(eps->V,m-1,&vj1));
  H[ldh*(m-2)+m-1] = norm2;

  CHKERRQ(BVDotColumn(eps->V,m,lhh));

  CHKERRQ(BVMultColumn(eps->V,-1.0,1.0,m,lhh));
  for (i=0;i<m;i++)
    H[ldh*(m-1)+i] += lhh[i];

  CHKERRQ(BVNormColumn(eps->V,m,NORM_2,beta));
  CHKERRQ(BVScaleColumn(eps->V,m,1.0 / *beta));
  *breakdown = PETSC_FALSE;

  if (m>100) CHKERRQ(PetscFree(lhh));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&t));
  PetscFunctionReturn(0);
}

/*
   EPSDelayedArnoldi1 - This function is similar to EPSDelayedArnoldi,
   but without reorthogonalization (only delayed normalization).
*/
PetscErrorCode EPSDelayedArnoldi1(EPS eps,PetscScalar *H,PetscInt ldh,PetscInt k,PetscInt *M,PetscReal *beta,PetscBool *breakdown)
{
  PetscInt       i,j,m=*M;
  PetscScalar    dot;
  PetscReal      norm=0.0;
  Vec            vj,vj1;

  PetscFunctionBegin;
  CHKERRQ(BVSetActiveColumns(eps->V,0,m));
  for (j=k;j<m;j++) {
    CHKERRQ(BVGetColumn(eps->V,j,&vj));
    CHKERRQ(BVGetColumn(eps->V,j+1,&vj1));
    CHKERRQ(STApply(eps->st,vj,vj1));
    CHKERRQ(BVRestoreColumn(eps->V,j+1,&vj1));
    if (j>k) {
      CHKERRQ(BVDotColumnBegin(eps->V,j+1,H+ldh*j));
      CHKERRQ(VecDotBegin(vj,vj,&dot));
      CHKERRQ(BVDotColumnEnd(eps->V,j+1,H+ldh*j));
      CHKERRQ(VecDotEnd(vj,vj,&dot));
      norm = PetscSqrtReal(PetscRealPart(dot));
      CHKERRQ(BVScaleColumn(eps->V,j,1.0/norm));
      H[ldh*(j-1)+j] = norm;
      for (i=0;i<j;i++) H[ldh*j+i] = H[ldh*j+i]/norm;
      H[ldh*j+j] = H[ldh*j+j]/dot;
      CHKERRQ(BVScaleColumn(eps->V,j+1,1.0/norm));
      *beta = norm;
    } else {  /* j==k */
      CHKERRQ(BVDotColumn(eps->V,j+1,H+ldh*j));
    }
    CHKERRQ(BVRestoreColumn(eps->V,j,&vj));
    CHKERRQ(BVMultColumn(eps->V,-1.0,1.0,j+1,H+ldh*j));
  }

  *breakdown = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*
   EPSKrylovConvergence_Filter - Specialized version for STFILTER.
*/
PetscErrorCode EPSKrylovConvergence_Filter(EPS eps,PetscBool getall,PetscInt kini,PetscInt nits,PetscReal beta,PetscReal gamma,PetscInt *kout)
{
  PetscInt       k,ninside,nconv;
  PetscScalar    re,im;
  PetscReal      resnorm;

  PetscFunctionBegin;
  ninside = 0;   /* count how many eigenvalues are located in the interval */
  for (k=kini;k<kini+nits;k++) {
    if (PetscRealPart(eps->eigr[k]) < gamma) break;
    ninside++;
  }
  eps->nev = ninside+kini;  /* adjust eigenvalue count */
  nconv = 0;   /* count how many eigenvalues satisfy the convergence criterion */
  for (k=kini;k<kini+ninside;k++) {
    /* eigenvalue */
    re = eps->eigr[k];
    im = eps->eigi[k];
    CHKERRQ(DSVectors(eps->ds,DS_MAT_X,&k,&resnorm));
    resnorm *= beta;
    /* error estimate */
    CHKERRQ((*eps->converged)(eps,re,im,resnorm,&eps->errest[k],eps->convergedctx));
    if (eps->errest[k] < eps->tol) nconv++;
    else break;
  }
  *kout = kini+nconv;
  CHKERRQ(PetscInfo(eps,"Found %" PetscInt_FMT " eigenvalue approximations inside the interval (gamma=%g), k=%" PetscInt_FMT " nconv=%" PetscInt_FMT "\n",ninside,(double)gamma,k,nconv));
  PetscFunctionReturn(0);
}

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
PetscErrorCode EPSKrylovConvergence(EPS eps,PetscBool getall,PetscInt kini,PetscInt nits,PetscReal beta,PetscReal betat,PetscReal corrf,PetscInt *kout)
{
  PetscInt       k,newk,newk2,marker,ld,inside;
  PetscScalar    re,im,*Zr,*Zi,*X;
  PetscReal      resnorm,gamma,lerrest;
  PetscBool      isshift,isfilter,refined,istrivial;
  Vec            x=NULL,y=NULL,w[3];

  PetscFunctionBegin;
  if (PetscUnlikely(eps->which == EPS_ALL)) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STFILTER,&isfilter));
    if (isfilter) {
      CHKERRQ(STFilterGetThreshold(eps->st,&gamma));
      CHKERRQ(EPSKrylovConvergence_Filter(eps,getall,kini,nits,beta,gamma,kout));
      PetscFunctionReturn(0);
    }
  }
  CHKERRQ(RGIsTrivial(eps->rg,&istrivial));
  if (PetscUnlikely(eps->trueres)) {
    CHKERRQ(BVCreateVec(eps->V,&x));
    CHKERRQ(BVCreateVec(eps->V,&y));
    CHKERRQ(BVCreateVec(eps->V,&w[0]));
    CHKERRQ(BVCreateVec(eps->V,&w[2]));
#if !defined(PETSC_USE_COMPLEX)
    CHKERRQ(BVCreateVec(eps->V,&w[1]));
#else
    w[1] = NULL;
#endif
  }
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(DSGetRefined(eps->ds,&refined));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isshift));
  marker = -1;
  if (eps->trackall) getall = PETSC_TRUE;
  for (k=kini;k<kini+nits;k++) {
    /* eigenvalue */
    re = eps->eigr[k];
    im = eps->eigi[k];
    if (!istrivial || eps->trueres || isshift || eps->conv==EPS_CONV_NORM) CHKERRQ(STBackTransform(eps->st,1,&re,&im));
    if (PetscUnlikely(!istrivial)) {
      CHKERRQ(RGCheckInside(eps->rg,1,&re,&im,&inside));
      if (marker==-1 && inside<0) marker = k;
      if (!(eps->trueres || isshift || eps->conv==EPS_CONV_NORM)) {  /* make sure eps->converged below uses the right value */
        re = eps->eigr[k];
        im = eps->eigi[k];
      }
    }
    newk = k;
    CHKERRQ(DSVectors(eps->ds,DS_MAT_X,&newk,&resnorm));
    if (PetscUnlikely(eps->trueres)) {
      CHKERRQ(DSGetArray(eps->ds,DS_MAT_X,&X));
      Zr = X+k*ld;
      if (newk==k+1) Zi = X+newk*ld;
      else Zi = NULL;
      CHKERRQ(EPSComputeRitzVector(eps,Zr,Zi,eps->V,x,y));
      CHKERRQ(DSRestoreArray(eps->ds,DS_MAT_X,&X));
      CHKERRQ(EPSComputeResidualNorm_Private(eps,PETSC_FALSE,re,im,x,y,w,&resnorm));
    }
    else if (!refined) resnorm *= beta*corrf;
    /* error estimate */
    CHKERRQ((*eps->converged)(eps,re,im,resnorm,&eps->errest[k],eps->convergedctx));
    if (marker==-1 && eps->errest[k] >= eps->tol) marker = k;
    if (PetscUnlikely(eps->twosided)) {
      newk2 = k;
      CHKERRQ(DSVectors(eps->ds,DS_MAT_Y,&newk2,&resnorm));
      resnorm *= betat;
      CHKERRQ((*eps->converged)(eps,re,im,resnorm,&lerrest,eps->convergedctx));
      eps->errest[k] = PetscMax(eps->errest[k],lerrest);
      if (marker==-1 && lerrest >= eps->tol) marker = k;
    }
    if (PetscUnlikely(newk==k+1)) {
      eps->errest[k+1] = eps->errest[k];
      k++;
    }
    if (marker!=-1 && !getall) break;
  }
  if (marker!=-1) k = marker;
  *kout = k;
  if (PetscUnlikely(eps->trueres)) {
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));
    CHKERRQ(VecDestroy(&w[0]));
    CHKERRQ(VecDestroy(&w[2]));
#if !defined(PETSC_USE_COMPLEX)
    CHKERRQ(VecDestroy(&w[1]));
#endif
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSPseudoLanczos(EPS eps,PetscReal *alpha,PetscReal *beta,PetscReal *omega,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscBool *symmlost,PetscReal *cos,Vec w)
{
  PetscInt       j,m = *M,i,ld,l;
  Vec            vj,vj1;
  PetscScalar    *hwork,lhwork[100];
  PetscReal      norm,norm1,norm2,t,sym=0.0,fro=0.0;
  PetscBLASInt   j_,one=1;

  PetscFunctionBegin;
  CHKERRQ(DSGetLeadingDimension(eps->ds,&ld));
  CHKERRQ(DSGetDimensions(eps->ds,NULL,&l,NULL,NULL));
  if (cos) *cos = 1.0;
  if (m > 100) CHKERRQ(PetscMalloc1(m,&hwork));
  else hwork = lhwork;

  CHKERRQ(BVSetActiveColumns(eps->V,0,m));
  for (j=k;j<m;j++) {
    CHKERRQ(BVGetColumn(eps->V,j,&vj));
    CHKERRQ(BVGetColumn(eps->V,j+1,&vj1));
    CHKERRQ(STApply(eps->st,vj,vj1));
    CHKERRQ(BVRestoreColumn(eps->V,j,&vj));
    CHKERRQ(BVRestoreColumn(eps->V,j+1,&vj1));
    CHKERRQ(BVOrthogonalizeColumn(eps->V,j+1,hwork,&norm,breakdown));
    alpha[j] = PetscRealPart(hwork[j]);
    beta[j] = PetscAbsReal(norm);
    if (j==k) {
      PetscReal *f;

      CHKERRQ(DSGetArrayReal(eps->ds,DS_MAT_T,&f));
      for (i=0;i<l;i++) hwork[i]  = 0.0;
      for (;i<j-1;i++)  hwork[i] -= f[2*ld+i];
      CHKERRQ(DSRestoreArrayReal(eps->ds,DS_MAT_T,&f));
    }
    hwork[j-1] -= beta[j-1];
    CHKERRQ(PetscBLASIntCast(j,&j_));
    sym = SlepcAbs(BLASnrm2_(&j_,hwork,&one),sym);
    fro = SlepcAbs(fro,SlepcAbs(alpha[j],beta[j]));
    if (j>0) fro = SlepcAbs(fro,beta[j-1]);
    if (sym/fro>PetscMax(PETSC_SQRT_MACHINE_EPSILON,10*eps->tol)) { *symmlost = PETSC_TRUE; *M=j+1; break; }
    omega[j+1] = (norm<0.0)? -1.0: 1.0;
    CHKERRQ(BVScaleColumn(eps->V,j+1,1.0/norm));
    /* */
    if (cos) {
      CHKERRQ(BVGetColumn(eps->V,j+1,&vj1));
      CHKERRQ(VecNorm(vj1,NORM_2,&norm1));
      CHKERRQ(BVApplyMatrix(eps->V,vj1,w));
      CHKERRQ(BVRestoreColumn(eps->V,j+1,&vj1));
      CHKERRQ(VecNorm(w,NORM_2,&norm2));
      t = 1.0/(norm1*norm2);
      if (*cos>t) *cos = t;
    }
  }
  if (m > 100) CHKERRQ(PetscFree(hwork));
  PetscFunctionReturn(0);
}
