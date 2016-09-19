/*

   SLEPc nonlinear eigensolver: "nleigs"

   Method: NLEIGS

   Algorithm:

       Fully rational Krylov method for nonlinear eigenvalue problems.

   References:

       [1] S. Guttel et al., "NLEIGS: A class of robust fully rational Krylov
           method for nonlinear eigenvalue problems", SIAM J. Sci. Comput.
           36(6):A2842-A2864, 2014.

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

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include <slepcblaslapack.h>

#define  MAX_LBPOINTS  100
#define  NDPOINTS      1e4
#define  MAX_NSHIFTS   100

typedef struct {
  PetscInt       nmat;      /* number of interpolation points */
  PetscScalar    *s,*xi;    /* Leja-Bagby points */
  PetscScalar    *beta;     /* scaling factors */
  Mat            *D;        /* divided difference matrices */
  PetscScalar    *coeffD;   /* coefficients for divided differences in split form */
  PetscInt       nshifts;   /* provided number of shifts */
  PetscScalar    *shifts;   /* user-provided shifts for the Rational Krylov variant */
  PetscInt       nshiftsw;  /* actual number of shifts (1 if Krylov-Schur) */
  PetscReal      ddtol;     /* tolerance for divided difference convergence */
  PetscInt       ddmaxit;   /* maximum number of divided difference terms */
  BV             W;         /* auxiliary BV object */
  PetscReal      keep;      /* restart parameter */
  PetscBool      lock;      /* locking/non-locking variant */
  PetscBool      trueres;   /* whether the true residual norm must be computed */
  PetscInt       idxrk;     /* index of next shift to use */
  KSP            *ksp;      /* ksp array for storing shift factorizations */
  Vec            vrn;       /* random vector with normally distributed value */
  void           *singularitiesctx;
  PetscErrorCode (*computesingularities)(NEP,PetscInt*,PetscScalar*,void*);
} NEP_NLEIGS;

typedef struct {
  PetscInt    nmat;
  PetscScalar coeff[MAX_NSHIFTS];
  Mat         A[MAX_NSHIFTS];
  Vec         t;
} ShellMatCtx;

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetShifts"
PETSC_STATIC_INLINE PetscErrorCode NEPNLEIGSSetShifts(NEP nep)
{
  NEP_NLEIGS *ctx = (NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  if (!ctx->nshifts) { 
    ctx->shifts = &nep->target;
    ctx->nshiftsw = 1;
  } else ctx->nshiftsw = ctx->nshifts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSBackTransform"
static PetscErrorCode NEPNLEIGSBackTransform(PetscObject ob,PetscInt n,PetscScalar *valr,PetscScalar *vali)
{
  NEP         nep;
  PetscInt    j;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar t;
#endif

  PetscFunctionBegin;
  nep = (NEP)ob;
#if !defined(PETSC_USE_COMPLEX)
  for (j=0;j<n;j++) {
    if (vali[j] == 0) valr[j] = 1.0 / valr[j] + nep->target;
    else {
      t = valr[j] * valr[j] + vali[j] * vali[j];
      valr[j] = valr[j] / t + nep->target;
      vali[j] = - vali[j] / t;
    }
  }
#else
  for (j=0;j<n;j++) {
    valr[j] = 1.0 / valr[j] + nep->target;
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSLejaBagbyPoints"
static PetscErrorCode NEPNLEIGSLejaBagbyPoints(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       i,k,ndpt=NDPOINTS,ndptx=NDPOINTS;
  PetscScalar    *ds,*dsi,*dxi,*nrs,*nrxi,*s=ctx->s,*xi=ctx->xi,*beta=ctx->beta;
  PetscReal      maxnrs,minnrxi;

  PetscFunctionBegin;
  ierr = PetscMalloc5(ndpt+1,&ds,ndpt+1,&dsi,ndpt,&dxi,ndpt+1,&nrs,ndpt,&nrxi);CHKERRQ(ierr);

  /* Discretize the target region boundary */
  ierr = RGComputeContour(nep->rg,ndpt,ds,dsi);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  for (i=0;i<ndpt;i++) if (dsi[i]!=0.0) break;
  if (i<ndpt) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"NLEIGS with real arithmetic requires the target set to be included in the real axis");
#endif
  /* Discretize the singularity region */
  if (ctx->computesingularities) {
    ierr = (ctx->computesingularities)(nep,&ndptx,dxi,ctx->singularitiesctx);CHKERRQ(ierr);
  } else ndptx = 0;

  /* Look for Leja-Bagby points in the discretization sets */
  s[0]    = ds[0];
  xi[0]   = (ndptx>0)?dxi[0]:PETSC_INFINITY;
  beta[0] = 1.0; /* scaling factors are also computed here */
  maxnrs  = 0.0;
  minnrxi = PETSC_MAX_REAL; 
  for (i=0;i<ndpt;i++) {
    nrs[i] = (ds[i]-s[0])/(1.0-ds[i]/xi[0]);
    if (PetscAbsScalar(nrs[i])>=maxnrs) {maxnrs = PetscAbsScalar(nrs[i]); s[1] = ds[i];}
  }
  for (i=1;i<ndptx;i++) {
    nrxi[i] = (dxi[i]-s[0])/(1.0-dxi[i]/xi[0]);
    if (PetscAbsScalar(nrxi[i])<=minnrxi) {minnrxi = PetscAbsScalar(nrxi[i]); xi[1] = dxi[i];}
  }
  if (ndptx<2) xi[1] = PETSC_INFINITY;

  beta[1] = maxnrs;
  for (k=2;k<ctx->ddmaxit;k++) {
    maxnrs = 0.0;
    minnrxi = PETSC_MAX_REAL;
    for (i=0;i<ndpt;i++) {
      nrs[i] *= ((ds[i]-s[k-1])/(1.0-ds[i]/xi[k-1]))/beta[k-1];
      if (PetscAbsScalar(nrs[i])>maxnrs) {maxnrs = PetscAbsScalar(nrs[i]); s[k] = ds[i];}
    }
    if (ndptx>=k) {
      for (i=1;i<ndptx;i++) {
        nrxi[i] *= ((dxi[i]-s[k-1])/(1.0-dxi[i]/xi[k-1]))/beta[k-1];
        if (PetscAbsScalar(nrxi[i])<minnrxi) {minnrxi = PetscAbsScalar(nrxi[i]); xi[k] = dxi[i];}
      }
    }  else xi[k] = PETSC_INFINITY;
    beta[k] = maxnrs;
  }
  ierr = PetscFree5(ds,dsi,dxi,nrs,nrxi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSEvalNRTFunct"
static PetscErrorCode NEPNLEIGSEvalNRTFunct(NEP nep,PetscInt k,PetscScalar sigma,PetscScalar *b)
{
  NEP_NLEIGS  *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt    i;
  PetscScalar *beta=ctx->beta,*s=ctx->s,*xi=ctx->xi;

  PetscFunctionBegin;
  b[0] = 1.0/beta[0];
  for (i=0;i<k;i++) {
    b[i+1] = ((sigma-s[i])*b[i])/(beta[i+1]*(1.0-sigma/xi[i]));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Fun"
static PetscErrorCode MatMult_Fun(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  ShellMatCtx    *ctx;
  PetscInt       i;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatMult(ctx->A[0],x,y);CHKERRQ(ierr);
  if (ctx->coeff[0]!=1.0) { ierr = VecScale(y,ctx->coeff[0]);CHKERRQ(ierr); }
  for (i=1;i<ctx->nmat;i++) {
    ierr = MatMult(ctx->A[i],x,ctx->t);CHKERRQ(ierr);
    ierr = VecAXPY(y,ctx->coeff[i],ctx->t);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Fun"
static PetscErrorCode MatMultTranspose_Fun(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  ShellMatCtx    *ctx;
  PetscInt       i;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatMultTranspose(ctx->A[0],x,y);CHKERRQ(ierr);
  if (ctx->coeff[0]!=1.0) { ierr = VecScale(y,ctx->coeff[0]);CHKERRQ(ierr); }
  for (i=1;i<ctx->nmat;i++) {
    ierr = MatMultTranspose(ctx->A[i],x,ctx->t);CHKERRQ(ierr);
    ierr = VecAXPY(y,ctx->coeff[i],ctx->t);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Fun"
static PetscErrorCode MatGetDiagonal_Fun(Mat A,Vec diag)
{
  PetscErrorCode ierr;
  ShellMatCtx    *ctx;
  PetscInt       i;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->A[0],diag);CHKERRQ(ierr);
  if (ctx->coeff[0]!=1.0) { ierr = VecScale(diag,ctx->coeff[0]);CHKERRQ(ierr); }
  for (i=1;i<ctx->nmat;i++) {
    ierr = MatGetDiagonal(ctx->A[i],ctx->t);CHKERRQ(ierr);
    ierr = VecAXPY(diag,ctx->coeff[i],ctx->t);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_Fun"
static PetscErrorCode MatDuplicate_Fun(Mat A,MatDuplicateOption op,Mat *B)
{
  PetscInt       n,i;
  ShellMatCtx    *ctxnew,*ctx;
  void           (*fun)();
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = PetscNew(&ctxnew);CHKERRQ(ierr);
  ctxnew->nmat = ctx->nmat;
  for (i=0;i<ctx->nmat;i++) {
    ierr = PetscObjectReference((PetscObject)ctx->A[i]);CHKERRQ(ierr);
    ctxnew->A[i] = ctx->A[i];
    ctxnew->coeff[i] = ctx->coeff[i];
  }
  ierr = MatGetSize(ctx->A[0],&n,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(ctx->t,&ctxnew->t);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_WORLD,n,n,n,n,(void*)ctxnew,B);CHKERRQ(ierr);
  ierr = MatShellGetOperation(A,MATOP_MULT,&fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_MULT,fun);CHKERRQ(ierr);
  ierr = MatShellGetOperation(A,MATOP_MULT_TRANSPOSE,&fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_MULT_TRANSPOSE,fun);CHKERRQ(ierr);
  ierr = MatShellGetOperation(A,MATOP_GET_DIAGONAL,&fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_GET_DIAGONAL,fun);CHKERRQ(ierr);
  ierr = MatShellGetOperation(A,MATOP_DUPLICATE,&fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_DUPLICATE,fun);CHKERRQ(ierr);
  ierr = MatShellGetOperation(A,MATOP_DESTROY,&fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_DESTROY,fun);CHKERRQ(ierr);
  ierr = MatShellGetOperation(A,MATOP_AXPY,&fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_AXPY,fun);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Fun"
static PetscErrorCode MatDestroy_Fun(Mat A)
{
  ShellMatCtx    *ctx;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBeginUser;
  if (A) {
    ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
    for (i=0;i<ctx->nmat;i++) {
      ierr = MatDestroy(&ctx->A[i]);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&ctx->t);CHKERRQ(ierr);
    ierr = PetscFree(ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAXPY_Fun"
static PetscErrorCode MatAXPY_Fun(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  ShellMatCtx    *ctxY,*ctxX;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBool      found;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(Y,(void**)&ctxY);CHKERRQ(ierr);
  ierr = MatShellGetContext(X,(void**)&ctxX);CHKERRQ(ierr);
  for (i=0;i<ctxX->nmat;i++) {
    found = PETSC_FALSE;
    for (j=0;!found&&j<ctxY->nmat;j++) {
      if (ctxX->A[i]==ctxY->A[j]) {
        found = PETSC_TRUE;
        ctxY->coeff[j] += a*ctxX->coeff[i];
      }
    }
    if (!found) {
      ctxY->coeff[ctxY->nmat] = a*ctxX->coeff[i];
      ctxY->A[ctxY->nmat++] = ctxX->A[i];
      ierr = PetscObjectReference((PetscObject)ctxX->A[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatScale_Fun"
static PetscErrorCode MatScale_Fun(Mat M,PetscScalar a)
{
  ShellMatCtx    *ctx;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
  for (i=0;i<ctx->nmat;i++) ctx->coeff[i] *= a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NLEIGSMatToMatShellArray"
static PetscErrorCode NLEIGSMatToMatShellArray(Mat M,Mat *Ms)
{
  PetscErrorCode ierr;
  ShellMatCtx    *ctx;
  PetscInt       n;
  PetscBool      has;
  
  PetscFunctionBegin;
  ierr = MatHasOperation(M,MATOP_DUPLICATE,&has);CHKERRQ(ierr);
  if (!has) SETERRQ(PetscObjectComm((PetscObject)M),1,"MatDuplicate operation required");
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ierr = MatDuplicate(M,MAT_COPY_VALUES,&ctx->A[0]);CHKERRQ(ierr);
  ctx->nmat = 1;
  ctx->coeff[0] = 1.0;
  ierr = MatCreateVecs(M,&ctx->t,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(M,&n,NULL);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject)M),n,n,n,n,(void*)ctx,Ms);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*Ms,MATOP_MULT,(void(*)())MatMult_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*Ms,MATOP_MULT_TRANSPOSE,(void(*)())MatMultTranspose_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*Ms,MATOP_GET_DIAGONAL,(void(*)())MatGetDiagonal_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*Ms,MATOP_DUPLICATE,(void(*)())MatDuplicate_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*Ms,MATOP_DESTROY,(void(*)())MatDestroy_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*Ms,MATOP_AXPY,(void(*)())MatAXPY_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*Ms,MATOP_SCALE,(void(*)())MatScale_Fun);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSNormEstimation"
static PetscErrorCode NEPNLEIGSNormEstimation(NEP nep,Mat M,PetscReal *norm,Vec *w)
{
  PetscScalar    *z,*x,*y;
  PetscReal      tr;
  Vec            X=w[0],Y=w[1];
  PetscInt       n,i;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscRandom    rand;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (!ctx->vrn) {
    /* generate a random vector with normally distributed entries with the Box-Muller transform */
    ierr = BVGetRandomContext(nep->V,&rand);CHKERRQ(ierr);
    ierr = MatCreateVecs(M,&ctx->vrn,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(X,rand);CHKERRQ(ierr);
    ierr = VecSetRandom(Y,rand);CHKERRQ(ierr);
    ierr = VecGetLocalSize(ctx->vrn,&n);CHKERRQ(ierr);
    ierr = VecGetArray(ctx->vrn,&z);CHKERRQ(ierr);
    ierr = VecGetArray(X,&x);CHKERRQ(ierr);
    ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
#if defined(PETSC_USE_COMPLEX)
      z[i] = PetscSqrtReal(-2.0*PetscLogReal(PetscRealPart(x[i])))*PetscCosReal(2.0*PETSC_PI*PetscRealPart(y[i]));
      z[i] += PETSC_i*(PetscSqrtReal(-2.0*PetscLogReal(PetscImaginaryPart(x[i])))*PetscCosReal(2.0*PETSC_PI*PetscImaginaryPart(y[i])));
#else
      z[i] = PetscSqrtReal(-2.0*PetscLogReal(x[i]))*PetscCosReal(2.0*PETSC_PI*y[i]);
#endif
    }
    ierr = VecRestoreArray(ctx->vrn,&z);CHKERRQ(ierr);
    ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
    ierr = VecNorm(ctx->vrn,NORM_2,&tr);CHKERRQ(ierr);
    ierr = VecScale(ctx->vrn,1/tr);CHKERRQ(ierr);
  }
  /* matrix-free norm estimator of Ipsen http://www4.ncsu.edu/~ipsen/ps/slides_ima.pdf */
  ierr = MatGetSize(M,&n,NULL);CHKERRQ(ierr);
  ierr = MatMult(M,ctx->vrn,X);CHKERRQ(ierr);
  ierr = VecNorm(X,NORM_2,norm);CHKERRQ(ierr);
  *norm *= PetscSqrtReal((PetscReal)n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSDividedDifferences_split"
static PetscErrorCode NEPNLEIGSDividedDifferences_split(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       k,j,i;
  PetscReal      norm0,norm,max;
  PetscScalar    *s=ctx->s,*beta=ctx->beta,*b,alpha,*coeffs;
  Mat            T,Ts;
  PetscBool      shell;

  PetscFunctionBegin;
  ierr = PetscMalloc1(nep->nt*ctx->ddmaxit,&ctx->coeffD);CHKERRQ(ierr);
  ierr = PetscMalloc2(ctx->ddmaxit+1,&b,ctx->ddmaxit+1,&coeffs);CHKERRQ(ierr);
  max = 0.0;
  for (j=0;j<nep->nt;j++) {
    ierr = FNEvaluateFunction(nep->f[j],s[0],ctx->coeffD+j);CHKERRQ(ierr);
    ctx->coeffD[j] /= beta[0];
    max = PetscMax(PetscAbsScalar(ctx->coeffD[j]),max);
  }
  norm0 = max;
  ctx->nmat = ctx->ddmaxit;
  for (k=1;k<ctx->ddmaxit;k++) {
    ierr = NEPNLEIGSEvalNRTFunct(nep,k,s[k],b);CHKERRQ(ierr);
    max = 0.0;
    for (i=0;i<nep->nt;i++) {
      ierr = FNEvaluateFunction(nep->f[i],s[k],ctx->coeffD+k*nep->nt+i);CHKERRQ(ierr);
      for (j=0;j<k;j++) {
        ctx->coeffD[k*nep->nt+i] -= b[j]*ctx->coeffD[i+nep->nt*j];
      }
      ctx->coeffD[k*nep->nt+i] /= b[k];
      max = PetscMax(PetscAbsScalar(ctx->coeffD[k*nep->nt+i]),max);
    }
    norm = max;
    if (norm/norm0 < ctx->ddtol) {
      ctx->nmat = k+1;
      break;
    } 
  }
  if (!ctx->ksp) { ierr = NEPNLEIGSGetKSPs(nep,&ctx->ksp);CHKERRQ(ierr); }
  ierr = PetscObjectTypeCompare((PetscObject)nep->A[0],MATSHELL,&shell);CHKERRQ(ierr);
  for (i=0;i<ctx->nshiftsw;i++) {
    ierr = NEPNLEIGSEvalNRTFunct(nep,ctx->nmat,ctx->shifts[i],coeffs);CHKERRQ(ierr);
    if (!shell) {
      ierr = MatDuplicate(nep->A[0],MAT_COPY_VALUES,&T);CHKERRQ(ierr);
    } else {
      ierr = NLEIGSMatToMatShellArray(nep->A[0],&T);CHKERRQ(ierr);
    }
    alpha = 0.0;
    for (j=0;j<ctx->nmat-1;j++) alpha += coeffs[j]*ctx->coeffD[j*nep->nt];
    ierr = MatScale(T,alpha);CHKERRQ(ierr);
    for (k=1;k<nep->nt;k++) {
      alpha = 0.0;
      for (j=0;j<ctx->nmat-1;j++) alpha += coeffs[j]*ctx->coeffD[j*nep->nt+k];
      if (shell) {
        ierr = NLEIGSMatToMatShellArray(nep->A[k],&Ts);CHKERRQ(ierr);
      }
      ierr = MatAXPY(T,alpha,shell?Ts:nep->A[k],nep->mstr);CHKERRQ(ierr);
      if (shell) {
        ierr = MatDestroy(&Ts);CHKERRQ(ierr);
      }
    }
    ierr = KSPSetOperators(ctx->ksp[i],T,T);CHKERRQ(ierr);
    ierr = KSPSetUp(ctx->ksp[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&T);CHKERRQ(ierr);
  }
  ierr = PetscFree2(b,coeffs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSDividedDifferences_callback"
static PetscErrorCode NEPNLEIGSDividedDifferences_callback(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       k,j,i;
  PetscReal      norm0,norm;
  PetscScalar    *s=ctx->s,*beta=ctx->beta,*b,*coeffs;
  Mat            *D=ctx->D,T;
  PetscBool      shell,has,vec=PETSC_FALSE;
  Vec            w[2];

  PetscFunctionBegin;
  ierr = PetscMalloc2(ctx->ddmaxit+1,&b,ctx->ddmaxit+1,&coeffs);CHKERRQ(ierr);
  T = nep->function;
  ierr = NEPComputeFunction(nep,s[0],T,T);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)T,MATSHELL,&shell);CHKERRQ(ierr);
  if (!shell) {
    ierr = MatDuplicate(T,MAT_COPY_VALUES,&D[0]);CHKERRQ(ierr);
  } else {
    ierr = NLEIGSMatToMatShellArray(T,&D[0]);CHKERRQ(ierr);
  }
  if (beta[0]!=1.0) {
    ierr = MatScale(D[0],1.0/beta[0]);CHKERRQ(ierr);
  }
  ierr = MatHasOperation(D[0],MATOP_NORM,&has);CHKERRQ(ierr);
  if (has) {
    ierr = MatNorm(D[0],NORM_FROBENIUS,&norm0);CHKERRQ(ierr);
  } else {
    ierr = MatCreateVecs(D[0],NULL,&w[0]);CHKERRQ(ierr);
    ierr = VecDuplicate(w[0],&w[1]);CHKERRQ(ierr);
    vec = PETSC_TRUE;
    ierr = NEPNLEIGSNormEstimation(nep,D[0],&norm0,w);CHKERRQ(ierr);
  }
  ctx->nmat = ctx->ddmaxit;
  for (k=1;k<ctx->ddmaxit;k++) {
    ierr = NEPNLEIGSEvalNRTFunct(nep,k,s[k],b);CHKERRQ(ierr);
    ierr = NEPComputeFunction(nep,s[k],T,T);CHKERRQ(ierr);
    if (!shell) {
      ierr = MatDuplicate(T,MAT_COPY_VALUES,&D[k]);CHKERRQ(ierr);
    } else {
      ierr = NLEIGSMatToMatShellArray(T,&D[k]);CHKERRQ(ierr);
    }
    for (j=0;j<k;j++) {
      ierr = MatAXPY(D[k],-b[j],D[j],nep->mstr);CHKERRQ(ierr);
    }
    ierr = MatScale(D[k],1.0/b[k]);CHKERRQ(ierr);
    ierr = MatHasOperation(D[k],MATOP_NORM,&has);CHKERRQ(ierr);
    if (has) {
      ierr = MatNorm(D[k],NORM_FROBENIUS,&norm);CHKERRQ(ierr);
    } else {
      if(!vec) {
        ierr = MatCreateVecs(D[k],NULL,&w[0]);CHKERRQ(ierr);
        ierr = VecDuplicate(w[0],&w[1]);CHKERRQ(ierr);
        vec = PETSC_TRUE;
      }
      ierr = NEPNLEIGSNormEstimation(nep,D[k],&norm,w);CHKERRQ(ierr);
    }
    if (norm/norm0 < ctx->ddtol) {
      ctx->nmat = k+1; /* increment (the last matrix is not used) */
      ierr = MatDestroy(&D[k]);CHKERRQ(ierr);
      break;
    } 
  }
  if (!ctx->ksp) { ierr = NEPNLEIGSGetKSPs(nep,&ctx->ksp);CHKERRQ(ierr); }
  for (i=0;i<ctx->nshiftsw;i++) {
    ierr = NEPNLEIGSEvalNRTFunct(nep,ctx->nmat,ctx->shifts[i],coeffs);CHKERRQ(ierr);
    ierr = MatDuplicate(ctx->D[0],MAT_COPY_VALUES,&T);CHKERRQ(ierr);
    if (coeffs[0]!=1.0) { ierr = MatScale(T,coeffs[0]);CHKERRQ(ierr); }
    for (j=1;j<ctx->nmat-1;j++) {
      ierr = MatAXPY(T,coeffs[j],ctx->D[j],nep->mstr);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(ctx->ksp[i],T,T);CHKERRQ(ierr);
    ierr = KSPSetUp(ctx->ksp[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&T);CHKERRQ(ierr);
  }
  ierr = PetscFree2(b,coeffs);CHKERRQ(ierr);
  if (vec) {
    ierr = VecDestroy(&w[0]);CHKERRQ(ierr);  
    ierr = VecDestroy(&w[1]);CHKERRQ(ierr);  
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSRitzVector"
static PetscErrorCode NEPNLEIGSRitzVector(NEP nep,PetscScalar *S,PetscInt ld,PetscInt nq,PetscScalar *H,PetscInt k,Vec t)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       deg=ctx->nmat-1,ldds,n;
  PetscBLASInt   nq_,lds_,n_,one=1,ldds_;
  PetscScalar    sone=1.0,zero=0.0,*x,*y,*X;

  PetscFunctionBegin;
  ierr = DSGetDimensions(nep->ds,&n,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(n+nq,&y);CHKERRQ(ierr);
  x = y+nq;
  ierr = DSGetLeadingDimension(nep->ds,&ldds);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nq,&nq_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldds,&ldds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(deg*ld,&lds_);CHKERRQ(ierr);
  ierr = DSGetArray(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
  if (ctx->nshifts) PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,H,&ldds_,X+k*ldds,&one,&zero,x,&one));
  else x = X+k*ldds;
  PetscStackCall("BLASgemv",BLASgemv_("N",&nq_,&n_,&sone,S,&lds_,x,&one,&zero,y,&one));
  ierr = DSRestoreArray(nep->ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(nep->V,0,nq);CHKERRQ(ierr);
  ierr = BVMultVec(nep->V,1.0,0.0,t,y);CHKERRQ(ierr);
  ierr = VecNormalize(t,NULL);CHKERRQ(ierr);
  ierr = PetscFree(y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSKrylovConvergence"
/*
   NEPKrylovConvergence - This is the analogue to EPSKrylovConvergence.
*/
static PetscErrorCode NEPNLEIGSKrylovConvergence(NEP nep,PetscScalar *S,PetscInt ld,PetscInt nq,PetscScalar *H,PetscBool getall,PetscInt kini,PetscInt nits,PetscScalar betak,PetscReal betah,PetscInt *kout,Vec *w)
{
  PetscErrorCode ierr;
  PetscInt       k,newk,marker,inside;
  PetscScalar    re,im;
  PetscReal      resnorm,tt;
  PetscBool      istrivial;
  Vec            t;
  NEP_NLEIGS     *ctx = (NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  t = w[0];
  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  marker = -1;
  if (nep->trackall) getall = PETSC_TRUE;
  for (k=kini;k<kini+nits;k++) {
    /* eigenvalue */
    re = nep->eigr[k];
    im = nep->eigi[k];
    if (!istrivial) {
      if (!ctx->nshifts) {
        ierr = NEPNLEIGSBackTransform((PetscObject)nep,1,&re,&im);CHKERRQ(ierr);
      }
      ierr = RGCheckInside(nep->rg,1,&re,&im,&inside);CHKERRQ(ierr);
      if (marker==-1 && inside<0) marker = k;
    }
    newk = k;
    ierr = DSVectors(nep->ds,DS_MAT_X,&newk,&resnorm);CHKERRQ(ierr);
    tt = (ctx->nshifts)?SlepcAbsEigenvalue(betak-nep->eigr[k]*betah,nep->eigi[k]*betah):betah;
    resnorm *=  PetscAbsReal(tt);
    /* error estimate */
    ierr = (*nep->converged)(nep,nep->eigr[k],nep->eigi[k],resnorm,&nep->errest[k],nep->convergedctx);CHKERRQ(ierr);
    if (ctx->trueres && (nep->errest[k] < nep->tol) ) {
      /* check explicit residual */
      ierr = NEPNLEIGSRitzVector(nep,S,ld,nq,H,k,t);CHKERRQ(ierr);
      ierr = NEPComputeResidualNorm_Private(nep,re,t,w+1,&resnorm);CHKERRQ(ierr);
      ierr = (*nep->converged)(nep,re,im,resnorm,&nep->errest[k],nep->convergedctx);CHKERRQ(ierr);
    }
    if (marker==-1 && nep->errest[k] >= nep->tol) marker = k;
    if (newk==k+1) {
      nep->errest[k+1] = nep->errest[k];
      k++;
    }
    if (marker!=-1 && !getall) break;
  }
  if (marker!=-1) k = marker;
  *kout = k;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetUp_NLEIGS"
PetscErrorCode NEPSetUp_NLEIGS(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       k,in;
  PetscScalar    zero=0.0;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  SlepcSC        sc;
  PetscBool      istrivial;

  PetscFunctionBegin;
  ierr = NEPSetDimensions_Default(nep,nep->nev,&nep->ncv,&nep->mpd);CHKERRQ(ierr);
  if (nep->ncv>nep->nev+nep->mpd) SETERRQ(PetscObjectComm((PetscObject)nep),1,"The value of ncv must not be larger than nev+mpd");
  if (!nep->max_it) nep->max_it = PetscMax(5000,2*nep->n/nep->ncv);
  if (!ctx->ddmaxit) ctx->ddmaxit = MAX_LBPOINTS;
  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (istrivial) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"NEPNLEIGS requires a nontrivial region defining the target set");
  ierr = RGCheckInside(nep->rg,1,&nep->target,&zero,&in);CHKERRQ(ierr);
  if (in<0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"The target is not inside the target set");
  if (!nep->which) nep->which = NEP_TARGET_MAGNITUDE;

  /* Initialize the NLEIGS context structure */
  k = ctx->ddmaxit;
  ierr = PetscMalloc4(k,&ctx->s,k,&ctx->xi,k,&ctx->beta,k,&ctx->D);CHKERRQ(ierr);
  nep->data = ctx;
  if (nep->tol==PETSC_DEFAULT) nep->tol = SLEPC_DEFAULT_TOL;
  if (ctx->ddtol==PETSC_DEFAULT) ctx->ddtol = nep->tol/10.0;
  if (!ctx->keep) ctx->keep = 0.5;

  /* Compute Leja-Bagby points and scaling values */
  ierr = NEPNLEIGSLejaBagbyPoints(nep);CHKERRQ(ierr);

  /* Compute the divided difference matrices */
  if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
    ierr = NEPNLEIGSDividedDifferences_split(nep);CHKERRQ(ierr);
  } else {
    ierr = NEPNLEIGSDividedDifferences_callback(nep);CHKERRQ(ierr);
  }
  ierr = NEPAllocateSolution(nep,ctx->nmat);CHKERRQ(ierr);
  ierr = NEPSetWorkVecs(nep,4);CHKERRQ(ierr);

  /* set-up DS and transfer split operator functions */
  ierr = DSSetType(nep->ds,ctx->nshifts?DSGNHEP:DSNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(nep->ds,nep->ncv+1);CHKERRQ(ierr);
  ierr = DSGetSlepcSC(nep->ds,&sc);CHKERRQ(ierr);
  if (!ctx->nshifts) {
    sc->map = NEPNLEIGSBackTransform;
    ierr = DSSetExtraRow(nep->ds,PETSC_TRUE);CHKERRQ(ierr);
  }
  sc->mapobj        = (PetscObject)nep;
  sc->rg            = nep->rg;
  sc->comparison    = nep->sc->comparison;
  sc->comparisonctx = nep->sc->comparisonctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPTOARSNorm2"
/*
  Norm of [sp;sq] 
*/
static PetscErrorCode NEPTOARSNorm2(PetscInt n,PetscScalar *S,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscBLASInt   n_,one=1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  *norm = BLASnrm2_(&n_,S,&one);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPTOAROrth2"
/*
 Computes GS orthogonalization   [z;x] - [Sp;Sq]*y,
 where y = ([Sp;Sq]'*[z;x]).
   k: Column from S to be orthogonalized against previous columns.
   Sq = Sp+ld
   dim(work)=k;
*/
static PetscErrorCode NEPTOAROrth2(NEP nep,PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt k,PetscScalar *y,PetscReal *norm,PetscBool *lindep,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscBLASInt   n_,lds_,k_,one=1;
  PetscScalar    sonem=-1.0,sone=1.0,szero=0.0,*x0,*x,*c;
  PetscInt       i,lds=deg*ld,n;
  PetscReal      eta,onorm;
  
  PetscFunctionBegin;
  ierr = BVGetOrthogonalization(nep->V,NULL,NULL,&eta,NULL);CHKERRQ(ierr);
  n = k+deg-1;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(deg*ld,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr); /* Number of vectors to orthogonalize against them */
  c = work;
  x0 = S+k*lds;
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S,&lds_,x0,&one,&szero,y,&one));
  for (i=1;i<deg;i++) {
    x = S+i*ld+k*lds;
    PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+i*ld,&lds_,x,&one,&sone,y,&one));
  }
  for (i=0;i<deg;i++) {
    x= S+i*ld+k*lds;
    PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+i*ld,&lds_,y,&one,&sone,x,&one));
  }
  ierr = NEPTOARSNorm2(lds,S+k*lds,&onorm);CHKERRQ(ierr);
  /* twice */
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S,&lds_,x0,&one,&szero,c,&one));
  for (i=1;i<deg;i++) {
    x = S+i*ld+k*lds;
    PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+i*ld,&lds_,x,&one,&sone,c,&one));
  }
  for (i=0;i<deg;i++) {
    x= S+i*ld+k*lds;
    PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+i*ld,&lds_,c,&one,&sone,x,&one));
  }
  for (i=0;i<k;i++) y[i] += c[i];
  if (norm) {
    ierr = NEPTOARSNorm2(lds,S+k*lds,norm);CHKERRQ(ierr);
    if (lindep) *lindep = (*norm < eta * onorm)?PETSC_TRUE:PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPTOARExtendBasis"
/*
  Extend the TOAR basis by applying the the matrix operator
  over a vector which is decomposed on the TOAR way
  Input:
    - S,V: define the latest Arnoldi vector (nv vectors in V)
  Output:
    - t: new vector extending the TOAR basis
    - r: temporally coefficients to compute the TOAR coefficients
         for the new Arnoldi vector
  Workspace: t_ (two vectors)
*/
static PetscErrorCode NEPTOARExtendBasis(NEP nep,PetscInt idxrktg,PetscScalar *S,PetscInt ls,PetscInt nv,BV V,Vec t,PetscScalar *r,PetscInt lr,Vec *t_)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       deg=ctx->nmat-1,k,j;
  Vec            v=t_[0],q=t_[1],w;
  PetscScalar    *beta=ctx->beta,*s=ctx->s,*xi=ctx->xi,*coeffs,sigma;

  PetscFunctionBegin;
  if (!ctx->ksp) { ierr = NEPNLEIGSGetKSPs(nep,&ctx->ksp);CHKERRQ(ierr); }
  sigma = ctx->shifts[idxrktg];
  ierr = BVSetActiveColumns(nep->V,0,nv);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->nmat-1,&coeffs);CHKERRQ(ierr);
  if (PetscAbsScalar(s[deg-2]-sigma)<100*PETSC_MACHINE_EPSILON) SETERRQ(PETSC_COMM_SELF,1,"Breakdown in NLEIGS");
  /* i-part stored in (i-1) position */
  for (j=0;j<nv;j++) {
    r[(deg-2)*lr+j] = (S[(deg-2)*ls+j]+(beta[deg-1]/xi[deg-2])*S[(deg-1)*ls+j])/(s[deg-2]-sigma);
  }
  ierr = BVSetActiveColumns(ctx->W,0,deg-1);CHKERRQ(ierr);
  ierr = BVGetColumn(ctx->W,deg-2,&w);CHKERRQ(ierr);
  ierr = BVMultVec(V,1.0,0.0,w,r+(deg-2)*lr);CHKERRQ(ierr);
  ierr = BVRestoreColumn(ctx->W,deg-2,&w);CHKERRQ(ierr);
  for (k=deg-2;k>0;k--) {
    if (PetscAbsScalar(s[k-1]-sigma)<100*PETSC_MACHINE_EPSILON) SETERRQ(PETSC_COMM_SELF,1,"Breakdown in NLEIGS");
    for (j=0;j<nv;j++) r[(k-1)*lr+j] = (S[(k-1)*ls+j]+(beta[k]/xi[k-1])*S[k*ls+j]-beta[k]*(1.0-sigma/xi[k-1])*r[(k)*lr+j])/(s[k-1]-sigma);
    ierr = BVGetColumn(ctx->W,k-1,&w);CHKERRQ(ierr);
    ierr = BVMultVec(V,1.0,0.0,w,r+(k-1)*lr);CHKERRQ(ierr);
    ierr = BVRestoreColumn(ctx->W,k-1,&w);CHKERRQ(ierr);
  }
  if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
    for (j=0;j<ctx->nmat-1;j++) coeffs[j] = ctx->coeffD[nep->nt*j];
    ierr = BVMultVec(ctx->W,1.0,0.0,v,coeffs);CHKERRQ(ierr);
    ierr = MatMult(nep->A[0],v,q);CHKERRQ(ierr);
    for (k=1;k<nep->nt;k++) {
      for (j=0;j<ctx->nmat-1;j++) coeffs[j] = ctx->coeffD[nep->nt*j+k];
      ierr = BVMultVec(ctx->W,1.0,0,v,coeffs);CHKERRQ(ierr);
      ierr = MatMult(nep->A[k],v,t);CHKERRQ(ierr);
      ierr = VecAXPY(q,1.0,t);CHKERRQ(ierr);
    }
    ierr = KSPSolve(ctx->ksp[idxrktg],q,t);CHKERRQ(ierr);
    ierr = VecScale(t,-1.0);CHKERRQ(ierr);
  } else {
    for (k=0;k<deg-1;k++) {
      ierr = BVGetColumn(ctx->W,k,&w);CHKERRQ(ierr);
      ierr = MatMult(ctx->D[k],w,q);CHKERRQ(ierr);
      ierr = BVRestoreColumn(ctx->W,k,&w);CHKERRQ(ierr);
      ierr = BVInsertVec(ctx->W,k,q);CHKERRQ(ierr);
    }
    for (j=0;j<ctx->nmat-1;j++) coeffs[j] = 1.0;
    ierr = BVMultVec(ctx->W,1.0,0.0,q,coeffs);CHKERRQ(ierr);
    ierr = KSPSolve(ctx->ksp[idxrktg],q,t);CHKERRQ(ierr);
    ierr = VecScale(t,-1.0);CHKERRQ(ierr);
  }
  ierr = PetscFree(coeffs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPTOARCoefficients"
/*
  Compute TOAR coefficients of the blocks of the new Arnoldi vector computed
*/
static PetscErrorCode NEPTOARCoefficients(NEP nep,PetscScalar sigma,PetscInt nv,PetscScalar *S,PetscInt ls,PetscScalar *r,PetscInt lr,PetscScalar *x,PetscScalar *work)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       k,j,d=ctx->nmat-1;
  PetscScalar    *t=work;

  PetscFunctionBegin;
  ierr = NEPNLEIGSEvalNRTFunct(nep,d-1,sigma,t);CHKERRQ(ierr);
  for (k=0;k<d-1;k++) {
    for (j=0;j<=nv;j++) r[k*lr+j] += t[k]*x[j];
  }
  for (j=0;j<=nv;j++) r[(d-1)*lr+j] = t[d-1]*x[j];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGS_RKcontinuation"
/*
  Compute continuation vector coefficients for the Rational-Krylov run.
  dim(work) >= (end-ini)*(end-ini+1) + end+1 + 2*(end-ini+1), dim(t) = end. 
*/
static PetscErrorCode NEPNLEIGS_RKcontinuation(NEP nep,PetscInt ini,PetscInt end,PetscScalar *K,PetscScalar *H,PetscInt ld,PetscScalar sigma,PetscScalar *S,PetscInt lds,PetscScalar *cont,PetscScalar *t,PetscScalar *work)
{
#if defined(PETSC_MISSING_LAPACK_GEQRF) || defined(SLEPC_MISSING_LAPACK_LARF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GEQRF/LARF - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscScalar    *x,*W,*tau,sone=1.0,szero=0.0;
  PetscInt       i,j,n1,n,nwu=0;
  PetscBLASInt   info,n_,n1_,one=1,dim,lds_;
  NEP_NLEIGS     *ctx = (NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  if (!ctx->nshifts || !end) {
    t[0] = 1;
    ierr = PetscMemcpy(cont,S+end*lds,lds*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    n   = end-ini;
    n1  = n+1;
    x   = work+nwu;
    nwu += end+1;
    tau = work+nwu;
    nwu += n;
    W   = work+nwu;
    nwu += n1*n;
    for (j=ini;j<end;j++) {
      for (i=ini;i<=end;i++) W[(j-ini)*n1+i-ini] = K[j*ld+i] -H[j*ld+i]*sigma;
    }
    ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(n1,&n1_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(end+1,&dim);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&n1_,&n_,W,&n1_,tau,work+nwu,&n1_,&info));
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGEQRF %d",info);
    for (i=0;i<end;i++) t[i] = 0.0;
    t[end] = 1.0; 
    for (j=n-1;j>=0;j--) {
      for (i=0;i<ini+j;i++) x[i] = 0.0;
      x[ini+j] = 1.0;
      for (i=j+1;i<n1;i++) x[i+ini] = W[i+n1*j]; 
      tau[j] = PetscConj(tau[j]);
      PetscStackCallBLAS("LAPACKlarf",LAPACKlarf_("L",&dim,&one,x,&one,tau+j,t,&dim,work+nwu));
    }
    ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&lds_,&n1_,&sone,S,&lds_,t,&one,&szero,cont,&one));
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSTOARrun"
/*
  Compute a run of Arnoldi iterations
*/
static PetscErrorCode NEPNLEIGSTOARrun(NEP nep,PetscInt *nq,PetscScalar *S,PetscInt ld,PetscScalar *K,PetscScalar *H,PetscInt ldh,BV V,PetscInt k,PetscInt *M,PetscBool *breakdown,Vec *t_)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx = (NEP_NLEIGS*)nep->data;
  PetscInt       i,j,p,m=*M,lwa,deg=ctx->nmat-1,lds=ld*deg,nqt=*nq;
  Vec            t=t_[0];
  PetscReal      norm;
  PetscScalar    *x,*work,*tt,sigma,*cont;
  PetscBool      lindep;

  PetscFunctionBegin;
  lwa = PetscMax(ld,deg)+(m+1)*(m+1)+4*(m+1);
  ierr = PetscMalloc4(ld,&x,lwa,&work,m+1,&tt,lds,&cont);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    sigma = ctx->shifts[(++(ctx->idxrk))%ctx->nshiftsw];

    /* Continuation vector */
    ierr = NEPNLEIGS_RKcontinuation(nep,0,j,K,H,ldh,sigma,S,lds,cont,tt,work);CHKERRQ(ierr);
    
    /* apply operator */
    ierr = BVGetColumn(nep->V,nqt,&t);CHKERRQ(ierr);
    ierr = NEPTOARExtendBasis(nep,(ctx->idxrk)%ctx->nshiftsw,cont,ld,nqt,V,t,S+(j+1)*lds,ld,t_+1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(nep->V,nqt,&t);CHKERRQ(ierr);

    /* orthogonalize */
    ierr = BVOrthogonalizeColumn(nep->V,nqt,x,&norm,&lindep);CHKERRQ(ierr);
    if (!lindep) {
      x[nqt] = norm;
      ierr = BVScaleColumn(nep->V,nqt,1.0/norm);CHKERRQ(ierr);
      nqt++;
    } else x[nqt] = 0.0;

    ierr = NEPTOARCoefficients(nep,sigma,*nq,cont,ld,S+(j+1)*lds,ld,x,work);CHKERRQ(ierr);

    /* Level-2 orthogonalization */
    ierr = NEPTOAROrth2(nep,S,ld,deg,j+1,H+j*ldh,&norm,breakdown,work);CHKERRQ(ierr);
    H[j+1+ldh*j] = norm;
    if (ctx->nshifts) {
      for (i=0;i<=j;i++) K[i+ldh*j] = sigma*H[i+ldh*j] + tt[i];
      K[j+1+ldh*j] = sigma*H[j+1+ldh*j];
    }
    *nq = nqt;
    if (*breakdown) {
      *M = j+1;
      break;
    }
    for (p=0;p<deg;p++) {
      for (i=0;i<=j+deg;i++) {
        S[i+p*ld+(j+1)*lds] /= norm;
      }
    }
  } 
  ierr = PetscFree4(x,work,tt,cont);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPTOARTrunc"
/* dim(work)=5*ld*lds dim(rwork)=6*n */
static PetscErrorCode NEPTOARTrunc(NEP nep,PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt *nq,PetscInt cs1,PetscScalar *work,PetscReal *rwork)
{
  PetscErrorCode ierr;
  PetscInt       lwa,nwu=0,nrwu=0;
  PetscInt       j,i,n,lds=deg*ld,rk=0,rs1;
  PetscScalar    *M,*V,*pU,t;
  PetscReal      *sg,tol;
  PetscBLASInt   cs1_,rs1_,cs1tdeg,n_,info,lw_;
  Mat            U;

  PetscFunctionBegin;
  rs1 = *nq;
  n = (rs1>deg*cs1)?deg*cs1:rs1;
  lwa = 5*ld*lds;
  M = work+nwu;
  nwu += rs1*cs1*deg;
  sg = rwork+nrwu;
  nrwu += n;
  pU = work+nwu;
  nwu += rs1*n;
  V = work+nwu;
  nwu += deg*cs1*n;
  for (i=0;i<cs1;i++) {
    for (j=0;j<deg;j++) {
      ierr = PetscMemcpy(M+(i+j*cs1)*rs1,S+i*lds+j*ld,rs1*sizeof(PetscScalar));CHKERRQ(ierr);
    } 
  }
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1,&cs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(rs1,&rs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1*deg,&cs1tdeg);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lwa-nwu,&lw_);CHKERRQ(ierr);
#if !defined (PETSC_USE_COMPLEX)
  PetscStackCall("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1tdeg,M,&rs1_,sg,pU,&rs1_,V,&n_,work+nwu,&lw_,&info));
#else
  PetscStackCall("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1tdeg,M,&rs1_,sg,pU,&rs1_,V,&n_,work+nwu,&lw_,rwork+nrwu,&info));  
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);
  
  /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,rs1,cs1+deg-1,pU,&U);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(nep->V,0,rs1);CHKERRQ(ierr);
  ierr = BVMultInPlace(nep->V,U,0,cs1+deg-1);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(nep->V,0,cs1+deg-1);CHKERRQ(ierr);
  ierr = MatDestroy(&U);CHKERRQ(ierr);  
  tol = PetscMax(rs1,deg*cs1)*PETSC_MACHINE_EPSILON*sg[0];
  for (i=0;i<PetscMin(n_,cs1tdeg);i++) if (sg[i]>tol) rk++;
  rk = PetscMin(cs1+deg-1,rk);
  
  /* Update S */
  ierr = PetscMemzero(S,lds*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<rk;i++) {
    t = sg[i];
    PetscStackCall("BLASscal",BLASscal_(&cs1tdeg,&t,V+i,&n_));
  }
  for (j=0;j<cs1;j++) {
    for (i=0;i<deg;i++) {
      ierr = PetscMemcpy(S+j*lds+i*ld,V+(cs1*i+j)*n,rk*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  *nq = rk;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPTOARSupdate"
/*
  S <- S*Q 
  columns s-s+ncu of S
  rows 0-sr of S
  size(Q) qr x ncu
  dim(work)=sr*ncu
*/
static PetscErrorCode NEPTOARSupdate(PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt sr,PetscInt s,PetscInt ncu,PetscInt qr,PetscScalar *Q,PetscInt ldq,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscScalar    a=1.0,b=0.0;
  PetscBLASInt   sr_,ncu_,ldq_,lds_,qr_;
  PetscInt       j,lds=deg*ld,i;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(sr,&sr_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(qr,&qr_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ncu,&ncu_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldq,&ldq_);CHKERRQ(ierr);
  for (i=0;i<deg;i++) {
    PetscStackCall("BLASgemm",BLASgemm_("N","N",&sr_,&ncu_,&qr_,&a,S+i*ld,&lds_,Q,&ldq_,&b,work,&sr_));
    for (j=0;j<ncu;j++) {
      ierr = PetscMemcpy(S+lds*(s+j)+i*ld,work+j*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSolve_NLEIGS"
PetscErrorCode NEPSolve_NLEIGS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx = (NEP_NLEIGS*)nep->data;
  PetscInt       i,j,k=0,l,nv=0,ld,lds,off,ldds,rs1,nq=0,newn;
  PetscInt       lwa,lrwa,nwu=0,nrwu=0,deg=ctx->nmat-1,nconv=0;
  PetscScalar    *S,*Q,*work,*H,*pU,*K,betak=0,*Hc,*eigr,*eigi;
  PetscReal      betah,norm,*rwork;
  PetscBool      breakdown=PETSC_FALSE,lindep;
  Mat            U;

  PetscFunctionBegin;
  ld = nep->ncv+deg;
  lds = deg*ld;
  lwa = (deg+6)*ld*lds;
  lrwa = 7*lds;
  ierr = DSGetLeadingDimension(nep->ds,&ldds);CHKERRQ(ierr);
  ierr = PetscMalloc4(lwa,&work,lrwa,&rwork,lds*ld,&S,ldds*ldds,&Hc);CHKERRQ(ierr);
  ierr = PetscMemzero(S,lds*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  if (!ctx->nshifts) {
    ierr = PetscMalloc2(nep->ncv,&eigr,nep->ncv,&eigi);CHKERRQ(ierr);
  } else { eigr = nep->eigr; eigi = nep->eigi; }
  ierr = BVDuplicateResize(nep->V,PetscMax(nep->nt-1,ctx->nmat-1),&ctx->W);CHKERRQ(ierr);

  /* Get the starting vector */
  for (i=0;i<deg;i++) {
    ierr = BVSetRandomColumn(nep->V,i);CHKERRQ(ierr);
    ierr = BVOrthogonalizeColumn(nep->V,i,S+i*ld,&norm,&lindep);CHKERRQ(ierr);
    if (!lindep) {
      ierr = BVScaleColumn(nep->V,i,1/norm);CHKERRQ(ierr);
      S[i+i*ld] = norm;
      nq++;
    }
  }
  if (!nq) SETERRQ(PetscObjectComm((PetscObject)nep),1,"NEP: Problem with initial vector");
  ierr = NEPTOARSNorm2(lds,S,&norm);CHKERRQ(ierr);
  for (j=0;j<deg;j++) {
    for (i=0;i<=j;i++) S[i+j*ld] /= norm;
  }

  /* Restart loop */
  l = 0;
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;
    
    /* Compute an nv-step Krylov relation */
    nv = PetscMin(nep->nconv+nep->mpd,nep->ncv);
    if (ctx->nshifts) { ierr = DSGetArray(nep->ds,DS_MAT_A,&K);CHKERRQ(ierr); }
    ierr = DSGetArray(nep->ds,ctx->nshifts?DS_MAT_B:DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = NEPNLEIGSTOARrun(nep,&nq,S,ld,K,H,ldds,nep->V,nep->nconv+l,&nv,&breakdown,nep->work);CHKERRQ(ierr);
    betah = PetscAbsScalar(H[(nv-1)*ldds+nv]);
    ierr = DSRestoreArray(nep->ds,ctx->nshifts?DS_MAT_B:DS_MAT_A,&H);CHKERRQ(ierr);
    if (ctx->nshifts) {
      betak = K[(nv-1)*ldds+nv];
      ierr = DSRestoreArray(nep->ds,DS_MAT_A,&K);CHKERRQ(ierr);
    }
    ierr = DSSetDimensions(nep->ds,nv,0,nep->nconv,nep->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(nep->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(nep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Solve projected problem */
    if (ctx->nshifts) {
      ierr = DSGetArray(nep->ds,DS_MAT_B,&H);CHKERRQ(ierr);
      ierr = PetscMemcpy(Hc,H,ldds*ldds*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = DSRestoreArray(nep->ds,DS_MAT_B,&H);CHKERRQ(ierr);
    }
    ierr = DSSolve(nep->ds,nep->eigr,nep->eigi);CHKERRQ(ierr);
    ierr = DSSort(nep->ds,nep->eigr,nep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);
    if (!ctx->nshifts) {
      ierr = DSUpdateExtraRow(nep->ds);CHKERRQ(ierr);
    }

    /* Check convergence */
    ierr = NEPNLEIGSKrylovConvergence(nep,S,ld,nq,Hc,PETSC_FALSE,nep->nconv,nv-nep->nconv,betak,betah,&k,nep->work);CHKERRQ(ierr);
    ierr = (*nep->stopping)(nep,nep->its,nep->max_it,k,nep->nev,&nep->reason,nep->stoppingctx);CHKERRQ(ierr);
    nconv = k;

    /* Update l */
    if (nep->reason != NEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
      if (!breakdown) {
        /* Prepare the Rayleigh quotient for restart */
        if (!ctx->nshifts) {
          ierr = DSTruncate(nep->ds,k+l);CHKERRQ(ierr);
          ierr = DSGetDimensions(nep->ds,&newn,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
          l = newn-k;
        } else {
          ierr = DSGetArray(nep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
          ierr = DSGetArray(nep->ds,DS_MAT_B,&H);CHKERRQ(ierr);
          ierr = DSGetArray(nep->ds,DS_MAT_A,&K);CHKERRQ(ierr);
          for (i=ctx->lock?k:0;i<k+l;i++) {
            H[k+l+i*ldds] = betah*Q[nv-1+i*ldds];
            K[k+l+i*ldds] = betak*Q[nv-1+i*ldds];
          }
          ierr = DSRestoreArray(nep->ds,DS_MAT_B,&H);CHKERRQ(ierr);
          ierr = DSRestoreArray(nep->ds,DS_MAT_A,&K);CHKERRQ(ierr);
          ierr = DSRestoreArray(nep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
          ierr = DSSetDimensions(nep->ds,k+l,0,nep->nconv,0);CHKERRQ(ierr);
        }
      }
    }
    if (!ctx->lock && l>0) { l += k; k = 0; }

    /* Update S */
    off = nep->nconv*ldds;
    ierr = DSGetArray(nep->ds,ctx->nshifts?DS_MAT_Z:DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = NEPTOARSupdate(S,ld,deg,nq,nep->nconv,k+l-nep->nconv,nv,Q+off,ldds,work+nwu);CHKERRQ(ierr);
    ierr = DSRestoreArray(nep->ds,ctx->nshifts?DS_MAT_Z:DS_MAT_Q,&Q);CHKERRQ(ierr);

    /* Copy last column of S */
    ierr = PetscMemcpy(S+lds*(k+l),S+lds*nv,lds*sizeof(PetscScalar));CHKERRQ(ierr);
    if (nep->reason == NEP_CONVERGED_ITERATING) {
      if (breakdown) {

        /* Stop if breakdown */
        ierr = PetscInfo2(nep,"Breakdown (it=%D norm=%g)\n",nep->its,(double)betah);CHKERRQ(ierr);
        nep->reason = NEP_DIVERGED_BREAKDOWN;
      } else {
        /* Truncate S */
        ierr = NEPTOARTrunc(nep,S,ld,deg,&nq,k+l+1,work+nwu,rwork+nrwu);CHKERRQ(ierr);
      }
    }
    nep->nconv = k;
    if (!ctx->nshifts) {
      for (i=0;i<nv;i++) { eigr[i] = nep->eigr[i]; eigi[i] = nep->eigi[i]; }
      ierr = NEPNLEIGSBackTransform((PetscObject)nep,nv,eigr,eigi);CHKERRQ(ierr);
    }
    ierr = NEPMonitor(nep,nep->its,nconv,eigr,eigi,nep->errest,nv);CHKERRQ(ierr);
  }
  nep->nconv = nconv;
  if (nep->nconv>0) {
    /* Extract invariant pair */
    ierr = NEPTOARTrunc(nep,S,ld,deg,&nq,nep->nconv,work+nwu,rwork+nrwu);CHKERRQ(ierr);
    /* Update vectors V = V*S or V=V*S*H */    
    rs1 = nep->nconv;
    if (ctx->nshifts) {
      ierr = DSGetArray(nep->ds,DS_MAT_B,&H);CHKERRQ(ierr);
      ierr = NEPTOARSupdate(S,ld,deg,rs1,0,nep->nconv,nep->nconv,H,ldds,work+nwu);CHKERRQ(ierr);
      ierr = DSRestoreArray(nep->ds,DS_MAT_B,&H);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(rs1*nep->nconv,&pU);CHKERRQ(ierr);
    for (i=0;i<nep->nconv;i++) {
      ierr = PetscMemcpy(pU+i*rs1,S+i*lds,rs1*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,rs1,nep->nconv,pU,&U);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(nep->V,0,rs1);CHKERRQ(ierr);
    ierr = BVMultInPlace(nep->V,U,0,nep->nconv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(nep->V,0,nep->nconv);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);
    ierr = PetscFree(pU);CHKERRQ(ierr);
  }
  /* truncate Schur decomposition and change the state to raw so that
     DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(nep->ds,nep->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(nep->ds,DS_STATE_RAW);CHKERRQ(ierr);

  ierr = PetscFree4(work,rwork,S,Hc);CHKERRQ(ierr);
  /* Map eigenvalues back to the original problem */
  if (!ctx->nshifts) {
    ierr = NEPNLEIGSBackTransform((PetscObject)nep,nep->nconv,nep->eigr,nep->eigi);CHKERRQ(ierr);
    ierr = PetscFree2(eigr,eigi);CHKERRQ(ierr);
  }
  ierr = BVDestroy(&ctx->W);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetSingularitiesFunction_NLEIGS"
static PetscErrorCode NEPNLEIGSSetSingularitiesFunction_NLEIGS(NEP nep,PetscErrorCode (*fun)(NEP,PetscInt*,PetscScalar*,void*),void *ctx)
{
  NEP_NLEIGS *nepctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  if (fun) nepctx->computesingularities = fun;
  if (ctx) nepctx->singularitiesctx     = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetSingularitiesFunction"
/*@C
   NEPNLEIGSSetSingularitiesFunction - Sets a user function to compute a discretization
   of the singularity set (where T(.) is not analytic).

   Logically Collective on NEP

   Input Parameters:
+  nep - the NEP context
.  fun - user function (if NULL then NEP retains any previously set value)
-  ctx - [optional] user-defined context for private data for the function
         (may be NULL, in which case NEP retains any previously set value)

   Calling Sequence of fun:
$   fun(NEP nep,PetscInt *maxnp,PetscScalar *xi,void *ctx)

+   nep   - the NEP context
.   maxnp - on input number of requested points in the discretization (can be set)
.   xi    - computed values of the discretization
-   ctx   - optional context, as set by NEPNLEIGSSetSingularitiesFunction()

   Note:
   The user-defined function can set a smaller value of maxnp if necessary.

   Level: intermediate

.seealso: NEPNLEIGSGetSingularitiesFunction()
@*/
PetscErrorCode NEPNLEIGSSetSingularitiesFunction(NEP nep,PetscErrorCode (*fun)(NEP,PetscInt*,PetscScalar*,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscTryMethod(nep,"NEPNLEIGSSetSingularitiesFunction_C",(NEP,PetscErrorCode(*)(NEP,PetscInt*,PetscScalar*,void*),void*),(nep,fun,ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetSingularitiesFunction_NLEIGS"
static PetscErrorCode NEPNLEIGSGetSingularitiesFunction_NLEIGS(NEP nep,PetscErrorCode (**fun)(NEP,PetscInt*,PetscScalar*,void*),void **ctx)
{
  NEP_NLEIGS *nepctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  if (fun) *fun = nepctx->computesingularities;
  if (ctx) *ctx = nepctx->singularitiesctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetSingularitiesFunction"
/*@C
   NEPNLEIGSGetSingularitiesFunction - Returns the Function and optionally the user
   provided context for computing a discretization of the singularity set.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  fun - location to put the function (or NULL)
-  ctx - location to stash the function context (or NULL)

   Level: advanced

.seealso: NEPNLEIGSSetSingularitiesFunction()
@*/
PetscErrorCode NEPNLEIGSGetSingularitiesFunction(NEP nep,PetscErrorCode (**fun)(NEP,PetscInt*,PetscScalar*,void*),void **ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscUseMethod(nep,"NEPNLEIGSGetSingularitiesFunction_C",(NEP,PetscErrorCode(**)(NEP,PetscInt*,PetscScalar*,void*),void**),(nep,fun,ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetRestart_NLEIGS"
static PetscErrorCode NEPNLEIGSSetRestart_NLEIGS(NEP nep,PetscReal keep)
{
  NEP_NLEIGS *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) ctx->keep = 0.5;
  else {
    if (keep<0.1 || keep>0.9) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument must be in the range [0.1,0.9]");
    ctx->keep = keep;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetRestart"
/*@
   NEPNLEIGSSetRestart - Sets the restart parameter for the NLEIGS
   method, in particular the proportion of basis vectors that must be kept
   after restart.

   Logically Collective on NEP

   Input Parameters:
+  nep  - the nonlinear eigensolver context
-  keep - the number of vectors to be kept at restart

   Options Database Key:
.  -nep_nleigs_restart - Sets the restart parameter

   Notes:
   Allowed values are in the range [0.1,0.9]. The default is 0.5.

   Level: advanced

.seealso: NEPNLEIGSGetRestart()
@*/
PetscErrorCode NEPNLEIGSSetRestart(NEP nep,PetscReal keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(nep,keep,2);
  ierr = PetscTryMethod(nep,"NEPNLEIGSSetRestart_C",(NEP,PetscReal),(nep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetRestart_NLEIGS"
static PetscErrorCode NEPNLEIGSGetRestart_NLEIGS(NEP nep,PetscReal *keep)
{
  NEP_NLEIGS *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  *keep = ctx->keep;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetRestart"
/*@
   NEPNLEIGSGetRestart - Gets the restart parameter used in the NLEIGS method.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameter:
.  keep - the restart parameter

   Level: advanced

.seealso: NEPNLEIGSSetRestart()
@*/
PetscErrorCode NEPNLEIGSGetRestart(NEP nep,PetscReal *keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(keep,2);
  ierr = PetscUseMethod(nep,"NEPNLEIGSGetRestart_C",(NEP,PetscReal*),(nep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetLocking_NLEIGS"
static PetscErrorCode NEPNLEIGSSetLocking_NLEIGS(NEP nep,PetscBool lock)
{
  NEP_NLEIGS *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  ctx->lock = lock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetLocking"
/*@
   NEPNLEIGSSetLocking - Choose between locking and non-locking variants of
   the NLEIGS method.

   Logically Collective on NEP

   Input Parameters:
+  nep  - the nonlinear eigensolver context
-  lock - true if the locking variant must be selected

   Options Database Key:
.  -nep_nleigs_locking - Sets the locking flag

   Notes:
   The default is to lock converged eigenpairs when the method restarts.
   This behaviour can be changed so that all directions are kept in the
   working subspace even if already converged to working accuracy (the
   non-locking variant).

   Level: advanced

.seealso: NEPNLEIGSGetLocking()
@*/
PetscErrorCode NEPNLEIGSSetLocking(NEP nep,PetscBool lock)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(nep,lock,2);
  ierr = PetscTryMethod(nep,"NEPNLEIGSSetLocking_C",(NEP,PetscBool),(nep,lock));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetLocking_NLEIGS"
static PetscErrorCode NEPNLEIGSGetLocking_NLEIGS(NEP nep,PetscBool *lock)
{
  NEP_NLEIGS *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  *lock = ctx->lock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetLocking"
/*@
   NEPNLEIGSGetLocking - Gets the locking flag used in the NLEIGS method.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameter:
.  lock - the locking flag

   Level: advanced

.seealso: NEPNLEIGSSetLocking()
@*/
PetscErrorCode NEPNLEIGSGetLocking(NEP nep,PetscBool *lock)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(lock,2);
  ierr = PetscUseMethod(nep,"NEPNLEIGSGetLocking_C",(NEP,PetscBool*),(nep,lock));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetInterpolation_NLEIGS"
static PetscErrorCode NEPNLEIGSSetInterpolation_NLEIGS(NEP nep,PetscReal tol,PetscInt maxits)
{
  NEP_NLEIGS *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  if (tol == PETSC_DEFAULT) {
    ctx->ddtol = PETSC_DEFAULT;
    nep->state = NEP_STATE_INITIAL;
  } else {
    if (tol <= 0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of tol. Must be > 0");
    ctx->ddtol = tol;
  }
  if (maxits == PETSC_DEFAULT || maxits == PETSC_DECIDE) {
    ctx->ddmaxit = 0;
    nep->state   = NEP_STATE_INITIAL;
  } else {
    if (maxits <= 0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of maxits. Must be > 0");
    ctx->ddmaxit = maxits;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetInterpolation"
/*@
   NEPNLEIGSSetInterpolation - Sets the tolerance and maximum iteration count used
   by the NLEIGS method when building the interpolation via divided differences.

   Logically Collective on NEP

   Input Parameters:
+  nep    - the nonlinear eigensolver context
.  tol    - the convergence tolerance
-  maxits - maximum number of iterations to use

   Options Database Key:
+  -nep_nleigs_interpolation_tol <tol> - Sets the convergence tolerance
-  -nep_nleigs_interpolation_max_it <maxits> - Sets the maximum number of iterations

   Notes:
   Use PETSC_DEFAULT for either argument to assign a reasonably good value.

   Level: advanced

.seealso: NEPNLEIGSGetInterpolation()
@*/
PetscErrorCode NEPNLEIGSSetInterpolation(NEP nep,PetscReal tol,PetscInt maxits)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(nep,tol,2);
  PetscValidLogicalCollectiveInt(nep,maxits,3);
  ierr = PetscTryMethod(nep,"NEPNLEIGSSetInterpolation_C",(NEP,PetscReal,PetscInt),(nep,tol,maxits));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetInterpolation_NLEIGS"
static PetscErrorCode NEPNLEIGSGetInterpolation_NLEIGS(NEP nep,PetscReal *tol,PetscInt *maxits)
{
  NEP_NLEIGS *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  if (tol)    *tol    = ctx->ddtol;
  if (maxits) *maxits = ctx->ddmaxit;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetInterpolation"
/*@
   NEPNLEIGSGetInterpolation - Gets the tolerance and maximum iteration count used
   by the NLEIGS method when building the interpolation via divided differences.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameter:
+  tol    - the convergence tolerance
-  maxits - maximum number of iterations

   Level: advanced

.seealso: NEPNLEIGSSetInterpolation()
@*/
PetscErrorCode NEPNLEIGSGetInterpolation(NEP nep,PetscReal *tol,PetscInt *maxits)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = PetscTryMethod(nep,"NEPNLEIGSGetInterpolation_C",(NEP,PetscReal*,PetscInt*),(nep,tol,maxits));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetTrueResidual_NLEIGS"
static PetscErrorCode NEPNLEIGSSetTrueResidual_NLEIGS(NEP nep,PetscBool trueres)
{
  NEP_NLEIGS *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  ctx->trueres = trueres;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetTrueResidual"
/*@
   NEPNLEIGSSetTrueResidual - Specifies if the solver must compute the true residual
   explicitly or not.

   Logically Collective on NEP

   Input Parameters:
+  nep - the nonlinear eigensolver context
-  trueres - whether true residuals are required or not

   Options Database Key:
.  -nep_nleigs_true_residual <boolean> - Sets/resets the boolean flag 'trueres'

   Notes:
   If the user sets trueres=PETSC_TRUE then the solver explicitly computes
   the true residual norm for each eigenpair approximation, and uses it for
   convergence testing. The default is to use the cheaper approximation 
   available from the (rational) Krylov iteration.

   Level: advanced

.seealso: NEPNLEIGSGetTrueResidual()
@*/
PetscErrorCode NEPNLEIGSSetTrueResidual(NEP nep,PetscBool trueres)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveBool(nep,trueres,2);
  ierr = PetscTryMethod(nep,"NEPNLEIGSSetTrueResidual_C",(NEP,PetscBool),(nep,trueres));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetTrueResidual_NLEIGS"
static PetscErrorCode NEPNLEIGSGetTrueResidual_NLEIGS(NEP nep,PetscBool *trueres)
{
  NEP_NLEIGS *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  *trueres = ctx->trueres;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetTrueResidual"
/*@
   NEPNLEIGSGetTrueResidual - Returns the flag indicating whether true
   residuals must be computed explicitly or not.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameter:
.  trueres - the returned flag

   Level: advanced

.seealso: NEPNLEIGSSetTrueResidual()
@*/
PetscErrorCode NEPNLEIGSGetTrueResidual(NEP nep,PetscBool *trueres)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(trueres,2);
  ierr = PetscTryMethod(nep,"NEPNLEIGSGetTrueResidual_C",(NEP,PetscBool*),(nep,trueres));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetRKShifts_NLEIGS"
static PetscErrorCode NEPNLEIGSSetRKShifts_NLEIGS(NEP nep,PetscInt ns,PetscScalar *shifts)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (ns<=0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_WRONG,"Number of shifts must be positive");
  if (ctx->nshifts) { ierr = PetscFree(ctx->shifts);CHKERRQ(ierr); }
  for (i=0;i<ctx->nshiftsw;i++) { ierr = KSPDestroy(&ctx->ksp[i]);CHKERRQ(ierr); }
  ierr = PetscFree(ctx->ksp);CHKERRQ(ierr);
  ctx->ksp = NULL;
  ierr = PetscMalloc1(ns,&ctx->shifts);CHKERRQ(ierr);
  for (i=0;i<ns;i++) ctx->shifts[i] = shifts[i];
  ctx->nshifts = ns;
  nep->state   = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetRKShifts"
/*@C
   NEPNLEIGSSetRKShifts - Sets a list of shifts to be used in the Rational
   Krylov method.

   Logically Collective on NEP

   Input Parameters:
+  nep    - the nonlinear eigensolver context
.  ns     - number of shifts
-  shifts - array of scalar values specifying the shifts

   Options Database Key:
.  -nep_nleigs_rk_shifts - Sets the list of shifts

   Notes:
   If only one shift is provided, the subspace is built with the simpler
   shift-and-invert Krylov-Schur.

   In the case of real scalars, complex shifts are not allowed. In the
   command line, a comma-separated list of complex values can be provided with
   the format [+/-][realnumber][+/-]realnumberi with no spaces, e.g.
   -nep_nleigs_rk_shifts 1.0+2.0i,1.5+2.0i,1.0+1.5i

   Level: advanced

.seealso: NEPNLEIGSGetRKShifts()
@*/
PetscErrorCode NEPNLEIGSSetRKShifts(NEP nep,PetscInt ns,PetscScalar *shifts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,ns,2);
  PetscValidPointer(nep,shifts);
  ierr = PetscTryMethod(nep,"NEPNLEIGSSetRKShifts_C",(NEP,PetscInt,PetscScalar*),(nep,ns,shifts));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetRKShifts_NLEIGS"
static PetscErrorCode NEPNLEIGSGetRKShifts_NLEIGS(NEP nep,PetscInt *ns,PetscScalar **shifts)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       i;

  PetscFunctionBegin;
  *ns = ctx->nshifts;
  if (ctx->nshifts) {
    ierr = PetscMalloc1(ctx->nshifts,shifts);CHKERRQ(ierr);
    for (i=0;i<ctx->nshifts;i++) (*shifts)[i] = ctx->shifts[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetRKShifts"
/*@C
   NEPNLEIGSGetRKShifts - Gets the list of shifts used in the Rational
   Krylov method.

   Not Collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameter:
+  ns     - number of shifts
-  shifts - array of shifts

   Level: advanced

.seealso: NEPNLEIGSSetRKShifts()
@*/
PetscErrorCode NEPNLEIGSGetRKShifts(NEP nep,PetscInt *ns,PetscScalar **shifts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(nep,ns);
  PetscValidPointer(nep,shifts);
  ierr = PetscTryMethod(nep,"NEPNLEIGSGetRKShifts_C",(NEP,PetscInt*,PetscScalar**),(nep,ns,shifts));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define SHIFTMAX 30

#undef __FUNCT__
#define __FUNCT__ "NEPSetFromOptions_NLEIGS"
PetscErrorCode NEPSetFromOptions_NLEIGS(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx = (NEP_NLEIGS*)nep->data;
  PetscInt       i,k;
  PetscBool      flg1,flg2,b;
  PetscReal      r;
  PetscScalar    array[SHIFTMAX];
  PC             pc;
  PCType         pctype;
  KSPType        ksptype;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"NEP NLEIGS Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-nep_nleigs_restart","Proportion of vectors kept after restart","NEPNLEIGSSetRestart",0.5,&r,&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = NEPNLEIGSSetRestart(nep,r);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-nep_nleigs_locking","Choose between locking and non-locking variants","NEPNLEIGSSetLocking",PETSC_FALSE,&b,&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = NEPNLEIGSSetLocking(nep,b);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-nep_nleigs_true_residual","Compute true residuals explicitly","NEPNLEIGSSetTrueResidual",PETSC_FALSE,&b,&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = NEPNLEIGSSetTrueResidual(nep,b);CHKERRQ(ierr);
  }
  ierr = NEPNLEIGSGetInterpolation(nep,&r,&i);CHKERRQ(ierr);
  if (!i) i = PETSC_DEFAULT;
  ierr = PetscOptionsInt("-nep_nleigs_interpolation_max_it","Maximum number of terms for interpolation via divided differences","NEPNLEIGSSetInterpolation",i,&i,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-nep_nleigs_interpolation_tol","Tolerance for interpolation via divided differences","NEPNLEIGSSetInterpolation",r,&r,&flg2);CHKERRQ(ierr);
  if (flg1 || flg2) {
    ierr = NEPNLEIGSSetInterpolation(nep,r,i);CHKERRQ(ierr);
  }
  k = SHIFTMAX;
  for (i=0;i<k;i++) array[i] = 0;
  ierr = PetscOptionsScalarArray("-nep_nleigs_rk_shifts","Shifts for Rational Krylov","NEPNLEIGSSetRKShifts",array,&k,&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = NEPNLEIGSSetRKShifts(nep,k,array);CHKERRQ(ierr);
  }

  if (!ctx->ksp) { ierr = NEPNLEIGSGetKSPs(nep,&ctx->ksp);CHKERRQ(ierr); }
  for (i=0;i<ctx->nshiftsw;i++) {
    ierr = KSPGetPC(ctx->ksp[i],&pc);CHKERRQ(ierr);
    ierr = KSPGetType(ctx->ksp[i],&ksptype);CHKERRQ(ierr);
    ierr = PCGetType(pc,&pctype);CHKERRQ(ierr);
    if (!pctype && !ksptype) {
      ierr = KSPSetType(ctx->ksp[i],KSPPREONLY);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(ctx->ksp[i],nep->function,nep->function_pre);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ctx->ksp[i]);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetKSPs_NLEIGS"
static PetscErrorCode NEPNLEIGSGetKSPs_NLEIGS(NEP nep,KSP **ksp)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx = (NEP_NLEIGS*)nep->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (!ctx->ksp) {
    ierr = NEPNLEIGSSetShifts(nep);CHKERRQ(ierr);
    ierr = PetscMalloc1(ctx->nshiftsw,&ctx->ksp);CHKERRQ(ierr);
    for (i=0;i<ctx->nshiftsw;i++) {
      ierr = KSPCreate(PetscObjectComm((PetscObject)nep),&ctx->ksp[i]);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ctx->ksp[i],((PetscObject)nep)->prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ctx->ksp[i],"nep_nleigs_");CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->ksp[i],(PetscObject)nep,1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->ksp[i]);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(ctx->ksp[i],PETSC_TRUE);CHKERRQ(ierr);
    }
  }
  *ksp = ctx->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSGetKSPs"
/*@C
   NEPNLEIGSGetKSPs - Retrieve the array of linear solver objects associated with
   the nonlinear eigenvalue solver.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  ksp - array of linear solver object

   Level: advanced
@*/
PetscErrorCode NEPNLEIGSGetKSPs(NEP nep,KSP **ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(ksp,2);
  ierr = PetscUseMethod(nep,"NEPNLEIGSGetKSPs_C",(NEP,KSP**),(nep,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPView_NLEIGS"
PetscErrorCode NEPView_NLEIGS(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscBool      isascii;
  PetscInt       i;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  NLEIGS: %d%% of basis vectors kept after restart\n",(int)(100*ctx->keep));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  NLEIGS: using the %slocking variant\n",ctx->lock?"":"non-");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  NLEIGS: divided difference terms: used=%D, max=%D\n",ctx->nmat-1,ctx->ddmaxit);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  NLEIGS: tolerance for divided difference convergence: %g\n",(double)ctx->ddtol);CHKERRQ(ierr);
    if (ctx->nshifts) {
      ierr = PetscViewerASCIIPrintf(viewer,"  NLEIGS: RK shifts: ");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      for (i=0;i<ctx->nshifts;i++) {
        ierr = SlepcSNPrintfScalar(str,50,ctx->shifts[i],PETSC_FALSE);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"%s%s",str,(i<ctx->nshifts-1)?",":"");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    }
    if (ctx->trueres) { ierr = PetscViewerASCIIPrintf(viewer,"  NLEIGS: computing true residuals for convergence check\n");CHKERRQ(ierr); }
    if (!ctx->ksp) { ierr = NEPNLEIGSGetKSPs(nep,&ctx->ksp);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(ctx->ksp[0],viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPReset_NLEIGS"
PetscErrorCode NEPReset_NLEIGS(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       k;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
    ierr = PetscFree(ctx->coeffD);CHKERRQ(ierr);
  } else {
    for (k=0;k<ctx->nmat;k++) { ierr = MatDestroy(&ctx->D[k]);CHKERRQ(ierr); }
  }
  if (ctx->vrn) {
    ierr = VecDestroy(&ctx->vrn);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPDestroy_NLEIGS"
PetscErrorCode NEPDestroy_NLEIGS(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       k;
  NEP_NLEIGS     *ctx = (NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  for (k=0;k<ctx->nshiftsw;k++) { ierr = KSPDestroy(&ctx->ksp[k]);CHKERRQ(ierr); }
  ierr = PetscFree(ctx->ksp);CHKERRQ(ierr);
  if (ctx->nshifts) { ierr = PetscFree(ctx->shifts);CHKERRQ(ierr); }
  ierr = PetscFree4(ctx->s,ctx->xi,ctx->beta,ctx->D);CHKERRQ(ierr);
  ierr = PetscFree(nep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetSingularitiesFunction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetSingularitiesFunction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetLocking_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetLocking_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetInterpolation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetInterpolation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetTrueResidual_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetTrueResidual_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetRKShifts_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetRKShifts_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetKSPs_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCreate_NLEIGS"
PETSC_EXTERN PetscErrorCode NEPCreate_NLEIGS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(nep,&ctx);CHKERRQ(ierr);
  nep->data = (void*)ctx;
  ctx->lock    = PETSC_TRUE;
  ctx->ddtol   = PETSC_DEFAULT;
  ctx->ddmaxit = 0;
  ctx->trueres = PETSC_FALSE;
  ctx->nshifts = 0;

  nep->ops->solve          = NEPSolve_NLEIGS;
  nep->ops->setup          = NEPSetUp_NLEIGS;
  nep->ops->setfromoptions = NEPSetFromOptions_NLEIGS;
  nep->ops->view           = NEPView_NLEIGS;
  nep->ops->destroy        = NEPDestroy_NLEIGS;
  nep->ops->reset          = NEPReset_NLEIGS;
  nep->ops->computevectors = NEPComputeVectors_Schur;
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetSingularitiesFunction_C",NEPNLEIGSSetSingularitiesFunction_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetSingularitiesFunction_C",NEPNLEIGSGetSingularitiesFunction_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetRestart_C",NEPNLEIGSSetRestart_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetRestart_C",NEPNLEIGSGetRestart_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetLocking_C",NEPNLEIGSSetLocking_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetLocking_C",NEPNLEIGSGetLocking_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetInterpolation_C",NEPNLEIGSSetInterpolation_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetInterpolation_C",NEPNLEIGSGetInterpolation_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetTrueResidual_C",NEPNLEIGSSetTrueResidual_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetTrueResidual_C",NEPNLEIGSGetTrueResidual_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetRKShifts_C",NEPNLEIGSSetRKShifts_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetRKShifts_C",NEPNLEIGSGetRKShifts_NLEIGS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetKSPs_C",NEPNLEIGSGetKSPs_NLEIGS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

