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
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

typedef struct {
  PetscInt       nmat;      /* number of interpolation points */
  PetscScalar    *s,*xi;    /* Leja-Bagby points */
  PetscScalar    *beta;     /* scaling factors */
  Mat            *D;        /* divided difference matrices */
  PetscScalar    *coeffD;   /* coefficients for divided differences in split form */
  PetscReal      ddtol;     /* tolerance for divided difference convergence */
  BV             W;         /* auxiliary BV object */
  PetscScalar    shift;     /* the target value */
  PetscReal      keep;      /* restart parameter */
  PetscBool      lock;      /* locking/non-locking variant */
  void           *singularitiesctx;
  PetscErrorCode (*computesingularities)(NEP,PetscInt*,PetscScalar*,void*);
} NEP_NLEIGS;

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSBackTransform"
static PetscErrorCode NEPNLEIGSBackTransform(PetscObject ob,PetscInt n,PetscScalar *valr,PetscScalar *vali)
{
  NEP         nep;
  NEP_NLEIGS  *ctx;
  PetscInt    j;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar t;
#endif

  PetscFunctionBegin;
  nep = (NEP)ob;
  ctx = (NEP_NLEIGS*)nep->data;
#if !defined(PETSC_USE_COMPLEX)
  for (j=0;j<n;j++) {
    if (vali[j] == 0) valr[j] = 1.0 / valr[j] + ctx->shift;
    else {
      t = valr[j] * valr[j] + vali[j] * vali[j];
      valr[j] = valr[j] / t + ctx->shift;
      vali[j] = - vali[j] / t;
    }
  }
#else
  for (j=0;j<n;j++) {
    valr[j] = 1.0 / valr[j] + ctx->shift;
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
  for (k=2;k<MAX_LBPOINTS;k++) {
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
#define __FUNCT__ "NEPNLEIGSDividedDifferences_split"
static PetscErrorCode NEPNLEIGSDividedDifferences_split(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       k,j,i;
  PetscReal      norm0,norm,max;
  PetscScalar    *s=ctx->s,*beta=ctx->beta,b[MAX_LBPOINTS+1],alpha,coeffs[MAX_LBPOINTS+1];
  Mat            T;

  PetscFunctionBegin;
  ierr = PetscMalloc1(nep->nt*MAX_LBPOINTS,&ctx->coeffD);CHKERRQ(ierr);
  max = 0.0;
  for (j=0;j<nep->nt;j++) {
    ierr = FNEvaluateFunction(nep->f[j],s[0],ctx->coeffD+j);CHKERRQ(ierr);
    ctx->coeffD[j] /= beta[0];
    max = PetscMax(PetscAbsScalar(ctx->coeffD[j]),max);
  }
  norm0 = max;
  ctx->nmat = MAX_LBPOINTS;
  for (k=1;k<MAX_LBPOINTS;k++) {
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
      ctx->nmat = k;
      break;
    } 
  }
  ierr = NEPNLEIGSEvalNRTFunct(nep,ctx->nmat,nep->target,coeffs);CHKERRQ(ierr);
  ierr = MatDuplicate(nep->A[0],MAT_COPY_VALUES,&T);CHKERRQ(ierr);
  alpha = 0.0;
  for (j=0;j<ctx->nmat;j++) alpha += coeffs[j]*ctx->coeffD[j*nep->nt];
  ierr = MatScale(T,alpha);CHKERRQ(ierr);
  for (k=1;k<nep->nt;k++) {
    alpha = 0.0;
    for (j=0;j<ctx->nmat;j++) alpha += coeffs[j]*ctx->coeffD[j*nep->nt+k];
    ierr = MatAXPY(T,alpha,nep->A[k],nep->mstr);CHKERRQ(ierr);
  }
  ctx->shift = nep->target;
  ierr = KSPSetOperators(nep->ksp,T,T);CHKERRQ(ierr);
  ierr = KSPSetUp(nep->ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSDividedDifferences_callback"
static PetscErrorCode NEPNLEIGSDividedDifferences_callback(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       k,j;
  PetscReal      norm0,norm;
  PetscScalar    *s=ctx->s,*beta=ctx->beta,b[MAX_LBPOINTS+1],coeffs[MAX_LBPOINTS+1];
  Mat            *D=ctx->D,T;

  PetscFunctionBegin;
  T = nep->function;
  ierr = NEPComputeFunction(nep,s[0],T,T);CHKERRQ(ierr);
  ierr = MatDuplicate(T,MAT_COPY_VALUES,&D[0]);CHKERRQ(ierr);
  if (beta[0]!=1.0) {
    ierr = MatScale(D[0],1.0/beta[0]);CHKERRQ(ierr);
  }
  ierr = MatNorm(D[0],NORM_FROBENIUS,&norm0);CHKERRQ(ierr);
  ctx->nmat = MAX_LBPOINTS;
  for (k=1;k<MAX_LBPOINTS;k++) {
    ierr = NEPNLEIGSEvalNRTFunct(nep,k,s[k],b);CHKERRQ(ierr);
    ierr = NEPComputeFunction(nep,s[k],T,T);CHKERRQ(ierr);
    ierr = MatDuplicate(T,MAT_COPY_VALUES,&D[k]);CHKERRQ(ierr);
    for (j=0;j<k;j++) {
      ierr = MatAXPY(D[k],-b[j],D[j],nep->mstr);CHKERRQ(ierr);
    }
    ierr = MatScale(D[k],1.0/b[k]);CHKERRQ(ierr);
    ierr = MatNorm(D[k],NORM_FROBENIUS,&norm);CHKERRQ(ierr);
    if (norm/norm0 < ctx->ddtol) {
      ctx->nmat = k;
      ierr = MatDestroy(&D[k]);CHKERRQ(ierr);
      break;
    } 
  }
  ierr = NEPNLEIGSEvalNRTFunct(nep,ctx->nmat,nep->target,coeffs);CHKERRQ(ierr);
  ierr = MatDuplicate(ctx->D[0],MAT_COPY_VALUES,&T);CHKERRQ(ierr);
  if (coeffs[0]!=1.0) { ierr = MatScale(T,coeffs[0]);CHKERRQ(ierr);}
  for (j=1;j<ctx->nmat;j++) {
    ierr = MatAXPY(T,coeffs[j],ctx->D[j],nep->mstr);CHKERRQ(ierr);
  }
  ctx->shift = nep->target;
  ierr = KSPSetOperators(nep->ksp,T,T);CHKERRQ(ierr);
  ierr = KSPSetUp(nep->ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSKrylovConvergence"
/*
   NEPKrylovConvergence - This is the analogue to EPSKrylovConvergence.
*/
static PetscErrorCode NEPNLEIGSKrylovConvergence(NEP nep,PetscBool getall,PetscInt kini,PetscInt nits,PetscReal beta,PetscInt *kout)
{
  PetscErrorCode ierr;
  PetscInt       k,newk,marker,inside;
  PetscScalar    re,im;
  PetscReal      resnorm;
  PetscBool      istrivial;

  PetscFunctionBegin;
  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  marker = -1;
  if (nep->trackall) getall = PETSC_TRUE;
  for (k=kini;k<kini+nits;k++) {
    /* eigenvalue */
    re = nep->eigr[k];
    im = nep->eigi[k];
    if (!istrivial) {
      ierr = NEPNLEIGSBackTransform((PetscObject)nep,1,&re,&im);CHKERRQ(ierr);
      ierr = RGCheckInside(nep->rg,1,&re,&im,&inside);CHKERRQ(ierr);
      if (marker==-1 && inside<0) marker = k;
      re = nep->eigr[k];
      im = nep->eigi[k];
    }
    newk = k;
    ierr = DSVectors(nep->ds,DS_MAT_X,&newk,&resnorm);CHKERRQ(ierr);
    resnorm *= beta;
    /* error estimate */
    ierr = (*nep->converged)(nep,re,im,resnorm,&nep->errest[k],nep->convergedctx);CHKERRQ(ierr);
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
  if (nep->ncv) { /* ncv set */
    if (nep->ncv<nep->nev) SETERRQ(PetscObjectComm((PetscObject)nep),1,"The value of ncv must be at least nev");
  } else if (nep->mpd) { /* mpd set */
    nep->ncv = PetscMin(nep->n,nep->nev+nep->mpd);
  } else { /* neither set: defaults depend on nev being small or large */
    if (nep->nev<500) nep->ncv = PetscMin(nep->n,PetscMax(2*nep->nev,nep->nev+15));
    else {
      nep->mpd = 500;
      nep->ncv = PetscMin(nep->n,nep->nev+nep->mpd);
    }
  }
  if (!nep->mpd) nep->mpd = nep->ncv;
  if (nep->ncv>nep->nev+nep->mpd) SETERRQ(PetscObjectComm((PetscObject)nep),1,"The value of ncv must not be larger than nev+mpd");
  if (!nep->max_it) nep->max_it = PetscMax(5000,2*nep->n/nep->ncv);
  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (istrivial) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"NEPNLEIGS requires a nontrivial region defining the target set");
  ierr = RGCheckInside(nep->rg,1,&nep->target,&zero,&in);CHKERRQ(ierr);
  if (in<0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"The target is not inside the target set");
  if (!nep->which) nep->which = NEP_TARGET_MAGNITUDE;

  /* Initialize the NLEIGS context structure */
  k = MAX_LBPOINTS;
  ierr = PetscMalloc4(k,&ctx->s,k,&ctx->xi,k,&ctx->beta,k,&ctx->D);CHKERRQ(ierr);
  nep->data = ctx;
  if (nep->tol==PETSC_DEFAULT) nep->tol = SLEPC_DEFAULT_TOL;
  ctx->ddtol = nep->tol/10.0;
  if (!ctx->keep) ctx->keep = 0.5;

  /* Compute Leja-Bagby points and scaling values */
  ierr = NEPNLEIGSLejaBagbyPoints(nep);CHKERRQ(ierr);

  /* Compute the divided difference matrices */
  if (!nep->ksp) { ierr = NEPGetKSP(nep,&nep->ksp);CHKERRQ(ierr); }
  if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
    ierr = NEPNLEIGSDividedDifferences_split(nep);CHKERRQ(ierr);
  } else {
    ierr = NEPNLEIGSDividedDifferences_callback(nep);CHKERRQ(ierr);
  }
  ierr = NEPAllocateSolution(nep,ctx->nmat);CHKERRQ(ierr);
  ierr = NEPSetWorkVecs(nep,4);CHKERRQ(ierr);

  /* set-up DS and transfer split operator functions */
  ierr = DSSetType(nep->ds,DSNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(nep->ds,nep->ncv+1);CHKERRQ(ierr);
  ierr = DSSetExtraRow(nep->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSGetSlepcSC(nep->ds,&sc);CHKERRQ(ierr);
  sc->map           = NEPNLEIGSBackTransform;
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
  PetscInt       nwu=0,i,lds=deg*ld,n;
  PetscReal      eta,onorm;
  
  PetscFunctionBegin;
  ierr = BVGetOrthogonalization(nep->V,NULL,NULL,&eta,NULL);CHKERRQ(ierr);
  n = k+deg-1;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(deg*ld,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr); /* Number of vectors to orthogonalize against them */
  c = work+nwu;
  nwu += k;
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
  }
  if (lindep) {
    *lindep = (*norm < eta * onorm)?PETSC_TRUE:PETSC_FALSE;
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
static PetscErrorCode NEPTOARExtendBasis(NEP nep,PetscScalar sigma,PetscScalar *S,PetscInt ls,PetscInt nv,BV V,Vec t,PetscScalar *r,PetscInt lr,Vec *t_)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       deg=ctx->nmat-1,k,j;
  Vec            v=t_[0],q=t_[1],w;
  PetscScalar    *beta=ctx->beta,*s=ctx->s,*xi=ctx->xi,*coeffs;

  PetscFunctionBegin;
  ierr = BVSetActiveColumns(nep->V,0,nv);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->nmat-1,&coeffs);CHKERRQ(ierr);
  if (PetscAbsScalar(s[deg-1]-sigma)<100*PETSC_MACHINE_EPSILON) SETERRQ(PETSC_COMM_SELF,1,"Breakdown in NLEIGS");
  for (j=0;j<nv;j++) {
    r[(deg-1)*lr+j] = (S[(deg-1)*ls+j]+(beta[deg]/xi[deg-1])*S[(deg)*ls+j])/(s[deg-1]-sigma);
  }
  ierr = BVSetActiveColumns(ctx->W,0,ctx->nmat-1);CHKERRQ(ierr);
  ierr = BVGetColumn(ctx->W,deg-1,&w);CHKERRQ(ierr);
  ierr = BVMultVec(V,1.0,0.0,w,r+(deg-1)*lr);CHKERRQ(ierr);
  ierr = BVRestoreColumn(ctx->W,deg-1,&w);CHKERRQ(ierr);
  for (k=deg-1;k>0;k--) {
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
    ierr = KSPSolve(nep->ksp,q,t);CHKERRQ(ierr);
    ierr = VecScale(t,-1.0);CHKERRQ(ierr);
  } else {
    for (k=0;k<ctx->nmat-1;k++) {
      ierr = BVGetColumn(ctx->W,k,&w);CHKERRQ(ierr);
      ierr = MatMult(ctx->D[k],w,q);CHKERRQ(ierr);
      ierr = BVRestoreColumn(ctx->W,k,&w);CHKERRQ(ierr);
      ierr = BVInsertVec(ctx->W,k,q);CHKERRQ(ierr);
    }
    for (j=0;j<ctx->nmat-1;j++) coeffs[j] = 1.0;
    ierr = BVMultVec(ctx->W,1.0,0.0,q,coeffs);CHKERRQ(ierr);
    ierr = KSPSolve(nep->ksp,q,t);CHKERRQ(ierr);
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
  PetscInt       k,j,d=ctx->nmat;
  PetscScalar    *t=work;

  PetscFunctionBegin;
  ierr = NEPNLEIGSEvalNRTFunct(nep,d-1,nep->target,t);CHKERRQ(ierr);
  for (k=0;k<d-1;k++) {
    for (j=0;j<=nv;j++) r[k*lr+j] += t[k]*x[j];
  }
  for (j=0;j<=nv;j++) r[(d-1)*lr+j] = t[d-1]*x[j];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPTOARrun"
/*
  Compute a run of Arnoldi iterations
*/
static PetscErrorCode NEPTOARrun(NEP nep,PetscInt *nq,PetscScalar *S,PetscInt ld,PetscScalar *H,PetscInt ldh,BV V,PetscInt k,PetscInt *M,PetscBool *breakdown,Vec *t_)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       i,j,p,m=*M,lwa,deg=ctx->nmat,lds=ld*deg,nqt=*nq;
  Vec            t=t_[0];
  PetscReal      norm;
  PetscScalar    *x,*work;
  PetscBool      lindep;

  PetscFunctionBegin;
  lwa = PetscMax(ld,deg);
  ierr = PetscMalloc2(ld,&x,lwa,&work);CHKERRQ(ierr);
  for (j=k;j<m;j++) {
    /* apply operator */
    ierr = BVGetColumn(nep->V,nqt,&t);CHKERRQ(ierr);
    ierr = NEPTOARExtendBasis(nep,ctx->shift,S+j*lds,ld,nqt,V,t,S+(j+1)*lds,ld,t_+1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(nep->V,nqt,&t);CHKERRQ(ierr);

    /* orthogonalize */
    ierr = BVOrthogonalizeColumn(nep->V,nqt,x,&norm,&lindep);CHKERRQ(ierr);
    if (!lindep) {
      x[nqt] = norm;
      ierr = BVScaleColumn(nep->V,nqt,1.0/norm);CHKERRQ(ierr);
      nqt++;
    }

    ierr = NEPTOARCoefficients(nep,ctx->shift,*nq,S+j*lds,ld,S+(j+1)*lds,ld,x,work);CHKERRQ(ierr);

    /* Level-2 orthogonalization */
    ierr = NEPTOAROrth2(nep,S,ld,deg,j+1,H+j*ldh,&norm,breakdown,work);CHKERRQ(ierr);
    H[j+1+ldh*j] = norm;
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
  ierr = PetscFree2(x,work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPTOARTrunc"
/* dim(work)=5*ld*lds dim(rwork)=6*n */
PetscErrorCode NEPTOARTrunc(NEP nep,PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt *nq,PetscInt cs1,PetscScalar *work,PetscReal *rwork)
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
PetscErrorCode NEPTOARSupdate(PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt sr,PetscInt s,PetscInt ncu,PetscInt qr,PetscScalar *Q,PetscInt ldq,PetscScalar *work)
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
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       i,j,k=0,l,nv=0,ld,lds,off,ldds,newn,rs1,nq=0;
  PetscInt       lwa,lrwa,nwu=0,nrwu=0,deg=ctx->nmat;
  PetscScalar    *S,*Q,*work,*H,*pU;
  PetscReal      beta,norm,*rwork;
  PetscBool      breakdown=PETSC_FALSE,lindep;
  Mat            U;
    
  PetscFunctionBegin;
  ld = nep->ncv+deg;
  lds = deg*ld;
  lwa = (deg+6)*ld*lds;
  lrwa = 7*lds;
  ierr = PetscMalloc3(lwa,&work,lrwa,&rwork,lds*ld,&S);CHKERRQ(ierr);
  ierr = PetscMemzero(S,lds*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(nep->ds,&ldds);CHKERRQ(ierr);
  ierr = BVDuplicateResize(nep->V,PetscMax(nep->nt-1,ctx->nmat-1),&ctx->W);CHKERRQ(ierr);

  /* Get the starting vector */
  for (i=0;i<deg;i++) {
    ierr = BVSetRandomColumn(nep->V,i,nep->rand);CHKERRQ(ierr);
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
    ierr = DSGetArray(nep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = NEPTOARrun(nep,&nq,S,ld,H,ldds,nep->V,nep->nconv+l,&nv,&breakdown,nep->work);CHKERRQ(ierr);
    beta = PetscAbsScalar(H[(nv-1)*ldds+nv]);
    ierr = DSRestoreArray(nep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = DSSetDimensions(nep->ds,nv,0,nep->nconv,nep->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(nep->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(nep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Solve projected problem */
    ierr = DSSolve(nep->ds,nep->eigr,nep->eigi);CHKERRQ(ierr);
    ierr = DSSort(nep->ds,nep->eigr,nep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);;
    ierr = DSUpdateExtraRow(nep->ds);CHKERRQ(ierr);

    /* Check convergence */
    ierr = NEPNLEIGSKrylovConvergence(nep,PETSC_FALSE,nep->nconv,nv-nep->nconv,beta,&k);CHKERRQ(ierr);
    ierr = (*nep->stopping)(nep,nep->its,nep->max_it,k,nep->nev,&nep->reason,nep->stoppingctx);CHKERRQ(ierr);

    /* Update l */
    if (nep->reason != NEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)*ctx->keep));
      if (!breakdown) {
        /* Prepare the Rayleigh quotient for restart */
        ierr = DSTruncate(nep->ds,k+l);CHKERRQ(ierr);
        ierr = DSGetDimensions(nep->ds,&newn,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
        l = newn-k;
      }
    }

    /* Update S */
    off = nep->nconv*ldds;
    ierr = DSGetArray(nep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = NEPTOARSupdate(S,ld,deg,nq,nep->nconv,k+l-nep->nconv,nv,Q+off,ldds,work+nwu);CHKERRQ(ierr);
    ierr = DSRestoreArray(nep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);

    /* Copy last column of S */
    ierr = PetscMemcpy(S+lds*(k+l),S+lds*nv,lds*sizeof(PetscScalar));CHKERRQ(ierr);

    if (nep->reason == NEP_CONVERGED_ITERATING) {
      if (breakdown) {

        /* Stop if breakdown */
        ierr = PetscInfo2(nep,"Breakdown (it=%D norm=%g)\n",nep->its,(double)beta);CHKERRQ(ierr);
        nep->reason = NEP_DIVERGED_BREAKDOWN;
      } else {
        /* Truncate S */
        ierr = NEPTOARTrunc(nep,S,ld,deg,&nq,k+l+1,work+nwu,rwork+nrwu);CHKERRQ(ierr);
      }
    }
    nep->nconv = k;
    ierr = NEPMonitor(nep,nep->its,nep->nconv,nep->eigr,nep->errest,nv);CHKERRQ(ierr);
  }
  if (nep->nconv>0) {
    /* Extract invariant pair */
    ierr = NEPTOARTrunc(nep,S,ld,deg,&nq,nep->nconv,work+nwu,rwork+nrwu);CHKERRQ(ierr);
    /* Update vectors V = V*S */    
    rs1 = nep->nconv;
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

  ierr = PetscFree3(work,rwork,S);CHKERRQ(ierr);
  /* Map eigenvalues back to the original problem */
  ierr = NEPNLEIGSBackTransform((PetscObject)nep,nep->nconv,nep->eigr,nep->eigi);CHKERRQ(ierr);
  ierr = BVDestroy(&ctx->W);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetSingularitiesFunction_NLEIGS"
static PetscErrorCode NEPNLEIGSSetSingularitiesFunction_NLEIGS(NEP nep,PetscErrorCode (*fun)(NEP,PetscInt*,PetscScalar*,void*),void *ctx)
{
  NEP_NLEIGS *nepctx = (NEP_NLEIGS*)nep->data;

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

   Level: beginner

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
  NEP_NLEIGS *nepctx = (NEP_NLEIGS*)nep->data;

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
  ierr = PetscTryMethod(nep,"NEPNLEIGSGetSingularitiesFunction_C",(NEP,PetscErrorCode(**)(NEP,PetscInt*,PetscScalar*,void*),void**),(nep,fun,ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetRestart_NLEIGS"
static PetscErrorCode NEPNLEIGSSetRestart_NLEIGS(NEP nep,PetscReal keep)
{
  NEP_NLEIGS *ctx = (NEP_NLEIGS*)nep->data;

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
  NEP_NLEIGS *ctx = (NEP_NLEIGS*)nep->data;

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
  ierr = PetscTryMethod(nep,"NEPNLEIGSGetRestart_C",(NEP,PetscReal*),(nep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSSetLocking_NLEIGS"
static PetscErrorCode NEPNLEIGSSetLocking_NLEIGS(NEP nep,PetscBool lock)
{
  NEP_NLEIGS *ctx = (NEP_NLEIGS*)nep->data;

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
  NEP_NLEIGS *ctx = (NEP_NLEIGS*)nep->data;

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
  ierr = PetscTryMethod(nep,"NEPNLEIGSGetLocking_C",(NEP,PetscBool*),(nep,lock));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetFromOptions_NLEIGS"
PetscErrorCode NEPSetFromOptions_NLEIGS(PetscOptionItems *PetscOptionsObject,NEP nep)
{
  PetscErrorCode ierr;
  PetscBool      flg,lock;
  PetscReal      keep;
  PC             pc;
  PCType         pctype;
  KSPType        ksptype;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"NEP NLEIGS Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-nep_nleigs_restart","Proportion of vectors kept after restart","NEPNLEIGSSetRestart",0.5,&keep,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = NEPNLEIGSSetRestart(nep,keep);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-nep_nleigs_locking","Choose between locking and non-locking variants","NEPNLEIGSSetLocking",PETSC_FALSE,&lock,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = NEPNLEIGSSetLocking(nep,lock);CHKERRQ(ierr);
  }
  if (!nep->ksp) { ierr = NEPGetKSP(nep,&nep->ksp);CHKERRQ(ierr); }
  ierr = KSPGetPC(nep->ksp,&pc);CHKERRQ(ierr);
  ierr = KSPGetType(nep->ksp,&ksptype);CHKERRQ(ierr);
  ierr = PCGetType(pc,&pctype);CHKERRQ(ierr);
  if (!pctype && !ksptype) {
    ierr = KSPSetType(nep->ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPView_NLEIGS"
PetscErrorCode NEPView_NLEIGS(NEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx = (NEP_NLEIGS*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  NLEIGS: %d%% of basis vectors kept after restart\n",(int)(100*ctx->keep));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  NLEIGS: using the %slocking variant\n",ctx->lock?"":"non-");CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPDestroy_NLEIGS"
PetscErrorCode NEPDestroy_NLEIGS(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  ierr = PetscFree4(ctx->s,ctx->xi,ctx->beta,ctx->D);CHKERRQ(ierr);
  ierr = PetscFree(nep->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetSingularitiesFunction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetSingularitiesFunction_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSSetLocking_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)nep,"NEPNLEIGSGetLocking_C",NULL);CHKERRQ(ierr);
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
  ctx->lock = PETSC_TRUE;

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
  PetscFunctionReturn(0);
}

