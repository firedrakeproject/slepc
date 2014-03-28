/*

   SLEPc nonlinear eigensolver: "nleigs"

   Method: NLEIGS

   References:

       [1] S. Guttel et al., "NLEIGS: A class of robust fully rational Krylov
           method for nonlinear eigenvalue problems", 2014.

   Last update: Mar 2013

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/nepimpl.h>         /*I "slepcnep.h" I*/
#include <slepcblaslapack.h>

#define  MAX_LBPOINTS  100
#define  DDTOL         1e-6

typedef struct {        /* context structure for the NLEIGS solver */
  PetscScalar  s0,s1;   /* target interval ends */
  PetscInt     ndpt,nmat; 
  PetscScalar  *s,*xi;  /* Leja-Bagby points */
  PetscScalar  *beta;   /* scaling factors */
  Mat          *D;      /* divided difference matrices */
  ST           st;
} NEP_NLEIGS;

typedef struct {
  PetscErrorCode (*comparison)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  void *comparisonctx;
  ST st;
} NEPSortForSTData;

#undef __FUNCT__
#define __FUNCT__ "NEPSortForSTFunc"
static PetscErrorCode NEPSortForSTFunc(PetscScalar ar,PetscScalar ai,
                                PetscScalar br,PetscScalar bi,PetscInt *r,void *ctx)
{
  NEPSortForSTData *data = (NEPSortForSTData*)ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = STBackTransform(data->st,1,&ar,&ai);CHKERRQ(ierr);
  ierr = STBackTransform(data->st,1,&br,&bi);CHKERRQ(ierr);
  ierr = (*data->comparison)(ar,ai,br,bi,r,data->comparisonctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSLejaBagbyPoints"
static PetscErrorCode NEPNLEIGSLejaBagbyPoints(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       i,k,ndpt=ctx->ndpt;
  PetscScalar    *ds,*dxi,*nrs,*nrxi,*s=ctx->s,*xi=ctx->xi,*beta=ctx->beta;
  PetscReal      h,maxnrs,minnrxi;

  PetscFunctionBegin;
  ierr = PetscMalloc4(ndpt,&ds,ndpt,&dxi,ndpt,&nrs,ndpt,&nrxi);CHKERRQ(ierr);
  /* Discretize the target region boundary (linspace)
    (this will be a region object function) */
  h = PetscAbsScalar(ctx->s1-ctx->s0)/(ndpt-1);
  for (i=0;i<ndpt;i++) ds[i] = ctx->s0+i*h;
  /* Discretize the singularity region (logspace)
    (by the moment is (0,-inf)~(10e-6,10e+6)) */
  h = 12.0/(ndpt-1);
  dxi[0] = -1e-6; dxi[ndpt-1] = -1e+6;
  for (i=1;i<ndpt-1;i++) dxi[i] = -PetscPowReal(10,-6+h*i);
  /* loop to look for Leja-Bagby points in the discretization sets 
     now both sets have the same number of points */
  s[0] = ds[0]; xi[0] = dxi[0];
  beta[0] = 1.0; /* scaling factors are also computed here */
  maxnrs = 0.0;
  minnrxi = PETSC_MAX_REAL; 
  for (i=0;i<ndpt;i++) {
    nrs[i]  = (ds[i]-s[0])/(1-ds[i]/xi[0]);
    if (PetscAbsScalar(nrs[i])>maxnrs) {maxnrs = PetscAbsScalar(nrs[i]); s[1] = ds[i];}
    nrxi[i] = (dxi[i]-s[0])/(1-dxi[i]/xi[0]);
    if (PetscAbsScalar(nrxi[i])<minnrxi) {minnrxi = PetscAbsScalar(nrxi[i]); xi[1] = dxi[i];}
  }
  beta[1] = maxnrs;
  for (k=2;k<MAX_LBPOINTS;k++) {
    maxnrs = 0.0;
    minnrxi = PETSC_MAX_REAL;
    for (i=0;i<ndpt;i++) {
      nrs[i]  *= ((ds[i]-s[k-1])/(1-ds[i]/xi[k-1]))/beta[k-1];
      if (PetscAbsScalar(nrs[i])>maxnrs) {maxnrs = PetscAbsScalar(nrs[i]); s[k] = ds[i];}
      nrxi[i] *= ((dxi[i]-s[k-1])/(1-dxi[i]/xi[k-1]))/beta[k-1];
      if (PetscAbsScalar(nrxi[i])<minnrxi) {minnrxi = PetscAbsScalar(nrxi[i]); xi[k] = dxi[i];}
    }
    beta[k] = maxnrs;
  }
  ierr = PetscFree4(ds,dxi,nrs,nrxi);CHKERRQ(ierr);
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
    b[i+1] = ((sigma-s[i])*b[i])/(beta[i+1]*(1-sigma/xi[i]));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPNLEIGSDividedDifferences"
static PetscErrorCode NEPNLEIGSDividedDifferences(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       k,j;
  PetscReal      norm0,norm;
  PetscScalar    *s=ctx->s,*beta=ctx->beta,b[MAX_LBPOINTS];
  Mat            *D=ctx->D,T=nep->function;
  MatStructure   str;
  PetscBool      flg=PETSC_TRUE;

  PetscFunctionBegin;
  ierr = NEPComputeFunction(nep,s[0],&T,&T,&str);CHKERRQ(ierr);
  ierr = MatDuplicate(T,MAT_COPY_VALUES,&D[0]);CHKERRQ(ierr);
  if (beta[0]!=1.0) {
    ierr = MatScale(D[0],1.0/beta[0]);CHKERRQ(ierr);
  }
  ierr = MatNorm(D[0],NORM_FROBENIUS,&norm0);CHKERRQ(ierr);
  ctx->nmat = MAX_LBPOINTS;
  for (k=1;k<MAX_LBPOINTS && flg;k++) {
    ierr = NEPNLEIGSEvalNRTFunct(nep,k,s[k],b);CHKERRQ(ierr);
    ierr = NEPComputeFunction(nep,s[k],&T,&T,&str);CHKERRQ(ierr);
    ierr = MatDuplicate(T,MAT_COPY_VALUES,&D[k]);CHKERRQ(ierr);
    for (j=0;j<k;j++) {
      ierr = MatAXPY(D[k],-b[j],D[j],str);CHKERRQ(ierr);
    }
    ierr = MatScale(D[k],1.0/b[k]);CHKERRQ(ierr);
    ierr = MatNorm(D[k],NORM_FROBENIUS,&norm);CHKERRQ(ierr);
    if (norm/norm0 < DDTOL) {
      flg = PETSC_FALSE;
      ctx->nmat = k;
      ierr = MatDestroy(&D[k]);CHKERRQ(ierr);
    } 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetUp_NLEIGS"
PetscErrorCode NEPSetUp_NLEIGS(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       k;
  PetscScalar    coeffs[MAX_LBPOINTS];
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;

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
  if (!nep->max_funcs) nep->max_funcs = nep->max_it;
  if (!ctx->st) { 
    ierr = STCreate(PetscObjectComm((PetscObject)nep),&ctx->st);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->st);CHKERRQ(ierr);
  }
  ierr = STSetDefaultShift(ctx->st,nep->target);CHKERRQ(ierr);
  ierr = STSetType(ctx->st,STSINVERT);CHKERRQ(ierr);
  if (!nep->which) nep->which = NEP_TARGET_MAGNITUDE;
  /* Initialize the NLEIGS context structure */
  k = MAX_LBPOINTS;
  ierr = PetscMalloc4(k,&ctx->s,k,&ctx->xi,k,&ctx->beta,k,&ctx->D);CHKERRQ(ierr);
  nep->data = ctx;
  ctx->s0 = -1.0;
  ctx->s1 = 1.0;
  ctx->ndpt = 1000; /* default number of discretization points */
  ierr = PetscOptionsGetScalar(NULL,"-nleigs_a",&ctx->s0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,"-nleigs_b",&ctx->s1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-nleigs_npts",&ctx->ndpt,NULL);CHKERRQ(ierr);
  ctx->ndpt = PetscMax(ctx->ndpt,2); /* ndpt=2 for no points other than s0 and s1*/
  /* Compute Leja-Bagby points and scaling values */
  ierr = NEPNLEIGSLejaBagbyPoints(nep);

  /* Compute the divided difference matrices */
  ierr = NEPNLEIGSDividedDifferences(nep);CHKERRQ(ierr);
  ierr = STSetOperators(ctx->st,ctx->nmat,ctx->D);CHKERRQ(ierr);
  ierr = STSetUp(ctx->st);CHKERRQ(ierr);
  ierr = NEPNLEIGSEvalNRTFunct(nep,ctx->nmat,nep->target,coeffs);CHKERRQ(ierr);
  ierr = STComputeSolveMat(ctx->st,1.0,coeffs);CHKERRQ(ierr);
  ierr = NEPAllocateSolution(nep,ctx->nmat);CHKERRQ(ierr);
  ierr = NEPSetWorkVecs(nep,4);CHKERRQ(ierr);

  /* set-up DS and transfer split operator functions */
  ierr = DSSetType(nep->ds,DSNHEP);CHKERRQ(ierr);
  ierr = DSAllocate(nep->ds,nep->ncv+1);CHKERRQ(ierr);
  ierr = DSSetExtraRow(nep->ds,PETSC_TRUE);CHKERRQ(ierr);
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
*/
static PetscErrorCode NEPTOAROrth2(PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt k,PetscScalar *y,PetscScalar *work,PetscInt nw)
{
  PetscErrorCode ierr;
  PetscBLASInt   n_,lds_,k_,one=1;
  PetscScalar    sonem=-1.0,sone=1.0,szero=0.0,*x0,*x,*c;
  PetscInt       lwa,nwu=0,i,lds=deg*ld,n;
  
  PetscFunctionBegin;
  n = k+deg-1;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(deg*ld,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr); /* Number of vectors to orthogonalize against them */
  lwa = k;
  if (!work||nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",6);
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
static PetscErrorCode NEPTOARExtendBasis(NEP nep,PetscScalar sigma,PetscScalar *S,PetscInt ls,PetscInt nv,Vec *V,Vec t,PetscScalar *r,PetscInt lr,Vec *t_,PetscInt nwv)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       deg=ctx->nmat,k,j;
  Vec            v=t_[0],q=t_[1];
  PetscScalar    *beta=ctx->beta,*s=ctx->s,*xi=ctx->xi;

  PetscFunctionBegin;
  for (j=0;j<nv;j++) {
    r[(deg-2)*lr+j] = (S[(deg-2)*ls+j]+(beta[deg-1]/xi[deg-2])*S[(deg-1)*ls+j])/(s[deg-2]-sigma);
  }
  ierr = SlepcVecMAXPBY(v,0.0,1.0,nv,r+(deg-2)*lr,V);CHKERRQ(ierr);
  ierr = STMatMult(ctx->st,deg-2,v,q);CHKERRQ(ierr);
  for (k=deg-2;k>0;k--) {
    for (j=0;j<nv;j++) r[(k-1)*lr+j] = (S[(k-1)*ls+j]+(beta[k]/xi[k-1])*S[k*ls+j]-beta[k]*(1-sigma/xi[k-1])*r[(k)*lr+j])/(s[k-1]-sigma);
    ierr = SlepcVecMAXPBY(v,0.0,1.0,nv,r+(k-1)*lr,V);CHKERRQ(ierr);
    ierr = STMatMult(ctx->st,k-1,v,t);CHKERRQ(ierr);
    ierr = VecAXPY(q,1.0,t);CHKERRQ(ierr);
  }
  ierr = STMatSolve(ctx->st,q,t);CHKERRQ(ierr);
  ierr = VecScale(t,-1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPTOARCoefficients"
/*
  Compute TOAR coefficients of the blocks of the new Arnoldi vector computed
*/
static PetscErrorCode NEPTOARCoefficients(NEP nep,PetscScalar sigma,PetscInt nv,PetscScalar *S,PetscInt ls,PetscScalar *r,PetscInt lr,PetscScalar *x)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       k,j,d=ctx->nmat;
  PetscScalar    t[d];

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
static PetscErrorCode NEPTOARrun(NEP nep,PetscScalar *S,PetscInt ld,PetscScalar *H,PetscInt ldh,Vec *V,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscScalar *work,PetscInt nw,Vec *t_,PetscInt nwv)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  PetscInt       i,j,p,m=*M,nwu=0,lwa,deg=ctx->nmat;
  PetscInt       lds=ld*deg;
  Vec            t=t_[0];
  PetscReal      norm;
  PetscScalar    sigma=0.0,x[ld];

  PetscFunctionBegin;
  if (!t_||nwv<4) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",12);
  lwa = ld;
  if (!work||nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",10);
  ierr = STGetShift(ctx->st,&sigma);CHKERRQ(ierr); 
  for (j=k;j<m;j++) {
    /* apply operator */
    ierr = NEPTOARExtendBasis(nep,sigma,S+j*lds,ld,j+deg,V,t,S+(j+1)*lds,ld,t_+1,2);CHKERRQ(ierr);
    /* orthogonalize */
    ierr = IPOrthogonalize(nep->ip,0,NULL,j+deg,NULL,nep->V,t,x,&norm,breakdown);CHKERRQ(ierr);
    x[j+deg] = norm;
    ierr = VecScale(t,1.0/norm);CHKERRQ(ierr);
    ierr = VecCopy(t,V[j+deg]);CHKERRQ(ierr);
    ierr = NEPTOARCoefficients(nep,sigma,j+deg,S+j*lds,ld,S+(j+1)*lds,ld,x);CHKERRQ(ierr);
    /* Level-2 orthogonalization */
    ierr = NEPTOAROrth2(S,ld,deg,j+1,H+j*ldh,work+nwu,lwa-nwu);CHKERRQ(ierr);
    ierr = NEPTOARSNorm2(lds,S+(j+1)*lds,&norm);CHKERRQ(ierr);
    for (p=0;p<deg;p++) {
      for (i=0;i<=j+deg;i++) {
        S[i+p*ld+(j+1)*lds] /= norm;
      }
    }
    H[j+1+ldh*j] = norm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPTOARTrunc"
PetscErrorCode NEPTOARTrunc(NEP nep,PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt rs1,PetscInt cs1,PetscScalar *work,PetscInt nw,PetscReal *rwork,PetscInt nrw)
{
  PetscErrorCode ierr;
  PetscInt       lwa,nwu=0,lrwa,nrwu=0;
  PetscInt       j,i,n,lds=deg*ld;
  PetscScalar    *M,*V,*U,t;
  PetscReal      *sg;
  PetscBLASInt   cs1_,rs1_,cs1tdeg,n_,info,lw_;

  PetscFunctionBegin;
  n = (rs1>deg*cs1)?deg*cs1:rs1;
  lwa = 5*ld*lds;
  lrwa = 6*n;
  if (!work||nw<lwa) {
    if (nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",6);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",5);
  }
  if (!rwork||nrw<lrwa) {
    if (nrw<lrwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",8);
    if (!rwork) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",7);
  }
  M = work+nwu;
  nwu += rs1*cs1*deg;
  sg = rwork+nrwu;
  nrwu += n;
  U = work+nwu;
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
  PetscStackCall("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1tdeg,M,&rs1_,sg,U,&rs1_,V,&n_,work+nwu,&lw_,&info));
#else
  PetscStackCall("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1tdeg,M,&rs1_,sg,U,&rs1_,V,&n_,work+nwu,&lw_,rwork+nrwu,&info));  
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);
  
  /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
  ierr = SlepcUpdateVectors(rs1,nep->V,0,cs1+deg-1,U,rs1,PETSC_FALSE);CHKERRQ(ierr);
  
  /* Update S */
  ierr = PetscMemzero(S,lds*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0;i<cs1+deg-1;i++) {
    t = sg[i];
    PetscStackCall("BLASscal",BLASscal_(&cs1tdeg,&t,V+i,&n_));
  }
  for (j=0;j<cs1;j++) {
    for (i=0;i<deg;i++) {
      ierr = PetscMemcpy(S+j*lds+i*ld,V+(cs1*i+j)*n,(cs1+deg-1)*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPTOARSupdate"
/*
  S <- S*Q 
  columns s-s+ncu of S
  rows 0-sr of S
  size(Q) qr x ncu
*/
PetscErrorCode NEPTOARSupdate(PetscScalar *S,PetscInt ld,PetscInt deg,PetscInt sr,PetscInt s,PetscInt ncu,PetscInt qr,PetscScalar *Q,PetscInt ldq,PetscScalar *work,PetscInt nw)
{
  PetscErrorCode ierr;
  PetscScalar    a=1.0,b=0.0;
  PetscBLASInt   sr_,ncu_,ldq_,lds_,qr_;
  PetscInt       lwa,j,lds=deg*ld,i;

  PetscFunctionBegin;
  lwa = sr*ncu;
  if (!work||nw<lwa) {
    if (nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",10);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",9);
  }
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
  PetscInt       i,j,k,l,nv=0,ld,lds,off,ldds,newn;
  PetscInt       lwa,lrwa,nwu=0,nrwu=0,deg=ctx->nmat;
  PetscScalar    *S,*Q,*work,*H;
  PetscReal      beta,norm,*rwork;
  PetscBool      breakdown;
  NEPSortForSTData  data;
/* /////// */
  PetscReal      re,im;
  PetscScalar    coeffs[ctx->nmat];
  Vec            x,t;
/* /////// */
    
  PetscFunctionBegin;
/* /////// */
  /* temporarily change eigenvalue comparison function */
  data.comparison    = nep->comparison;
  data.comparisonctx = nep->comparisonctx;
  data.st            = ctx->st;
  nep->comparison    = NEPSortForSTFunc;
  nep->comparisonctx = &data;
  ierr = DSSetEigenvalueComparison(nep->ds,nep->comparison,nep->comparisonctx);CHKERRQ(ierr);
/* /////// */

  /* Restart loop */
  ld = nep->ncv+deg;
  lds = deg*ld;
  lwa = (deg+5)*ld*lds;
  lrwa = 7*lds;
  ierr = PetscMalloc3(lwa,&work,lrwa,&rwork,lds*ld,&S);CHKERRQ(ierr);
  ierr = PetscMemzero(S,lds*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(nep->ds,&ldds);CHKERRQ(ierr);

  /* Get the starting Lanczos vector */
  if (nep->nini==0) {  
    ierr = SlepcVecSetRandom(nep->V[0],nep->rand);CHKERRQ(ierr);
  }
  ierr = IPNorm(nep->ip,nep->V[0],&norm);CHKERRQ(ierr);
  ierr = VecScale(nep->V[0],1/norm);CHKERRQ(ierr);
  S[0] = norm;
  for (i=1;i<deg;i++) {
    ierr = SlepcVecSetRandom(nep->V[i],nep->rand);CHKERRQ(ierr);
    ierr = IPOrthogonalize(nep->ip,0,NULL,i,NULL,nep->V,nep->V[i],S+i*ld,&norm,NULL);CHKERRQ(ierr);
    ierr = VecScale(nep->V[i],1/norm);CHKERRQ(ierr);
    S[i+i*ld] = norm;
    if (norm<PETSC_MACHINE_EPSILON) SETERRQ(PetscObjectComm((PetscObject)nep),1,"Problem with initial vector");
  }
  ierr = NEPTOARSNorm2(lds,S,&norm);CHKERRQ(ierr);
  for (j=0;j<deg;j++) {
    for (i=0;i<=j;i++) S[i+j*ld] /= norm;
  }
  /* Restart loop */
  l = 0;
  while (nep->reason == NEP_CONVERGED_ITERATING) {
    nep->its++;
    
    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(nep->nconv+nep->mpd,nep->ncv);
    ierr = DSGetArray(nep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = NEPTOARrun(nep,S,ld,H,ldds,nep->V,nep->nconv+l,&nv,&breakdown,work+nwu,lwa-nwu,nep->work,4);CHKERRQ(ierr);
    beta = PetscAbsScalar(H[(nv-1)*ldds+nv]);
    ierr = DSRestoreArray(nep->ds,DS_MAT_A,&H);CHKERRQ(ierr);
    ierr = DSSetDimensions(nep->ds,nv,0,nep->nconv,nep->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(nep->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(nep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }

    /* Solve projected problem */
    ierr = DSSolve(nep->ds,nep->eig,NULL);CHKERRQ(ierr);
    ierr = DSSort(nep->ds,nep->eig,NULL,NULL,NULL,NULL);CHKERRQ(ierr);;
    ierr = DSUpdateExtraRow(nep->ds);CHKERRQ(ierr);

    /* Check convergence */
    ierr = NEPKrylovConvergence(nep,PETSC_FALSE,nep->nconv,nv-nep->nconv,nv,beta,&k);CHKERRQ(ierr);
    if (nep->its >= nep->max_it) nep->reason = NEP_DIVERGED_MAX_IT;
    if (k >= nep->nev) nep->reason = NEP_CONVERGED_TOL;

    /* Update l */
    if (nep->reason != NEP_CONVERGED_ITERATING || breakdown) l = 0;
    else {
      l = PetscMax(1,(PetscInt)((nv-k)/2));
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
    ierr = NEPTOARSupdate(S,ld,deg,nv+deg,nep->nconv,k+l-nep->nconv,nv,Q+off,ldds,work+nwu,lwa-nwu);CHKERRQ(ierr);
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
        ierr = NEPTOARTrunc(nep,S,ld,deg,nv+deg,k+l+1,work+nwu,lwa-nwu,rwork+nrwu,lrwa-nrwu);CHKERRQ(ierr);
      }
    }
    nep->nconv = k;
    ierr = NEPMonitor(nep,nep->its,nep->nconv,nep->eig,nep->errest,nv);CHKERRQ(ierr);
  }
  if (nep->nconv>0) {
    /* Extract invariant pair */
    ierr = NEPTOARTrunc(nep,S,ld,deg,nv+deg,nep->nconv,work+nwu,lwa-nwu,rwork+nrwu,lrwa-nrwu);CHKERRQ(ierr);
    /* Update vectors V = V*S */    
    ierr = SlepcUpdateVectors(nep->nconv+deg-1,nep->V,0,nep->nconv,S,lds,PETSC_FALSE);CHKERRQ(ierr);
  }
  /* truncate Schur decomposition and change the state to raw so that
     DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(nep->ds,nep->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(nep->ds,DS_STATE_RAW);CHKERRQ(ierr);

  /* Compute eigenvectors */
  if (nep->nconv > 0) {
    ierr = NEPComputeVectors_Schur(nep);CHKERRQ(ierr);
  }
  ierr = PetscFree3(work,rwork,S);CHKERRQ(ierr);
  /* /////// */
  /* restore comparison function */
  nep->comparison    = data.comparison;
  nep->comparisonctx = data.comparisonctx;
  /* /////// */
  /* Map eigenvalues back to the original problem */
  ierr = STBackTransform(ctx->st,nep->nconv,nep->eig,NULL);CHKERRQ(ierr);

  /* /////// */
  /* For testing, before destroying the solver context 
     the residual for the interpolation is shown */
  if (nep->nconv>0) {
    ierr = VecDuplicate(nep->V[0],&x);CHKERRQ(ierr);
    ierr = VecDuplicate(nep->V[0],&t);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Showing absolute residual for the interpolation function eigenpairs\n",nep->nconv);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate eigenpairs: %d\n",nep->nconv);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations: %d\n\n",nep->its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "           k              ||Q%d(k)x||           error\n"
         "   ----------------- ------------------ ------------------\n",deg);CHKERRQ(ierr);
    for (i=0;i<nep->nconv;i++) {
      ierr = NEPNLEIGSEvalNRTFunct(nep,ctx->nmat,nep->eig[i],coeffs);CHKERRQ(ierr);
      ierr = STMatMult(ctx->st,0,nep->V[i],x);CHKERRQ(ierr);
      for (j=1;j<ctx->nmat;j++) {
        ierr = STMatMult(ctx->st,j,nep->V[i],t);CHKERRQ(ierr);
        ierr = VecAXPY(x,coeffs[j],t);CHKERRQ(ierr);
      }
      ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(nep->eig[i]);
      im = PetscImaginaryPart(nep->eig[i]);
#else
      re = nep->eig[i];
      im = 0.0;
#endif
      if (im!=0.0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD," %9f%+9f j %12g\n",(double)re,(double)im,(double)norm);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12f         %12g\n",(double)re,(double)norm);CHKERRQ(ierr);
      }
    }
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&t);CHKERRQ(ierr);
  }
/* /////// */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPDestroy_NLEIGS"
PetscErrorCode NEPDestroy_NLEIGS(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       k;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  ierr = STDestroy(&(ctx)->st);CHKERRQ(ierr);
  for (k=0;k<ctx->nmat;k++) {
    ierr = MatDestroy(&ctx->D[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(ctx->s,ctx->xi,ctx->beta,ctx->D);CHKERRQ(ierr);
  ierr = PetscFree(nep->data);CHKERRQ(ierr);
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

  nep->ops->solve          = NEPSolve_NLEIGS;
  nep->ops->setup          = NEPSetUp_NLEIGS;
  nep->ops->reset          = NEPReset_Default;
  nep->ops->destroy        = NEPDestroy_NLEIGS;
  PetscFunctionReturn(0);
}

