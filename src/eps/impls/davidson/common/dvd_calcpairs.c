/*
  SLEPc eigensolver: "davidson"

  Step: calc the best eigenpairs in the subspace V.

  For that, performs these steps:
    1) Update W <- A * V
    2) Update H <- V' * W
    3) Obtain eigenpairs of H
    4) Select some eigenpairs
    5) Compute the Ritz pairs of the selected ones

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

#include "davidson.h"
#include <slepcblaslapack.h>

PetscErrorCode dvd_calcpairs_proj(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_qz_start(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_qz_d(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_projeig_solve(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_selectPairs(dvdDashboard *d, PetscInt n);
PetscErrorCode dvd_calcpairs_X(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                               Vec *X);
PetscErrorCode dvd_calcpairs_Y(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                               Vec *Y);
PetscErrorCode dvd_calcpairs_res_0(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                                   Vec *R);
PetscErrorCode dvd_calcpairs_eig_res_0(dvdDashboard *d,PetscInt r_s,PetscInt r_e,Vec *R);
PetscErrorCode dvd_calcpairs_proj_res(dvdDashboard *d, PetscInt r_s,
                                      PetscInt r_e, Vec *R);
PetscErrorCode dvd_calcpairs_updateV0(dvdDashboard *d, DvdReduction *r,
                                      DvdMult_copy_func **sr);
PetscErrorCode dvd_calcpairs_updateV1(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateW0(dvdDashboard *d, DvdReduction *r,
                                      DvdMult_copy_func **sr);
PetscErrorCode dvd_calcpairs_updateW1(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateAV0(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateAV1(dvdDashboard *d, DvdReduction *r,
                                       DvdMult_copy_func **sr);
PetscErrorCode dvd_calcpairs_updateBV0(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateBV1(dvdDashboard *d, DvdReduction *r,
                                       DvdMult_copy_func **sr);
PETSC_STATIC_INLINE PetscErrorCode dvd_calcpairs_updateBV0_gen(dvdDashboard *d,Vec *real_BV,PetscInt *size_cX,Vec **BV,PetscInt *size_BV,PetscInt *max_size_BV,PetscBool BV_shift,PetscInt *cX_in_proj,DSMatType MT);

/**** Control routines ********************************************************/
#undef __FUNCT__  
#define __FUNCT__ "dvd_calcpairs_qz"
PetscErrorCode dvd_calcpairs_qz(dvdDashboard *d, dvdBlackboard *b,
                                EPSOrthType orth, IP ipI,
                                PetscInt cX_proj, PetscBool harm)
{
  PetscErrorCode ierr;
  PetscInt       i,max_cS;
  PetscBool      std_probl,her_probl,ind_probl,her_ind_probl;
  DSType         dstype;
  const char     *prefix;
  PetscErrorCode (*f)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  void           *ctx;

  PetscFunctionBegin;
  std_probl = DVD_IS(d->sEP, DVD_EP_STD)?PETSC_TRUE:PETSC_FALSE;
  her_probl = DVD_IS(d->sEP, DVD_EP_HERMITIAN)?PETSC_TRUE:PETSC_FALSE;
  ind_probl = DVD_IS(d->sEP, DVD_EP_INDEFINITE)?PETSC_TRUE:PETSC_FALSE;
  her_ind_probl = (her_probl || ind_probl)? PETSC_TRUE:PETSC_FALSE;

  /* Setting configuration constrains */
#if !defined(PETSC_USE_COMPLEX)
  /* if the last converged eigenvalue is complex its conjugate pair is also
     converged */
  b->max_nev = PetscMax(b->max_nev, d->nev+(her_probl && !d->B?0:1));
#else
  b->max_nev = PetscMax(b->max_nev, d->nev);
#endif
  b->max_size_proj = PetscMax(b->max_size_proj, b->max_size_V+cX_proj);
  d->size_real_V = b->max_size_V+b->max_nev;
  d->W_shift = d->B?PETSC_TRUE:PETSC_FALSE;
  d->size_real_W = harm?(b->max_size_V+(d->W_shift?b->max_nev:b->max_size_cP)):0;
  d->size_real_AV = b->max_size_V+b->max_size_cP;
  d->size_BDS = 0;
  if (d->B && her_ind_probl && (orth == EPS_ORTH_I || orth == EPS_ORTH_BOPT)) {
    d->size_real_BV = b->size_V; d->BV_shift = PETSC_TRUE;
    if (orth == EPS_ORTH_BOPT) d->size_BDS = d->eps->nds;
  } else if (d->B) { 
    d->size_real_BV = b->max_size_V + b->max_size_P; d->BV_shift = PETSC_FALSE;
  } else {
    d->size_real_BV = 0; d->BV_shift = PETSC_FALSE;
  }
  b->own_vecs+= d->size_real_V + d->size_real_W + d->size_real_AV +
                d->size_real_BV + d->size_BDS;
  b->own_scalars+= b->max_size_proj*b->max_size_proj*2*(std_probl?1:2) +
                                              /* H, G?, S, T? */
                   b->max_nev*b->max_nev*(her_ind_probl?0:(!d->B?1:2)) +
                                                /* cS?, cT? */
                   FromRealToScalar(d->size_real_V)*(ind_probl?1:0) + /* nBV */
                   FromRealToScalar(b->max_size_proj)*(ind_probl?1:0) + /* nBpX */
                   (d->eps->arbit_func? b->size_V*2 : 0); /* rr, ri */
  b->max_size_auxV = PetscMax(b->max_size_auxV, b->max_size_X);
                                                /* updateV0 */
  max_cS = PetscMax(b->max_size_X,cX_proj);
  b->max_size_auxS = PetscMax(PetscMax(
    b->max_size_auxS,
    b->max_size_X*b->max_size_proj*2*(std_probl?1:2) + /* updateAV1,BV1 */
      max_cS*b->max_nev*(her_ind_probl?0:(!d->B?1:2)) + /* updateV0,W0 */
                                                     /* SlepcReduction: in */
      PetscMax(
        b->max_size_X*b->max_size_proj*2*(std_probl?1:2) + /* updateAV1,BV1 */
          max_cS*b->max_nev*(her_ind_probl?0:(!d->B?1:2)), /* updateV0,W0 */
                                                    /* SlepcReduction: out */
        PetscMax(
          b->max_size_proj*b->max_size_proj, /* updateAV0,BV0 */
          b->max_size_proj+b->max_nev))), /* dvd_orth */
    std_probl?0:(b->max_size_proj*11+16) /* projeig */);
#if defined(PETSC_USE_COMPLEX)
  b->max_size_auxS = PetscMax(b->max_size_auxS, b->max_size_V);
                                           /* dvd_calcpairs_projeig_eig */
#endif

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    d->max_cX_in_proj = cX_proj;
    d->max_size_P = b->max_size_P;
    d->real_V = b->free_vecs; b->free_vecs+= d->size_real_V;
    if (harm) {
      d->real_W = b->free_vecs; b->free_vecs+= d->size_real_W;
    } else {
      d->real_W = PETSC_NULL;
    }
    d->real_AV = d->AV = b->free_vecs; b->free_vecs+= d->size_real_AV;
    d->max_size_proj = b->max_size_proj;
    d->real_H = b->free_scalars; b->free_scalars+= b->max_size_proj*b->max_size_proj;
    d->ldH = b->max_size_proj;
    d->S = b->free_scalars; b->free_scalars+= b->max_size_proj*b->max_size_proj;
    if (!her_ind_probl) {
      d->cS = b->free_scalars; b->free_scalars+= b->max_nev*b->max_nev;
      d->max_size_cS = d->ldcS = b->max_nev;
    } else {
      d->cS = PETSC_NULL;
      d->max_size_cS = d->ldcS = 0;
      d->orthoV_type = orth;
      if (ind_probl) {
        d->real_nBV = (PetscReal*)b->free_scalars; b->free_scalars+= FromRealToScalar(d->size_real_V);
        d->nBpX = (PetscReal*)b->free_scalars; b->free_scalars+= FromRealToScalar(d->max_size_proj);
      } else
        d->real_nBV = d->nBDS = d->nBpX = PETSC_NULL;
    }
    d->ipV = ipI;
    d->ipW = ipI;
    if (orth == EPS_ORTH_BOPT) {
      d->BDS = b->free_vecs; b->free_vecs+= d->eps->nds;
      for (i=0; i<d->eps->nds; i++) {
        ierr = MatMult(d->B, d->eps->defl[i], d->BDS[i]);CHKERRQ(ierr);
      }
    } else
      d->BDS = PETSC_NULL;
    if (d->B) {
      d->real_BV = b->free_vecs; b->free_vecs+= d->size_real_BV;
    } else {
      d->size_real_BV = 0;
      d->real_BV = PETSC_NULL;
      d->BV_shift = PETSC_FALSE;
    }
    if (!std_probl) {
      d->real_G = b->free_scalars; b->free_scalars+= b->max_size_proj*b->max_size_proj;
      d->T = b->free_scalars; b->free_scalars+= b->max_size_proj*b->max_size_proj;
    } else {
      d->real_G = PETSC_NULL;
      d->T = PETSC_NULL;
    }
    if (d->B && !her_ind_probl) {
      d->cT = b->free_scalars; b->free_scalars+= b->max_nev*b->max_nev;
      d->ldcT = b->max_nev;
    } else {
      d->cT = PETSC_NULL;
      d->ldcT = 0;
    }
    if (d->eps->arbit_func) {
      d->eps->rr = b->free_scalars; b->free_scalars+= b->size_V;
      d->eps->ri = b->free_scalars; b->free_scalars+= b->size_V;
    } else {
      d->eps->rr = PETSC_NULL;
      d->eps->ri = PETSC_NULL;
    }
    /* Create a DS if the method works with Schur decompositions */
    if (d->cS) {
      ierr = DSCreate(((PetscObject)d->eps->ds)->comm,&d->conv_ps);CHKERRQ(ierr);
      ierr = DSSetType(d->conv_ps,d->cT ? DSGNHEP : DSNHEP);CHKERRQ(ierr);
      /* Transfer as much as possible options from eps->ds to conv_ps */
      ierr = DSGetOptionsPrefix(d->eps->ds,&prefix);CHKERRQ(ierr);
      ierr = DSSetOptionsPrefix(d->conv_ps,prefix);CHKERRQ(ierr);
      ierr = DSSetFromOptions(d->conv_ps);CHKERRQ(ierr);
      ierr = DSGetEigenvalueComparison(d->eps->ds,&f,&ctx);CHKERRQ(ierr);
      ierr = DSSetEigenvalueComparison(d->conv_ps,f,ctx);CHKERRQ(ierr);
      ierr = DSAllocate(d->conv_ps,b->max_nev);CHKERRQ(ierr);
    } else {
      d->conv_ps = PETSC_NULL;
    }
    d->calcPairs = dvd_calcpairs_proj;
    d->calcpairs_residual = dvd_calcpairs_res_0;
    d->calcpairs_residual_eig = dvd_calcpairs_eig_res_0;
    d->calcpairs_proj_res = dvd_calcpairs_proj_res;
    d->calcpairs_selectPairs = dvd_calcpairs_selectPairs;
    d->ipI = ipI;
    /* Create and configure a DS for solving the projected problems */
    if (d->real_W) {    /* If we use harmonics */
      dstype = DSGNHEP;
    } else {
      if (ind_probl) {
        dstype = DSGHIEP;
      } else if (std_probl) {
        dstype = her_probl ? DSHEP : DSNHEP;
      } else {
        dstype = her_probl ? DSGHEP : DSGNHEP;
      }
    }
    d->ps = d->eps->ds;
    ierr = DSSetType(d->ps,dstype);CHKERRQ(ierr);
    ierr = DSAllocate(d->ps,d->max_size_proj);CHKERRQ(ierr);

    DVD_FL_ADD(d->startList, dvd_calcpairs_qz_start);
    DVD_FL_ADD(d->destroyList, dvd_calcpairs_qz_d);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_calcpairs_qz_start"
PetscErrorCode dvd_calcpairs_qz_start(dvdDashboard *d)
{
  PetscBool her_probl,ind_probl,her_ind_probl;
  PetscInt  i;

  PetscFunctionBegin;
  her_probl = DVD_IS(d->sEP, DVD_EP_HERMITIAN)?PETSC_TRUE:PETSC_FALSE;
  ind_probl = DVD_IS(d->sEP, DVD_EP_INDEFINITE)?PETSC_TRUE:PETSC_FALSE;
  her_ind_probl = (her_probl || ind_probl)? PETSC_TRUE:PETSC_FALSE;

  d->size_V = 0;
  d->V = d->real_V;
  d->cX = d->real_V;
  d->size_cX = 0;
  d->max_size_V = d->size_real_V;
  d->W = d->real_W;
  d->max_size_W = d->size_real_W;
  d->size_W = 0;
  d->size_AV = 0;
  d->AV = d->real_AV;
  d->max_size_AV = d->size_real_AV;
  d->size_H = 0;
  d->H = d->real_H;
  if (d->cS) for (i=0; i<d->max_size_cS*d->max_size_cS; i++) d->cS[i] = 0.0;
  d->size_BV = 0;
  d->BV = d->real_BV;
  d->max_size_BV = d->size_real_BV;
  d->size_G = 0;
  d->G = d->real_G;
  if (d->cT) for (i=0; i<d->max_size_cS*d->max_size_cS; i++) d->cT[i] = 0.0;
  d->cY = d->B && !her_ind_probl ? d->W : PETSC_NULL;
  d->BcX = d->orthoV_type == EPS_ORTH_I && d->B && her_probl ? d->BcX : PETSC_NULL;
  d->size_cY = 0;
  d->size_BcX = 0;
  d->cX_in_V = d->cX_in_H = d->cX_in_G = d->cX_in_W = d->cX_in_AV = d->cX_in_BV = 0;
  d->nBV = d->nBcX = d->real_nBV;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_calcpairs_qz_d"
PetscErrorCode dvd_calcpairs_qz_d(dvdDashboard *d)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DSDestroy(&d->conv_ps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_proj"
PetscErrorCode dvd_calcpairs_proj(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  DvdReduction    r;
#define MAX_OPS 7
  DvdReductionChunk
                  ops[MAX_OPS];
  DvdMult_copy_func
                  sr[MAX_OPS], *sr0 = sr;
  PetscInt        size_in, i;
  PetscScalar     *in = d->auxS, *out;
  PetscBool       stdp;

  PetscFunctionBegin;
  stdp = DVD_IS(d->sEP, DVD_EP_STD)?PETSC_TRUE:PETSC_FALSE;
  size_in =
    (d->size_cX+d->V_tra_s-d->cX_in_H)*d->V_tra_s*(d->cT?2:(d->cS?1:0)) + /* updateV0,W0 */
    (d->size_H*(d->V_new_e-d->V_new_s)*2+
      (d->V_new_e-d->V_new_s)*(d->V_new_e-d->V_new_s))*(!stdp?2:1); /* updateAV1,BV1 */
    
  out = in+size_in;

  /* Check consistency */
  if (2*size_in > d->size_auxS) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  /* Prepare reductions */
  ierr = SlepcAllReduceSumBegin(ops, MAX_OPS, in, out, size_in, &r,
                                ((PetscObject)d->V[0])->comm);CHKERRQ(ierr);
  /* Allocate size_in */
  d->auxS+= size_in;
  d->size_auxS-= size_in;

  /* Update AV, BV, W and the projected matrices */
  /* 1. S <- S*MT */
  ierr = dvd_calcpairs_updateV0(d, &r, &sr0);CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateW0(d, &r, &sr0);CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateAV0(d);CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateBV0(d);CHKERRQ(ierr);
  /* 2. V <- orth(V, V_new) */ 
  ierr = dvd_calcpairs_updateV1(d);CHKERRQ(ierr);
  /* 3. AV <- [AV A * V(V_new_s:V_new_e-1)] */
  /* Check consistency */
  if (d->size_AV != d->V_new_s) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");
  for (i=d->V_new_s; i<d->V_new_e; i++) {
    ierr = MatMult(d->A, d->V[i], d->AV[i]);CHKERRQ(ierr);
  }
  d->size_AV = d->V_new_e;
  /* 4. BV <- [BV B * V(V_new_s:V_new_e-1)] */
  if (d->B && d->orthoV_type != EPS_ORTH_BOPT) {
    /* Check consistency */
    if (d->size_BV != d->V_new_s) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");
    for (i=d->V_new_s; i<d->V_new_e; i++) {
      ierr = MatMult(d->B, d->V[i], d->BV[i]);CHKERRQ(ierr);
    }
    d->size_BV = d->V_new_e;
  }
  /* 5 <- W <- [W f(AV,BV)] */
  ierr = dvd_calcpairs_updateW1(d);CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateAV1(d, &r, &sr0);CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateBV1(d, &r, &sr0);CHKERRQ(ierr);

  /* Deallocate size_in */
  d->auxS-= size_in;
  d->size_auxS+= size_in;

  /* Do reductions */
  ierr = SlepcAllReduceSumEnd(&r);CHKERRQ(ierr);

  /* Perform the transformation on the projected problem */
  if (d->calcpairs_proj_trans) {
    ierr = d->calcpairs_proj_trans(d);CHKERRQ(ierr);
  }

  d->V_tra_s = d->V_tra_e = 0;
  d->V_new_s = d->V_new_e;

  /* Solve the projected problem */
  if (d->size_H>0) {
    ierr = dvd_calcpairs_projeig_solve(d);CHKERRQ(ierr);
  }

  /* Check consistency */
  if (d->size_V != d->V_new_e || d->size_V+d->cX_in_H != d->size_H || d->cX_in_V != d->cX_in_H ||
      d->size_V != d->size_AV || d->cX_in_H != d->cX_in_AV ||
        (DVD_ISNOT(d->sEP, DVD_EP_STD) && (
          d->size_V+d->cX_in_G != d->size_G || d->cX_in_H != d->cX_in_G ||
          d->size_H != d->size_G || (d->BV && (
            d->size_V != d->size_BV || d->cX_in_H != d->cX_in_BV)))) ||
      (d->W && d->size_W != d->size_V)) {
    SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");
  }
  PetscFunctionReturn(0);
#undef MAX_OPS
}

/**** Basic routines **********************************************************/

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateV0"
/* auxV: V_tra_s, DvdMult_copy_func: 1 */
PetscErrorCode dvd_calcpairs_updateV0(dvdDashboard *d, DvdReduction *r,
                                      DvdMult_copy_func **sr)
{
  PetscErrorCode  ierr;
  PetscInt        rm,i,ld;
  PetscScalar     *pQ;

  PetscFunctionBegin;
  if (d->V_tra_s == 0 && d->V_tra_e == 0) PetscFunctionReturn(0);

  /* Update nBcX and nBV */
  if (d->nBcX && d->nBpX && d->nBV) {
    d->nBV+= d->V_tra_s;
    for (i=0; i<d->V_tra_s; i++) d->nBcX[d->size_cX+i] = d->nBpX[i];
    for (i=d->V_tra_s; i<d->V_tra_e; i++) d->nBV[i-d->V_tra_s] = d->nBpX[i];
  }

  /* cX <- [cX V*MT(0:V_tra_s-1)], V <- V*MT(V_tra_s:V_tra_e) */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->real_V,&d->size_cX,&d->V,&d->size_V,&d->max_size_V,PETSC_TRUE,&d->cX_in_V,DS_MAT_Q);CHKERRQ(ierr);

  /* Udpate cS for standard problems */
  if (d->cS && !d->cT && !d->cY && (d->V_tra_s > d->max_cX_in_proj || d->size_cX >= d->nev)) {
    /* Check consistency */
    if (d->size_cS+d->V_tra_s != d->size_cX) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

    /* auxV <- AV * ps.Q(0:V_tra_e-1) */
    rm = d->size_cX>=d->nev?0:d->max_cX_in_proj;
    ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
    ierr = DSGetArray(d->ps,DS_MAT_Q,&pQ);CHKERRQ(ierr);
    ierr = SlepcUpdateVectorsZ(d->auxV,0.0,1.0,d->AV-d->cX_in_AV,d->size_AV+d->cX_in_AV,pQ,ld,d->size_MT,d->V_tra_s-rm);CHKERRQ(ierr);
    ierr = DSRestoreArray(d->ps,DS_MAT_Q,&pQ);CHKERRQ(ierr);

    /* cS(:, size_cS:) <- cX' * auxV */
    ierr = VecsMultS(&d->cS[d->ldcS*d->size_cS], 0, d->ldcS, d->cX, 0, d->size_cX-rm, d->auxV, 0, d->V_tra_s-rm, r, (*sr)++);CHKERRQ(ierr);
    d->size_cS+= d->V_tra_s-rm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateV1"
/* auxS: size_cX+V_new_e+1 */
PetscErrorCode dvd_calcpairs_updateV1(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  Vec             *cX = d->BcX? d->BcX : ((d->cY && !d->W)? d->cY : d->cX);

  PetscFunctionBegin;
  if (d->V_new_s == d->V_new_e) PetscFunctionReturn(0);

  /* Check consistency */
  if (d->size_V != d->V_new_s) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  /* V <- gs([cX V(0:V_new_s-1)], V(V_new_s:V_new_e-1)) */
  if (d->orthoV_type == EPS_ORTH_BOPT) {
    ierr = dvd_BorthV_faster(d->ipV,d->eps->defl,d->BDS,d->nBDS,d->eps->nds,d->cX,d->real_BV,d->nBcX,d->size_cX,d->V,d->BV,d->nBV,d->V_new_s,d->V_new_e,d->auxS,d->eps->rand);CHKERRQ(ierr);
    d->size_BV = d->V_new_e;
  } else if (DVD_IS(d->sEP, DVD_EP_INDEFINITE)) {
    ierr = dvd_BorthV_stable(d->ipV,d->eps->defl,d->nBDS,d->eps->nds,d->cX,d->nBcX,d->size_cX,d->V,d->nBV,d->V_new_s,d->V_new_e,d->auxS,d->eps->rand);CHKERRQ(ierr);
  } else {
    ierr = dvd_orthV(d->ipV,d->eps->defl,d->eps->nds,cX,d->size_cX,d->V,d->V_new_s,d->V_new_e,d->auxS,d->eps->rand);CHKERRQ(ierr);
  }
  d->size_V = d->V_new_e;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateW0"
/* auxV: V_tra_s, DvdMult_copy_func: 2 */
PetscErrorCode dvd_calcpairs_updateW0(dvdDashboard *d, DvdReduction *r,
                                      DvdMult_copy_func **sr)
{
  PetscErrorCode  ierr;
  PetscInt        rm,ld;
  PetscScalar     *pQ;

  PetscFunctionBegin;
  if (d->V_tra_s == 0 && d->V_tra_e == 0) PetscFunctionReturn(0);

  /* cY <- [cY W*ps.Z(0:V_tra_s-1)], W <- W*ps.Z(V_tra_s:V_tra_e) */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->real_W,&d->size_cY,&d->W,&d->size_W,&d->max_size_W,d->W_shift,&d->cX_in_W,DS_MAT_Z);CHKERRQ(ierr);

  /* Udpate cS and cT */
  if (d->cT && (d->V_tra_s > d->max_cX_in_proj || d->size_cX >= d->nev)) {
    /* Check consistency */
    if (d->size_cS+d->V_tra_s != d->size_cX || (d->W && d->size_cY != d->size_cX)) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

    ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
    ierr = DSGetArray(d->ps,DS_MAT_Q,&pQ);CHKERRQ(ierr);
    /* auxV <- AV * ps.Q(0:V_tra_e-1) */
    rm = d->size_cX>=d->nev?0:d->max_cX_in_proj;
    ierr = SlepcUpdateVectorsZ(d->auxV,0.0,1.0,d->AV-d->cX_in_H,d->size_AV-d->cX_in_H,pQ,ld,d->size_MT,d->V_tra_s-rm);CHKERRQ(ierr);

    /* cS(:, size_cS:) <- cY' * auxV */
    ierr = VecsMultS(&d->cS[d->ldcS*d->size_cS], 0, d->ldcS, d->cY?d->cY:d->cX, 0, d->size_cX-rm, d->auxV, 0, d->V_tra_s-rm, r, (*sr)++);CHKERRQ(ierr);

    /* auxV <- BV * ps.Q(0:V_tra_e-1) */
    ierr = SlepcUpdateVectorsZ(d->auxV,0.0,1.0,d->BV-d->cX_in_H,d->size_BV-d->cX_in_H,pQ,ld,d->size_MT,d->V_tra_s-rm);CHKERRQ(ierr);
    ierr = DSRestoreArray(d->ps,DS_MAT_Q,&pQ);CHKERRQ(ierr);

    /* cT(:, size_cS:) <- cY' * auxV */
    ierr = VecsMultS(&d->cT[d->ldcS*d->size_cS], 0, d->ldcS, d->cY?d->cY:d->cX, 0, d->size_cX-rm, d->auxV, 0, d->V_tra_s-rm, r, (*sr)++);CHKERRQ(ierr);
    
    d->size_cS+= d->V_tra_s-rm;
    d->size_cT+= d->V_tra_s-rm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateW1"
/* auxS: size_cX+V_new_e+1 */
PetscErrorCode dvd_calcpairs_updateW1(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  Vec             *cY = d->cY?d->cY:d->cX;

  PetscFunctionBegin;
  if (!d->W || d->V_new_s == d->V_new_e) PetscFunctionReturn(0);

  /* Check consistency */
  if (d->size_W != d->V_new_s) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  /* Update W */
  ierr = d->calcpairs_W(d);CHKERRQ(ierr);

  /* W <- gs([cY W(0:V_new_s-1)], W(V_new_s:V_new_e-1)) */
  ierr = dvd_orthV(d->ipW, PETSC_NULL, 0, cY, d->size_cX, d->W-d->cX_in_W, d->V_new_s+d->cX_in_W, d->V_new_e+d->cX_in_W, d->auxS, d->eps->rand);CHKERRQ(ierr);
  d->size_W = d->V_new_e;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateAV0"
/* auxS: size_H*(V_tra_e-V_tra_s) */
PetscErrorCode dvd_calcpairs_updateAV0(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscInt        cMT,tra_s,ld;
  PetscScalar     *pQ,*pZ;

  PetscFunctionBegin;
  if (d->V_tra_s == 0 && d->V_tra_e == 0) PetscFunctionReturn(0);

  /* AV(V_tra_s-cp-1:) = cAV*ps.Q(V_tra_s:) */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->real_AV,PETSC_NULL,&d->AV,&d->size_AV,&d->max_size_AV,PETSC_FALSE,&d->cX_in_AV,DS_MAT_Q);CHKERRQ(ierr);
  tra_s = PetscMax(d->V_tra_s-d->max_cX_in_proj,0);
  cMT = d->V_tra_e - tra_s;

  /* Update H <- ps.Z(tra_s)' * (H * ps.Q(tra_s:)) */
  ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
  ierr = DSGetArray(d->ps,DS_MAT_Q,&pQ);CHKERRQ(ierr);
  if (d->W) {
    ierr = DSGetArray(d->ps,DS_MAT_Z,&pZ);CHKERRQ(ierr);
  } else {
    pZ = pQ;
  }
  ierr = SlepcDenseMatProdTriang(d->auxS,0,d->ldH,d->H,d->sH,d->ldH,d->size_H,d->size_H,PETSC_FALSE,&pQ[ld*tra_s],0,ld,d->size_MT,cMT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = SlepcDenseMatProdTriang(d->H,d->sH,d->ldH,&pZ[ld*tra_s],0,ld,d->size_MT,cMT,PETSC_TRUE,d->auxS,0,d->ldH,d->size_H,cMT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DSRestoreArray(d->ps,DS_MAT_Q,&pQ);CHKERRQ(ierr);
  if (d->W) {
    ierr = DSRestoreArray(d->ps,DS_MAT_Z,&pZ);CHKERRQ(ierr);
  }
  d->size_H = cMT;
  d->cX_in_H = d->cX_in_AV;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateAV1"
/* DvdMult_copy_func: 2 */
PetscErrorCode dvd_calcpairs_updateAV1(dvdDashboard *d, DvdReduction *r,
                                       DvdMult_copy_func **sr)
{
  PetscErrorCode  ierr;
  Vec             *W = d->W?d->W:d->V;

  PetscFunctionBegin;
  if (d->V_new_s == d->V_new_e) PetscFunctionReturn(0);

  /* Check consistency */
  if (d->size_H != d->V_new_s+d->cX_in_H || d->size_V != d->V_new_e) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  /* H = [H               W(old)'*AV(new);
          W(new)'*AV(old) W(new)'*AV(new) ],
     being old=0:V_new_s-1, new=V_new_s:V_new_e-1 */
  ierr = VecsMultS(d->H,d->sH,d->ldH,W-d->cX_in_H,d->V_new_s+d->cX_in_H, d->V_new_e+d->cX_in_H, d->AV-d->cX_in_H,d->V_new_s+d->cX_in_H,d->V_new_e+d->cX_in_H, r, (*sr)++);CHKERRQ(ierr);
  d->size_H = d->V_new_e+d->cX_in_H;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateBV0"
/* auxS: max(BcX*(size_cX+V_new_e+1), size_G*(V_tra_e-V_tra_s)) */
PetscErrorCode dvd_calcpairs_updateBV0(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscInt        cMT,tra_s,i,ld;
  PetscBool       lindep;
  PetscReal       norm;
  PetscScalar     *pQ,*pZ;

  PetscFunctionBegin;
  if (d->V_tra_s == 0 && d->V_tra_e == 0) PetscFunctionReturn(0);

  /* BV <- BV*MT */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->real_BV,PETSC_NULL,&d->BV,&d->size_BV,&d->max_size_BV,d->BV_shift,&d->cX_in_BV,DS_MAT_Q);CHKERRQ(ierr);

  /* If BcX, BcX <- orth(BcX) */
  if (d->BcX) {
    for (i=0; i<d->V_tra_s; i++) {
      ierr = IPOrthogonalize(d->ipI, 0, PETSC_NULL, d->size_BcX+i, PETSC_NULL,
                             d->BcX, d->BcX[d->size_BcX+i], PETSC_NULL,
                             &norm, &lindep);CHKERRQ(ierr);
      if (lindep) SETERRQ(PETSC_COMM_SELF,1, "Error during orth(BcX, B*cX(new))");
      ierr = VecScale(d->BcX[d->size_BcX+i], 1.0/norm);CHKERRQ(ierr);
    }
    d->size_BcX+= d->V_tra_s;
  }

  /* Update G <- ps.Z' * (G * ps.Q) */
  if (d->G) {
    tra_s = PetscMax(d->V_tra_s-d->max_cX_in_proj,0);
    cMT = d->V_tra_e - tra_s;
    ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
    ierr = DSGetArray(d->ps,DS_MAT_Q,&pQ);CHKERRQ(ierr);
    if (d->W) {
      ierr = DSGetArray(d->ps,DS_MAT_Z,&pZ);CHKERRQ(ierr);
    } else {
      pZ = pQ;
    }
    ierr = SlepcDenseMatProdTriang(d->auxS,0,d->ldH,d->G,d->sG,d->ldH,d->size_G,d->size_G,PETSC_FALSE,&pQ[ld*tra_s],0,ld,d->size_MT,cMT,PETSC_FALSE);CHKERRQ(ierr);
    ierr = SlepcDenseMatProdTriang(d->G,d->sG,d->ldH,&pZ[ld*tra_s],0,ld,d->size_MT,cMT,PETSC_TRUE,d->auxS,0,d->ldH,d->size_G,cMT,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DSRestoreArray(d->ps,DS_MAT_Q,&pQ);CHKERRQ(ierr);
    if (d->W) {
      ierr = DSRestoreArray(d->ps,DS_MAT_Z,&pZ);CHKERRQ(ierr);
    }
    d->size_G = cMT;
    d->cX_in_G = d->cX_in_V;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateBV1"
/* DvdMult_copy_func: 2 */
PetscErrorCode dvd_calcpairs_updateBV1(dvdDashboard *d, DvdReduction *r,
                                       DvdMult_copy_func **sr)
{
  PetscErrorCode  ierr;
  Vec             *W = d->W?d->W:d->V, *BV = d->BV?d->BV:d->V;

  PetscFunctionBegin;
  if (!d->G || d->V_new_s == d->V_new_e) PetscFunctionReturn(0);

  /* G = [G               W(old)'*BV(new);
          W(new)'*BV(old) W(new)'*BV(new) ],
     being old=0:V_new_s-1, new=V_new_s:V_new_e-1 */
  ierr = VecsMultS(d->G,d->sG,d->ldH,W-d->cX_in_G,d->V_new_s+d->cX_in_G,d->V_new_e+d->cX_in_G,BV-d->cX_in_G,d->V_new_s+d->cX_in_G,d->V_new_e+d->cX_in_G,r,(*sr)++);CHKERRQ(ierr);
  d->size_G = d->V_new_e+d->cX_in_G;
  PetscFunctionReturn(0);
}

/* in complex, d->size_H real auxiliar values are needed */
#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_projeig_solve"
PetscErrorCode dvd_calcpairs_projeig_solve(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscScalar     *A;
  PetscInt        ld,i;

  PetscFunctionBegin;
  ierr = DSSetDimensions(d->ps,d->size_H,PETSC_IGNORE,0,0);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
  ierr = DSGetArray(d->ps,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = SlepcDenseCopyTriang(A,0,ld,d->H,d->sH,d->ldH,d->size_H,d->size_H);CHKERRQ(ierr);
  ierr = DSRestoreArray(d->ps,DS_MAT_A,&A);CHKERRQ(ierr);
  if (d->G) {
    ierr = DSGetArray(d->ps,DS_MAT_B,&A);CHKERRQ(ierr);
    ierr = SlepcDenseCopyTriang(A,0,ld,d->G,d->sG,d->ldH,d->size_H,d->size_H);CHKERRQ(ierr);
    ierr = DSRestoreArray(d->ps,DS_MAT_B,&A);CHKERRQ(ierr);
  }
  /* Set the signature on projected matrix B */
  if (DVD_IS(d->sEP, DVD_EP_INDEFINITE)) {
    ierr = DSGetArray(d->ps,DS_MAT_B,&A);CHKERRQ(ierr);
    ierr = PetscMemzero(A,sizeof(PetscScalar)*d->size_H*ld);CHKERRQ(ierr);
    for (i=0; i<d->size_H; i++) {
      A[i+ld*i] = d->nBV[i];
    }
    ierr = DSRestoreArray(d->ps,DS_MAT_B,&A);CHKERRQ(ierr);
  }
  ierr = DSSetState(d->ps,DS_STATE_RAW);CHKERRQ(ierr);
  ierr = DSSolve(d->ps,d->eigr-d->cX_in_H,d->eigi-d->cX_in_H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_apply_arbitrary_func"
PetscErrorCode dvd_calcpairs_apply_arbitrary_func(dvdDashboard *d,PetscInt r_s,PetscInt r_e,PetscScalar **rr_,PetscScalar **ri_)
{
  PetscInt        i,k,ld;
  PetscScalar     *pX,*rr,*ri,ar,ai;
  Vec             *X = d->auxV,xr,xi;
  PetscErrorCode  ierr;
#if !defined(PETSC_USE_COMPLEX)
  PetscInt        j;
#endif
  
  PetscFunctionBegin;
  /* Quick exit without neither arbitrary selection nor harmonic extraction */
  if (!d->eps->arbit_func && !d->calcpairs_eig_backtrans) {
    *rr_ = d->eigr-d->cX_in_H;
    *ri_ = d->eigi-d->cX_in_H;
    PetscFunctionReturn(0);
  }

  /* Quick exit without arbitrary selection, but with harmonic extraction */
  if (!d->eps->arbit_func && d->calcpairs_eig_backtrans) {
    *rr_ = rr = d->auxS;
    *ri_ = ri = d->auxS+r_e-r_s;
    for (i=r_s; i<r_e; i++) {
      ierr = d->calcpairs_eig_backtrans(d,d->eigr[i],d->eigi[i],&rr[i-r_s],&ri[i-r_s]);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
  *rr_ = rr = d->eps->rr + d->eps->nconv;
  *ri_ = ri = d->eps->ri + d->eps->nconv;
  for (i=r_s; i<r_e; i++) {
    k = i;
    ierr = DSVectors(d->ps,DS_MAT_X,&k,PETSC_NULL);CHKERRQ(ierr);
    ierr = DSNormalize(d->ps,DS_MAT_X,i);CHKERRQ(ierr);
    ierr = DSGetArray(d->ps,DS_MAT_X,&pX);CHKERRQ(ierr);
    ierr = dvd_improvex_compute_X(d,i,k+1,X,pX,ld);CHKERRQ(ierr);
    ierr = DSRestoreArray(d->ps,DS_MAT_X,&pX);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    if (d->nX[i] != 1.0) {
      for (j=i; j<k+1; j++) {
        ierr = VecScale(X[j-i],1/d->nX[i]);CHKERRQ(ierr);
      }
    }
    xr = X[0];
    xi = X[1];
    if (i == k) {
      ierr = VecZeroEntries(xi);CHKERRQ(ierr);
    }
#else
    xr = X[0];
    xi = PETSC_NULL;
    if (d->nX[i] != 1.0) {
      ierr = VecScale(xr,1.0/d->nX[i]);CHKERRQ(ierr);
    }
#endif
    if (d->calcpairs_eig_backtrans) {
      ierr = d->calcpairs_eig_backtrans(d,d->eigr[i],d->eigi[i],&ar,&ai);CHKERRQ(ierr);
    } else {
      ar = d->eigr[i];
      ai = d->eigi[i];
    }
    ierr = (d->eps->arbit_func)(ar,ai,xr,xi,&rr[i-r_s],&ri[i-r_s],d->eps->arbit_ctx);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    if (i != k) {
      rr[i+1-r_s] = rr[i-r_s];
      ri[i+1-r_s] = ri[i-r_s];
      i++;
    }
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_selectPairs"
PetscErrorCode dvd_calcpairs_selectPairs(dvdDashboard *d, PetscInt n)
{
  PetscInt        k;
  PetscScalar     *rr,*ri;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  n = PetscMin(n,d->size_H-d->cX_in_H);
  /* Put the best n pairs at the beginning. Useful for restarting */
  ierr = DSSetDimensions(d->ps,PETSC_IGNORE,PETSC_IGNORE,d->cX_in_H,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = dvd_calcpairs_apply_arbitrary_func(d,d->cX_in_H,d->size_H,&rr,&ri);CHKERRQ(ierr);
  k = n;
  ierr = DSSort(d->ps,d->eigr-d->cX_in_H,d->eigi-d->cX_in_H,rr,ri,&k);CHKERRQ(ierr);
  /* Put the best pair at the beginning. Useful to check its residual */
#if !defined(PETSC_USE_COMPLEX)
  if (n != 1 && (n != 2 || d->eigi[0] == 0.0))
#else
  if (n != 1)
#endif
  {
    ierr = dvd_calcpairs_apply_arbitrary_func(d,d->cX_in_H,d->size_H,&rr,&ri);CHKERRQ(ierr);
    k = 1;
    ierr = DSSort(d->ps,d->eigr-d->cX_in_H,d->eigi-d->cX_in_H,rr,ri,&k);CHKERRQ(ierr);
  }
  if (d->calcpairs_eigs_trans) {
    ierr = d->calcpairs_eigs_trans(d);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_res_0"
/* Compute the residual vectors R(i) <- (AV - BV*eigr(i))*pX(i), and also
   the norm associated to the Schur pair, where i = r_s..r_e
*/
PetscErrorCode dvd_calcpairs_res_0(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                             Vec *R)
{
  PetscInt        i,ldpX;
  PetscScalar     *pX;
  PetscErrorCode  ierr;
  Vec             *BV = d->BV?d->BV:d->V;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(d->ps,&ldpX);CHKERRQ(ierr);
  ierr = DSGetArray(d->ps,DS_MAT_Q,&pX);CHKERRQ(ierr);
  for (i=r_s; i<r_e; i++) {
    /* nX(i) <- ||X(i)|| */
    if (d->correctXnorm) {
      /* R(i) <- V*pX(i) */
      ierr = SlepcUpdateVectorsZ(&R[i-r_s],0.0,1.0,&d->V[-d->cX_in_H],d->size_V+d->cX_in_H,&pX[ldpX*(i+d->cX_in_H)],ldpX,d->size_H,1);CHKERRQ(ierr);
      ierr = VecNorm(R[i-r_s],NORM_2,&d->nX[i]);CHKERRQ(ierr);
    } else {
      d->nX[i] = 1.0;
    }
    /* R(i-r_s) <- AV*pX(i) */
    ierr = SlepcUpdateVectorsZ(&R[i-r_s],0.0,1.0,&d->AV[-d->cX_in_H],d->size_AV+d->cX_in_H,&pX[ldpX*(i+d->cX_in_H)],ldpX,d->size_H,1);CHKERRQ(ierr);
    /* R(i-r_s) <- R(i-r_s) - eigr(i)*BV*pX(i) */
    ierr = SlepcUpdateVectorsZ(&R[i-r_s],1.0,-d->eigr[i+d->cX_in_H],&BV[-d->cX_in_H],d->size_V+d->cX_in_H,&pX[ldpX*(i+d->cX_in_H)],ldpX,d->size_H,1);CHKERRQ(ierr);
  }
  ierr = DSRestoreArray(d->ps,DS_MAT_Q,&pX);CHKERRQ(ierr);
  ierr = d->calcpairs_proj_res(d, r_s, r_e, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_proj_res"
PetscErrorCode dvd_calcpairs_proj_res(dvdDashboard *d, PetscInt r_s,
                                      PetscInt r_e, Vec *R)
{
  PetscInt        i;
  PetscErrorCode  ierr;
  PetscBool       lindep;
  Vec             *cX;

  PetscFunctionBegin;
  /* If exists the BcX, R <- orth(BcX, R), nR[i] <- ||R[i]|| */
  if (d->BcX)
    cX = d->BcX;

  /* If exists left subspace, R <- orth(cY, R), nR[i] <- ||R[i]|| */
  else if (d->cY)
    cX = d->cY;

  /* If fany configurations, R <- orth(cX, R), nR[i] <- ||R[i]|| */
  else if (!(DVD_IS(d->sEP, DVD_EP_STD) && DVD_IS(d->sEP, DVD_EP_HERMITIAN)))
    cX = d->cX;

  /* Otherwise, nR[i] <- ||R[i]|| */
  else
    cX = PETSC_NULL;

  if (cX) {
    if (cX && d->orthoV_type == EPS_ORTH_BOPT) {
      Vec auxV;
      ierr = VecDuplicate(d->auxV[0],&auxV);CHKERRQ(ierr);
      for (i=0; i<r_e-r_s; i++) {
        ierr = IPBOrthogonalize(d->ipV,d->eps->nds,d->eps->defl,d->BDS,d->nBDS,d->size_cX,PETSC_NULL,d->cX,d->real_BV,d->nBcX,R[i],auxV,PETSC_NULL,&d->nR[r_s+i],&lindep);CHKERRQ(ierr);
      }
      ierr = VecDestroy(&auxV);CHKERRQ(ierr);
    } else if (DVD_IS(d->sEP, DVD_EP_INDEFINITE)) {
      for (i=0; i<r_e-r_s; i++) {
        ierr = IPPseudoOrthogonalize(d->ipV,d->size_cX,cX,d->nBcX,R[i],PETSC_NULL,&d->nR[r_s+i],&lindep);CHKERRQ(ierr);
      }
    } else {
      for (i=0; i<r_e-r_s; i++) {
        ierr = IPOrthogonalize(d->ipI,0,PETSC_NULL,d->size_cX,PETSC_NULL,cX,R[i],PETSC_NULL,&d->nR[r_s+i],&lindep);CHKERRQ(ierr);
      }
    }
    if (lindep || (PetscAbs(d->nR[r_s+i]) < PETSC_MACHINE_EPSILON)) {
      ierr = PetscInfo2(d->eps,"The computed eigenvector residual %D is too low, %G!\n",r_s+i,d->nR[r_s+i]);CHKERRQ(ierr);
    }
  }
  if (!cX || (cX && d->orthoV_type == EPS_ORTH_BOPT)) {
    for (i=0;i<r_e-r_s;i++) {
      ierr = VecNormBegin(R[i],NORM_2,&d->nR[r_s+i]);CHKERRQ(ierr);
    }
    for (i=0;i<r_e-r_s;i++) {
      ierr = VecNormEnd(R[i],NORM_2,&d->nR[r_s+i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_eig_res_0"
/* Compute the residual vectors R(i) <- (AV - BV*eigr(i))*pX(i), and also
   the norm associated to the eigenpair, where i = r_s..r_e
   R, vectors of Vec of size r_e-r_s,
   auxV, PetscMax(r_e+cX_in_H, 2*(r_e-r_s)) vectors,
   auxS, auxiliar vector of size (d->size_cX+r_e)^2+6(d->size_cX+r_e)+(r_e-r_s)*d->size_H
*/
PetscErrorCode dvd_calcpairs_eig_res_0(dvdDashboard *d,PetscInt r_s,PetscInt r_e,Vec *R)
{
  PetscInt        i,size_in,n,ld,ldc,k;
  PetscErrorCode  ierr;
  Vec             *Bx;
  PetscScalar     *cS,*cT,*pcX,*pX,*pX0;
  DvdReduction    r;
  DvdReductionChunk
                  ops[2];
  DvdMult_copy_func
                  sr[2];
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar     b[8];
  Vec             X[4];
#endif

  PetscFunctionBegin;
  /* Quick return */
  if (!d->cS) PetscFunctionReturn(0);

  size_in = (d->size_cX+r_e)*(d->cX_in_AV+r_e)*(d->cT?2:1);
  /* Check consistency */
  if (d->size_auxV < PetscMax(2*(r_e-r_s),d->cX_in_AV+r_e) || d->size_auxS < PetscMax(d->size_H*(r_e-r_s) /* pX0 */, 2*size_in /* SlepcAllReduceSum */)) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  n = d->size_cX+r_e;
  ierr = DSSetDimensions(d->conv_ps,n,PETSC_IGNORE,0,0);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(d->conv_ps,&ldc);CHKERRQ(ierr);
  ierr = DSGetArray(d->conv_ps,DS_MAT_A,&cS);CHKERRQ(ierr);
  ierr = SlepcDenseCopyTriang(cS,0,ldc,d->cS,0,d->ldcS,d->size_cS,d->size_cS);CHKERRQ(ierr);
  if (d->cT) {
    ierr = DSGetArray(d->conv_ps,DS_MAT_B,&cT);CHKERRQ(ierr);
    ierr = SlepcDenseCopyTriang(cT,0,ldc,d->cT,0,d->ldcT,d->size_cS,d->size_cS);CHKERRQ(ierr);
  }
  ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
  ierr = DSGetArray(d->ps,DS_MAT_Q,&pX);CHKERRQ(ierr);
  /* Prepare reductions */
  ierr = SlepcAllReduceSumBegin(ops,2,d->auxS,d->auxS+size_in,size_in,&r,((PetscObject)d->V[0])->comm);CHKERRQ(ierr);
  /* auxV <- AV * pX(0:r_e+cX_in_H) */
  ierr = SlepcUpdateVectorsZ(d->auxV,0.0,1.0,d->AV-d->cX_in_AV,d->size_AV+d->cX_in_AV,pX,ld,d->size_H,d->cX_in_AV+r_e);CHKERRQ(ierr);
  /* cS(:, size_cS:) <- cX' * auxV */
  ierr = VecsMultS(&cS[ldc*d->size_cS],0,ldc,d->cY?d->cY:d->cX,0,d->size_cX+r_e,d->auxV,0,d->cX_in_AV+r_e,&r,&sr[0]);CHKERRQ(ierr);

  if (d->cT) {
    /* R <- BV * pX(0:r_e+cX_in_H) */
    ierr = SlepcUpdateVectorsZ(d->auxV,0.0,1.0,d->BV-d->cX_in_BV,d->size_BV+d->cX_in_BV,pX,ld,d->size_G,d->cX_in_BV+r_e);CHKERRQ(ierr);
    /* cT(:, size_cS:) <- cX' * auxV */
    ierr = VecsMultS(&cT[ldc*d->size_cT],0,ldc,d->cY?d->cY:d->cX,0,d->size_cY+r_e,d->auxV,0,d->cX_in_BV+r_e,&r,&sr[1]);CHKERRQ(ierr);
  }
  /* Do reductions */
  ierr = SlepcAllReduceSumEnd(&r);CHKERRQ(ierr);

  ierr = DSRestoreArray(d->conv_ps,DS_MAT_A,&cS);CHKERRQ(ierr);
  if (d->cT) {
    ierr = DSRestoreArray(d->conv_ps,DS_MAT_B,&cT);CHKERRQ(ierr);
  }
  ierr = DSSetState(d->conv_ps,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
  /* eig(S,T) */
  k = d->size_cX+r_s;
  ierr = DSVectors(d->conv_ps,DS_MAT_X,&k,PETSC_NULL);CHKERRQ(ierr);
  ierr = DSNormalize(d->conv_ps,DS_MAT_X,d->size_cX+r_s);CHKERRQ(ierr);
  /* pX0 <- ps.Q(0:d->cX_in_AV+r_e-1) * conv_ps.X(size_cX-cX_in_H:) */
  pX0 = d->auxS;
  ierr = DSGetArray(d->conv_ps,DS_MAT_X,&pcX);CHKERRQ(ierr);
  ierr = SlepcDenseMatProd(pX0,d->size_H,0.0,1.0,pX,ld,d->size_H,d->cX_in_AV+r_e,PETSC_FALSE,&pcX[(d->size_cX-d->cX_in_H)*ldc],ldc,n,r_e-r_s,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DSRestoreArray(d->ps,DS_MAT_Q,&pX);CHKERRQ(ierr);
  /* auxV <- cX(0:size_cX-cX_in_AV)*conv_ps.X + V*pX0 */
  ierr = SlepcUpdateVectorsZ(d->auxV,0.0,1.0,d->cX,d->size_cX-d->cX_in_AV,pcX,ldc,n,r_e-r_s);CHKERRQ(ierr);
  ierr = DSRestoreArray(d->conv_ps,DS_MAT_X,&pcX);CHKERRQ(ierr);
  ierr = SlepcUpdateVectorsZ(d->auxV,d->size_cX-d->cX_in_AV==0?0.0:1.0,1.0,d->V-d->cX_in_AV,d->size_V+d->cX_in_AV,pX0,d->size_H,d->size_H,r_e-r_s);CHKERRQ(ierr);
  /* R <- A*auxV */
  for (i=0; i<r_e-r_s; i++) {
    ierr = MatMult(d->A,d->auxV[i],R[i]);CHKERRQ(ierr);
  }
  /* Bx <- B*auxV */
  if (d->B) {
    Bx = &d->auxV[r_e-r_s];
    for (i=0; i<r_e-r_s; i++) {
      ierr = MatMult(d->B,d->auxV[i],Bx[i]);CHKERRQ(ierr);
    }
  } else {
    Bx = d->auxV;
  }
  /* R <- (A - eig*B)*V*pX */
  for (i=0;i<r_e-r_s;i++) {
#if !defined(PETSC_USE_COMPLEX)
    if (d->eigi[r_s+i] != 0.0) {
      /* [Ax_i Ax_i+1 Bx_i Bx_i+1]*= [   1        0 
                                         0        1
                                      -eigr_i -eigi_i
                                       eigi_i -eigr_i] */
      b[0] = b[5] = 1.0;
      b[2] = b[7] = -d->eigr[r_s+i];
      b[6] = -(b[3] = d->eigi[r_s+i]);
      b[1] = b[4] = 0.0;
      X[0] = R[i]; X[1] = R[i+1]; X[2] = Bx[i]; X[3] = Bx[i+1];
      ierr = SlepcUpdateVectorsD(X,4,1.0,b,4,4,2,d->auxS,d->size_auxS);CHKERRQ(ierr);
      i++;
    } else
#endif
    {
      /* R <- Ax -eig*Bx */
      ierr = VecAXPBY(R[i], -d->eigr[r_s+i], 1.0, Bx[i]);CHKERRQ(ierr);
    }
  }
  /* nR <- ||R|| */
  for (i=0;i<r_e-r_s;i++) {
    ierr = VecNormBegin(R[i],NORM_2,&d->nR[r_s+i]);CHKERRQ(ierr);
  }
  for (i=0;i<r_e-r_s;i++) {
    ierr = VecNormEnd(R[i],NORM_2,&d->nR[r_s+i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/**** Pattern routines ********************************************************/

/* BV <- BV*MT */
#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateBV0_gen"
PETSC_STATIC_INLINE PetscErrorCode dvd_calcpairs_updateBV0_gen(dvdDashboard *d,Vec *real_BV,PetscInt *size_cBV,Vec **BV,PetscInt *size_BV,PetscInt *max_size_BV,PetscBool BV_shift,PetscInt *cX_in_proj,DSMatType mat)
{
  PetscErrorCode  ierr;
  PetscInt        cMT,rm,cp,tra_s,i,ld;
  Vec             *nBV;
  PetscScalar     *MT;

  PetscFunctionBegin;
  if (!real_BV || !*BV || (d->V_tra_s == 0 && d->V_tra_e == 0)) PetscFunctionReturn(0);

  ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
  ierr = DSGetArray(d->ps,mat,&MT);CHKERRQ(ierr);
  if (d->V_tra_s > d->max_cX_in_proj && !BV_shift) {
    tra_s = PetscMax(d->V_tra_s-d->max_cX_in_proj, 0);
    cMT = d->V_tra_e - tra_s;
    rm = d->V_tra_s - tra_s;
    cp = PetscMin(d->max_cX_in_proj - rm, *cX_in_proj);
    nBV = real_BV+d->max_cX_in_proj;
    /* BV(-cp-rm:-1-rm) <- BV(-cp:-1) */
    for (i=-cp; i<0; i++) {
      ierr = VecCopy((*BV)[i], nBV[i-rm]);CHKERRQ(ierr);
    }
    /* BV(-rm:) <- BV*MT(tra_s:V_tra_e-1) */
    ierr = SlepcUpdateVectorsZ(&nBV[-rm],0.0,1.0,*BV-*cX_in_proj,*size_BV+*cX_in_proj,&MT[ld*tra_s],ld,d->size_MT,cMT);CHKERRQ(ierr);
    *size_BV = d->V_tra_e  - d->V_tra_s;
    *max_size_BV-= nBV - *BV;
    *BV = nBV;
    if (cX_in_proj && d->max_cX_in_proj>0) *cX_in_proj = cp+rm;
  } else if (d->V_tra_s <= d->max_cX_in_proj || BV_shift) {
    /* [BcX BV] <- [BcX BV*MT] */
    ierr = SlepcUpdateVectorsZ(*BV-*cX_in_proj,0.0,1.0,*BV-*cX_in_proj,*size_BV+*cX_in_proj,MT,ld,d->size_MT,d->V_tra_e);CHKERRQ(ierr);
    *BV+= d->V_tra_s-*cX_in_proj;
    *max_size_BV-= d->V_tra_s-*cX_in_proj;
    *size_BV = d->V_tra_e  - d->V_tra_s;
    if (size_cBV && BV_shift) *size_cBV = *BV - real_BV; 
    if (d->max_cX_in_proj>0) *cX_in_proj = PetscMin(*BV - real_BV, d->max_cX_in_proj);
  } else { /* !BV_shift */
    /* BV <- BV*MT(V_tra_s:) */
    ierr = SlepcUpdateVectorsZ(*BV,0.0,1.0,*BV,*size_BV,&MT[d->V_tra_s*ld],ld,d->size_MT,d->V_tra_e-d->V_tra_s);CHKERRQ(ierr);
    *size_BV = d->V_tra_e - d->V_tra_s;
  }
  ierr = DSRestoreArray(d->ps,mat,&MT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
