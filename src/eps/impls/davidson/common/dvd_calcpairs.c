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
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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
PetscErrorCode dvd_calcpairs_projeig_eig(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_projeig_qz_std(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_projeig_qz_gen(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_selectPairs_qz(dvdDashboard *d, PetscInt n);
PetscErrorCode dvd_calcpairs_selectPairs_eig(dvdDashboard *d, PetscInt n);
PetscErrorCode dvd_calcpairs_X(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                               Vec *X);
PetscErrorCode dvd_calcpairs_Y(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                               Vec *Y);
PetscErrorCode dvd_calcpairs_res_0(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                                   Vec *R);
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
PETSC_STATIC_INLINE PetscErrorCode dvd_calcpairs_updateBV0_gen(dvdDashboard *d,Vec *real_BV,PetscInt *size_cX,Vec **BV,PetscInt *size_BV,PetscInt *max_size_BV,PetscBool BV_shift,PetscInt *cX_in_proj,PetscScalar *MTX);

/**** Control routines ********************************************************/
#undef __FUNCT__  
#define __FUNCT__ "dvd_calcpairs_qz"
PetscErrorCode dvd_calcpairs_qz(dvdDashboard *d, dvdBlackboard *b,
                                orthoV_type_t orth, IP ipI,
                                PetscInt cX_proj, PetscBool harm)
{
  PetscErrorCode  ierr;
  PetscInt        i;
  PetscBool       std_probl,her_probl;

  PetscFunctionBegin;

  std_probl = DVD_IS(d->sEP, DVD_EP_STD)?PETSC_TRUE:PETSC_FALSE;
  her_probl = DVD_IS(d->sEP, DVD_EP_HERMITIAN)?PETSC_TRUE:PETSC_FALSE;

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
  if (d->B && her_probl && (orth == DVD_ORTHOV_I || orth == DVD_ORTHOV_BOneMV)) {
    d->size_real_BV = b->size_V; d->BV_shift = PETSC_TRUE;
    if (orth == DVD_ORTHOV_BOneMV) d->size_BDS = d->eps->nds;
  } else if (d->B) { 
    d->size_real_BV = b->max_size_V + b->max_size_P; d->BV_shift = PETSC_FALSE;
  } else {
    d->size_real_BV = 0; d->BV_shift = PETSC_FALSE;
  }
  b->own_vecs+= d->size_real_V + d->size_real_W + d->size_real_AV +
                d->size_real_BV + d->size_BDS;
  b->own_scalars+= b->max_size_proj*b->max_size_proj*2*(std_probl?1:2) +
                                              /* H, G?, S, T? */
                   b->max_size_proj*b->max_size_proj*(std_probl?1:2) +
                                              /* pX, pY? */
                   b->max_nev*b->max_nev*(her_probl?0:(!d->B?1:2)); 
                                                /* cS?, cT? */
  b->max_size_auxV = PetscMax(b->max_size_auxV, b->max_size_X);
                                                /* updateV0 */
  b->max_size_auxS = PetscMax(PetscMax(
    b->max_size_auxS,
    b->max_size_X*b->max_size_proj*2*(!d->B?1:2) + /* updateAV1,BV1 */
      b->max_size_X*b->max_nev*(her_probl?0:(!d->B?1:2)) + /* updateV0,W0 */
                                                     /* SlepcReduction: in */
      PetscMax(
        b->max_size_X*b->max_size_proj*2*(!d->B?1:2) + /* updateAV1,BV1 */
        b->max_size_X*b->max_nev*(her_probl?0:(!d->B?1:2)), /* updateV0,W0 */
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
    d->pX = b->free_scalars; b->free_scalars+= b->max_size_proj*b->max_size_proj;
    d->S = b->free_scalars; b->free_scalars+= b->max_size_proj*b->max_size_proj;
    if (!her_probl) {
      d->cS = b->free_scalars; b->free_scalars+= b->max_nev*b->max_nev;
      d->max_size_cS = d->ldcS = b->max_nev;
    } else {
      d->cS = PETSC_NULL;
      d->max_size_cS = d->ldcS = 0;
    }
    d->ipV = ipI;
    d->ipW = ipI;
    if (orth == DVD_ORTHOV_BOneMV) {
      d->BDS = b->free_vecs; b->free_vecs+= d->eps->nds;
      for (i=0; i<d->eps->nds; i++) {
        ierr = MatMult(d->B, d->eps->DS[i], d->BDS[i]); CHKERRQ(ierr);
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
      d->pY = b->free_scalars; b->free_scalars+= b->max_size_proj*b->max_size_proj;
    } else {
      d->real_G = PETSC_NULL;
      d->T = PETSC_NULL;
      d->pY = PETSC_NULL;
    }
    if (d->B && !her_probl) {
      d->cT = b->free_scalars; b->free_scalars+= b->max_nev*b->max_nev;
      d->ldcT = b->max_nev;
    } else {
      d->cT = PETSC_NULL;
      d->ldcT = 0;
    }

    d->calcPairs = dvd_calcpairs_proj;
    d->calcpairs_residual = dvd_calcpairs_res_0;
    d->calcpairs_proj_res = dvd_calcpairs_proj_res;
    d->calcpairs_selectPairs = PETSC_NULL;
    d->ipI = ipI;
    DVD_FL_ADD(d->startList, dvd_calcpairs_qz_start);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_calcpairs_qz_start"
PetscErrorCode dvd_calcpairs_qz_start(dvdDashboard *d)
{
  PetscBool       std_probl, her_probl;
  PetscInt        i;

  PetscFunctionBegin;

  std_probl = DVD_IS(d->sEP, DVD_EP_STD)?PETSC_TRUE:PETSC_FALSE;
  her_probl = DVD_IS(d->sEP, DVD_EP_HERMITIAN)?PETSC_TRUE:PETSC_FALSE;

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
  d->cY = d->B && !her_probl ? d->W : PETSC_NULL;
  d->BcX = d->orthoV_type == DVD_ORTHOV_I && d->B && her_probl ? d->BcX : PETSC_NULL;
  d->size_cY = 0;
  d->size_BcX = 0;
  d->cX_in_V = d->cX_in_H = d->cX_in_G = d->cX_in_W = d->cX_in_AV = d->cX_in_BV = 0;

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_proj"
PetscErrorCode dvd_calcpairs_proj(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  DvdReduction    r;
  static const PetscInt MAX_OPS = 7;
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
    (d->size_cX+d->V_tra_s)*d->V_tra_s*(d->cT?2:(d->cS?1:0)) + /* updateV0,W0 */
    d->size_H*d->V_tra_e*(!stdp?2:1) + /* updateAV1,BV1 */
    (d->size_H*(d->V_new_e-d->V_new_s)*2+
      (d->V_new_e-d->V_new_s)*(d->V_new_e-d->V_new_s))*(!stdp?2:1); /* updateAV0,BV0 */
    
  out = in+size_in;

   /* Check consistency */
   if (2*size_in > d->size_auxS) { SETERRQ(PETSC_COMM_SELF,1, "Consistency broken!"); }

  /* Prepare reductions */
  ierr = SlepcAllReduceSumBegin(ops, MAX_OPS, in, out, size_in, &r,
                                ((PetscObject)d->V[0])->comm); CHKERRQ(ierr);
  /* Allocate size_in */
  d->auxS+= size_in;
  d->size_auxS-= size_in;

  /* Update AV, BV, W and the projected matrices */
  /* 1. S <- S*MT */
  ierr = dvd_calcpairs_updateV0(d, &r, &sr0); CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateW0(d, &r, &sr0); CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateAV0(d); CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateBV0(d); CHKERRQ(ierr);
  /* 2. V <- orth(V, V_new) */ 
  ierr = dvd_calcpairs_updateV1(d); CHKERRQ(ierr);
  /* 3. AV <- [AV A * V(V_new_s:V_new_e-1)] */
  /* Check consistency */
  if (d->size_AV != d->V_new_s) { SETERRQ(PETSC_COMM_SELF,1, "Consistency broken!"); }
  for (i=d->V_new_s; i<d->V_new_e; i++) {
    ierr = MatMult(d->A, d->V[i], d->AV[i]); CHKERRQ(ierr);
  }
  d->size_AV = d->V_new_e;
  /* 4. BV <- [BV B * V(V_new_s:V_new_e-1)] */
  if (d->B && d->orthoV_type != DVD_ORTHOV_BOneMV) {
    /* Check consistency */
    if (d->size_BV != d->V_new_s) { SETERRQ(PETSC_COMM_SELF,1, "Consistency broken!"); }
    for (i=d->V_new_s; i<d->V_new_e; i++) {
      ierr = MatMult(d->B, d->V[i], d->BV[i]); CHKERRQ(ierr);
    }
    d->size_BV = d->V_new_e;
  }
  /* 5 <- W <- [W f(AV,BV)] */
  ierr = dvd_calcpairs_updateW1(d); CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateAV1(d, &r, &sr0); CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateBV1(d, &r, &sr0); CHKERRQ(ierr);

  /* Deallocate size_in */
  d->auxS-= size_in;
  d->size_auxS+= size_in;

  /* Do reductions */
  ierr = SlepcAllReduceSumEnd(&r); CHKERRQ(ierr);

  /* Perform the transformation on the projected problem */
  if (d->calcpairs_proj_trans) {
    ierr = d->calcpairs_proj_trans(d); CHKERRQ(ierr);
  }

  d->V_tra_s = d->V_tra_e = 0;
  d->V_new_s = d->V_new_e;

  /* Solve the projected problem */
  if (DVD_IS(d->sEP, DVD_EP_STD)) {
    if (DVD_IS(d->sEP, DVD_EP_HERMITIAN)) {
      ierr = dvd_calcpairs_projeig_eig(d); CHKERRQ(ierr);
    } else {
      ierr = dvd_calcpairs_projeig_qz_std(d); CHKERRQ(ierr);
    }
  } else {
    ierr = dvd_calcpairs_projeig_qz_gen(d); CHKERRQ(ierr);
  }

  /* Check consistency */
  if (d->size_V != d->V_new_e || d->size_V+d->cX_in_H != d->size_H || d->cX_in_V != d->cX_in_H ||
      d->size_V != d->size_AV || d->cX_in_H != d->cX_in_AV ||
        (DVD_ISNOT(d->sEP, DVD_EP_STD) && (
          d->size_V+d->cX_in_G != d->size_G || d->cX_in_H != d->cX_in_G ||
          d->size_H != d->size_G || (d->BV && (
            d->size_V != d->size_BV || d->cX_in_H != d->cX_in_BV)))) ||
      (d->W && d->size_W != d->size_V)) {
    SETERRQ(PETSC_COMM_SELF,1, "Consistency broken!");
  }

  PetscFunctionReturn(0);
}

/**** Basic routines **********************************************************/

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateV0"
/* auxV: V_tra_s, DvdMult_copy_func: 1 */
PetscErrorCode dvd_calcpairs_updateV0(dvdDashboard *d, DvdReduction *r,
                                      DvdMult_copy_func **sr)
{
  PetscErrorCode  ierr;
  PetscInt        rm;

  PetscFunctionBegin;

  if (d->V_tra_s == 0 && d->V_tra_e == 0) { PetscFunctionReturn(0); }

  /* cX <- [cX V*MT(0:V_tra_s-1)], V <- V*MT(V_tra_s:V_tra_e) */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->real_V,&d->size_cX,&d->V,&d->size_V,&d->max_size_V,PETSC_TRUE,&d->cX_in_V,d->MTX);CHKERRQ(ierr);

  /* Udpate cS for standard problems */
  if (d->cS && !d->cT && !d->cY && (d->V_tra_s > d->max_cX_in_proj || d->size_cX >= d->nev)) {
    /* Check consistency */
    if (d->size_cS+d->V_tra_s != d->size_cX) { SETERRQ(PETSC_COMM_SELF,1, "Consistency broken!"); }

    /* auxV <- AV * MTX(0:V_tra_e-1) */
    ierr = SlepcUpdateVectorsZ(d->auxV, 0.0, 1.0, d->AV-d->cX_in_AV, d->size_AV+d->cX_in_AV, d->MTX, d->ldMTX, d->size_MT, d->V_tra_s-d->max_cX_in_proj); CHKERRQ(ierr);

    /* cS(:, size_cS:) <- cX' * auxV */
    rm = d->size_cX>=d->nev?0:d->max_cX_in_proj;
    ierr = VecsMultS(&d->cS[d->ldcS*d->size_cS], 0, d->ldcS, d->cX, 0, d->size_cX-rm, d->auxV, 0, d->V_tra_s-rm, r, (*sr)++); CHKERRQ(ierr);
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
  Vec             *cX = d->BcX? d->BcX : ( (d->cY && !d->W)? d->cY : d->cX );

  PetscFunctionBegin;

  if (d->V_new_s == d->V_new_e) { PetscFunctionReturn(0); }

  /* Check consistency */
  if (d->size_V != d->V_new_s) { SETERRQ(PETSC_COMM_SELF,1, "Consistency broken!"); }

  /* V <- gs([cX V(0:V_new_s-1)], V(V_new_s:V_new_e-1)) */
  if (d->orthoV_type == DVD_ORTHOV_BOneMV) {
    ierr = dvd_BorthV(d->ipV, d->eps->DS, d->BDS, d->eps->nds, d->cX, d->real_BV,
                      d->size_cX, d->V, d->BV, d->V_new_s, d->V_new_e,
                      d->auxS, d->eps->rand); CHKERRQ(ierr);
    d->size_BV = d->V_new_e;
  } else {
    ierr = dvd_orthV(d->ipV, d->eps->DS, d->eps->nds, cX, d->size_cX, d->V,
                   d->V_new_s, d->V_new_e, d->auxS, d->eps->rand);
    CHKERRQ(ierr);
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
  PetscInt        rm;

  PetscFunctionBegin;

  if (!d->W || (d->V_tra_s == 0 && d->V_tra_e == 0)) { PetscFunctionReturn(0); }

  /* cY <- [cY W*MTY(0:V_tra_s-1)], W <- W*MTY(V_tra_s:V_tra_e) */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->real_W,&d->size_cY,&d->W,&d->size_W,&d->max_size_W,d->W_shift,&d->cX_in_W,d->MTY);CHKERRQ(ierr);

  /* Udpate cS and cT */
  if (d->cY && d->cT && (d->V_tra_s > d->max_cX_in_proj || d->size_cX >= d->nev)) {
    /* Check consistency */
    if (d->size_cS+d->V_tra_s != d->size_cY) { SETERRQ(PETSC_COMM_SELF,1, "Consistency broken!"); }

    /* auxV <- AV * MTX(0:V_tra_e-1) */
    rm = d->size_cX>=d->nev?0:d->max_cX_in_proj;
    ierr = SlepcUpdateVectorsZ(d->auxV, 0.0, 1.0, d->AV-d->cX_in_H, d->size_AV-d->cX_in_H, d->MTX, d->ldMTX, d->size_MT, d->V_tra_s-d->max_cX_in_proj); CHKERRQ(ierr);

    /* cS(:, size_cS:) <- cY' * auxV */
    ierr = VecsMultS(&d->cS[d->ldcS*d->size_cS], 0, d->ldcS, d->cY, 0, d->size_cY-rm, d->auxV, 0, d->V_tra_s-rm, r, (*sr)++); CHKERRQ(ierr);

    /* auxV <- BV * MTX(0:V_tra_e-1) */
    ierr = SlepcUpdateVectorsZ(d->auxV, 0.0, 1.0, d->BV-d->cX_in_H, d->size_BV-d->cX_in_H, d->MTX, d->ldMTX, d->size_MT, d->V_tra_s-d->max_cX_in_proj); CHKERRQ(ierr);

    /* cT(:, size_cS:) <- cY' * auxV */
    ierr = VecsMultS(&d->cT[d->ldcS*d->size_cS], 0, d->ldcS, d->cY, 0, d->size_cY-rm, d->auxV, 0, d->V_tra_s-rm, r, (*sr)++); CHKERRQ(ierr);
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

  PetscFunctionBegin;

  if (!d->W || (d->V_new_s == d->V_new_e)) { PetscFunctionReturn(0); }

  /* Check consistency */
  if (d->size_W != d->V_new_s) { SETERRQ(PETSC_COMM_SELF,1, "Consistency broken!"); }

  /* Update W */
  ierr = d->calcpairs_W(d); CHKERRQ(ierr);

  /* W <- gs([cY W(0:V_new_s-1)], W(V_new_s:V_new_e-1)) */
  ierr = dvd_orthV(d->ipW, PETSC_NULL, 0, d->cY, d->size_cY, d->W, d->V_new_s,
                   d->V_new_e, d->auxS, d->eps->rand);
  CHKERRQ(ierr);
  d->size_W = d->V_new_e;

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateAV0"
/* auxS: size_H*(V_tra_e-V_tra_s) */
PetscErrorCode dvd_calcpairs_updateAV0(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscScalar     *MTY = d->W?d->MTY:d->MTX;
  PetscInt        cMT, tra_s, rm, cp;

  PetscFunctionBegin;

  if (d->V_tra_s == 0 && d->V_tra_e == 0) { PetscFunctionReturn(0); }

  /* AV(V_tra_s-cp-1:) = cAV*MTX(V_tra_s:) */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->real_AV,PETSC_NULL,&d->AV,&d->size_AV,&d->max_size_AV,PETSC_FALSE,&d->cX_in_AV,d->MTX);CHKERRQ(ierr);
  tra_s = PetscMax(d->V_tra_s-d->max_cX_in_proj, 0);
  cMT = d->V_tra_e - tra_s;
  rm = d->V_tra_s - tra_s;
  cp = PetscMin(d->max_cX_in_proj - rm, d->cX_in_H);

  /* Update H <- MTY(tra_s)' * (H * MTX(tra_s:)) */
  ierr = SlepcDenseMatProdTriang(d->auxS, 0, d->ldH, d->H, d->sH, d->ldH, d->size_H, d->size_H, PETSC_FALSE, &d->MTX[d->ldMTX*tra_s], 0, d->ldMTX, d->size_MT, cMT, PETSC_FALSE); CHKERRQ(ierr);
  ierr = SlepcDenseMatProdTriang(d->H, d->sH, d->ldH, &MTY[d->ldMTX*tra_s], 0, d->ldMTX, d->size_MT, cMT, PETSC_TRUE, d->auxS, 0, d->ldH, d->size_H, cMT, PETSC_FALSE); CHKERRQ(ierr);
  d->size_H = cMT;
  d->cX_in_H = cp+rm;

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

  if (d->V_new_s == d->V_new_e) { PetscFunctionReturn(0); }

  /* Check consistency */
  if (d->size_H != d->V_new_s+d->cX_in_H || d->size_V != d->V_new_e) {
    SETERRQ(PETSC_COMM_SELF,1, "Consistency broken!");
  }

  /* H = [H               W(old)'*AV(new);
          W(new)'*AV(old) W(new)'*AV(new) ],
     being old=0:V_new_s-1, new=V_new_s:V_new_e-1 */
  ierr = VecsMultS(d->H,d->sH,d->ldH,W-d->cX_in_H,d->V_new_s+d->cX_in_H, d->V_new_e+d->cX_in_H, d->AV-d->cX_in_H,d->V_new_s+d->cX_in_H,d->V_new_e+d->cX_in_H, r, (*sr)++); CHKERRQ(ierr);
  d->size_H = d->V_new_e+d->cX_in_H;

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateBV0"
/* auxS: max(BcX*(size_cX+V_new_e+1), size_G*(V_tra_e-V_tra_s)) */
PetscErrorCode dvd_calcpairs_updateBV0(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscScalar     *MTY = d->W?d->MTY:d->MTX;
  PetscInt        cMT, rm, cp, tra_s, i;
  PetscBool       lindep;
  PetscReal       norm;

  PetscFunctionBegin;

  if (d->V_tra_s == 0 && d->V_tra_e == 0) { PetscFunctionReturn(0); }

  /* BV <- BV*MTX */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->real_BV,PETSC_NULL,&d->BV,&d->size_BV,&d->max_size_BV,d->BV_shift,&d->cX_in_BV,d->MTX);CHKERRQ(ierr);

  /* If BcX, BcX <- orth(BcX) */
  if (d->BcX) {
    for (i=0; i<d->V_tra_s; i++) {
      ierr = IPOrthogonalize(d->ipI, 0, PETSC_NULL, d->size_BcX+i, PETSC_NULL,
                             d->BcX, d->BcX[d->size_BcX+i], PETSC_NULL,
                             &norm, &lindep); CHKERRQ(ierr);
      if(lindep) SETERRQ(((PetscObject)d->ipI)->comm,1, "Error during orth(BcX, B*cX(new))");
      ierr = VecScale(d->BcX[d->size_BcX+i], 1.0/norm); CHKERRQ(ierr);
    }
    d->size_BcX+= d->V_tra_s;
  }

  /* Update G <- MTY' * (G * MTX) */
  if (d->G) {
    tra_s = PetscMax(d->V_tra_s-d->max_cX_in_proj, 0);
    cMT = d->V_tra_e - tra_s;
    rm = d->V_tra_s - tra_s;
    cp = PetscMin(d->max_cX_in_proj - rm, d->cX_in_G);
    ierr = SlepcDenseMatProdTriang(d->auxS, 0, d->ldH, d->G, d->sG, d->ldH, d->size_G, d->size_G, PETSC_FALSE, &d->MTX[d->ldMTX*tra_s], 0, d->ldMTX, d->size_MT, cMT, PETSC_FALSE); CHKERRQ(ierr);
    ierr = SlepcDenseMatProdTriang(d->G, d->sG, d->ldH, &MTY[d->ldMTX*tra_s], 0, d->ldMTX, d->size_MT, cMT, PETSC_TRUE, d->auxS, 0, d->ldH, d->size_G, cMT, PETSC_FALSE); CHKERRQ(ierr);
    d->size_G = cMT;
    d->cX_in_G = cp+rm;
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

  if (!d->G || d->V_new_s == d->V_new_e) { PetscFunctionReturn(0); }

  /* G = [G               W(old)'*BV(new);
          W(new)'*BV(old) W(new)'*BV(new) ],
     being old=0:V_new_s-1, new=V_new_s:V_new_e-1 */
  ierr = VecsMultS(d->G,d->sG,d->ldH,W-d->cX_in_G,d->V_new_s+d->cX_in_G,d->V_new_e+d->cX_in_G,BV-d->cX_in_G,d->V_new_s+d->cX_in_G,d->V_new_e+d->cX_in_G,r,(*sr)++); CHKERRQ(ierr);
  d->size_G = d->V_new_e+d->cX_in_G;

  PetscFunctionReturn(0);
}

/* in complex, d->size_H real auxiliar values are needed */
#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_projeig_eig"
PetscErrorCode dvd_calcpairs_projeig_eig(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscReal       *w;
#if defined(PETSC_USE_COMPLEX)
  PetscInt        i;
#endif

  PetscFunctionBegin;

  /* S <- H */
  d->ldS = d->ldpX = d->size_H;
  ierr = SlepcDenseCopyTriang(d->S, DVD_MAT_LTRIANG, d->size_H, d->H, d->sH, d->ldH,
                              d->size_H, d->size_H); CHKERRQ(ierr);

  /* S = pX' * L * pX */
#if !defined(PETSC_USE_COMPLEX)
  w = d->eigr-d->cX_in_H;
#else
  w = (PetscReal*)d->auxS;
#endif
  ierr = EPSDenseHEP(d->size_H, d->S, d->ldS, w, d->pX); CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  for (i=0; i<d->size_H; i++) d->eigr[i-d->cX_in_H] = w[i];
#endif

  d->calcpairs_selectPairs = dvd_calcpairs_selectPairs_eig;

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_projeig_qz_std"
PetscErrorCode dvd_calcpairs_projeig_qz_std(dvdDashboard *d)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* S <- H */
  d->ldS = d->ldpX = d->size_H;
  ierr = SlepcDenseCopyTriang(d->S, 0, d->size_H, d->H, d->sH, d->ldH,
                              d->size_H, d->size_H);

  /* S = pX' * H * pX */
  ierr = EPSDenseHessenberg(d->size_H, 0, d->S, d->ldS, d->pX); CHKERRQ(ierr);
  ierr = EPSDenseSchur(d->size_H, 0, d->S, d->ldS, d->pX, d->eigr-d->cX_in_H, d->eigi-d->cX_in_H);
  CHKERRQ(ierr);

  d->calcpairs_selectPairs = dvd_calcpairs_selectPairs_qz;

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_projeig_qz_gen"
/*
  auxS(dgges) = size_H (beta) + 8*size_H+16 (work)
  auxS(zgges) = size_H (beta) + 1+2*size_H (work) + 8*size_H (rwork)
*/
PetscErrorCode dvd_calcpairs_projeig_qz_gen(dvdDashboard *d)
{
#if defined(SLEPC_MISSING_LAPACK_GGES)
  PetscFunctionBegin;
  SETERRQ(((PetscObject)(d->eps))->comm,PETSC_ERR_SUP,"GGES - Lapack routine is unavailable.");
#else
  PetscErrorCode  ierr;
  PetscScalar     *beta = d->auxS;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar     *auxS = beta + d->size_H;
  PetscBLASInt    n_auxS = d->size_auxS - d->size_H;
#else
  PetscReal       *auxR = (PetscReal*)(beta + d->size_H);
  PetscScalar     *auxS = (PetscScalar*)(auxR+8*d->size_H);
  PetscBLASInt    n_auxS = d->size_auxS - 9*d->size_H;
#endif
  PetscInt        i;
  PetscBLASInt    info,n,a;

  PetscFunctionBegin;
  /* S <- H, T <- G */
  d->ldS = d->ldT = d->ldpX = d->ldpY = d->size_H;
  ierr = SlepcDenseCopyTriang(d->S, 0, d->size_H, d->H, d->sH, d->ldH,
                              d->size_H, d->size_H);CHKERRQ(ierr);
  ierr = SlepcDenseCopyTriang(d->T, 0, d->size_H, d->G, d->sG, d->ldH,
                              d->size_H, d->size_H);CHKERRQ(ierr);

  /* S = Z'*H*Q, T = Z'*G*Q */
  n = d->size_H;
#if !defined(PETSC_USE_COMPLEX)
  LAPACKgges_(d->pY?"V":"N", "V", "N", PETSC_NULL, &n, d->S, &n, d->T, &n,
              &a, d->eigr-d->cX_in_H, d->eigi-d->cX_in_H, beta, d->pY, &n, d->pX, &n,
              auxS, &n_auxS, PETSC_NULL, &info);
#else
  LAPACKgges_(d->pY?"V":"N", "V", "N", PETSC_NULL, &n, d->S, &n, d->T, &n,
              &a, d->eigr-d->cX_in_H, beta, d->pY, &n, d->pX, &n,
              auxS, &n_auxS, auxR, PETSC_NULL, &info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Error in Lapack GGES %d", info);

  /* eigr[i] <- eigr[i] / beta[i] */
  for (i=0; i<d->size_H; i++)
    d->eigr[i-d->cX_in_H] /= beta[i],
    d->eigi[i-d->cX_in_H] /= beta[i];

  d->calcpairs_selectPairs = dvd_calcpairs_selectPairs_qz;

  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_selectPairs_eig"
PetscErrorCode dvd_calcpairs_selectPairs_eig(dvdDashboard *d, PetscInt n)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = EPSSortDenseHEP(d->eps, d->size_H, 0, d->eigr-d->cX_in_H, d->pX, d->ldpX);
  CHKERRQ(ierr);

  if (d->calcpairs_eigs_trans) {
    ierr = d->calcpairs_eigs_trans(d); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_selectPairs_qz"
PetscErrorCode dvd_calcpairs_selectPairs_qz(dvdDashboard *d, PetscInt n)
{
  PetscErrorCode  ierr;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar     s;
  PetscInt        i, j;
#endif
  PetscFunctionBegin;

  if ((d->ldpX != d->size_H) ||
      ( d->T &&
        ((d->ldS != d->ldT) || (d->ldpX != d->ldpY) ||
         (d->ldpX != d->size_H)) ) ) {
     SETERRQ(PETSC_COMM_SELF,1, "Error before ordering eigenpairs");
  }

  if (d->T) {
    ierr = EPSSortDenseSchurGeneralized(d->eps, d->size_H, 0, n, d->S, d->T,
                                        d->ldS, d->pY, d->pX, d->eigr-d->cX_in_H,
                                        d->eigi-d->cX_in_H); CHKERRQ(ierr);
  } else {
    ierr = EPSSortDenseSchur(d->eps, d->size_H, 0, d->S, d->ldS, d->pX,
                             d->eigr-d->cX_in_H, d->eigi-d->cX_in_H); CHKERRQ(ierr);
  }

  if (d->calcpairs_eigs_trans) {
    ierr = d->calcpairs_eigs_trans(d); CHKERRQ(ierr);
  }

  /* Some functions need the diagonal elements in T be real */
#if defined(PETSC_USE_COMPLEX)
  if (d->T) for(i=0; i<d->size_H; i++)
    if (PetscImaginaryPart(d->T[d->ldT*i+i]) != 0.0) {
      s = PetscConj(d->T[d->ldT*i+i])/PetscAbsScalar(d->T[d->ldT*i+i]);
      for(j=0; j<=i; j++)
        d->T[d->ldT*i+j] = PetscRealPart(d->T[d->ldT*i+j]*s),
        d->S[d->ldS*i+j]*= s;
      for(j=0; j<d->size_H; j++) d->pX[d->ldpX*i+j]*= s;
    }
#endif

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_res_0"
/* Compute the residual vectors R(i) <- (AV - BV*eigr(i))*pX(i), and also
   the norm, where i = r_s..r_e
*/
PetscErrorCode dvd_calcpairs_res_0(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                             Vec *R)
{
  PetscInt        i;
  PetscErrorCode  ierr;
  Vec             *BV = d->BV?d->BV:d->V;

  PetscFunctionBegin;

  for (i=r_s; i<r_e; i++) {
    /* nX(i) <- ||X(i)|| */
    if (d->correctXnorm) {
      /* R(i) <- V*pX(i) */
      ierr = SlepcUpdateVectorsZ(&R[i-r_s], 0.0, 1.0,
        &d->V[-d->cX_in_H], d->size_V+d->cX_in_H,
        &d->pX[d->ldpX*(i+d->cX_in_H)], d->ldpX, d->size_H, 1); CHKERRQ(ierr);
      ierr = VecNorm(R[i-r_s], NORM_2, &d->nX[i]);CHKERRQ(ierr);
    } else
      d->nX[i] = 1.0;

    /* R(i-r_s) <- AV*pX(i) */
    ierr = SlepcUpdateVectorsZ(&R[i-r_s], 0.0, 1.0,
      &d->AV[-d->cX_in_H], d->size_AV+d->cX_in_H,
      &d->pX[d->ldpX*(i+d->cX_in_H)], d->ldpX, d->size_H, 1); CHKERRQ(ierr);

    /* R(i-r_s) <- R(i-r_s) - eigr(i)*BV*pX(i) */
    ierr = SlepcUpdateVectorsZ(&R[i-r_s], 1.0, -d->eigr[i+d->cX_in_H],
      &BV[-d->cX_in_H], d->size_V+d->cX_in_H,
      &d->pX[d->ldpX*(i+d->cX_in_H)], d->ldpX, d->size_H, 1); CHKERRQ(ierr);
  }

  ierr = d->calcpairs_proj_res(d, r_s, r_e, R); CHKERRQ(ierr);

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

  if (cX) for (i=0; i<r_e-r_s; i++) {
    ierr = IPOrthogonalize(d->ipI, 0, PETSC_NULL, d->size_cX, PETSC_NULL,
                           cX, R[i], PETSC_NULL, &d->nR[r_s+i], &lindep);
    CHKERRQ(ierr);
    if(lindep || (d->nR[r_s+i] < PETSC_MACHINE_EPSILON)) {
      ierr = PetscInfo2(d->eps,"The computed eigenvector residual %D is too low, %G!\n",r_s+i,d->nR[r_s+i]);CHKERRQ(ierr);
    }

  } else for(i=0; i<r_e-r_s; i++) {
    ierr = VecNorm(R[i], NORM_2, &d->nR[r_s+i]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/**** Pattern routines ********************************************************/

/* BV <- BV*MTX */
#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateBV0_gen"
PETSC_STATIC_INLINE PetscErrorCode dvd_calcpairs_updateBV0_gen(dvdDashboard *d,Vec *real_BV,PetscInt *size_cBV,Vec **BV,PetscInt *size_BV,PetscInt *max_size_BV,PetscBool BV_shift,PetscInt *cX_in_proj,PetscScalar *MTX)
{
  PetscErrorCode  ierr;
  PetscInt        cMT, rm, cp, tra_s, i;
  Vec             *nBV;

  PetscFunctionBegin;

  if (!real_BV || !*BV || (d->V_tra_s == 0 && d->V_tra_e == 0)) { PetscFunctionReturn(0); }

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
    /* BV(-rm:) <- BV*MTX(tra_s:V_tra_e-1) */
    ierr = SlepcUpdateVectorsZ(&nBV[-rm], 0.0, 1.0, *BV-*cX_in_proj, *size_BV+*cX_in_proj, &MTX[d->ldMTX*tra_s], d->ldMTX, d->size_MT, cMT); CHKERRQ(ierr);
    *size_BV = d->V_tra_e  - d->V_tra_s;
    *max_size_BV-= nBV - *BV;
    *BV = nBV;
    if (cX_in_proj && d->max_cX_in_proj>0) *cX_in_proj = cp+rm;
  } else if (d->V_tra_s <= d->max_cX_in_proj || BV_shift) {
    /* [BcX BV] <- [BcX BV*MTX] */
    ierr = SlepcUpdateVectorsZ(*BV-*cX_in_proj, 0.0, 1.0, *BV-*cX_in_proj, *size_BV+*cX_in_proj, MTX, d->ldMTX, d->size_MT, d->V_tra_e); CHKERRQ(ierr);
    *BV+= d->V_tra_s-*cX_in_proj;
    *max_size_BV-= d->V_tra_s-*cX_in_proj;
    *size_BV = d->V_tra_e  - d->V_tra_s;
    if (size_cBV && BV_shift) *size_cBV = *BV - real_BV; 
    if (d->max_cX_in_proj>0) *cX_in_proj = PetscMin(*BV - real_BV, d->max_cX_in_proj);
  } else { /* !BV_shift */
    /* BV <- BV*MTX(V_tra_s:) */
    ierr = SlepcUpdateVectorsZ(*BV, 0.0, 1.0, *BV, *size_BV,
      &MTX[d->V_tra_s*d->ldMTX], d->ldMTX, d->size_MT, d->V_tra_e-d->V_tra_s);
    CHKERRQ(ierr);
    *size_BV = d->V_tra_e - d->V_tra_s;
  }

  PetscFunctionReturn(0);
}
