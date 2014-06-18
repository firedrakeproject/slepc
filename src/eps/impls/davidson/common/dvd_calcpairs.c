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

#include "davidson.h"

PetscErrorCode dvd_calcpairs_proj(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_qz_start(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_qz_d(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_projeig_solve(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_selectPairs(dvdDashboard *d,PetscInt n);
PetscErrorCode dvd_calcpairs_X(dvdDashboard *d,PetscInt r_s,PetscInt r_e,Vec *X);
PetscErrorCode dvd_calcpairs_Y(dvdDashboard *d,PetscInt r_s,PetscInt r_e,Vec *Y);
PetscErrorCode dvd_calcpairs_res_0(dvdDashboard *d,PetscInt r_s,PetscInt r_e);
PetscErrorCode dvd_calcpairs_eig_res_0(dvdDashboard *d,PetscInt r_s,PetscInt r_e);
PetscErrorCode dvd_calcpairs_proj_res(dvdDashboard *d,PetscInt r_s,PetscInt r_e,Vec *R);
PetscErrorCode dvd_calcpairs_updateV0(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateV1(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateW0(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateW1(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateAV0(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateAV1(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateBV0(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateBV1(dvdDashboard *d);
PETSC_STATIC_INLINE PetscErrorCode dvd_calcpairs_updateBV0_gen(dvdDashboard *d,BV bv,DSMatType MT);

/**** Control routines ********************************************************/
#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_qz"
PetscErrorCode dvd_calcpairs_qz(dvdDashboard *d,dvdBlackboard *b,EPSOrthType orth,PetscInt cX_proj,PetscBool harm)
{
  PetscErrorCode ierr;
  PetscInt       max_cS;
  PetscBool      std_probl,her_probl,ind_probl,her_ind_probl;
  DSType         dstype;
  const char     *prefix;
  PetscErrorCode (*f)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  void           *ctx;
  Vec            v1;

  PetscFunctionBegin;
  std_probl = DVD_IS(d->sEP, DVD_EP_STD)?PETSC_TRUE:PETSC_FALSE;
  her_probl = DVD_IS(d->sEP, DVD_EP_HERMITIAN)?PETSC_TRUE:PETSC_FALSE;
  ind_probl = DVD_IS(d->sEP, DVD_EP_INDEFINITE)?PETSC_TRUE:PETSC_FALSE;
  her_ind_probl = (her_probl || ind_probl)? PETSC_TRUE:PETSC_FALSE;

  /* Setting configuration constrains */
  b->max_size_proj = PetscMax(b->max_size_proj, b->max_size_V+cX_proj);
  d->W_shift = d->B?PETSC_TRUE:PETSC_FALSE;
  if (d->B && her_ind_probl && orth == EPS_ORTH_I) d->BV_shift = PETSC_TRUE;
  else d->BV_shift = PETSC_FALSE;
  b->own_scalars+= b->max_size_proj*b->max_size_proj*2*(std_probl?1:2) +
                                              /* H, G?, S, T? */
                   b->max_nev*b->max_nev*(her_ind_probl?0:(!d->B?1:2)) +
                                                /* cS?, cT? */
                   FromRealToScalar(d->eps->ncv)*(ind_probl?1:0) + /* nBV */
                   FromRealToScalar(b->max_size_proj)*(ind_probl?1:0) + /* nBpX */
                   (d->eps->arbitrary? b->size_V*2 : 0); /* rr, ri */
  b->max_size_auxV = PetscMax(PetscMax(b->max_size_auxV,
                    b->max_size_X),  /* updateV0 */
                    2);              /* arbitrary */
 
  max_cS = PetscMax(b->max_size_X,cX_proj);
  b->max_size_auxS = PetscMax(PetscMax(
    b->max_size_auxS,
    b->max_size_proj*b->max_size_proj*2*(std_probl?1:2) + /* updateAV1,BV1 */
      max_cS*b->max_nev*(her_ind_probl?0:(!d->B?1:2)) + /* updateV0,W0 */
                                                     /* SlepcReduction: in */
      PetscMax(
        b->max_size_proj*b->max_size_proj*2*(std_probl?1:2) + /* updateAV1,BV1 */
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
    d->max_size_proj = b->max_size_proj;
    d->real_H = b->free_scalars; b->free_scalars+= b->max_size_proj*b->max_size_proj;
    d->ldH = b->max_size_proj;
    d->S = b->free_scalars; b->free_scalars+= b->max_size_proj*b->max_size_proj;
    if (!her_ind_probl) {
      d->cS = b->free_scalars; b->free_scalars+= b->max_nev*b->max_nev;
      d->max_size_cS = d->ldcS = b->max_nev;
    } else {
      d->cS = NULL;
      d->max_size_cS = d->ldcS = 0;
      d->orthoV_type = orth;
      if (ind_probl) {
        d->real_nBV = (PetscReal*)b->free_scalars; b->free_scalars+= FromRealToScalar(d->eps->ncv);
        d->nBpX = (PetscReal*)b->free_scalars; b->free_scalars+= FromRealToScalar(d->max_size_proj);
      } else d->real_nBV = d->nBpX = NULL;
    }
    if (!std_probl) {
      d->real_G = b->free_scalars; b->free_scalars+= b->max_size_proj*b->max_size_proj;
      d->T = b->free_scalars; b->free_scalars+= b->max_size_proj*b->max_size_proj;
    } else {
      d->real_G = NULL;
      d->T = NULL;
    }
    if (d->B && !her_ind_probl) {
      d->cT = b->free_scalars; b->free_scalars+= b->max_nev*b->max_nev;
      d->ldcT = b->max_nev;
    } else {
      d->cT = NULL;
      d->ldcT = 0;
    }
    /* Create a DS if the method works with Schur decompositions */
    if (d->cS) {
      ierr = DSCreate(PetscObjectComm((PetscObject)d->eps->ds),&d->conv_ps);CHKERRQ(ierr);
      ierr = DSSetType(d->conv_ps,d->cT ? DSGNHEP : DSNHEP);CHKERRQ(ierr);
      /* Transfer as much as possible options from eps->ds to conv_ps */
      ierr = DSGetOptionsPrefix(d->eps->ds,&prefix);CHKERRQ(ierr);
      ierr = DSSetOptionsPrefix(d->conv_ps,prefix);CHKERRQ(ierr);
      ierr = DSSetFromOptions(d->conv_ps);CHKERRQ(ierr);
      ierr = DSGetEigenvalueComparison(d->eps->ds,&f,&ctx);CHKERRQ(ierr);
      ierr = DSSetEigenvalueComparison(d->conv_ps,f,ctx);CHKERRQ(ierr);
      ierr = DSAllocate(d->conv_ps,b->max_nev);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)d->eps,(PetscObject)d->conv_ps);CHKERRQ(ierr);
    } else {
      d->conv_ps = NULL;
    }
    d->calcPairs = dvd_calcpairs_proj;
    d->calcpairs_residual = dvd_calcpairs_res_0;
    d->calcpairs_residual_eig = dvd_calcpairs_eig_res_0;
    d->calcpairs_proj_res = dvd_calcpairs_proj_res;
    d->calcpairs_selectPairs = dvd_calcpairs_selectPairs;
    /* Create and configure a DS for solving the projected problems */
    if (d->W) {    /* If we use harmonics */
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
    /* Create various vector basis */
    if (harm) {
      ierr = BVDuplicate(d->eps->V,&d->W);CHKERRQ(ierr);
      ierr = BVResize(d->W,d->eps->ncv,PETSC_FALSE);CHKERRQ(ierr);
    } else d->W = NULL;
    ierr = BVDuplicate(d->eps->V,&d->AX);CHKERRQ(ierr);
    ierr = BVResize(d->AX,d->eps->ncv,PETSC_FALSE);CHKERRQ(ierr);
    ierr = BVDuplicate(d->eps->V,&d->auxBV);CHKERRQ(ierr);
    ierr = BVResize(d->auxBV,d->eps->ncv,PETSC_FALSE);CHKERRQ(ierr);
    if (d->B) {
      ierr = BVDuplicate(d->eps->V,&d->BX);CHKERRQ(ierr);
      ierr = BVResize(d->BX,d->eps->ncv,PETSC_FALSE);CHKERRQ(ierr);
    } else d->BX = NULL;
    ierr = BVGetColumn(d->eps->V,0,&v1);CHKERRQ(ierr);
    ierr = SlepcVecPoolCreate(v1,0,&d->auxV);CHKERRQ(ierr);
    ierr = BVRestoreColumn(d->eps->V,0,&v1);CHKERRQ(ierr);

    DVD_FL_ADD(d->startList, dvd_calcpairs_qz_start);
    DVD_FL_ADD(d->destroyList, dvd_calcpairs_qz_d);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_qz_start"
PetscErrorCode dvd_calcpairs_qz_start(dvdDashboard *d)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = BVSetActiveColumns(d->eps->V,0,0);CHKERRQ(ierr);
  if (d->W) { ierr = BVSetActiveColumns(d->W,0,0);CHKERRQ(ierr); }
  ierr = BVSetActiveColumns(d->AX,0,0);CHKERRQ(ierr);
  if (d->BX) { ierr = BVSetActiveColumns(d->BX,0,0);CHKERRQ(ierr); }
  d->size_H = 0;
  d->H = d->real_H;
  if (d->cS) for (i=0; i<d->max_size_cS*d->max_size_cS; i++) d->cS[i] = 0.0;
  d->size_G = 0;
  d->G = d->real_G;
  if (d->cT) for (i=0; i<d->max_size_cS*d->max_size_cS; i++) d->cT[i] = 0.0;
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
  ierr = BVDestroy(&d->W);CHKERRQ(ierr);
  ierr = BVDestroy(&d->AX);CHKERRQ(ierr);
  ierr = BVDestroy(&d->BX);CHKERRQ(ierr);
  ierr = SlepcVecPoolDestroy(&d->auxV);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_proj"
PetscErrorCode dvd_calcpairs_proj(dvdDashboard *d)
{
  PetscErrorCode ierr;
  PetscInt       i,l,k,lv,kv,lax,kax,lbx,kbx,lw,kw;
  Vec            v1,v2;

  PetscFunctionBegin;
  /* Update AV, BV, W and the projected matrices */
  /* 1. S <- S*MT */
  ierr = dvd_calcpairs_updateV0(d);CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateW0(d);CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateAV0(d);CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateBV0(d);CHKERRQ(ierr);
  /* 2. V <- orth(V, V_new) */
  ierr = dvd_calcpairs_updateV1(d);CHKERRQ(ierr);
  /* 3. AV <- [AV A * V(V_new_s:V_new_e-1)] */
  /* Check consistency */
  ierr = BVGetActiveColumns(d->AX,&l,&k);CHKERRQ(ierr);
  if (k-l != d->V_new_s) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");
  for (i=d->V_new_s; i<d->V_new_e; i++) {
    ierr = BVGetColumn(d->eps->V,i,&v1);CHKERRQ(ierr);
    ierr = BVGetColumn(d->AX,i,&v2);CHKERRQ(ierr);
    ierr = MatMult(d->A,v1,v2);CHKERRQ(ierr);
    ierr = BVRestoreColumn(d->eps->V,i,&v1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(d->AX,i,&v2);CHKERRQ(ierr);
  }
  ierr = BVSetActiveColumns(d->AX,l,l+d->V_new_e);CHKERRQ(ierr);
  /* 4. BV <- [BV B * V(V_new_s:V_new_e-1)] */
  if (d->B) {
    /* Check consistency */
    ierr = BVGetActiveColumns(d->BX,&l,&k);CHKERRQ(ierr);
    if (k-l != d->V_new_s) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");
    for (i=d->V_new_s; i<d->V_new_e; i++) {
      ierr = BVGetColumn(d->eps->V,i,&v1);CHKERRQ(ierr);
      ierr = BVGetColumn(d->BX,i,&v2);CHKERRQ(ierr);
      ierr = MatMult(d->B,v1,v2);CHKERRQ(ierr);
      ierr = BVRestoreColumn(d->eps->V,i,&v1);CHKERRQ(ierr);
      ierr = BVRestoreColumn(d->BX,i,&v2);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(d->BX,l,l+d->V_new_e);CHKERRQ(ierr);
  }
  /* 5 <- W <- [W f(AV,BV)] */
  ierr = dvd_calcpairs_updateW1(d);CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateAV1(d);CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateBV1(d);CHKERRQ(ierr);

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
  ierr = BVGetActiveColumns(d->eps->V,&lv,&kv);CHKERRQ(ierr);
  ierr = BVGetActiveColumns(d->AX,&lax,&kax);CHKERRQ(ierr);
  if (d->BX) { ierr = BVGetActiveColumns(d->BX,&lbx,&kbx);CHKERRQ(ierr); }
  if (d->W) { ierr = BVGetActiveColumns(d->W,&lw,&kw);CHKERRQ(ierr); }
  if (kv-lv != d->V_new_e || kv-lv+d->cX_in_H != d->size_H || d->cX_in_V != d->cX_in_H ||
      kv-lv != kax-lax || d->cX_in_H != d->cX_in_AV ||
        (DVD_ISNOT(d->sEP, DVD_EP_STD) && (
          kv-lv+d->cX_in_G != d->size_G || d->cX_in_H != d->cX_in_G ||
          d->size_H != d->size_G || (d->BX && (
            kv-lv != kbx-lbx || d->cX_in_H != d->cX_in_BV)))) ||
      (d->W && kw-lw != kv-lv)) {
    SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");
  }
  PetscFunctionReturn(0);
}

/**** Basic routines **********************************************************/

#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_updateV0"
/* auxV: V_tra_s */
PetscErrorCode dvd_calcpairs_updateV0(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscInt        j,ld,k,l;
  PetscScalar     *array;
  Mat             Q,S;

  PetscFunctionBegin;
  if (d->V_tra_s == 0 && d->V_tra_e == 0) PetscFunctionReturn(0);

  /* Update nBcX and nBV */
  /*if (d->nBcX && d->nBpX && d->nBV) {
    d->nBV+= d->V_tra_s;
    for (i=0; i<d->V_tra_s; i++) d->nBcX[d->size_cX+i] = d->nBpX[i];
    for (i=d->V_tra_s; i<d->V_tra_e; i++) d->nBV[i-d->V_tra_s] = d->nBpX[i];
  }*/

  /* cX <- [cX V*MT(0:V_tra_s-1)], V <- V*MT(V_tra_s:V_tra_e) */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->eps->V,DS_MAT_Q);CHKERRQ(ierr);

  /* Udpate cS for standard problems */
  ierr = BVGetActiveColumns(d->eps->V,&l,&k);CHKERRQ(ierr);
  if (d->cS && !d->cT && l >= d->nev) {
    /* Check consistency */
    if (d->size_cS+d->V_tra_s != l) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

    /* auxV <- AV * ps.Q(0:V_tra_e-1) */
    ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
    ierr = DSGetMat(d->ps,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(d->auxBV,0,d->V_tra_e);CHKERRQ(ierr);
    ierr = BVMult(d->auxBV,1.0,0.0,d->AX,Q);CHKERRQ(ierr);
    ierr = MatDestroy(&Q);CHKERRQ(ierr);

    /* cS(:, size_cS:) <- cX' * auxV */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,l,l,NULL,&S);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(d->eps->V,0,l);CHKERRQ(ierr);
    ierr = BVDot(d->auxBV,d->eps->V,S);CHKERRQ(ierr);
    ierr = MatDenseGetArray(S,&array);CHKERRQ(ierr);
    for (j=0;j<l;j++) {
      ierr = PetscMemcpy(&d->cS[d->ldcS*(d->size_cS+j)],array+j*l,l*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(S,&array);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(d->eps->V,l,k);CHKERRQ(ierr);
    d->size_cS+= d->V_tra_s;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_updateV1"
/* auxS: size_cX+V_new_e+1 */
PetscErrorCode dvd_calcpairs_updateV1(dvdDashboard *d)
{
  PetscErrorCode ierr;
  PetscInt       l,k;

  PetscFunctionBegin;
  if (d->V_new_s == d->V_new_e) PetscFunctionReturn(0);

  /* Check consistency */
  ierr = BVGetActiveColumns(d->eps->V,&l,&k);CHKERRQ(ierr);
  if (k-l != d->V_new_s) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  /* V <- gs([cX V(0:V_new_s-1)], V(V_new_s:V_new_e-1)) */
  ierr = dvd_orthV(d->eps->V,d->V_new_s,d->V_new_e,d->eps->rand);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(d->eps->V,l,d->V_new_e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_updateW0"
/* auxV: V_tra_s, DvdMult_copy_func: 2 */
PetscErrorCode dvd_calcpairs_updateW0(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscInt        j,ld,l,k,lv,kv;
  PetscScalar     *array;
  Mat             S,T,Q;

  PetscFunctionBegin;
  if (d->V_tra_s == 0 && d->V_tra_e == 0) PetscFunctionReturn(0);

  /* cY <- [cY W*ps.Z(0:V_tra_s-1)], W <- W*ps.Z(V_tra_s:V_tra_e) */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->W,DS_MAT_Z);CHKERRQ(ierr);

  /* Udpate cS and cT */
  ierr = BVGetActiveColumns(d->eps->V,&lv,&kv);CHKERRQ(ierr);
  if (d->cT && l >= d->nev) {
    /* Check consistency */
    ierr = BVGetActiveColumns(d->W,&l,&k);CHKERRQ(ierr);
    if (d->size_cS+d->V_tra_s != lv || (d->W && k-l != lv)) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

    ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
    /* auxV <- AV * ps.Q(0:V_tra_e-1) */
    ierr = DSGetMat(d->ps,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(d->auxBV,0,d->V_tra_e);CHKERRQ(ierr);
    ierr = BVMult(d->auxBV,1.0,0.0,d->AX,Q);CHKERRQ(ierr);

    /* cS(:, size_cS:) <- cY' * auxV */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,l,l,NULL,&S);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(d->W,0,l);CHKERRQ(ierr);
    ierr = BVDot(d->auxBV,d->W,S);CHKERRQ(ierr);
    ierr = MatDenseGetArray(S,&array);CHKERRQ(ierr);
    for (j=0;j<l;j++) {
      ierr = PetscMemcpy(&d->cS[d->ldcS*(d->size_cS+j)],array+j*l,l*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(S,&array);CHKERRQ(ierr);

    /* auxV <- BV * ps.Q(0:V_tra_e-1) */
    ierr = BVMult(d->auxBV,1.0,0.0,d->BX,Q);CHKERRQ(ierr);
    ierr = MatDestroy(&Q);CHKERRQ(ierr);

    /* cT(:, size_cS:) <- cY' * auxV */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,l,l,NULL,&T);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(d->W,0,l);CHKERRQ(ierr);
    ierr = BVDot(d->auxBV,d->W,T);CHKERRQ(ierr);
    ierr = MatDenseGetArray(T,&array);CHKERRQ(ierr);
    for (j=0;j<l;j++) {
      ierr = PetscMemcpy(&d->cT[d->ldcT*(d->size_cT+j)],array+j*l,l*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(T,&array);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(d->W,l,k);CHKERRQ(ierr);

    ierr = BVGetActiveColumns(d->eps->V,&d->size_cS,NULL);CHKERRQ(ierr);
    d->size_cT = d->size_cS;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_updateW1"
/* auxS: size_cX+V_new_e+1 */
PetscErrorCode dvd_calcpairs_updateW1(dvdDashboard *d)
{
  PetscErrorCode ierr;
  PetscInt       l,k;

  PetscFunctionBegin;
  if (!d->W || d->V_new_s == d->V_new_e) PetscFunctionReturn(0);

  /* Check consistency */
  ierr = BVGetActiveColumns(d->W,&l,&k);CHKERRQ(ierr);
  if (k-l != d->V_new_s) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  /* Update W */
  ierr = d->calcpairs_W(d);CHKERRQ(ierr);

  /* W <- gs([cY W(0:V_new_s-1)], W(V_new_s:V_new_e-1)) */
  ierr = dvd_orthV(d->W,d->V_new_s,d->V_new_e,d->eps->rand);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(d->W,l,d->V_new_e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_updateAV0"
/* auxS: size_H*(V_tra_e-V_tra_s) */
PetscErrorCode dvd_calcpairs_updateAV0(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscInt        cMT,ld;
  PetscScalar     *pQ,*pZ;

  PetscFunctionBegin;
  if (d->V_tra_s == 0 && d->V_tra_e == 0) PetscFunctionReturn(0);

  /* AV(V_tra_s-cp-1:) = cAV*ps.Q(V_tra_s:) */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->AX,DS_MAT_Q);CHKERRQ(ierr);
  cMT = d->V_tra_e - d->V_tra_s;

  /* Update H <- ps.Z(tra_s)' * (H * ps.Q(tra_s:)) */
  ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
  ierr = DSGetArray(d->ps,DS_MAT_Q,&pQ);CHKERRQ(ierr);
  if (d->W) {
    ierr = DSGetArray(d->ps,DS_MAT_Z,&pZ);CHKERRQ(ierr);
  } else pZ = pQ;
  ierr = SlepcDenseMatProdTriang(d->auxS,0,d->ldH,d->H,d->sH,d->ldH,d->size_H,d->size_H,PETSC_FALSE,&pQ[ld*d->V_tra_s],0,ld,d->size_MT,cMT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = SlepcDenseMatProdTriang(d->H,d->sH,d->ldH,&pZ[ld*d->V_tra_s],0,ld,d->size_MT,cMT,PETSC_TRUE,d->auxS,0,d->ldH,d->size_H,cMT,PETSC_FALSE);CHKERRQ(ierr);
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
PetscErrorCode dvd_calcpairs_updateAV1(dvdDashboard *d)
{
  PetscErrorCode ierr;
  PetscInt       k,l;

  PetscFunctionBegin;
  if (d->V_new_s == d->V_new_e) PetscFunctionReturn(0);

  /* Check consistency */
  ierr = BVGetActiveColumns(d->eps->V,&l,&k);CHKERRQ(ierr);
  if (d->size_H != d->V_new_s+d->cX_in_H || k-l != d->V_new_e) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  /* H = [H               W(old)'*AV(new);
          W(new)'*AV(old) W(new)'*AV(new) ],
     being old=0:V_new_s-1, new=V_new_s:V_new_e-1 */
  ierr = BVMultS(d->AX,d->W?d->W:d->eps->V,d->H,d->ldH);CHKERRQ(ierr);
  d->size_H = d->V_new_e+d->cX_in_H;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_updateBV0"
/* auxS: max(BcX*(size_cX+V_new_e+1), size_G*(V_tra_e-V_tra_s)) */
PetscErrorCode dvd_calcpairs_updateBV0(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscInt        cMT,tra_s,ld;
  PetscScalar     *pQ,*pZ;

  PetscFunctionBegin;
  if (d->V_tra_s == 0 && d->V_tra_e == 0) PetscFunctionReturn(0);

  /* BV <- BV*MT */
  ierr = dvd_calcpairs_updateBV0_gen(d,d->BX,DS_MAT_Q);CHKERRQ(ierr);

  /* Update G <- ps.Z' * (G * ps.Q) */
  if (d->G) {
    tra_s = PetscMax(d->V_tra_s-d->max_cX_in_proj,0);
    cMT = d->V_tra_e - tra_s;
    ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
    ierr = DSGetArray(d->ps,DS_MAT_Q,&pQ);CHKERRQ(ierr);
    if (d->W) {
      ierr = DSGetArray(d->ps,DS_MAT_Z,&pZ);CHKERRQ(ierr);
    } else pZ = pQ;
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
PetscErrorCode dvd_calcpairs_updateBV1(dvdDashboard *d)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!d->G || d->V_new_s == d->V_new_e) PetscFunctionReturn(0);

  /* G = [G               W(old)'*BV(new);
          W(new)'*BV(old) W(new)'*BV(new) ],
     being old=0:V_new_s-1, new=V_new_s:V_new_e-1 */
  ierr = BVMultS(d->BX?d->BX:d->eps->V,d->W?d->W:d->eps->V,d->G,d->ldH);CHKERRQ(ierr);
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
  ierr = DSSetDimensions(d->ps,d->size_H,0,0,0);CHKERRQ(ierr);
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
#define __FUNCT__ "dvd_calcpairs_apply_arbitrary"
PetscErrorCode dvd_calcpairs_apply_arbitrary(dvdDashboard *d,PetscInt r_s,PetscInt r_e,PetscScalar **rr_,PetscScalar **ri_)
{
  PetscInt        i,k,ld;
  PetscScalar     *pX,*rr,*ri,ar,ai;
  Vec             *X,xr,xi;
  PetscErrorCode  ierr;
#if defined(PETSC_USE_COMPLEX)
  PetscInt        N=2;
#else
  PetscInt        N=1,j;
#endif

  PetscFunctionBegin;
  /* Quick exit without neither arbitrary selection nor harmonic extraction */
  if (!d->eps->arbitrary && !d->calcpairs_eig_backtrans) {
    *rr_ = d->eigr-d->cX_in_H;
    *ri_ = d->eigi-d->cX_in_H;
    PetscFunctionReturn(0);
  }

  /* Quick exit without arbitrary selection, but with harmonic extraction */
  if (!d->eps->arbitrary && d->calcpairs_eig_backtrans) {
    *rr_ = rr = d->auxS;
    *ri_ = ri = d->auxS+r_e-r_s;
    for (i=r_s; i<r_e; i++) {
      ierr = d->calcpairs_eig_backtrans(d,d->eigr[i],d->eigi[i],&rr[i-r_s],&ri[i-r_s]);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  ierr = SlepcVecPoolGetVecs(d->auxV,N,&X);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
  *rr_ = rr = d->eps->rr + d->eps->nconv;
  *ri_ = ri = d->eps->ri + d->eps->nconv;
  for (i=r_s; i<r_e; i++) {
    k = i;
    ierr = DSVectors(d->ps,DS_MAT_X,&k,NULL);CHKERRQ(ierr);
    ierr = DSNormalize(d->ps,DS_MAT_X,i);CHKERRQ(ierr);
    ierr = DSGetArray(d->ps,DS_MAT_X,&pX);CHKERRQ(ierr);
    ierr = dvd_improvex_compute_X(d,i,k+1,X,pX,ld);CHKERRQ(ierr);
    ierr = DSRestoreArray(d->ps,DS_MAT_X,&pX);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    if (d->nX[i] != 1.0) {
      for (j=i; j<k+1; j++) {
        ierr = VecScale(X[j-i],1.0/d->nX[i]);CHKERRQ(ierr);
      }
    }
    xr = X[0];
    xi = X[1];
    if (i == k) {
      ierr = VecZeroEntries(xi);CHKERRQ(ierr);
    }
#else
    xr = X[0];
    xi = NULL;
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
    ierr = (d->eps->arbitrary)(ar,ai,xr,xi,&rr[i-r_s],&ri[i-r_s],d->eps->arbitraryctx);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    if (i != k) {
      rr[i+1-r_s] = rr[i-r_s];
      ri[i+1-r_s] = ri[i-r_s];
      i++;
    }
#endif
  }
  ierr = SlepcVecPoolRestoreVecs(d->auxV,N,&X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_selectPairs"
PetscErrorCode dvd_calcpairs_selectPairs(dvdDashboard *d,PetscInt n)
{
  PetscInt        k;
  PetscScalar     *rr,*ri;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  n = PetscMin(n,d->size_H-d->cX_in_H);
  /* Put the best n pairs at the beginning. Useful for restarting */
  ierr = DSSetDimensions(d->ps,0,0,d->cX_in_H,0);CHKERRQ(ierr);
  ierr = dvd_calcpairs_apply_arbitrary(d,d->cX_in_H,d->size_H,&rr,&ri);CHKERRQ(ierr);
  k = n;
  ierr = DSSort(d->ps,d->eigr-d->cX_in_H,d->eigi-d->cX_in_H,rr,ri,&k);CHKERRQ(ierr);
  /* Put the best pair at the beginning. Useful to check its residual */
#if !defined(PETSC_USE_COMPLEX)
  if (n != 1 && (n != 2 || d->eigi[0] == 0.0))
#else
  if (n != 1)
#endif
  {
    ierr = dvd_calcpairs_apply_arbitrary(d,d->cX_in_H,d->size_H,&rr,&ri);CHKERRQ(ierr);
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
PetscErrorCode dvd_calcpairs_res_0(dvdDashboard *d,PetscInt r_s,PetscInt r_e)
{
  PetscInt        i,ldpX;
  PetscScalar     *pX;
  PetscErrorCode  ierr;
  BV              BX = d->BX?d->BX:d->eps->V;
  Vec             *R;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(d->ps,&ldpX);CHKERRQ(ierr);
  ierr = DSGetArray(d->ps,DS_MAT_Q,&pX);CHKERRQ(ierr);
  /* nX(i) <- ||X(i)|| */
  ierr = dvd_improvex_compute_X(d,r_s,r_e,NULL,pX,ldpX);CHKERRQ(ierr);
  ierr = SlepcVecPoolGetVecs(d->auxV,r_e-r_s,&R);CHKERRQ(ierr);
  for (i=r_s; i<r_e; i++) {
    /* R(i-r_s) <- AV*pX(i) */
    ierr = BVMultVec(d->AX,1.0,0.0,R[i-r_s],&pX[ldpX*(i+d->cX_in_H)]);CHKERRQ(ierr);
    /* R(i-r_s) <- R(i-r_s) - eigr(i)*BX*pX(i) */
    ierr = BVMultVec(BX,-d->eigr[i+d->cX_in_H],1.0,R[i-r_s],&pX[ldpX*(i+d->cX_in_H)]);CHKERRQ(ierr);
  }
  ierr = DSRestoreArray(d->ps,DS_MAT_Q,&pX);CHKERRQ(ierr);
  ierr = d->calcpairs_proj_res(d,r_s,r_e,R);CHKERRQ(ierr);
  ierr = SlepcVecPoolRestoreVecs(d->auxV,r_e-r_s,&R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_proj_res"
PetscErrorCode dvd_calcpairs_proj_res(dvdDashboard *d,PetscInt r_s,PetscInt r_e,Vec *R)
{
  PetscInt        i,l,k;
  PetscErrorCode  ierr;
  PetscBool       lindep;
  BV              cX;

  PetscFunctionBegin;
  /* If exists left subspace, R <- orth(cY, R), nR[i] <- ||R[i]|| */
  if (d->W) cX = d->W;

  /* If not HEP, R <- orth(cX, R), nR[i] <- ||R[i]|| */
  else if (!(DVD_IS(d->sEP, DVD_EP_STD) && DVD_IS(d->sEP, DVD_EP_HERMITIAN))) cX = d->eps->V;

  /* Otherwise, nR[i] <- ||R[i]|| */
  else cX = NULL;

  if (cX) {
    ierr = BVGetActiveColumns(cX,&l,&k);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(cX,0,l);CHKERRQ(ierr);
    for (i=0; i<r_e-r_s; i++) {
      ierr = BVOrthogonalizeVec(cX,R[i],NULL,&d->nR[r_s+i],&lindep);CHKERRQ(ierr);
    }
    ierr = BVSetActiveColumns(cX,l,k);CHKERRQ(ierr);
    if (lindep || (PetscAbs(d->nR[r_s+i]) < PETSC_MACHINE_EPSILON)) {
      ierr = PetscInfo2(d->eps,"The computed eigenvector residual %D is too low, %g!\n",r_s+i,(double)(d->nR[r_s+i]));CHKERRQ(ierr);
    }
  } else {
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
PetscErrorCode dvd_calcpairs_eig_res_0(dvdDashboard *d,PetscInt r_s,PetscInt r_e)
{
  PetscInt        i,n,ld,ldc,l,k,j;
  PetscErrorCode  ierr;
  Vec             *Bx,auxV;
  PetscScalar     *cS,*cT,*pcX,*pX,*pX0,*array;
  Mat             Q,S;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar     b[8];
  Vec             X[4];
#endif
  Vec             *R;

  PetscFunctionBegin;
  /* Quick return */
  if (!d->cS) PetscFunctionReturn(0);

  /* Check consistency */
  if (d->size_auxS < d->size_H*(r_e-r_s) /* pX0 */) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  /*
    Compute expanded cS = conv_ps.A, cT = conv_ps.B:
    conv_ps.A = [ cX'*A*cX    cX'*A*X ]
                [  X'*A*cX     X'*A*X ], where cX'*A*cX = cS and X = V*ps.Q
  */ 
  ierr = BVGetActiveColumns(d->eps->V,&l,NULL);CHKERRQ(ierr);
  n = l+r_e;
  ierr = DSSetDimensions(d->conv_ps,n,0,0,0);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(d->conv_ps,&ldc);CHKERRQ(ierr);
  ierr = DSGetArray(d->conv_ps,DS_MAT_A,&cS);CHKERRQ(ierr);
  ierr = SlepcDenseCopyTriang(cS,0,ldc,d->cS,0,d->ldcS,d->size_cS,d->size_cS);CHKERRQ(ierr);
  if (d->cT) {
    ierr = DSGetArray(d->conv_ps,DS_MAT_B,&cT);CHKERRQ(ierr);
    ierr = SlepcDenseCopyTriang(cT,0,ldc,d->cT,0,d->ldcT,d->size_cS,d->size_cS);CHKERRQ(ierr);
  }
  /* auxV <- A*X = AV * pX(0:r_e+cX_in_H) */
  ierr = SlepcVecPoolGetVecs(d->auxV,r_e-r_s,&R);CHKERRQ(ierr);
  ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
  ierr = DSGetMat(d->ps,DS_MAT_Q,&Q);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(d->auxBV,0,d->V_tra_e);CHKERRQ(ierr);
  ierr = BVMult(d->auxBV,1.0,0.0,d->AX,Q);CHKERRQ(ierr);
  /* cS(:, size_cS:) <- cX' * auxV */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,l,l,NULL,&S);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(d->eps->V,0,l);CHKERRQ(ierr);
  ierr = BVDot(d->auxBV,d->eps->V,S);CHKERRQ(ierr);
  ierr = MatDenseGetArray(S,&array);CHKERRQ(ierr);
  for (j=0;j<l;j++) {
    ierr = PetscMemcpy(&cS[ld*(d->size_cS+j)],array+j*l,l*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(S,&array);CHKERRQ(ierr);

  if (d->cT) {
    /* R <- BV * pX(0:r_e+cX_in_H) */
    ierr = BVMult(d->auxBV,1.0,0.0,d->BX,Q);CHKERRQ(ierr);
    /* cT(:, size_cS:) <- cX' * auxV */
    ierr = BVDot(d->auxBV,d->eps->V,S);CHKERRQ(ierr);
    ierr = MatDenseGetArray(S,&array);CHKERRQ(ierr);
    for (j=0;j<l;j++) {
      ierr = PetscMemcpy(&cT[ld*(d->size_cS+j)],array+j*l,l*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(S,&array);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&Q);CHKERRQ(ierr);

  ierr = DSRestoreArray(d->conv_ps,DS_MAT_A,&cS);CHKERRQ(ierr);
  if (d->cT) {
    ierr = DSRestoreArray(d->conv_ps,DS_MAT_B,&cT);CHKERRQ(ierr);
  }
  ierr = DSSetState(d->conv_ps,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
  /* eig(S,T) */
  k = l+r_s;
  ierr = DSVectors(d->conv_ps,DS_MAT_X,&k,NULL);CHKERRQ(ierr);
  ierr = DSNormalize(d->conv_ps,DS_MAT_X,l+r_s);CHKERRQ(ierr);
  /* pX0 <- ps.Q(0:d->cX_in_AV+r_e-1) * conv_ps.X(size_cX-cX_in_H:) */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,d->size_H,l,NULL,&S);CHKERRQ(ierr);
  ierr = MatDenseGetArray(S,&pX0);CHKERRQ(ierr);
  ierr = DSGetArray(d->conv_ps,DS_MAT_X,&pcX);CHKERRQ(ierr);
  ierr = SlepcDenseMatProd(pX0,d->size_H,0.0,1.0,&pX[(d->cX_in_AV+r_s)*ld],ld,d->size_H,r_e-r_s,PETSC_FALSE,&pcX[l+l*ldc],ldc,r_e+d->cX_in_H,r_e-r_s,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DSRestoreArray(d->ps,DS_MAT_Q,&pX);CHKERRQ(ierr);
  ierr = DSRestoreArray(d->conv_ps,DS_MAT_X,&pcX);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(S,&pX0);CHKERRQ(ierr);
  /* auxV <- cX(0:size_cX-cX_in_AV)*conv_ps.X + V*pX0 */
  ierr = DSGetMat(d->conv_ps,DS_MAT_X,&Q);CHKERRQ(ierr);
  ierr = BVMult(d->auxBV,1.0,0.0,d->eps->V,Q);CHKERRQ(ierr);
  ierr = MatDestroy(&Q);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(d->eps->V,l,k);CHKERRQ(ierr);
  ierr = BVMult(d->auxBV,1.0,1.0,d->eps->V,S);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  /* nX <- ||auxV|| */
  for (i=0;i<r_e-r_s;i++) {
    ierr = BVNormColumn(d->auxBV,i,NORM_2,&d->nX[r_s+i]);CHKERRQ(ierr);
  }
  /* R <- A*auxV */
  for (i=0; i<r_e-r_s; i++) {
    ierr = BVGetColumn(d->auxBV,i,&auxV);CHKERRQ(ierr);
    ierr = MatMult(d->A,auxV,R[i]);CHKERRQ(ierr);
    ierr = BVRestoreColumn(d->auxBV,i,&auxV);CHKERRQ(ierr);
  }
  /* Bx <- B*auxV */
  ierr = SlepcVecPoolGetVecs(d->auxV,r_e-r_s,&Bx);CHKERRQ(ierr);
  for (i=0; i<r_e-r_s; i++) {
    ierr = BVGetColumn(d->auxBV,i,&auxV);CHKERRQ(ierr);
    if (d->B) {
      ierr = MatMult(d->B,auxV,Bx[i]);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(auxV,Bx[i]);CHKERRQ(ierr);
    }
    ierr = BVRestoreColumn(d->auxBV,i,&auxV);CHKERRQ(ierr);
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
  ierr = SlepcVecPoolGetVecs(d->auxV,r_e-r_s,&Bx);CHKERRQ(ierr);
  /* nR <- ||R|| */
  for (i=0;i<r_e-r_s;i++) {
    ierr = VecNormBegin(R[i],NORM_2,&d->nR[r_s+i]);CHKERRQ(ierr);
  }
  for (i=0;i<r_e-r_s;i++) {
    ierr = VecNormEnd(R[i],NORM_2,&d->nR[r_s+i]);CHKERRQ(ierr);
  }
  ierr = SlepcVecPoolRestoreVecs(d->auxV,r_e-r_s,&R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/**** Pattern routines ********************************************************/

/* BV <- BV*MT */
#undef __FUNCT__
#define __FUNCT__ "dvd_calcpairs_updateBV0_gen"
PETSC_STATIC_INLINE PetscErrorCode dvd_calcpairs_updateBV0_gen(dvdDashboard *d,BV bv,DSMatType mat)
{
  PetscErrorCode  ierr;
  PetscInt        ld,l,k;
  Mat             MT;

  PetscFunctionBegin;
  ierr = DSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
  ierr = BVGetActiveColumns(bv,&l,&k);CHKERRQ(ierr);
  ierr = DSGetMat(d->ps,mat,&MT);CHKERRQ(ierr);
  ierr = BVMultInPlace(bv,MT,d->V_tra_s,d->V_tra_e);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(bv,l,d->V_tra_e);CHKERRQ(ierr);
  ierr = MatDestroy(&MT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
