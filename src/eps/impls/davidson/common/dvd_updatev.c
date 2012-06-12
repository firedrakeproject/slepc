/*
  SLEPc eigensolver: "davidson"

  Step: test for restarting, updateV, restartV

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

PetscErrorCode dvd_updateV_start(dvdDashboard *d);
PetscBool dvd_isrestarting_fullV(dvdDashboard *d);
PetscErrorCode dvd_managementV_basic_d(dvdDashboard *d);
PetscErrorCode dvd_updateV_extrapol(dvdDashboard *d);
PetscErrorCode dvd_updateV_conv_gen(dvdDashboard *d);
PetscErrorCode dvd_updateV_restart_gen(dvdDashboard *d);
PetscErrorCode dvd_updateV_update_gen(dvdDashboard *d);
PetscErrorCode dvd_updateV_conv_finish(dvdDashboard *d);
PetscErrorCode dvd_updateV_testConv(dvdDashboard *d, PetscInt s, PetscInt pre,
                                    PetscInt e, Vec *auxV, PetscScalar *auxS,
                                    PetscInt *nConv);

typedef struct {
  PetscInt
    min_size_V,     /* restart with this number of eigenvectors */
    plusk,          /* when restart, save plusk vectors from last iteration */
    mpd;            /* max size of the searching subspace */
  void
    *old_updateV_data;
                    /* old updateV data */
  isRestarting_type
    old_isRestarting;
                    /* old isRestarting */
  PetscScalar
    *oldU,          /* previous projected right igenvectors */
    *oldV;          /* previous projected left eigenvectors */
  PetscInt
    ldoldU,         /* leading dimension of oldU */
    size_oldU;      /* size of oldU */
  PetscBool 
    allResiduals;   /* if computing all the residuals */
} dvdManagV_basic;

#define _Ceil(A,B) ((A)/(B)+((A)%(B)==0?0:1))
#define FromRealToScalar(S) ((PetscInt)_Ceil((S)*sizeof(PetscReal),sizeof(PetscScalar)))

#undef __FUNCT__  
#define __FUNCT__ "dvd_managementV_basic"
PetscErrorCode dvd_managementV_basic(dvdDashboard *d, dvdBlackboard *b,
                                     PetscInt bs, PetscInt mpd,
                                     PetscInt min_size_V,
                                     PetscInt plusk, PetscBool harm,
                                     PetscBool allResiduals)
{
  PetscErrorCode  ierr;
  dvdManagV_basic *data;
#if !defined(PETSC_USE_COMPLEX)
  PetscBool       her_probl, std_probl;
#endif

  PetscFunctionBegin;
  /* Setting configuration constrains */
#if !defined(PETSC_USE_COMPLEX)
  /* if the last converged eigenvalue is complex its conjugate pair is also
     converged */
  her_probl = DVD_IS(d->sEP, DVD_EP_HERMITIAN)?PETSC_TRUE:PETSC_FALSE;
  std_probl = DVD_IS(d->sEP, DVD_EP_STD)?PETSC_TRUE:PETSC_FALSE;
  b->max_size_X = PetscMax(b->max_size_X, bs+(her_probl && std_probl)?0:1);
#else
  b->max_size_X = PetscMax(b->max_size_X, bs);
#endif

  b->max_size_V = PetscMax(b->max_size_V, mpd);
  min_size_V = PetscMin(min_size_V, mpd-bs);
  b->max_size_auxV = PetscMax(b->max_size_auxV, 1); /* dvd_updateV_testConv */
  b->max_size_auxS = PetscMax(b->max_size_auxS, b->max_size_V*2 /* SlepcDenseOrth  */ );
  b->size_V = PetscMax(b->size_V, b->max_size_V + b->max_size_P + b->max_nev);
  b->own_scalars+= b->size_V*2 /* eigr, eigr */ +
                   b->size_V /* nR */   +
                   b->size_V /* nX */   +
                   b->size_V /* errest */ +
                   b->max_size_V*b->max_size_V*(harm?2:1)*(plusk>0?2:1)
                                               /* MTX,MTY?,oldU,oldV? */;
  b->max_size_oldX = plusk;

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscMalloc(sizeof(dvdManagV_basic), &data); CHKERRQ(ierr);
    data->mpd = b->max_size_V;
    data->min_size_V = min_size_V;
    d->bs = bs;
    d->max_size_X = b->max_size_X;
    data->plusk = plusk;
    data->allResiduals = allResiduals;

    d->size_real_eigr = b->size_V;
    d->real_eigr = b->free_scalars; b->free_scalars+= b->size_V;
    d->real_eigi = b->free_scalars; b->free_scalars+= b->size_V;
    d->real_nR = (PetscReal*)b->free_scalars; b->free_scalars+= FromRealToScalar(b->size_V);
    d->real_nX = (PetscReal*)b->free_scalars; b->free_scalars+= FromRealToScalar(b->size_V);
    d->real_errest = (PetscReal*)b->free_scalars; b->free_scalars+= FromRealToScalar(b->size_V);
    d->MTX = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    if (plusk > 0) {
      data->oldU = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    }
    if (harm) {
      d->MTY = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
      if (plusk > 0) {
        data->oldV = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
      }
    } else {
      d->MTY = PETSC_NULL;
      data->oldV = PETSC_NULL;
    }

    data->old_updateV_data = d->updateV_data;
    d->updateV_data = data;
    data->old_isRestarting = d->isRestarting;
    d->isRestarting = dvd_isrestarting_fullV;
    d->updateV = dvd_updateV_extrapol;
    d->preTestConv = dvd_updateV_testConv;
    DVD_FL_ADD(d->startList, dvd_updateV_start);
    DVD_FL_ADD(d->endList, dvd_updateV_conv_finish);
    DVD_FL_ADD(d->destroyList, dvd_managementV_basic_d);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_start"
PetscErrorCode dvd_updateV_start(dvdDashboard *d)
{
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;
  PetscInt        i;

  PetscFunctionBegin;

  d->size_cX = 0;
  d->eigr = d->ceigr = d->real_eigr;
  d->eigi = d->ceigi = d->real_eigi;
#if defined(PETSC_USE_COMPLEX)
  for(i=0; i<d->size_real_V; i++) d->eigi[i] = 0.0;
#endif
  d->nR = d->real_nR;
  for(i=0; i<d->size_real_V; i++) d->nR[i] = PETSC_MAX_REAL;
  d->nX = d->real_nX;
  d->errest = d->real_errest;
  for(i=0; i<d->size_real_V; i++) d->errest[i] = PETSC_MAX_REAL;
  data->ldoldU = 0;
  data->oldV = PETSC_NULL;
  d->ldMTY = 0;
  data->size_oldU = 0;
  d->nconv = 0;
  d->npreconv = 0;
  d->V_tra_s = d->V_tra_e = d->V_new_s = d->V_new_e = 0;
  d->size_D = 0;

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_isrestarting_fullV"
PetscBool dvd_isrestarting_fullV(dvdDashboard *d)
{
  PetscBool       restart;
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;

  PetscFunctionBegin;

  restart = (d->size_V + d->max_size_X > PetscMin(data->mpd,d->max_size_V))?
                PETSC_TRUE:PETSC_FALSE;

  /* Check old isRestarting function */
  if (!restart && data->old_isRestarting)
    restart = data->old_isRestarting(d);

  PetscFunctionReturn(restart);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_managementV_basic_d"
PetscErrorCode dvd_managementV_basic_d(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;

  PetscFunctionBegin;

  /* Restore changes in dvdDashboard */
  d->updateV_data = data->old_updateV_data;
  
  /* Free local data */
  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_extrapol"
PetscErrorCode dvd_updateV_extrapol(dvdDashboard *d)
{
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = d->calcpairs_selectPairs(d, data->min_size_V); CHKERRQ(ierr);

  /* If the subspaces doesn't need restart, add new vector */
  if (!d->isRestarting(d)) {
    d->size_D = 0;
    ierr = dvd_updateV_update_gen(d); CHKERRQ(ierr);

    /* If some vector were add, exit */
    if (d->size_D > 0) { PetscFunctionReturn(0); }
  }

  /* If some eigenpairs were converged, lock them  */
  if (d->npreconv > 0) {
    i = d->npreconv;
    ierr = dvd_updateV_conv_gen(d); CHKERRQ(ierr);

    /* If some eigenpair was locked, exit */
    if (i > d->npreconv) { PetscFunctionReturn(0); }
  }

  /* Else, a restarting is performed */
  ierr = dvd_updateV_restart_gen(d); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_conv_gen"
PetscErrorCode dvd_updateV_conv_gen(dvdDashboard *d)
{
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;
  PetscInt        npreconv,ld,cMT,cMTX;
  PetscErrorCode  ierr;
  PetscScalar     *pQ,*pZ;
#if !defined(PETSC_USE_COMPLEX)
  PetscInt        i;
#endif

  PetscFunctionBegin;

  npreconv = d->npreconv;
  /* Constrains the converged pairs to nev */
#if !defined(PETSC_USE_COMPLEX)
  /* Tries to maintain together conjugate eigenpairs */
  for(i = 0; (i + (d->eigi[i]!=0.0?1:0) < npreconv) && (d->nconv + i < d->nev); i+= (d->eigi[i]!=0.0?2:1));
  npreconv = i;
#else
  npreconv = PetscMax(PetscMin(d->nev - d->nconv, npreconv), 0);
#endif
  /* Quick exit */
  if (npreconv == 0) { PetscFunctionReturn(0); }

  npreconv+= d->cX_in_H;
  ierr = PSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
  d->ldMTX = d->ldMTY = d->size_H;
  d->size_MT = d->size_H;
  cMT = d->size_H - npreconv;
  /* Harmonics restarts wiht right eigenvectors, and other with the left ones.
     If the problem is standard or hermitian, left and right vectors are the same */
  if (d->W||!d->cY||d->BcX||DVD_IS(d->sEP,DVD_EP_STD)||DVD_IS(d->sEP,DVD_EP_HERMITIAN)) {
    /* MTX <- ps.Q */
    ierr = PSGetArray(d->ps,PS_MAT_Q,&pQ);CHKERRQ(ierr);
    ierr = SlepcDenseCopy(d->MTX,d->ldMTX,pQ,ld,d->size_H,d->size_H);CHKERRQ(ierr);
    ierr = PSRestoreArray(d->ps,PS_MAT_Q,&pQ);CHKERRQ(ierr);
  } else {
    /* MTX <- [ps.Q(0:npreconv-1) ps.Z(npreconv:size_H-1)] */
    ierr = PSGetArray(d->ps,PS_MAT_Q,&pQ);CHKERRQ(ierr);
    ierr = SlepcDenseCopy(d->MTX,d->ldMTX,pQ,ld,d->size_H,npreconv);CHKERRQ(ierr);
    ierr = PSRestoreArray(d->ps,PS_MAT_Q,&pQ);CHKERRQ(ierr);
    ierr = PSGetArray(d->ps,PS_MAT_Z,&pZ);CHKERRQ(ierr);
    ierr = SlepcDenseCopy(&d->MTX[d->ldMTX*npreconv],d->ldMTX,&pZ[ld*npreconv],ld,d->size_H,cMT);CHKERRQ(ierr);
    ierr = PSRestoreArray(d->ps,PS_MAT_Z,&pZ);CHKERRQ(ierr);
    ierr = SlepcDenseOrth(d->MTX,d->ldMTX,d->size_H,d->size_H,d->auxS,d->size_auxS,&cMTX);CHKERRQ(ierr);
    cMT = cMTX - npreconv;
  }

  if (d->MTY) {
    /* MTY <- ps.Z */
    ierr = PSGetArray(d->ps,PS_MAT_Z,&pZ);CHKERRQ(ierr);
    ierr = SlepcDenseCopy(d->MTY,d->ldMTY,pZ,ld,d->size_H,d->size_H);CHKERRQ(ierr);
    ierr = PSRestoreArray(d->ps,PS_MAT_Z,&pZ);CHKERRQ(ierr);
  }
  /* Lock the converged pairs */
  d->eigr+= npreconv-d->cX_in_H;
#if !defined(PETSC_USE_COMPLEX)
  if (d->eigi) d->eigi+= npreconv-d->cX_in_H;
#endif
  d->nconv+= npreconv-d->cX_in_H;
  d->errest+= npreconv-d->cX_in_H;
  /* Notify the changes in V and update the other subspaces */
  d->V_tra_s = npreconv;          d->V_tra_e = d->size_H;
  d->V_new_s = cMT;               d->V_new_e = d->V_new_s;
  /* Remove oldU */
  data->size_oldU = 0;

  d->npreconv-= npreconv-d->cX_in_H;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_conv_finish"
PetscErrorCode dvd_updateV_conv_finish(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;
#if defined(PETSC_USE_COMPLEX)
  PetscInt        i, j;
  PetscScalar     s;
#endif  

  PetscFunctionBegin;

  /* Some functions need the diagonal elements in cT be real */
#if defined(PETSC_USE_COMPLEX)
  if (d->cT) for(i=0; i<d->nconv; i++) {
    s = PetscConj(d->cT[d->ldcT*i+i])/PetscAbsScalar(d->cT[d->ldcT*i+i]);
    for(j=0; j<=i; j++)
      d->cT[d->ldcT*i+j] = PetscRealPart(d->cT[d->ldcT*i+j]*s),
      d->cS[d->ldcS*i+j]*= s;
    ierr = VecScale(d->cX[i], s); CHKERRQ(ierr);
  }
#endif
  ierr = d->calcpairs_selectPairs(d, data->min_size_V); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
 
#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_restart_gen"
PetscErrorCode dvd_updateV_restart_gen(dvdDashboard *d)
{
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;
  PetscInt        size_plusk,size_X,i,j,ld,cMTX,cMTY;
  PetscScalar     *pX,*pZ;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* Select size_X desired pairs from V */
  size_X = PetscMin(PetscMin(data->min_size_V,
                             d->size_V ),
                             d->max_size_V );

  /* Add plusk eigenvectors from the previous iteration */
  size_plusk = PetscMax(0, PetscMin(PetscMin(data->plusk,
                                    data->size_oldU ),
                                    d->max_size_V - size_X ));

  /* Harmonics restarts wiht right eigenvectors, and other with the left ones */
  ierr = PSGetArray(d->ps,(d->W||!d->cY||d->BcX)?PS_MAT_Q:PS_MAT_Z,&pX);CHKERRQ(ierr);
  ierr = PSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);

  /* MTX <- orth([pX(0:size_X-1) [oldU(0:size_plusk-1); 0] ]) */
  d->ldMTX = d->size_MT = d->size_H;
  ierr = SlepcDenseCopy(d->MTX,d->ldMTX,pX,ld,d->size_H,size_X);CHKERRQ(ierr);
  ierr = PSRestoreArray(d->ps,(d->W||!d->cY||d->BcX)?PS_MAT_Q:PS_MAT_Z,&pX);CHKERRQ(ierr);
  if (size_plusk > 0) {
    ierr = SlepcDenseCopy(&d->MTX[d->ldMTX*size_X],d->ldMTX,data->oldU,data->ldoldU,data->size_oldU,size_plusk);CHKERRQ(ierr);
    for(i=size_X; i<size_X+size_plusk; i++) {
      for(j=data->size_oldU; j<d->size_H; j++) {
        d->MTX[j*d->ldMTX+i] = 0.0;
      }
    }
    ierr = SlepcDenseOrth(d->MTX,d->ldMTX,d->size_V,size_X+size_plusk,d->auxS,d->size_auxS,&cMTX);CHKERRQ(ierr);
  } else {
    cMTX = size_X;
  }

  if (d->MTY) {
    /* MTY <- orth([pY(0:size_X-1) [oldV(0:size_plusk-1); 0] ]) */
    d->ldMTY = d->ldMTX;
    ierr = PSGetArray(d->ps,PS_MAT_Z,&pZ);CHKERRQ(ierr);
    ierr = SlepcDenseCopy(d->MTY,d->ldMTY,pZ,ld,d->size_H,size_X);CHKERRQ(ierr);
    ierr = PSRestoreArray(d->ps,PS_MAT_Z,&pZ);CHKERRQ(ierr);
    if (size_plusk > 0) {
      ierr = SlepcDenseCopy(&d->MTY[d->ldMTY*size_X],d->ldMTY,data->oldV,data->ldoldU,data->size_oldU,size_plusk);CHKERRQ(ierr);
      for(i=size_X; i<size_X+size_plusk; i++) {
        for(j=data->size_oldU; j<d->size_H; j++) {
          d->MTY[j*d->ldMTY+i] = 0.0;
        }
      }
      ierr = SlepcDenseOrth(d->MTY,d->ldMTY,d->size_V,size_X+size_plusk,d->auxS,d->size_auxS,&cMTY);CHKERRQ(ierr);
      cMTX = PetscMin(cMTX, cMTY);
    }
  }

  /* Notify the changes in V and update the other subspaces */
  d->V_tra_s = d->cX_in_H;                  d->V_tra_e = cMTX;
  d->V_new_s = d->V_tra_e-d->cX_in_H; d->V_new_e = d->V_new_s;

  /* Remove oldU */
  data->size_oldU = 0;

  /* Remove npreconv */
  d->npreconv = 0;
    
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_update_gen"
PetscErrorCode dvd_updateV_update_gen(dvdDashboard *d)
{
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;
  PetscInt        size_D,ld;
  PetscScalar     *pQ,*pZ;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* Select the desired pairs */
  size_D = PetscMin(PetscMin(PetscMin(d->bs,
                                      d->size_V ),
                                      d->max_size_V-d->size_V ),
                                      d->size_H );
  if (size_D == 0) {
    ierr = PetscInfo2(d->eps, "MON: D:%D H:%D\n", size_D, d->size_H); CHKERRQ(ierr);
    ierr = d->initV(d); CHKERRQ(ierr);
    ierr = d->calcPairs(d); CHKERRQ(ierr);
  }

  /* Fill V with D */
  ierr = d->improveX(d, d->V+d->size_V, d->max_size_V-d->size_V, 0, size_D, &size_D); CHKERRQ(ierr);

  /* If D is empty, exit */
  d->size_D = size_D;
  if (size_D == 0) { PetscFunctionReturn(0); }

  /* Get the residual of all pairs */
  ierr = dvd_updateV_testConv(d,size_D,size_D,data->allResiduals?d->size_V:size_D,d->auxV,d->auxS,PETSC_NULL);CHKERRQ(ierr);

  /* Notify the changes in V */
  d->V_tra_s = 0;                 d->V_tra_e = 0;
  d->V_new_s = d->size_V;         d->V_new_e = d->size_V+size_D;

  /* Save the projected eigenvectors */
  if (data->plusk > 0) {
    data->ldoldU = data->size_oldU = d->size_H;
    ierr = PSGetLeadingDimension(d->ps,&ld);CHKERRQ(ierr);
    ierr = PSGetArray(d->ps,PS_MAT_Q,&pQ);CHKERRQ(ierr);
    ierr = SlepcDenseCopy(data->oldU,data->ldoldU,pQ,ld,d->size_H,d->size_H);CHKERRQ(ierr);
    ierr = PSRestoreArray(d->ps,PS_MAT_Q,&pQ);CHKERRQ(ierr);
    if (d->cY) {
      ierr = PSGetArray(d->ps,PS_MAT_Z,&pZ);CHKERRQ(ierr);
      ierr = SlepcDenseCopy(data->oldV,data->ldoldU,pZ,ld,d->size_H,d->size_H);CHKERRQ(ierr);
      ierr = PSRestoreArray(d->ps,PS_MAT_Z,&pZ);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_updateV_testConv"
/* auxV: (by calcpairs_residual_eig) */
PetscErrorCode dvd_updateV_testConv(dvdDashboard *d, PetscInt s, PetscInt pre,
                                    PetscInt e, Vec *auxV, PetscScalar *auxS,
                                    PetscInt *nConv)
{
  PetscInt        i,j,b;
  PetscReal       norm;
  PetscErrorCode  ierr;
  PetscBool       conv, c;
  dvdManagV_basic *data = (dvdManagV_basic*)d->updateV_data;

  PetscFunctionBegin;
  
  if (nConv) *nConv = s;
  for(i=s, conv=PETSC_TRUE;
      (conv || data->allResiduals) && (i < e);
      i+=b) {
#if !defined(PETSC_USE_COMPLEX)
    b = d->eigi[i]!=0.0?2:1; 
#else
    b = 1;
#endif
    if (i+b-1 >= pre) {
      ierr = d->calcpairs_residual(d, i, i+b, auxV);
      CHKERRQ(ierr);
    }
    /* Test the Schur vector */
    for (j=0,c=PETSC_TRUE; j<b && c; j++) {
      norm = d->nR[i+j]/d->nX[i+j]; 
      c = d->testConv(d, d->eigr[i+j], d->eigi[i+j], norm, &d->errest[i+j]);
    }
    /* Test the eigenvector */
    if (d->eps->trueres && conv && c) {
      ierr = d->calcpairs_residual_eig(d,i,i+b,auxV);CHKERRQ(ierr);
      for (j=0,c=PETSC_TRUE; j<b && c; j++) {
        norm = d->nR[i+j]/d->nX[i+j];
        c = d->testConv(d, d->eigr[i+j], d->eigi[i+j], norm, &d->errest[i+j]);
      }
    }
    if (conv && c) { if (nConv) *nConv = i+b; }
    else conv = PETSC_FALSE;
  }
  pre = PetscMax(pre, i);

#if !defined(PETSC_USE_COMPLEX)
  /* Enforce converged conjugate complex eigenpairs */
  if (nConv) {
    for(j=0; j<*nConv; j++) if(d->eigi[j] != 0.0) j++;
    if(j > *nConv) (*nConv)--;
  }
#endif
  for(i=pre; i<e; i++) d->errest[i] = d->nR[i] = PETSC_MAX_REAL;
  
  PetscFunctionReturn(0);
}
