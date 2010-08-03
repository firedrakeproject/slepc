/*
  SLEPc eigensolver: "davidson"

  Step: init subspace V

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

PetscErrorCode dvd_initV_classic_0(dvdDashboard *d);
PetscErrorCode dvd_initV_classic_d(dvdDashboard *d);

PetscErrorCode dvd_initV_user_0(dvdDashboard *d);
PetscErrorCode dvd_initV_user_d(dvdDashboard *d);

PetscErrorCode dvd_initV_krylov_0(dvdDashboard *d);
PetscErrorCode dvd_initV_krylov_d(dvdDashboard *d);

/*
  Fill V with a random subspace
*/

typedef struct {
  PetscInt k;           /* number of vectors initialized */
  void *old_initV_data; /* old initV data */
} dvdInitV_Classic;

#undef __FUNCT__  
#define __FUNCT__ "dvd_initV_classic"
PetscErrorCode dvd_initV_classic(dvdDashboard *d, dvdBlackboard *b, PetscInt k)
{
  PetscErrorCode  ierr;
  dvdInitV_Classic
                  *data;

  PetscFunctionBegin;

  /* Setting configuration constrains */
  b->max_size_V = PetscMax(b->max_size_V, k);

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscMalloc(sizeof(dvdInitV_Classic), &data); CHKERRQ(ierr);
    data->k = k;
    data->old_initV_data = d->initV_data;
    d->initV_data = data;
    d->initV = dvd_initV_classic_0;
    DVD_FL_ADD(d->destroyList, dvd_initV_classic_d);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_initV_classic_0"
PetscErrorCode dvd_initV_classic_0(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdInitV_Classic
                  *data = (dvdInitV_Classic*)d->initV_data;
  PetscInt        i;

  PetscFunctionBegin;

  /* Generate a set of random initial vectors and orthonormalize them */
  for (i=0; i<PetscMin(data->k,d->max_size_V); i++) {
    ierr = SlepcVecSetRandom(d->V[i], d->eps->rand); CHKERRQ(ierr);
  }
  d->size_V = i;
  d->V_imm_s = 0; d->V_imm_e = 0;
  d->V_tra_s = 0; d->V_tra_e = 0;
  d->V_new_s = 0; d->V_new_e = i;
 
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_initV_classic_d"
PetscErrorCode dvd_initV_classic_d(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdInitV_Classic
                  *data = (dvdInitV_Classic*)d->initV_data;

  PetscFunctionBegin;

  /* Restore changes in dvdDashboard */
  d->initV_data = data->old_initV_data;

  /* Free local data */
  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
  Fill V with user vectors
*/

typedef struct {
  Vec *userV;           /* custom initial search subspace */
  PetscInt size_userV,  /* size of userV */
    k;                  /* desired initial subspace size */
  void *old_initV_data; /* old initV data */
} dvdInitV_User;

#undef __FUNCT__  
#define __FUNCT__ "dvd_initV_user"
PetscErrorCode dvd_initV_user(dvdDashboard *d, dvdBlackboard *b, Vec *userV,
                        PetscInt size_userV, PetscInt k)
{
  PetscErrorCode  ierr;
  dvdInitV_User   *data;

  PetscFunctionBegin;

  /* Setting configuration constrains */
  b->max_size_V = PetscMax(b->max_size_V, k);

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscMalloc(sizeof(dvdInitV_User), &data); CHKERRQ(ierr);
    data->k = k;
    data->size_userV = size_userV;
    data->userV = userV;
    data->old_initV_data = d->initV_data;
    d->initV_data = data;
    d->initV = dvd_initV_user_0;
    DVD_FL_ADD(d->destroyList, dvd_initV_user_d);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_initV_user_0"
PetscErrorCode dvd_initV_user_0(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdInitV_User   *data = (dvdInitV_User*)d->initV_data;
  PetscInt        i;

  PetscFunctionBegin;

  /* Generate a set of random initial vectors and orthonormalize them */
  for (i=0; i<PetscMin(data->size_userV,d->max_size_V); i++) {
    ierr = VecCopy(data->userV[i], d->V[i]); CHKERRQ(ierr);
  }
  for (; i<PetscMin(data->k,d->max_size_V); i++) {
    ierr = SlepcVecSetRandom(d->V[i], d->eps->rand); CHKERRQ(ierr);
  }
  d->size_V = i;
  d->V_imm_s = 0; d->V_imm_e = 0;
  d->V_tra_s = 0; d->V_tra_e = 0;
  d->V_new_s = 0; d->V_new_e = i;
 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_initV_user_d"
PetscErrorCode dvd_initV_user_d(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdInitV_User   *data = (dvdInitV_User*)d->initV_data;

  PetscFunctionBegin;

  /* Restore changes in dvdDashboard */
  d->initV_data = data->old_initV_data;

  /* Free local data */
  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/*
  Start with a krylov subspace with the matrix A
*/

typedef struct {
  PetscInt k;           /* number of steps of arnoldi */
  void *old_initV_data; /* old initV data */
} dvdInitV_Krylov;

#undef __FUNCT__  
#define __FUNCT__ "dvd_initV_krylov"
PetscErrorCode dvd_initV_krylov(dvdDashboard *d, dvdBlackboard *b, PetscInt k)
{
  PetscErrorCode  ierr;
  dvdInitV_Krylov *data;

  PetscFunctionBegin;

  /* Setting configuration constrains */
  b->max_size_auxV = PetscMax(b->max_size_auxV, 2);

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscMalloc(sizeof(dvdInitV_Krylov), &data); CHKERRQ(ierr);
    data->k = k;
    data->old_initV_data = d->initV_data;
    d->initV_data = data;
    d->initV = dvd_initV_krylov_0;
    DVD_FL_ADD(d->destroyList, dvd_initV_krylov_d);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_initV_krylov_0"
PetscErrorCode dvd_initV_krylov_0(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdInitV_Krylov *data = (dvdInitV_Krylov*)d->initV_data;
  PetscReal       norm;
  PetscInt        i;
  Vec             *cX = d->BcX? d->BcX : ( (d->cY && !d->W)? d->cY : d->cX );

  PetscFunctionBegin;

  /* Generate a random vector for starting the arnoldi method */
  ierr = SlepcVecSetRandom(d->V[0], d->eps->rand); CHKERRQ(ierr);
  ierr = IPNorm(d->ipV, d->V[0], &norm); CHKERRQ(ierr);
  ierr = VecScale(d->V[0], 1.0/norm); CHKERRQ(ierr);

  /* Perform k steps of Arnoldi with the operator K^{-1}*(t[1]*A-t[2]*B) */
  for (i=1; i<PetscMin(data->k,d->max_size_V); i++) {
   /* aux <- theta[1]A*in - theta[0]*B*in */
    if (d->B) {
      ierr = MatMult(d->A, d->V[i-1], d->V[i]); CHKERRQ(ierr);
      ierr = MatMult(d->B, d->V[i-1], d->auxV[0]); CHKERRQ(ierr);
      ierr = VecAXPBY(d->V[i], -d->target[0], d->target[1], d->auxV[0]);
      CHKERRQ(ierr);
    } else {
      ierr = MatMult(d->A, d->V[i-1], d->V[i]); CHKERRQ(ierr);
      ierr = VecAXPBY(d->V[i], -d->target[0], d->target[1], d->V[i-1]);
      CHKERRQ(ierr);
    }
    ierr = dvd_orthV(d->ipV, d->eps->DS, d->eps->nds, cX, d->size_cX, d->V, i,
                     i+1, d->auxS, d->auxV[0], d->eps->rand); CHKERRQ(ierr);
  }

  d->size_V = i;
  d->V_imm_s = 0; d->V_imm_e = 0;
  d->V_tra_s = 0; d->V_tra_e = 0;
  d->V_new_s = 0; d->V_new_e = i;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "dvd_initV_krylov_d"
PetscErrorCode dvd_initV_krylov_d(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdInitV_Krylov *data = (dvdInitV_Krylov*)d->initV_data;

  PetscFunctionBegin;

  /* Restore changes in dvdDashboard */
  d->initV_data = data->old_initV_data;

  /* Free local data */
  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
