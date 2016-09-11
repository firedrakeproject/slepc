/*
  SLEPc eigensolver: "davidson"

  Step: init subspace V

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

#include "davidson.h"

typedef struct {
  PetscInt k;                 /* desired initial subspace size */
  PetscInt user;              /* number of user initial vectors */
  void     *old_initV_data;   /* old initV data */
} dvdInitV;

#undef __FUNCT__
#define __FUNCT__ "dvd_initV_classic_0"
static PetscErrorCode dvd_initV_classic_0(dvdDashboard *d)
{
  PetscErrorCode ierr;
  dvdInitV       *data = (dvdInitV*)d->initV_data;
  PetscInt       i,user = PetscMin(data->user,d->eps->mpd), l,k;

  PetscFunctionBegin;
  ierr = BVGetActiveColumns(d->eps->V,&l,&k);CHKERRQ(ierr);
  /* User vectors are added at the beginning, so no active column should be in V */
  if (data->user>0&&l>0) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");
  /* Generate a set of random initial vectors and orthonormalize them */
  for (i=l+user;i<l+data->k && i<d->eps->ncv && i-l<d->eps->mpd;i++) {
    ierr = BVSetRandomColumn(d->eps->V,i);CHKERRQ(ierr);
  }
  d->V_tra_s = 0; d->V_tra_e = 0;
  d->V_new_s = 0; d->V_new_e = i-l;

  /* After that the user vectors will be destroyed */
  data->user = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_initV_krylov_0"
static PetscErrorCode dvd_initV_krylov_0(dvdDashboard *d)
{
  PetscErrorCode ierr;
  dvdInitV       *data = (dvdInitV*)d->initV_data;
  PetscInt       i,user = PetscMin(data->user,d->eps->mpd),l,k;
  Vec            av,v1,v2;

  PetscFunctionBegin;
  ierr = BVGetActiveColumns(d->eps->V,&l,&k);CHKERRQ(ierr);
  /* User vectors are added at the beginning, so no active column should be in V */
  if (data->user>0&&l>0) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");

  /* If needed, generate a random vector for starting the arnoldi method */
  if (user == 0) {
    ierr = BVSetRandomColumn(d->eps->V,l);CHKERRQ(ierr);
    user = 1;
  }

  /* Perform k steps of Arnoldi with the operator K^{-1}*(t[1]*A-t[2]*B) */
  ierr = dvd_orthV(d->eps->V,l,l+user);CHKERRQ(ierr);
  for (i=l+user;i<l+data->k && i<d->eps->ncv && i-l<d->eps->mpd;i++) {
    /* aux <- theta[1]A*in - theta[0]*B*in */
    ierr = BVGetColumn(d->eps->V,i,&v1);CHKERRQ(ierr);
    ierr = BVGetColumn(d->eps->V,i-user,&v2);CHKERRQ(ierr);
    ierr = BVGetColumn(d->auxBV,0,&av);CHKERRQ(ierr);
    if (d->B) {
      ierr = MatMult(d->A,v2,v1);CHKERRQ(ierr);
      ierr = MatMult(d->B,v2,av);CHKERRQ(ierr);
      ierr = VecAXPBY(av,d->target[1],-d->target[0],v1);CHKERRQ(ierr);
    } else {
      ierr = MatMult(d->A,v2,av);CHKERRQ(ierr);
      ierr = VecAXPBY(av,-d->target[0],d->target[1],v2);CHKERRQ(ierr);
    }
    ierr = d->improvex_precond(d,0,av,v1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(d->eps->V,i,&v1);CHKERRQ(ierr);
    ierr = BVRestoreColumn(d->eps->V,i-user,&v2);CHKERRQ(ierr);
    ierr = BVRestoreColumn(d->auxBV,0,&av);CHKERRQ(ierr);
    ierr = dvd_orthV(d->eps->V,i,i+1);CHKERRQ(ierr);
  }

  d->V_tra_s = 0; d->V_tra_e = 0;
  d->V_new_s = 0; d->V_new_e = i-l;

  /* After that the user vectors will be destroyed */
  data->user = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_initV_d"
static PetscErrorCode dvd_initV_d(dvdDashboard *d)
{
  PetscErrorCode ierr;
  dvdInitV       *data = (dvdInitV*)d->initV_data;

  PetscFunctionBegin;
  /* Restore changes in dvdDashboard */
  d->initV_data = data->old_initV_data;

  /* Free local data */
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_initV"
PetscErrorCode dvd_initV(dvdDashboard *d, dvdBlackboard *b, PetscInt k,PetscInt user, PetscBool krylov)
{
  PetscErrorCode ierr;
  dvdInitV       *data;

  PetscFunctionBegin;
  /* Setting configuration constrains */
  b->max_size_V = PetscMax(b->max_size_V, k);

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscNewLog(d->eps,&data);CHKERRQ(ierr);
    data->k = k;
    data->user = user;
    data->old_initV_data = d->initV_data;
    d->initV_data = data;
    if (krylov) d->initV = dvd_initV_krylov_0;
    else d->initV = dvd_initV_classic_0;
    ierr = EPSDavidsonFLAdd(&d->destroyList,dvd_initV_d);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_orthV"
PetscErrorCode dvd_orthV(BV V,PetscInt V_new_s,PetscInt V_new_e)
{
  PetscErrorCode ierr;
  PetscInt       i,j,l,k;
  PetscBool      lindep;
  PetscReal      norm;

  PetscFunctionBegin;
  ierr = BVGetActiveColumns(V,&l,&k);CHKERRQ(ierr);
  for (i=V_new_s;i<V_new_e;i++) {
    ierr = BVSetActiveColumns(V,0,i);CHKERRQ(ierr);
    for (j=0;j<3;j++) {
      if (j>0) {
        ierr = BVSetRandomColumn(V,i);CHKERRQ(ierr);
        ierr = PetscInfo1(V,"Orthonormalization problems adding the vector %D to the searching subspace\n",i);CHKERRQ(ierr);
      }
      ierr = BVOrthogonalizeColumn(V,i,NULL,&norm,&lindep);CHKERRQ(ierr);
      if (!lindep && (PetscAbsReal(norm) > PETSC_SQRT_MACHINE_EPSILON)) break;
    }
    if (lindep || (PetscAbsReal(norm) < PETSC_SQRT_MACHINE_EPSILON)) SETERRQ(PetscObjectComm((PetscObject)V),1, "Error during the orthonormalization of the vectors");
    ierr = BVScaleColumn(V,i,1.0/norm);CHKERRQ(ierr);
  }
  ierr = BVSetActiveColumns(V,l,k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

