/*
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

#define DVD_CHECKSUM(b) ((b)->max_size_V + (b)->max_size_oldX)

#undef __FUNCT__
#define __FUNCT__ "dvd_schm_basic_preconf"
PetscErrorCode dvd_schm_basic_preconf(dvdDashboard *d,dvdBlackboard *b,PetscInt mpd,PetscInt min_size_V,PetscInt bs,PetscInt ini_size_V,PetscInt size_initV,PetscInt plusk,HarmType_t harmMode,KSP ksp,InitType_t init,PetscBool allResiduals,PetscBool orth,PetscInt cX_proj,PetscInt cX_impr,PetscBool doubleexp)
{
  PetscErrorCode ierr;
  PetscInt       check_sum0,check_sum1;

  PetscFunctionBegin;
  ierr = PetscMemzero(b,sizeof(dvdBlackboard));CHKERRQ(ierr);
  b->state = DVD_STATE_PRECONF;

  for (check_sum0=-1,check_sum1=DVD_CHECKSUM(b); check_sum0 != check_sum1; check_sum0 = check_sum1, check_sum1 = DVD_CHECKSUM(b)) {

    /* Setup basic management of V */
    ierr = dvd_managementV_basic(d,b,bs,mpd,min_size_V,plusk,PetscNot(harmMode==DVD_HARM_NONE),allResiduals);CHKERRQ(ierr);

    /* Setup the initial subspace for V */
    ierr = dvd_initV(d,b,ini_size_V,size_initV,(init==DVD_INITV_KRYLOV)?PETSC_TRUE:PETSC_FALSE);CHKERRQ(ierr);

    /* Setup the convergence in order to use the SLEPc convergence test */
    ierr = dvd_testconv_slepc(d,b);CHKERRQ(ierr);

    /* Setup Raileigh-Ritz for selecting the best eigenpairs in V */
    ierr = dvd_calcpairs_qz(d,b,orth,cX_proj,PetscNot(harmMode==DVD_HARM_NONE));CHKERRQ(ierr);
    if (harmMode != DVD_HARM_NONE) {
      ierr = dvd_harm_conf(d,b,harmMode,PETSC_FALSE,0.0);CHKERRQ(ierr);
    }

    /* Setup the method for improving the eigenvectors */
    if (doubleexp) {
      ierr = dvd_improvex_gd2(d,b,ksp,bs);CHKERRQ(ierr);
    } else {
      ierr = dvd_improvex_jd(d,b,ksp,bs,cX_impr,PETSC_FALSE);CHKERRQ(ierr);
      ierr = dvd_improvex_jd_proj_uv(d,b,DVD_PROJ_KZX);CHKERRQ(ierr);
      ierr = dvd_improvex_jd_lit_const(d,b,0,0.0,0.0);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_schm_basic_conf"
PetscErrorCode dvd_schm_basic_conf(dvdDashboard *d,dvdBlackboard *b,PetscInt mpd,PetscInt min_size_V,PetscInt bs,PetscInt ini_size_V,PetscInt size_initV,PetscInt plusk,HarmType_t harmMode,PetscBool fixedTarget,PetscScalar t,KSP ksp,PetscReal fix,InitType_t init,PetscBool allResiduals,PetscBool orth,PetscInt cX_proj,PetscInt cX_impr,PetscBool dynamic,PetscBool doubleexp)
{
  PetscInt       check_sum0,check_sum1,maxits;
  PetscReal      tol;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  b->state = DVD_STATE_CONF;
  check_sum0 = DVD_CHECKSUM(b);

  /* Setup basic management of V */
  ierr = dvd_managementV_basic(d,b,bs,mpd,min_size_V,plusk,PetscNot(harmMode==DVD_HARM_NONE),allResiduals);CHKERRQ(ierr);

  /* Setup the initial subspace for V */
  ierr = dvd_initV(d,b,ini_size_V,size_initV,(init==DVD_INITV_KRYLOV)?PETSC_TRUE:PETSC_FALSE);CHKERRQ(ierr);

  /* Setup the convergence in order to use the SLEPc convergence test */
  ierr = dvd_testconv_slepc(d,b);CHKERRQ(ierr);

  /* Setup Raileigh-Ritz for selecting the best eigenpairs in V */
  ierr = dvd_calcpairs_qz(d,b,orth,cX_proj,PetscNot(harmMode==DVD_HARM_NONE));CHKERRQ(ierr);
  if (harmMode != DVD_HARM_NONE) {
    ierr = dvd_harm_conf(d,b,harmMode,fixedTarget,t);CHKERRQ(ierr);
  }

  /* Setup the method for improving the eigenvectors */
  if (doubleexp) {
    ierr = dvd_improvex_gd2(d,b,ksp,bs);CHKERRQ(ierr);
  } else {
    ierr = dvd_improvex_jd(d,b,ksp,bs,cX_impr,dynamic);CHKERRQ(ierr);
    ierr = dvd_improvex_jd_proj_uv(d,b,DVD_PROJ_KZX);CHKERRQ(ierr);
    ierr = KSPGetTolerances(ksp,&tol,NULL,NULL,&maxits);CHKERRQ(ierr);
    ierr = dvd_improvex_jd_lit_const(d,b,maxits,tol,fix);CHKERRQ(ierr);
  }

  check_sum1 = DVD_CHECKSUM(b);
  if (check_sum0 != check_sum1) SETERRQ(PETSC_COMM_SELF,1, "Something awful happened");
  PetscFunctionReturn(0);
}
