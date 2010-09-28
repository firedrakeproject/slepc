/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#define DVD_CHECKSUM(b) \
  ( (b)->max_size_V + (b)->max_size_auxV + (b)->max_size_auxS + \
    (b)->own_vecs + (b)->own_scalars + (b)->max_size_oldX )

#undef __FUNCT__  
#define __FUNCT__ "dvd_schm_basic_preconf"
PetscErrorCode dvd_schm_basic_preconf(dvdDashboard *d, dvdBlackboard *b,
  PetscInt max_size_V, PetscInt mpd, PetscInt min_size_V, PetscInt bs,
  PetscInt ini_size_V, PetscInt size_initV, PetscInt plusk, PC pc,
  HarmType_t harmMode, KSP ksp, InitType_t init, PetscTruth allResiduals)
{
  PetscErrorCode ierr;
  PetscInt       check_sum0, check_sum1;

  PetscFunctionBegin;

  ierr = PetscMemzero(b, sizeof(dvdBlackboard)); CHKERRQ(ierr);
  b->state = DVD_STATE_PRECONF;

  for(check_sum0=-1,check_sum1=DVD_CHECKSUM(b); check_sum0 != check_sum1;
      check_sum0 = check_sum1, check_sum1 = DVD_CHECKSUM(b)) {
    b->own_vecs = b->own_scalars = 0;

    /* Setup basic management of V */
    ierr = dvd_managementV_basic(d, b, bs, max_size_V, mpd, min_size_V, plusk,
                               harmMode==DVD_HARM_NONE?PETSC_FALSE:PETSC_TRUE,
                               allResiduals);
    CHKERRQ(ierr);
  
    /* Setup the initial subspace for V */
    if (size_initV) {
      ierr = dvd_initV_user(d, b, size_initV, ini_size_V); CHKERRQ(ierr);
    } else switch(init) {
    case DVD_INITV_CLASSIC:
      ierr = dvd_initV_classic(d, b, ini_size_V); CHKERRQ(ierr); break;
    case DVD_INITV_KRYLOV:
      ierr = dvd_initV_krylov(d, b, ini_size_V); CHKERRQ(ierr); break;
    }
  
    /* Setup the convergence in order to use the SLEPc convergence test */
    ierr = dvd_testconv_slepc(d, b);CHKERRQ(ierr);
  
    /* Setup Raileigh-Ritz for selecting the best eigenpairs in V */
    ierr = dvd_calcpairs_qz(d, b, PETSC_NULL); CHKERRQ(ierr);
    if (harmMode != DVD_HARM_NONE) {
      ierr = dvd_harm_conf(d, b, harmMode, PETSC_FALSE, 0.0); CHKERRQ(ierr);
    }
  
    /* Setup the preconditioner */
    ierr = dvd_static_precond_PC(d, b, pc); CHKERRQ(ierr);

    /* Setup the method for improving the eigenvectors */
    ierr = dvd_improvex_jd(d, b, ksp, bs); CHKERRQ(ierr);
    ierr = dvd_improvex_jd_proj_uv(d, b, DVD_PROJ_KBXX); CHKERRQ(ierr);
    ierr = dvd_improvex_jd_lit_const(d, b, 0, 0.0, 0.0); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_schm_basic_conf"
PetscErrorCode dvd_schm_basic_conf(dvdDashboard *d, dvdBlackboard *b,
  PetscInt max_size_V, PetscInt mpd, PetscInt min_size_V, PetscInt bs,
  PetscInt ini_size_V, PetscInt size_initV, PetscInt plusk, PC pc,
  IP ip, HarmType_t harmMode, PetscTruth fixedTarget, PetscScalar t, KSP ksp,
  PetscReal fix, InitType_t init, PetscTruth allResiduals)
{
  PetscInt        check_sum0, check_sum1, maxits;
  Vec             *fv;
  PetscScalar     *fs;
  PetscReal       tol;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  b->state = DVD_STATE_CONF;
  check_sum0 = DVD_CHECKSUM(b);
  b->own_vecs = 0; b->own_scalars = 0;
  fv = b->free_vecs; fs = b->free_scalars;

  /* Setup basic management of V */
  ierr = dvd_managementV_basic(d, b, bs, max_size_V, mpd, min_size_V, plusk,
                        harmMode==DVD_HARM_NONE?PETSC_FALSE:PETSC_TRUE,
                        allResiduals);
  CHKERRQ(ierr);

  /* Setup the initial subspace for V */
  if (size_initV) {
    ierr = dvd_initV_user(d, b, size_initV, ini_size_V); CHKERRQ(ierr);
  } else switch(init) {
  case DVD_INITV_CLASSIC:
    ierr = dvd_initV_classic(d, b, ini_size_V); CHKERRQ(ierr); break;
  case DVD_INITV_KRYLOV:
    ierr = dvd_initV_krylov(d, b, ini_size_V); CHKERRQ(ierr); break;
  }

  /* Setup the convergence in order to use the SLEPc convergence test */
  ierr = dvd_testconv_slepc(d, b); CHKERRQ(ierr);

  /* Setup Raileigh-Ritz for selecting the best eigenpairs in V */
  ierr = dvd_calcpairs_qz(d, b, ip); CHKERRQ(ierr);
  if (harmMode != DVD_HARM_NONE) {
    ierr = dvd_harm_conf(d, b, harmMode, fixedTarget, t); CHKERRQ(ierr);
  }

  /* Setup the preconditioner */
  ierr = dvd_static_precond_PC(d, b, pc); CHKERRQ(ierr);

  /* Setup the method for improving the eigenvectors */
  ierr = dvd_improvex_jd(d, b, ksp, bs); CHKERRQ(ierr);
  ierr = dvd_improvex_jd_proj_uv(d, b, DVD_IS(d->sEP, DVD_EP_HERMITIAN)?
                                              DVD_PROJ_KBXZ:DVD_PROJ_KBXY);
  CHKERRQ(ierr);
  ierr = KSPGetTolerances(ksp, &tol, PETSC_NULL, PETSC_NULL, &maxits);
  CHKERRQ(ierr);
  ierr = dvd_improvex_jd_lit_const(d, b, maxits, tol, fix); CHKERRQ(ierr);

  check_sum1 = DVD_CHECKSUM(b);
  if ((check_sum0 != check_sum1) ||
      (b->free_vecs - fv > b->own_vecs) ||
      (b->free_scalars - fs > b->own_scalars))
    SETERRQ(1, "Something awful happened!");
    
  PetscFunctionReturn(0);
}
