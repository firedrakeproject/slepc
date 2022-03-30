/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "davidson.h"

#define DVD_CHECKSUM(b) ((b)->max_size_V + (b)->max_size_oldX)

PetscErrorCode dvd_schm_basic_preconf(dvdDashboard *d,dvdBlackboard *b,PetscInt mpd,PetscInt min_size_V,PetscInt bs,PetscInt ini_size_V,PetscInt size_initV,PetscInt plusk,HarmType_t harmMode,KSP ksp,InitType_t init,PetscBool allResiduals,PetscBool orth,PetscBool doubleexp)
{
  PetscInt       check_sum0,check_sum1;

  PetscFunctionBegin;
  PetscCall(PetscMemzero(b,sizeof(dvdBlackboard)));
  b->state = DVD_STATE_PRECONF;

  for (check_sum0=-1,check_sum1=DVD_CHECKSUM(b); check_sum0 != check_sum1; check_sum0 = check_sum1, check_sum1 = DVD_CHECKSUM(b)) {

    /* Setup basic management of V */
    PetscCall(dvd_managementV_basic(d,b,bs,mpd,min_size_V,plusk,PetscNot(harmMode==DVD_HARM_NONE),allResiduals));

    /* Setup the initial subspace for V */
    PetscCall(dvd_initV(d,b,ini_size_V,size_initV,(init==DVD_INITV_KRYLOV)?PETSC_TRUE:PETSC_FALSE));

    /* Setup the convergence in order to use the SLEPc convergence test */
    PetscCall(dvd_testconv_slepc(d,b));

    /* Setup Raileigh-Ritz for selecting the best eigenpairs in V */
    PetscCall(dvd_calcpairs_qz(d,b,orth,PetscNot(harmMode==DVD_HARM_NONE)));
    if (harmMode != DVD_HARM_NONE) PetscCall(dvd_harm_conf(d,b,harmMode,PETSC_FALSE,0.0));

    /* Setup the method for improving the eigenvectors */
    if (doubleexp) PetscCall(dvd_improvex_gd2(d,b,ksp,bs));
    else {
      PetscCall(dvd_improvex_jd(d,b,ksp,bs,PETSC_FALSE));
      PetscCall(dvd_improvex_jd_proj_uv(d,b));
      PetscCall(dvd_improvex_jd_lit_const(d,b,0,0.0,0.0));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode dvd_schm_basic_conf(dvdDashboard *d,dvdBlackboard *b,PetscInt mpd,PetscInt min_size_V,PetscInt bs,PetscInt ini_size_V,PetscInt size_initV,PetscInt plusk,HarmType_t harmMode,PetscBool fixedTarget,PetscScalar t,KSP ksp,PetscReal fix,InitType_t init,PetscBool allResiduals,PetscBool orth,PetscBool dynamic,PetscBool doubleexp)
{
  PetscInt       check_sum0,check_sum1,maxits;
  PetscReal      tol;

  PetscFunctionBegin;
  b->state = DVD_STATE_CONF;
  check_sum0 = DVD_CHECKSUM(b);

  /* Setup basic management of V */
  PetscCall(dvd_managementV_basic(d,b,bs,mpd,min_size_V,plusk,PetscNot(harmMode==DVD_HARM_NONE),allResiduals));

  /* Setup the initial subspace for V */
  PetscCall(dvd_initV(d,b,ini_size_V,size_initV,(init==DVD_INITV_KRYLOV)?PETSC_TRUE:PETSC_FALSE));

  /* Setup the convergence in order to use the SLEPc convergence test */
  PetscCall(dvd_testconv_slepc(d,b));

  /* Setup Raileigh-Ritz for selecting the best eigenpairs in V */
  PetscCall(dvd_calcpairs_qz(d,b,orth,PetscNot(harmMode==DVD_HARM_NONE)));
  if (harmMode != DVD_HARM_NONE) PetscCall(dvd_harm_conf(d,b,harmMode,fixedTarget,t));

  /* Setup the method for improving the eigenvectors */
  if (doubleexp) PetscCall(dvd_improvex_gd2(d,b,ksp,bs));
  else {
    PetscCall(dvd_improvex_jd(d,b,ksp,bs,dynamic));
    PetscCall(dvd_improvex_jd_proj_uv(d,b));
    PetscCall(KSPGetTolerances(ksp,&tol,NULL,NULL,&maxits));
    PetscCall(dvd_improvex_jd_lit_const(d,b,maxits,tol,fix));
  }

  check_sum1 = DVD_CHECKSUM(b);
  PetscAssert(check_sum0==check_sum1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Something awful happened");
  PetscFunctionReturn(0);
}
