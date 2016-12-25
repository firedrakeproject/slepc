/*
  SLEPc eigensolver: "davidson"

  Step: test for convergence

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

static PetscBool dvd_testconv_slepc_0(dvdDashboard *d,PetscScalar eigvr,PetscScalar eigvi,PetscReal r,PetscReal *err)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*d->eps->converged)(d->eps,eigvr,eigvi,r,err,d->eps->convergedctx);CHKERRABORT(PetscObjectComm((PetscObject)d->eps),ierr);
  PetscFunctionReturn(PetscNot(*err>=d->eps->tol));
}

PetscErrorCode dvd_testconv_slepc(dvdDashboard *d, dvdBlackboard *b)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscFree(d->testConv_data);CHKERRQ(ierr);
    d->testConv = dvd_testconv_slepc_0;
  }
  PetscFunctionReturn(0);
}

