/*
   Modification of the *temp* implementation of the BLOPEX multivector in order
   to wrap created PETSc vectors as multivectors.

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

#include <slepc/private/bvimpl.h>
#include <stdlib.h>
#include <blopex_interpreter.h>
#include <blopex_temp_multivector.h>
#include "slepc-interface.h"

static void* mv_TempMultiVectorCreateFromBV(void* ii_,BlopexInt n,void* sample)
{
  PetscErrorCode          ierr;
  BV                      bv = (BV)sample;
  Vec                     v;
  PetscInt                i,l,k,nc,useconstr=PETSC_FALSE,flg;
  mv_TempMultiVector      *x;
  mv_InterfaceInterpreter *ii = (mv_InterfaceInterpreter*)ii_;

  x = (mv_TempMultiVector*)malloc(sizeof(mv_TempMultiVector));
  if (!x) SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Allocation for x failed");

  x->interpreter = ii;
  x->numVectors  = n;

  x->vector = (void**)calloc(n,sizeof(void*));
  if (!x->vector) SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Allocation for x->vector failed");

  x->ownsVectors = 1;
  x->mask = NULL;
  x->ownsMask = 0;

  ierr = BVGetActiveColumns(bv,&l,&k);CHKERRABORT(PETSC_COMM_SELF,ierr);
  ierr = PetscObjectComposedDataGetInt((PetscObject)bv,slepc_blopex_useconstr,useconstr,flg);CHKERRABORT(PETSC_COMM_SELF,ierr);
  if (!l && useconstr) {
    ierr = BVGetNumConstraints(bv,&nc);CHKERRABORT(PETSC_COMM_SELF,ierr);
    l = -nc;
  }
  if (n != k-l) SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_PLIB,"BV active columns plus constraints do not match argument n");
  for (i=0;i<n;i++) {
    ierr = BVGetColumn(bv,l+i,&v);CHKERRABORT(PETSC_COMM_SELF,ierr);
    ierr = PetscObjectReference((PetscObject)v);CHKERRABORT(PETSC_COMM_SELF,ierr);
    x->vector[i] = (void*)v;
    ierr = BVRestoreColumn(bv,l+i,&v);CHKERRABORT(PETSC_COMM_SELF,ierr);
  }
  return x;
}

static void mv_TempMultiPETSCVectorDestroy(void* x_)
{
  mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

  if (!x) return;

  if (x->ownsVectors && x->vector) free(x->vector);
  if (x->mask && x->ownsMask) free(x->mask);
  free(x);
}

/*
    Create an InterfaceInterpreter using the PETSc implementation
    but overloading CreateMultiVector that doesn't create any
    new vector.
*/
int SLEPCSetupInterpreter(mv_InterfaceInterpreter *i)
{
  PETSCSetupInterpreter(i);
  i->CreateMultiVector = mv_TempMultiVectorCreateFromBV;

  return 0;
}

/*
    Change the multivector destructor in order to destroy the multivector
    structure without destroy the PETSc vectors.
*/
void SLEPCSetupInterpreterForDignifiedDeath(mv_InterfaceInterpreter *i)
{
  i->DestroyMultiVector = mv_TempMultiPETSCVectorDestroy;
}

