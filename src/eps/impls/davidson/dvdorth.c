/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#undef __FUNCT__
#define __FUNCT__ "dvd_orthV"
PetscErrorCode dvd_orthV(BV V,PetscInt V_new_s,PetscInt V_new_e,PetscRandom rand)
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
        ierr = BVSetRandomColumn(V,i,rand);CHKERRQ(ierr);
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
