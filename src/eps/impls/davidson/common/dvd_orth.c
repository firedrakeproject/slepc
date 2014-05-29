/*
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

#undef __FUNCT__
#define __FUNCT__ "dvd_orthV"
/* auxS: size_cX+V_new_e */
PetscErrorCode dvd_orthV(IP ip,Vec *defl,PetscInt size_DS,Vec *cX,PetscInt size_cX,Vec *V,PetscInt V_new_s,PetscInt V_new_e,PetscScalar *auxS,PetscRandom rand)
{
  PetscErrorCode  ierr;
  PetscInt        i, j;
  PetscBool       lindep;
  PetscReal       norm;
  PetscScalar     *auxS0 = auxS;

  PetscFunctionBegin;
  /* Orthonormalize V with IP */
  for (i=V_new_s;i<V_new_e;i++) {
    for (j=0;j<3;j++) {
      if (j>0) { ierr = SlepcVecSetRandom(V[i], rand);CHKERRQ(ierr); }
      if (cX + size_cX == V) {
        /* If cX and V are contiguous, orthogonalize in one step */
        ierr = IPOrthogonalize(ip, size_DS, defl, size_cX+i, NULL, cX,
                               V[i], auxS0, &norm, &lindep);CHKERRQ(ierr);
      } else if (defl) {
        /* Else orthogonalize first against defl, and then against cX and V */
        ierr = IPOrthogonalize(ip, size_DS, defl, size_cX, NULL, cX,
                               V[i], auxS0, NULL, &lindep);CHKERRQ(ierr);
        if (!lindep) {
          ierr = IPOrthogonalize(ip, 0, NULL, i, NULL, V,
                                 V[i], auxS0, &norm, &lindep);CHKERRQ(ierr);
        }
      } else {
        /* Else orthogonalize first against cX and then against V */
        ierr = IPOrthogonalize(ip, size_cX, cX, i, NULL, V,
                               V[i], auxS0, &norm, &lindep);CHKERRQ(ierr);
      }
      if (!lindep && (norm > PETSC_SQRT_MACHINE_EPSILON)) break;
      ierr = PetscInfo1(ip,"Orthonormalization problems adding the vector %D to the searching subspace\n",i);CHKERRQ(ierr);
    }
    if (lindep || (norm < PETSC_SQRT_MACHINE_EPSILON)) SETERRQ(PetscObjectComm((PetscObject)ip),1, "Error during orthonormalization of eigenvectors");
    ierr = VecScale(V[i],1.0/norm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_BorthV_stable"
/* auxS: size_cX+V_new_e+1 */
PetscErrorCode dvd_BorthV_stable(IP ip,Vec *defl,PetscReal *BDSn,PetscInt size_DS,Vec *cX,PetscReal *BcXn,PetscInt size_cX,Vec *V,PetscReal *BVn,PetscInt V_new_s,PetscInt V_new_e,PetscScalar *auxS,PetscRandom rand)
{
  PetscErrorCode  ierr;
  PetscInt        i, j;
  PetscBool       lindep;
  PetscReal       norm;
  PetscScalar     *auxS0 = auxS;

  PetscFunctionBegin;
  /* Orthonormalize V with IP */
  for (i=V_new_s;i<V_new_e;i++) {
    for (j=0;j<3;j++) {
      if (j>0) {
        ierr = SlepcVecSetRandom(V[i],rand);CHKERRQ(ierr);
        ierr = PetscInfo1(ip,"Orthonormalization problems adding the vector %d to the searching subspace\n",i);CHKERRQ(ierr);
      }
      /* Orthogonalize against the deflation, if needed */
      if (defl) {
        ierr = IPPseudoOrthogonalize(ip,size_DS,defl,BDSn,V[i],auxS0,NULL,&lindep);CHKERRQ(ierr);
        if (lindep) continue;
      }
      /* If cX and V are contiguous, orthogonalize in one step */
      if (cX + size_cX == V) {
        ierr = IPPseudoOrthogonalize(ip,size_cX+i,cX,BcXn,V[i],auxS0,&norm,&lindep);CHKERRQ(ierr);
      /* Else orthogonalize first against cX and then against V */
      } else {
        ierr = IPPseudoOrthogonalize(ip,size_cX,cX,BcXn,V[i],auxS0,NULL,&lindep);CHKERRQ(ierr);
        if (lindep) continue;
        ierr = IPPseudoOrthogonalize(ip,i,V,BVn,V[i],auxS0,&norm,&lindep);CHKERRQ(ierr);
      }
      if (!lindep && (PetscAbs(norm) > PETSC_MACHINE_EPSILON)) break;
    }
    if (lindep || (PetscAbs(norm) < PETSC_MACHINE_EPSILON)) {
        SETERRQ(PetscObjectComm((PetscObject)ip),1, "Error during the orthonormalization of the eigenvectors");
    }
    if (BVn) BVn[i] = norm > 0.0 ? 1.0 : -1.0;
    norm = PetscAbs(norm);
    ierr = VecScale(V[i],1.0/norm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
