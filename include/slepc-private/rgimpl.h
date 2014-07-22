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

#if !defined(_RGIMPL)
#define _RGIMPL

#include <slepcrg.h>
#include <slepc-private/slepcimpl.h>

typedef struct _RGOps *RGOps;

struct _RGOps {
  PetscErrorCode (*contour)(RG,PetscInt,PetscScalar*,PetscScalar*);
  PetscErrorCode (*checkinside)(RG,PetscScalar,PetscScalar,PetscBool*);
  PetscErrorCode (*setfromoptions)(RG);
  PetscErrorCode (*view)(RG,PetscViewer);
  PetscErrorCode (*destroy)(RG);
};

struct _p_RG {
  PETSCHEADER(struct _RGOps);
  PetscBool   complement;    /* region is the complement of the specified one */
  void        *data;
};

#endif
