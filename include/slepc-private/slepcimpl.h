/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#ifndef _SLEPCIMPL
#define _SLEPCIMPL

#include <slepcsys.h>

/* context for monitors of type XXXMonitorConverged */
struct _n_SlepcConvMonitor {
  PetscViewer viewer;
  PetscInt    oldnconv;
};
typedef struct _n_SlepcConvMonitor* SlepcConvMonitor;

/* Private functions that are shared by several classes */

extern PetscErrorCode DenseSelectedEvec(PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscInt,PetscBool,PetscInt,PetscScalar*);
extern PetscErrorCode SlepcConvMonitorDestroy(SlepcConvMonitor *ctx);

extern PetscErrorCode SlepcCompareLargestMagnitude(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
extern PetscErrorCode SlepcCompareSmallestMagnitude(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
extern PetscErrorCode SlepcCompareLargestReal(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
extern PetscErrorCode SlepcCompareSmallestReal(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
extern PetscErrorCode SlepcCompareLargestImaginary(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
extern PetscErrorCode SlepcCompareSmallestImaginary(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
extern PetscErrorCode SlepcCompareTargetMagnitude(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
extern PetscErrorCode SlepcCompareTargetReal(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
extern PetscErrorCode SlepcCompareTargetImaginary(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);

#endif
