/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

#if !defined(__SVECIMPL_H)
#define __SVECIMPL_H

typedef struct {
  Vec       v;
  PetscBool mpi;    /* true if either VECMPI or VECMPICUSP */
  PetscBool cuda;   /* true if either VECSEQCUDA or VECMPICUDA */
} BV_SVEC;

PETSC_INTERN PetscErrorCode BVMult_Svec_CUDA(BV,PetscScalar,PetscScalar,BV,Mat);

#endif

