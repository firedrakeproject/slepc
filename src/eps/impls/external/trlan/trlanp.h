/*
   Private data structure used by the TRLAN interface

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

#if !defined(__TRLANP_H)
#define __TRLANP_H

typedef struct {
  PetscBLASInt       maxlan;
  PetscBLASInt       restart;
  PetscReal          *work;
  PetscBLASInt       lwork;
} EPS_TRLAN;

/*
   Definition of routines from the TRLAN package
   These are real case. TRLAN currently only has DOUBLE PRECISION version
*/

#if defined(SLEPC_TRLAN_HAVE_UNDERSCORE)
#define TRLan_ trlan77_
#elif defined(SLEPC_TRLAN_HAVE_CAPS)
#define TRLan_ TRLAN77
#else
#define TRLan_ trlan77
#endif

PETSC_EXTERN void TRLan_(PetscBLASInt(*op)(PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*),PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);

#endif

