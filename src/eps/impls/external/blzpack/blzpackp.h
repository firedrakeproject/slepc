/*
   Private data structure used by the BLZPACK interface

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

#if !defined(__BLZPACKP_H)
#define __BLZPACKP_H

typedef struct {
  PetscBLASInt         block_size;      /* block size */
  PetscBLASInt         slice;           /* use spectrum slicing */
  PetscBLASInt         nsteps;          /* maximum number of steps per run */
  PetscBLASInt         *istor;
  PetscReal            *rstor;
  PetscScalar          *u;
  PetscScalar          *v;
  PetscScalar          *eig;
} EPS_BLZPACK;

/*
   Definition of routines from the BLZPACK package
*/

#if defined(SLEPC_BLZPACK_HAVE_UNDERSCORE)
#define SLEPC_BLZPACK(lcase,ucase) lcase##_
#elif defined(SLEPC_BLZPACK_HAVE_CAPS)
#define SLEPC_BLZPACK(lcase,ucase) ucase
#else
#define SLEPC_BLZPACK(lcase,ucase) lcase
#endif

/*
    These are real case, current version of BLZPACK only supports real
    matrices
*/

#if defined(PETSC_USE_SINGLE)
/*
   For these machines we must call the single precision Fortran version
*/
#define BLZpack_ SLEPC_BLZPACK(blzdrs,BLZDRS)
#else
#define BLZpack_ SLEPC_BLZPACK(blzdrd,BLZDRD)
#endif

#define BLZistorr_ SLEPC_BLZPACK(istorr,ISTORR)
#define BLZrstorr_ SLEPC_BLZPACK(rstorr,RSTORR)

PETSC_EXTERN void BLZpack_(PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*);

PETSC_EXTERN PetscBLASInt BLZistorr_(PetscBLASInt*,const char*,int);
PETSC_EXTERN PetscReal BLZrstorr_(PetscReal*,char*,int);

#endif

