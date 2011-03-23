/*
   Private data structure used by the ARPACK interface

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain
 
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

#if !defined(__ARPACKP_H)
#define __ARPACKP_H

#include <private/epsimpl.h>                /*I "slepceps.h" I*/

typedef struct {
  PetscBool   *select;
  PetscScalar *workev;
  PetscScalar *workd;
  PetscScalar *workl;
  PetscBLASInt
              lworkl;
#if defined(PETSC_USE_COMPLEX)
  PetscReal  *rwork;
#endif
} EPS_ARPACK;

/*
   Definition of routines from the ARPACK package
*/

#if defined(SLEPC_ARPACK_HAVE_UNDERSCORE)
#define SLEPC_ARPACK(lcase,ucase) lcase##_
#elif defined(SLEPC_ARPACK_HAVE_CAPS)
#define SLEPC_ARPACK(lcase,ucase) ucase
#else
#define SLEPC_ARPACK(lcase,ucase) lcase
#endif

#if !defined(PETSC_USE_COMPLEX)

/*
    These are real case 
*/

#if defined(PETSC_USES_FORTRAN_SINGLE) 
/*
   For these machines we must call the single precision Fortran version
*/
#define ARnaupd_ SLEPC_ARPACK(psnaupd,PSNAUPD)
#define ARneupd_ SLEPC_ARPACK(psneupd,PSNEUPD)
#define ARsaupd_ SLEPC_ARPACK(pssaupd,PSSAUPD)
#define ARseupd_ SLEPC_ARPACK(psseupd,PSSEUPD)

#else

#define ARnaupd_ SLEPC_ARPACK(pdnaupd,PDNAUPD)
#define ARneupd_ SLEPC_ARPACK(pdneupd,PDNEUPD)
#define ARsaupd_ SLEPC_ARPACK(pdsaupd,PDSAUPD)
#define ARseupd_ SLEPC_ARPACK(pdseupd,PDSEUPD)

#endif

#else
/*
   Complex 
*/
#if defined(PETSC_USE_SINGLE) 

#define ARnaupd_ SLEPC_ARPACK(pcnaupd,PCNAUPD)
#define ARneupd_ SLEPC_ARPACK(pcneupd,PCNEUPD)

#else

#define ARnaupd_ SLEPC_ARPACK(pznaupd,PZNAUPD)
#define ARneupd_ SLEPC_ARPACK(pzneupd,PZNEUPD)

#endif

#endif

EXTERN_C_BEGIN

extern void   ARsaupd_(MPI_Fint*,PetscBLASInt*,char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,
                       PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,
                       PetscBLASInt*,PetscBLASInt*,int,int);
extern void   ARseupd_(MPI_Fint*,PetscBool*,char*,PetscBool*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,
                       char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,
                       PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,
                       PetscBLASInt*,int,int,int);

#if !defined(PETSC_USE_COMPLEX)
extern void   ARnaupd_(MPI_Fint*,PetscBLASInt*,char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,
                       PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,
                       PetscBLASInt*,PetscBLASInt*,int,int);
extern void   ARneupd_(MPI_Fint*,PetscBool*,char*,PetscBool*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,
                       PetscReal*,PetscReal*,char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,
                       PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,
                       PetscBLASInt*,PetscBLASInt*,int,int,int);
#else
extern void   ARnaupd_(MPI_Fint*,PetscBLASInt*,char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,
                       PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,
                       PetscBLASInt*,PetscReal*,PetscBLASInt*,int,int);
extern void   ARneupd_(MPI_Fint*,PetscBool*,char*,PetscBool*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,
                       PetscScalar*,char*,PetscBLASInt*,const char*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,
                       PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,
                       PetscReal*,PetscBLASInt*,int,int,int);
#endif

EXTERN_C_END

#endif

