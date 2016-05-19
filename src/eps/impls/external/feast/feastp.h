/*
   Private data structure used by the FEAST interface

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

#if !defined(__FEASTP_H)
#define __FEASTP_H

typedef struct {
  PetscScalar  *work1,*work2,*Aq,*Bq;   /* workspace */
  PetscBLASInt npoints;                 /* number of contour points */
} EPS_FEAST;

/*
   Definition of routines from the FEAST package
*/

#if defined(SLEPC_FEAST_HAVE_UNDERSCORE)
#define SLEPC_FEAST(lcase,ucase) lcase##_
#elif defined(SLEPC_FEAST_HAVE_CAPS)
#define SLEPC_FEAST(lcase,ucase) ucase
#else
#define SLEPC_FEAST(lcase,ucase) lcase
#endif

#if defined(PETSC_USE_COMPLEX)

#if defined(PETSC_USE_REAL_SINGLE)

#if defined(SLEPC_FEAST_HAVE_UNDERSCORE)
#define SLEPC_FEASTM(lcase,ucase) cfeast_h##lcase##_
#elif defined(SLEPC_FEAST_HAVE_CAPS)
#define SLEPC_FEASTM(lcase,ucase) CFEAST_H##ucase
#else
#define SLEPC_FEASTM(lcase,ucase) cfeast_h##lcase
#endif

#else

#if defined(SLEPC_FEAST_HAVE_UNDERSCORE)
#define SLEPC_FEASTM(lcase,ucase) zfeast_h##lcase##_
#elif defined(SLEPC_FEAST_HAVE_CAPS)
#define SLEPC_FEASTM(lcase,ucase) ZFEAST_H##ucase
#else
#define SLEPC_FEASTM(lcase,ucase) zfeast_h##lcase
#endif

#endif

#else

#if defined(PETSC_USE_REAL_SINGLE)

#if defined(SLEPC_FEAST_HAVE_UNDERSCORE)
#define SLEPC_FEASTM(lcase,ucase) sfeast_s##lcase##_
#elif defined(SLEPC_FEAST_HAVE_CAPS)
#define SLEPC_FEASTM(lcase,ucase) SFEAST_S##ucase
#else
#define SLEPC_FEASTM(lcase,ucase) sfeast_s##lcase
#endif

#else

#if defined(SLEPC_FEAST_HAVE_UNDERSCORE)
#define SLEPC_FEASTM(lcase,ucase) dfeast_s##lcase##_
#elif defined(SLEPC_FEAST_HAVE_CAPS)
#define SLEPC_FEASTM(lcase,ucase) DFEAST_S##ucase
#else
#define SLEPC_FEASTM(lcase,ucase) dfeast_s##lcase
#endif

#endif

#endif

#define FEASTinit_(a) SLEPC_FEAST(feastinit,FEASTINIT) ((a))
#define FEASTrci_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r) SLEPC_FEASTM(rci,RCI) ((a),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r))

PETSC_EXTERN void   SLEPC_FEAST(feastinit,FEASTINIT)(PetscBLASInt*);
PETSC_EXTERN void   SLEPC_FEASTM(rci,RCI)(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);

#endif

