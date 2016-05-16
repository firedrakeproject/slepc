/*
   SLEPc mathematics include file. Defines basic operations and functions.
   This file is included by slepcsys.h and should not be used directly.

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

#if !defined(__SLEPCMATH_H)
#define __SLEPCMATH_H

/*
    Default tolerance for the different solvers, depending on the precision
*/
#if defined(PETSC_USE_REAL_SINGLE)
#  define SLEPC_DEFAULT_TOL   1e-6
#elif defined(PETSC_USE_REAL_DOUBLE)
#  define SLEPC_DEFAULT_TOL   1e-8
#elif defined(PETSC_USE_REAL___FLOAT128)
#  define SLEPC_DEFAULT_TOL   1e-16
#else
#  define SLEPC_DEFAULT_TOL   1e-7
#endif

/*@C
   SlepcAbs - Returns sqrt(x**2+y**2), taking care not to cause unnecessary
   overflow. It is based on LAPACK's DLAPY2.

   Not Collective

   Input parameters:
.  x,y - the real numbers

   Output parameter:
.  return - the result

   Note:
   This function is not available from Fortran.

   Level: developer
@*/
PETSC_STATIC_INLINE PetscReal SlepcAbs(PetscReal x,PetscReal y)
{
  PetscReal w,z,t,xabs=PetscAbs(x),yabs=PetscAbs(y);

  w = PetscMax(xabs,yabs);
  z = PetscMin(xabs,yabs);
  if (z == 0.0) return w;
  t = z/w;
  return w*PetscSqrtReal(1.0+t*t);
}

/*MC
   SlepcAbsEigenvalue - Returns the absolute value of a complex number given
   its real and imaginary parts.

   Synopsis:
   PetscReal SlepcAbsEigenvalue(PetscScalar x,PetscScalar y)

   Not Collective

   Input parameters:
+  x  - the real part of the complex number
-  y  - the imaginary part of the complex number

   Notes:
   This function computes sqrt(x**2+y**2), taking care not to cause unnecessary
   overflow. It is based on LAPACK's DLAPY2.

   This function is not available from Fortran.

   Level: developer
M*/
#if !defined(PETSC_USE_COMPLEX)
#define SlepcAbsEigenvalue(x,y) SlepcAbs(x,y)
#else
#define SlepcAbsEigenvalue(x,y) PetscAbsScalar(x)
#endif

#endif

