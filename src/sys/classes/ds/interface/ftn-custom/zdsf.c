/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepcds.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dsvectors_                DSVECTORS
#define dssort_                   DSSORT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dsvectors_                dsvectors
#define dssort_                   dssort
#endif

SLEPC_EXTERN void dsvectors_(DS *ds,DSMatType *mat,PetscInt *j,PetscReal *rnorm,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(j);
  CHKFORTRANNULLREAL(rnorm);
  *ierr = DSVectors(*ds,*mat,j,rnorm);
}

SLEPC_EXTERN void dssort_(DS *ds,PetscScalar *eigr,PetscScalar *eigi,PetscScalar *rr,PetscScalar *ri,PetscInt *k,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(eigr);
  CHKFORTRANNULLSCALAR(eigi);
  CHKFORTRANNULLSCALAR(rr);
  CHKFORTRANNULLSCALAR(ri);
  CHKFORTRANNULLINTEGER(k);
  *ierr = DSSort(*ds,eigr,eigi,rr,ri,k);
}
