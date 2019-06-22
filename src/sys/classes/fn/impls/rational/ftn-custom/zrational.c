/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepc/private/fnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fnrationalsetnumerator0_    FNRATIONALSETNUMERATOR0
#define fnrationalsetnumerator1_    FNRATIONALSETNUMERATOR1
#define fnrationalsetdenominator0_  FNRATIONALSETDENOMINATOR0
#define fnrationalsetdenominator1_  FNRATIONALSETDENOMINATOR1
#define fnrationalgetnumerator00_   FNRATIONALGETNUMERATOR00
#define fnrationalgetnumerator10_   FNRATIONALGETNUMERATOR10
#define fnrationalgetnumerator01_   FNRATIONALGETNUMERATOR01
#define fnrationalgetnumerator11_   FNRATIONALGETNUMERATOR11
#define fnrationalgetdenominator00_ FNRATIONALGETDENOMINATOR00
#define fnrationalgetdenominator10_ FNRATIONALGETDENOMINATOR10
#define fnrationalgetdenominator01_ FNRATIONALGETDENOMINATOR01
#define fnrationalgetdenominator11_ FNRATIONALGETDENOMINATOR11
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fnrationalsetnumerator0_    fnrationalsetnumerator0
#define fnrationalsetnumerator1_    fnrationalsetnumerator1
#define fnrationalsetdenominator0_  fnrationalsetdenominator0
#define fnrationalsetdenominator1_  fnrationalsetdenominator1
#define fnrationalgetnumerator00_   fnrationalgetnumerator00
#define fnrationalgetnumerator10_   fnrationalgetnumerator10
#define fnrationalgetnumerator01_   fnrationalgetnumerator01
#define fnrationalgetnumerator11_   fnrationalgetnumerator11
#define fnrationalgetdenominator00_ fnrationalgetdenominator00
#define fnrationalgetdenominator10_ fnrationalgetdenominator10
#define fnrationalgetdenominator01_ fnrationalgetdenominator01
#define fnrationalgetdenominator11_ fnrationalgetdenominator11
#endif

SLEPC_EXTERN void PETSC_STDCALL fnrationalsetnumerator_(FN *fn,PetscInt *np,PetscScalar *pcoeff,int *ierr)
{
  CHKFORTRANNULLSCALAR(pcoeff);
  *ierr = FNRationalSetNumerator(*fn,*np,pcoeff);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalsetnumerator0_(FN *fn,PetscInt *np,PetscScalar *pcoeff,int *ierr)
{
  fnrationalsetnumerator_(fn,np,pcoeff,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalsetnumerator1_(FN *fn,PetscInt *np,PetscScalar *pcoeff,int *ierr)
{
  fnrationalsetnumerator_(fn,np,pcoeff,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalsetdenominator_(FN *fn,PetscInt *nq,PetscScalar *qcoeff,int *ierr)
{
  CHKFORTRANNULLSCALAR(qcoeff);
  *ierr = FNRationalSetDenominator(*fn,*nq,qcoeff);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalsetdenominator0_(FN *fn,PetscInt *nq,PetscScalar *qcoeff,int *ierr)
{
  fnrationalsetdenominator_(fn,nq,qcoeff,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalsetdenominator1_(FN *fn,PetscInt *nq,PetscScalar *qcoeff,int *ierr)
{
  fnrationalsetdenominator_(fn,nq,qcoeff,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalgetnumerator_(FN *fn,PetscInt *np,PetscScalar *pcoeff,int *ierr)
{
  PetscScalar *ocoeff;
  PetscInt    n;

  CHKFORTRANNULLSCALAR(pcoeff);
  *ierr = FNRationalGetNumerator(*fn,&n,&ocoeff); if (*ierr) return;
  if (pcoeff) { *ierr = PetscArraycpy(pcoeff,ocoeff,n); if (*ierr) return; }
  *np = n;
  *ierr = PetscFree(ocoeff);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalgetnumerator00_(FN *fn,PetscInt *np,PetscScalar *qcoeff,int *ierr)
{
  fnrationalgetnumerator_(fn,np,qcoeff,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalgetnumerator10_(FN *fn,PetscInt *np,PetscScalar *qcoeff,int *ierr)
{
  fnrationalgetnumerator_(fn,np,qcoeff,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalgetnumerator01_(FN *fn,PetscInt *np,PetscScalar *qcoeff,int *ierr)
{
  fnrationalgetnumerator_(fn,np,qcoeff,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalgetnumerator11_(FN *fn,PetscInt *np,PetscScalar *qcoeff,int *ierr)
{
  fnrationalgetnumerator_(fn,np,qcoeff,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalgetdenominator_(FN *fn,PetscInt *nq,PetscScalar *qcoeff,int *ierr)
{
  PetscScalar *ocoeff;
  PetscInt    n;

  CHKFORTRANNULLSCALAR(qcoeff);
  *ierr = FNRationalGetDenominator(*fn,&n,&ocoeff); if (*ierr) return;
  if (qcoeff) { *ierr = PetscArraycpy(qcoeff,ocoeff,n); if (*ierr) return; }
  *nq = n;
  *ierr = PetscFree(ocoeff);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalgetdenominator00_(FN *fn,PetscInt *nq,PetscScalar *qcoeff,int *ierr)
{
  fnrationalgetdenominator_(fn,nq,qcoeff,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalgetdenominator10_(FN *fn,PetscInt *nq,PetscScalar *qcoeff,int *ierr)
{
  fnrationalgetdenominator_(fn,nq,qcoeff,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalgetdenominator01_(FN *fn,PetscInt *nq,PetscScalar *qcoeff,int *ierr)
{
  fnrationalgetdenominator_(fn,nq,qcoeff,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL fnrationalgetdenominator11_(FN *fn,PetscInt *nq,PetscScalar *qcoeff,int *ierr)
{
  fnrationalgetdenominator_(fn,nq,qcoeff,ierr);
}

