/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepc/private/epsimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define epskrylovschursetsubintervals_    EPSKRYLOVSCHURSETSUBINTERVALS
#define epskrylovschurgetsubintervals_    EPSKRYLOVSCHURGETSUBINTERVALS
#define epskrylovschurgetinertias_        EPSKRYLOVSCHURGETINERTIAS
#define epskrylovschurgetsubcomminfo_     EPSKRYLOVSCHURGETSUBCOMMINFO
#define epskrylovschurgetsubcommpairs_    EPSKRYLOVSCHURGETSUBCOMMPAIRS
#define epskrylovschurgetsubcommmats_     EPSKRYLOVSCHURGETSUBCOMMMATS
#define epskrylovschurupdatesubcommmats_  EPSKRYLOVSCHURUPDATESUBCOMMMATS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define epskrylovschursetsubintervals_    epskrylovschursetsubintervals
#define epskrylovschurgetsubintervals_    epskrylovschurgetsubintervals
#define epskrylovschurgetinertias_        epskrylovschurgetinertias
#define epskrylovschurgetsubcomminfo_     epskrylovschurgetsubcomminfo
#define epskrylovschurgetsubcommpairs_    epskrylovschurgetsubcommpairs
#define epskrylovschurgetsubcommmats_     epskrylovschurgetsubcommmats
#define epskrylovschurupdatesubcommmats_  epskrylovschurupdatesubcommmats
#endif

PETSC_EXTERN void PETSC_STDCALL epskrylovschursetsubintervals_(EPS *eps,PetscReal *subint,PetscErrorCode *ierr)
{
  CHKFORTRANNULLREAL(subint);
  *ierr = EPSKrylovSchurSetSubintervals(*eps,subint);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschurgetsubintervals_(EPS *eps,PetscReal *subint,PetscErrorCode *ierr)
{
  PetscReal *osubint;
  PetscInt  npart;

  CHKFORTRANNULLREAL(subint);
  *ierr = EPSKrylovSchurGetSubintervals(*eps,&osubint); if (*ierr) return;
  *ierr = EPSKrylovSchurGetPartitions(*eps,&npart); if (*ierr) return;
  *ierr = PetscMemcpy(subint,osubint,(npart+1)*sizeof(PetscReal)); if (*ierr) return;
  *ierr = PetscFree(osubint);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschurgetinertias_(EPS *eps,PetscInt *nshift,PetscReal *shifts,PetscInt *inertias,PetscErrorCode *ierr)
{
  PetscReal *oshifts;
  PetscInt  *oinertias;
  PetscInt  n;

  CHKFORTRANNULLREAL(shifts);
  CHKFORTRANNULLINTEGER(inertias);
  *ierr = EPSKrylovSchurGetInertias(*eps,&n,&oshifts,&oinertias); if (*ierr) return;
  if (shifts) { *ierr = PetscMemcpy(shifts,oshifts,n*sizeof(PetscReal)); if (*ierr) return; }
  if (inertias) { *ierr = PetscMemcpy(inertias,oinertias,n*sizeof(PetscInt)); if (*ierr) return; }
  *nshift = n;
  *ierr = PetscFree(oshifts);
  *ierr = PetscFree(oinertias);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschurgetsubcomminfo_(EPS *eps,PetscInt *k,PetscInt *n,Vec *v,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(v);
  *ierr = EPSKrylovSchurGetSubcommInfo(*eps,k,n,v);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschurgetsubcommpairs_(EPS *eps,PetscInt *i,PetscScalar *eig,Vec *v,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(v);
  *ierr = EPSKrylovSchurGetSubcommPairs(*eps,*i,eig,*v);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschurgetsubcommmats_(EPS *eps,Mat *A,Mat *B,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(A);
  CHKFORTRANNULLOBJECT(B);
  *ierr = EPSKrylovSchurGetSubcommMats(*eps,A,B);
}

PETSC_EXTERN void PETSC_STDCALL epskrylovschurupdatesubcommmats_(EPS *eps,PetscScalar *s,PetscScalar *a,Mat *Au,PetscScalar *t,PetscScalar *b,Mat *Bu,MatStructure *str,PetscBool *globalup,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(Au);
  CHKFORTRANNULLOBJECTDEREFERENCE(Bu);
  *ierr = EPSKrylovSchurUpdateSubcommMats(*eps,*s,*a,*Au,*t,*b,*Bu,*str,*globalup);
}

