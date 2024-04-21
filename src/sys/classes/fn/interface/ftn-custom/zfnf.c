/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepcfn.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fndestroy_                 FNDESTROY
#define fnview_                    FNVIEW
#define fnviewfromoptions_         FNVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fndestroy_                 fndestroy
#define fnview_                    fnview
#define fnviewfromoptions_         fnviewfromoptions
#endif

SLEPC_EXTERN void fndestroy_(FN *fn,PetscErrorCode *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(fn);
  *ierr = FNDestroy(fn); if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(fn);
}

SLEPC_EXTERN void fnview_(FN *fn,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = FNView(*fn,v);
}

SLEPC_EXTERN void fnviewfromoptions_(FN *fn,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = FNViewFromOptions(*fn,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
