/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "private/fortranimpl.h"
#include "slepceps.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define epsprimmegetmethod_  EPSPRIMMEGETMETHOD
#define epsprimmegetprecond_ EPSPRIMMEGETPRECOND
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define epsprimmegetmethod_  epsprimmegetmethod
#define epsprimmegetprecond_ epsprimmegetprecond
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL  epsprimmegetmethod_(EPS *eps,EPSPRIMMEMethod *method, PetscErrorCode *ierr ){
  *ierr = EPSPRIMMEGetMethod(*eps,method);
}

void PETSC_STDCALL  epsprimmegetprecond_(EPS *eps,EPSPRIMMEPrecond *precond, PetscErrorCode *ierr ){
  *ierr = EPSPRIMMEGetPrecond(*eps,precond);
}

EXTERN_C_END

