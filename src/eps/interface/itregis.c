/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/epsimpl.h>  /*I "slepceps.h" I*/

PETSC_EXTERN PetscErrorCode EPSCreate_Power(EPS);
PETSC_EXTERN PetscErrorCode EPSCreate_Subspace(EPS);
PETSC_EXTERN PetscErrorCode EPSCreate_Arnoldi(EPS);
PETSC_EXTERN PetscErrorCode EPSCreate_Lanczos(EPS);
PETSC_EXTERN PetscErrorCode EPSCreate_KrylovSchur(EPS);
#if defined(SLEPC_HAVE_ARPACK)
PETSC_EXTERN PetscErrorCode EPSCreate_ARPACK(EPS);
#endif
PETSC_EXTERN PetscErrorCode EPSCreate_LAPACK(EPS);
#if defined(SLEPC_HAVE_BLZPACK) && !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN PetscErrorCode EPSCreate_BLZPACK(EPS);
#endif
#if defined(SLEPC_HAVE_TRLAN) && !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN PetscErrorCode EPSCreate_TRLAN(EPS);
#endif
#if defined(SLEPC_HAVE_BLOPEX)
PETSC_EXTERN PetscErrorCode EPSCreate_BLOPEX(EPS);
#endif
#if defined(SLEPC_HAVE_PRIMME)
PETSC_EXTERN PetscErrorCode EPSCreate_PRIMME(EPS eps);
#endif
PETSC_EXTERN PetscErrorCode EPSCreate_GD(EPS eps);
PETSC_EXTERN PetscErrorCode EPSCreate_JD(EPS eps);
PETSC_EXTERN PetscErrorCode EPSCreate_RQCG(EPS eps);

#undef __FUNCT__
#define __FUNCT__ "EPSRegisterAll"
/*@C
  EPSRegisterAll - Registers all the eigenvalue solvers in the EPS package.

  Not Collective

  Level: advanced

.seealso:  EPSRegister()
@*/
PetscErrorCode EPSRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  EPSRegisterAllCalled = PETSC_TRUE;
  ierr = EPSRegister(EPSKRYLOVSCHUR,path,"EPSCreate_KrylovSchur",EPSCreate_KrylovSchur);CHKERRQ(ierr);
  ierr = EPSRegister(EPSPOWER,path,"EPSCreate_Power",EPSCreate_Power);CHKERRQ(ierr);
  ierr = EPSRegister(EPSSUBSPACE,path,"EPSCreate_Subspace",EPSCreate_Subspace);CHKERRQ(ierr);
  ierr = EPSRegister(EPSARNOLDI,path,"EPSCreate_Arnoldi",EPSCreate_Arnoldi);CHKERRQ(ierr);
  ierr = EPSRegister(EPSLANCZOS,path,"EPSCreate_Lanczos",EPSCreate_Lanczos);CHKERRQ(ierr);
  ierr = EPSRegister(EPSGD,path,"EPSCreate_GD",EPSCreate_GD);CHKERRQ(ierr);
  ierr = EPSRegister(EPSJD,path,"EPSCreate_JD",EPSCreate_JD);CHKERRQ(ierr);
  ierr = EPSRegister(EPSRQCG,path,"EPSCreate_RQCG",EPSCreate_RQCG);CHKERRQ(ierr);
  ierr = EPSRegister(EPSLAPACK,path,"EPSCreate_LAPACK",EPSCreate_LAPACK);CHKERRQ(ierr);
#if defined(SLEPC_HAVE_ARPACK)
  ierr = EPSRegister(EPSARPACK,path,"EPSCreate_ARPACK",EPSCreate_ARPACK);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_BLZPACK) && !defined(PETSC_USE_COMPLEX)
  ierr = EPSRegister(EPSBLZPACK,path,"EPSCreate_BLZPACK",EPSCreate_BLZPACK);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_TRLAN) && !defined(PETSC_USE_COMPLEX)
  ierr = EPSRegister(EPSTRLAN,path,"EPSCreate_TRLAN",EPSCreate_TRLAN);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_BLOPEX)
  ierr = EPSRegister(EPSBLOPEX,path,"EPSCreate_BLOPEX",EPSCreate_BLOPEX);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_PRIMME)
  ierr = EPSRegister(EPSPRIMME,path,"EPSCreate_PRIMME",EPSCreate_PRIMME);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
