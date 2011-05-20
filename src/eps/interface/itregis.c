/*
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

#include <private/epsimpl.h>  /*I "slepceps.h" I*/

EXTERN_C_BEGIN
extern PetscErrorCode EPSCreate_Power(EPS);
extern PetscErrorCode EPSCreate_Subspace(EPS);
extern PetscErrorCode EPSCreate_Arnoldi(EPS);
extern PetscErrorCode EPSCreate_Lanczos(EPS);
extern PetscErrorCode EPSCreate_KrylovSchur(EPS);
extern PetscErrorCode EPSCreate_DSITRLanczos(EPS);
#if defined(SLEPC_HAVE_ARPACK)
extern PetscErrorCode EPSCreate_ARPACK(EPS);
#endif
extern PetscErrorCode EPSCreate_LAPACK(EPS);
#if defined(SLEPC_HAVE_BLZPACK) && !defined(PETSC_USE_COMPLEX)
extern PetscErrorCode EPSCreate_BLZPACK(EPS);
#endif
#if defined(SLEPC_HAVE_TRLAN) && !defined(PETSC_USE_COMPLEX)
extern PetscErrorCode EPSCreate_TRLAN(EPS);
#endif
#if defined(PETSC_HAVE_BLOPEX)
extern PetscErrorCode EPSCreate_BLOPEX(EPS);
#endif
#if defined(SLEPC_HAVE_PRIMME)
extern PetscErrorCode EPSCreate_PRIMME(EPS eps);
#endif
extern PetscErrorCode EPSCreate_GD(EPS eps);
extern PetscErrorCode EPSCreate_JD(EPS eps);
EXTERN_C_END
  
#undef __FUNCT__  
#define __FUNCT__ "EPSRegisterAll"
/*@C
  EPSRegisterAll - Registers all the eigenvalue solvers in the EPS package.

  Not Collective

  Level: advanced

.seealso:  EPSRegisterDynamic()
@*/
PetscErrorCode EPSRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  EPSRegisterAllCalled = PETSC_TRUE;
  ierr = EPSRegisterDynamic(EPSKRYLOVSCHUR,path,"EPSCreate_KRYLOVSCHUR",EPSCreate_KrylovSchur);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSPOWER,path,"EPSCreate_POWER",EPSCreate_Power);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSSUBSPACE,path,"EPSCreate_SUBSPACE",EPSCreate_Subspace);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSARNOLDI,path,"EPSCreate_ARNOLDI",EPSCreate_Arnoldi);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSLANCZOS,path,"EPSCreate_LANCZOS",EPSCreate_Lanczos);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSDSITRLANCZOS,path,"EPSCreate_DSITRLANCZOS",EPSCreate_DSITRLanczos);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSGD,path,"EPSCreate_GD",EPSCreate_GD);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSJD,path,"EPSCreate_JD",EPSCreate_JD);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSLAPACK,path,"EPSCreate_LAPACK",EPSCreate_LAPACK);CHKERRQ(ierr);
#if defined(SLEPC_HAVE_ARPACK)
  ierr = EPSRegisterDynamic(EPSARPACK,path,"EPSCreate_ARPACK",EPSCreate_ARPACK);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_BLZPACK) && !defined(PETSC_USE_COMPLEX)
  ierr = EPSRegisterDynamic(EPSBLZPACK,path,"EPSCreate_BLZPACK",EPSCreate_BLZPACK);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_TRLAN) && !defined(PETSC_USE_COMPLEX)
  ierr = EPSRegisterDynamic(EPSTRLAN,path,"EPSCreate_TRLAN",EPSCreate_TRLAN);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_BLOPEX)
  ierr = EPSRegisterDynamic(EPSBLOPEX,path,"EPSCreate_BLOPEX",EPSCreate_BLOPEX);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_PRIMME)
  ierr = EPSRegisterDynamic(EPSPRIMME,path,"EPSCreate_PRIMME",EPSCreate_PRIMME);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
