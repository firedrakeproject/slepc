/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/eps/epsimpl.h"  /*I "slepceps.h" I*/

EXTERN_C_BEGIN
EXTERN PetscErrorCode EPSCreate_POWER(EPS);
EXTERN PetscErrorCode EPSCreate_SUBSPACE(EPS);
EXTERN PetscErrorCode EPSCreate_ARNOLDI(EPS);
EXTERN PetscErrorCode EPSCreate_LANCZOS(EPS);
EXTERN PetscErrorCode EPSCreate_KRYLOVSCHUR(EPS);
#if defined(SLEPC_HAVE_ARPACK)
EXTERN PetscErrorCode EPSCreate_ARPACK(EPS);
#endif
EXTERN PetscErrorCode EPSCreate_LAPACK(EPS);
#if defined(SLEPC_HAVE_BLZPACK) && !defined(PETSC_USE_COMPLEX)
EXTERN PetscErrorCode EPSCreate_BLZPACK(EPS);
#endif
#if defined(SLEPC_HAVE_TRLAN) && !defined(PETSC_USE_COMPLEX)
EXTERN PetscErrorCode EPSCreate_TRLAN(EPS);
#endif
#if defined(PETSC_HAVE_BLOPEX) && !defined(PETSC_USE_COMPLEX)
EXTERN PetscErrorCode EPSCreate_BLOPEX(EPS);
#endif
#if defined(SLEPC_HAVE_PRIMME)
EXTERN PetscErrorCode EPSCreate_PRIMME(EPS eps);
#endif
EXTERN_C_END
  
#undef __FUNCT__  
#define __FUNCT__ "EPSRegisterAll"
/*@C
  EPSRegisterAll - Registers all the eigenvalue solvers in the EPS package.

  Not Collective

  Level: advanced

.seealso:  EPSRegisterDynamic()
@*/
PetscErrorCode EPSRegisterAll(char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = EPSRegisterDynamic(EPSPOWER, path,"EPSCreate_POWER", 
		  EPSCreate_POWER);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSSUBSPACE, path,"EPSCreate_SUBSPACE", 
		  EPSCreate_SUBSPACE);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSARNOLDI, path,"EPSCreate_ARNOLDI", 
		  EPSCreate_ARNOLDI);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSLANCZOS, path,"EPSCreate_LANCZOS", 
		  EPSCreate_LANCZOS);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSKRYLOVSCHUR, path,"EPSCreate_KRYLOVSCHUR", 
		  EPSCreate_KRYLOVSCHUR);CHKERRQ(ierr);
#if defined(SLEPC_HAVE_ARPACK)
  ierr = EPSRegisterDynamic(EPSARPACK, path,"EPSCreate_ARPACK", 
		  EPSCreate_ARPACK);CHKERRQ(ierr);
#endif
  ierr = EPSRegisterDynamic(EPSLAPACK, path,"EPSCreate_LAPACK", 
		  EPSCreate_LAPACK);CHKERRQ(ierr);
#if defined(SLEPC_HAVE_BLZPACK) && !defined(PETSC_USE_COMPLEX)
  ierr = EPSRegisterDynamic(EPSBLZPACK, path,"EPSCreate_BLZPACK", 
		  EPSCreate_BLZPACK);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_TRLAN) && !defined(PETSC_USE_COMPLEX)
  ierr = EPSRegisterDynamic(EPSTRLAN, path,"EPSCreate_TRLAN", 
		  EPSCreate_TRLAN);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_BLOPEX) && !defined(PETSC_USE_COMPLEX)
  ierr = EPSRegisterDynamic(EPSBLOPEX, path,"EPSCreate_BLOPEX", 
		  EPSCreate_BLOPEX);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_PRIMME)
  ierr = EPSRegisterDynamic(EPSPRIMME, path, "EPSCreate_PRIMME", 
                            EPSCreate_PRIMME);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
