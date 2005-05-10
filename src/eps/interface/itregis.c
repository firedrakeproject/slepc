
#include "src/eps/epsimpl.h"  /*I "slepceps.h" I*/

EXTERN_C_BEGIN
EXTERN PetscErrorCode EPSCreate_POWER(EPS);
EXTERN PetscErrorCode EPSCreate_SUBSPACE(EPS);
EXTERN PetscErrorCode EPSCreate_ARNOLDI(EPS);
/* EXTERN PetscErrorCode EPSCreate_ARNOLDI2(EPS); */
EXTERN PetscErrorCode EPSCreate_LANCZOS(EPS);
#if defined(SLEPC_HAVE_ARPACK)
EXTERN PetscErrorCode EPSCreate_ARPACK(EPS);
#endif
EXTERN PetscErrorCode EPSCreate_LAPACK(EPS);
#if defined(SLEPC_HAVE_BLZPACK) && !defined(PETSC_USE_COMPLEX)
EXTERN PetscErrorCode EPSCreate_BLZPACK(EPS);
#endif
#if defined(SLEPC_HAVE_PLANSO) && !defined(PETSC_USE_COMPLEX)
EXTERN PetscErrorCode EPSCreate_PLANSO(EPS);
#endif
#if defined(SLEPC_HAVE_TRLAN) && !defined(PETSC_USE_COMPLEX)
EXTERN PetscErrorCode EPSCreate_TRLAN(EPS);
#endif
#if defined(PETSC_HAVE_HYPRE) && !defined(PETSC_USE_COMPLEX)
EXTERN PetscErrorCode EPSCreate_LOBPCG(EPS);
#endif
EXTERN_C_END
  
/*
    This is used by EPSSetType() to make sure that at least one 
    EPSRegisterAll() is called. In general, if there is more than one
    DLL, then EPSRegisterAll() may be called several times.
*/

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
/*  ierr = EPSRegisterDynamic(EPSARNOLDI2, path,"EPSCreate_ARNOLDI2", 
		  EPSCreate_ARNOLDI2);CHKERRQ(ierr); */
  ierr = EPSRegisterDynamic(EPSLANCZOS, path,"EPSCreate_LANCZOS", 
		  EPSCreate_LANCZOS);CHKERRQ(ierr);
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
#if defined(SLEPC_HAVE_PLANSO) && !defined(PETSC_USE_COMPLEX)
  ierr = EPSRegisterDynamic(EPSPLANSO, path,"EPSCreate_PLANSO", 
		  EPSCreate_PLANSO);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_TRLAN) && !defined(PETSC_USE_COMPLEX)
  ierr = EPSRegisterDynamic(EPSTRLAN, path,"EPSCreate_TRLAN", 
		  EPSCreate_TRLAN);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HYPRE) && !defined(PETSC_USE_COMPLEX)
  ierr = EPSRegisterDynamic(EPSLOBPCG, path,"EPSCreate_LOBPCG", 
		  EPSCreate_LOBPCG);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
