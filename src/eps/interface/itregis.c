
#include "src/eps/epsimpl.h"  /*I "slepceps.h" I*/

EXTERN_C_BEGIN
extern int EPSCreate_PREONLY(EPS);
extern int EPSCreate_POWER(EPS);
extern int EPSCreate_SUBSPACE(EPS);
extern int EPSCreate_ARNOLDI(EPS);
extern int EPSCreate_SRRIT(EPS);
#if defined(SLEPC_HAVE_ARPACK)
extern int EPSCreate_ARPACK(EPS);
#endif
extern int EPSCreate_LAPACK(EPS);
#if defined(SLEPC_HAVE_BLZPACK) && !defined(PETSC_USE_COMPLEX)
extern int EPSCreate_BLZPACK(EPS);
#endif
#if defined(SLEPC_HAVE_PLANSO) && !defined(PETSC_USE_COMPLEX)
extern int EPSCreate_PLANSO(EPS);
#endif
#if defined(SLEPC_HAVE_TRLAN) && !defined(PETSC_USE_COMPLEX)
extern int EPSCreate_TRLAN(EPS);
#endif
EXTERN_C_END
  
/*
    This is used by EPSSetType() to make sure that at least one 
    EPSRegisterAll() is called. In general, if there is more than one
    DLL, then EPSRegisterAll() may be called several times.
*/
extern PetscTruth EPSRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "EPSRegisterAll"
/*@C
  EPSRegisterAll - Registers all the eigenvalue solvers in the EPS package.

  Not Collective

  Level: advanced

.seealso:  EPSRegisterDestroy()
@*/
int EPSRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  EPSRegisterAllCalled = PETSC_TRUE;

  ierr = EPSRegisterDynamic(EPSPOWER, path,"EPSCreate_POWER", 
		  EPSCreate_POWER);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSSUBSPACE, path,"EPSCreate_SUBSPACE", 
		  EPSCreate_SUBSPACE);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSARNOLDI, path,"EPSCreate_ARNOLDI", 
		  EPSCreate_ARNOLDI);CHKERRQ(ierr);
  ierr = EPSRegisterDynamic(EPSSRRIT, path,"EPSCreate_SRRIT", 
		  EPSCreate_SRRIT);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

