
#include "src/svd/svdimpl.h"  /*I "slepcsvd.h" I*/

EXTERN_C_BEGIN
EXTERN PetscErrorCode SVDCreate_EIGENSOLVER(SVD);
EXTERN PetscErrorCode SVDCreate_LAPACK(SVD);
EXTERN PetscErrorCode SVDCreate_LANCZOS(SVD);
EXTERN_C_END
  
/*
    This is used by SVDSetType() to make sure that at least one 
    SVDRegisterAll() is called. In general, if there is more than one
    DLL, then SVDRegisterAll() may be called several times.
*/

#undef __FUNCT__  
#define __FUNCT__ "SVDRegisterAll"
/*@C
  SVDRegisterAll - Registers all the singular value solvers in the SVD package.

  Not Collective

  Level: advanced

.seealso:  SVDRegisterDynamic()
@*/
PetscErrorCode SVDRegisterAll(char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  ierr = SVDRegisterDynamic(SVDEIGENSOLVER, path,"SVDCreate_EIGENSOLVER", 
		  SVDCreate_EIGENSOLVER);CHKERRQ(ierr);
  ierr = SVDRegisterDynamic(SVDLAPACK, path,"SVDCreate_LAPACK", 
		  SVDCreate_LAPACK);CHKERRQ(ierr);
  ierr = SVDRegisterDynamic(SVDLANCZOS, path,"SVDCreate_LANCZOS", 
		  SVDCreate_LANCZOS);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
