
#include "src/st/stimpl.h"          /*I   "slepcst.h"   I*/

EXTERN_C_BEGIN
extern int STCreate_None(ST);
extern int STCreate_Shell(ST);
extern int STCreate_Shift(ST);
extern int STCreate_Sinvert(ST);
EXTERN_C_END

extern PetscTruth STRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "STRegisterAll"
/*@C
   STRegisterAll - Registers all of the spectral transformations in the ST package.

   Not Collective

   Input Parameter:
.  path - the library where the routines are to be found (optional)

   Level: advanced

.seealso: STRegisterDynamic(), STRegisterDestroy()
@*/
int STRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  STRegisterAllCalled = PETSC_TRUE;

  ierr = STRegisterDynamic(STNONE  ,path,"STCreate_None",STCreate_None);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STSHELL ,path,"STCreate_Shell",STCreate_Shell);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STSHIFT ,path,"STCreate_Shift",STCreate_Shift);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STSINV  ,path,"STCreate_Sinvert",STCreate_Sinvert);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


