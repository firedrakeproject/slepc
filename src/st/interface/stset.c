
/*
    Routines to set ST methods and options.
*/

#include "src/st/stimpl.h"      /*I "slepcst.h" I*/
#include "petscsys.h"

PetscTruth STRegisterAllCalled = PETSC_FALSE;
/*
   Contains the list of registered EPS routines
*/
PetscFList STList = 0;

#undef __FUNCT__  
#define __FUNCT__ "STSetType"
/*@C
   STSetType - Builds ST for a particular spectral transformation.

   Collective on ST

   Input Parameter:
+  st   - the spectral transformation context.
-  type - a known type

   Options Database Key:
.  -st_type <type> - Sets ST type

   Use -help for a list of available transformations

   Notes:
   See "slepc/include/slepcst.h" for available transformations 

   Normally, it is best to use the EPSSetFromOptions() command and
   then set the ST type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the many different transformations. 

   Level: intermediate

.seealso: EPSSetType()

@*/
int STSetType(ST st,STType type)
{
  int ierr,(*r)(ST);
  PetscTruth match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)st,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (st->ops->destroy) {ierr =  (*st->ops->destroy)(st);CHKERRQ(ierr);}
  ierr = PetscFListDestroy(&st->qlist);CHKERRQ(ierr);
  st->data        = 0;
  st->setupcalled = 0;

  /* Get the function pointers for the method requested */
  if (!STRegisterAllCalled) {ierr = STRegisterAll(0); CHKERRQ(ierr);}

  /* Determine the STCreateXXX routine for a particular type */
  ierr =  PetscFListFind(st->comm, STList, type,(void (**)(void)) &r );CHKERRQ(ierr);
  if (!r) SETERRQ1(1,"Unable to find requested ST type %s",type);
  if (st->data) {ierr = PetscFree(st->data);CHKERRQ(ierr);}

  st->ops->destroy         = (int (*)(ST )) 0;
  st->ops->view            = (int (*)(ST,PetscViewer) ) 0;
  st->ops->apply           = (int (*)(ST,Vec,Vec) ) 0;
  st->ops->applyB          = STDefaultApplyB;
  st->ops->applynoB        = (int (*)(ST,Vec,Vec) ) 0;
  st->ops->setup           = (int (*)(ST) ) 0;
  st->ops->setfromoptions  = (int (*)(ST) ) 0;
  st->ops->presolve        = (int (*)(ST) ) 0;
  st->ops->postsolve       = (int (*)(ST) ) 0;
  st->ops->backtr          = (int (*)(ST,PetscScalar*,PetscScalar*) ) 0;

  /* Call the STCreateXXX routine for this particular type */
  ierr = (*r)(st);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)st,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STRegisterDestroy"
/*@C
   STRegisterDestroy - Frees the list of spectral transformations that were
   registered by STRegisterDynamic().

   Not Collective

   Level: advanced

.seealso: STRegisterAll(), STRegisterAll()

@*/
int STRegisterDestroy(void)
{
  int ierr;

  PetscFunctionBegin;
  if (STList) {
    ierr = PetscFListDestroy(&STList);CHKERRQ(ierr);
    STList = 0;
  }
  STRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetType"
/*@C
   STGetType - Gets the ST type name (as a string) from the ST context.

   Not Collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  name - name of the spectral transformation 

   Level: intermediate

.seealso: STSetType()

@*/
int STGetType(ST st,STType *meth)
{
  PetscFunctionBegin;
  *meth = (STType) st->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetFromOptions"
/*@
   STSetFromOptions - Sets ST options from the options database.
   This routine must be called before STSetUp() if the user is to be
   allowed to set the type of transformation.

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Level: beginner

.seealso: 

@*/
int STSetFromOptions(ST st)
{
  int        ierr;
  char       type[256];
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);

  if (!STRegisterAllCalled) {ierr = STRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsBegin(st->comm,st->prefix,"Spectral Transformation (ST) Options","ST");CHKERRQ(ierr);
    ierr = PetscOptionsList("-st_type","Spectral Transformation type","STSetType",STList,(char*)(st->type_name?st->type_name:STSHIFT),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = STSetType(st,type);CHKERRQ(ierr);
    }
    /*
      Set the type if it was never set.
    */
    if (!st->type_name) {
      ierr = STSetType(st,STSHIFT);CHKERRQ(ierr);
    }

    if (st->numberofshifts>0) {
      ierr = PetscOptionsScalar("-st_shift","Value of the shift","STSetShift",st->sigma,&st->sigma,PETSC_NULL); CHKERRQ(ierr);
    }

    if (st->ops->setfromoptions) {
      ierr = (*st->ops->setfromoptions)(st);CHKERRQ(ierr);
    }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (st->ksp) { ierr = KSPSetFromOptions(st->ksp);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

