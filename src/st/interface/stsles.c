
/*
    The ST (spectral transformation) interface routines related to the
    SLES object associated to it.
*/

#include "src/st/stimpl.h"            /*I "slepcst.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "STAssociatedSLESSolve"
/*@C
   STAssociatedSLESSolve - Solve the linear system of equations associated
   to the spectral transformation.

   Collective on ST

   Input Parameters:
.  st - the spectral transformation context
.  b  - right hand side vector

   Output  Parameter:
.  x - computed solution

   Level: developer

.seealso: STGetSLES(), SLESSolve()
@*/
int STAssociatedSLESSolve(ST st,Vec b,Vec x)
{
  int   its,ierr;
  KSP   ksp;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (!st->sles) { SETERRQ(PETSC_ERR_SUP,"ST has no associated SLES"); }
  ierr = SLESSolve(st->sles,b,x,&its);CHKERRQ(ierr);
  ierr = SLESGetKSP(st->sles,&ksp);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
  if (reason<0) { SETERRQ1(0,"Warning: SLES did not converge (%d)",reason); }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetSLES"
/*@
   STSetSLES - Sets the SLES object associated with the spectral 
   transformation.

   Not collective

   Input Parameters:
+  st   - the spectral transformation context
-  sles - the linear system context

   Level: advanced

@*/
int STSetSLES(ST st,SLES sles)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE);
  PetscValidHeaderSpecific(sles,SLES_COOKIE);
  PetscCheckSameComm(st,sles);
  st->sles = sles;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetSLES"
/*@
   STGetSLES - Gets the SLES object associated with the spectral
   transformation.

   Not collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  sles - the linear system context

   Notes:
   On output, the value of sles can be PETSC_NULL if the combination of 
   eigenproblem type and selected transformation does not require to 
   solve a linear system of equations.
   
   Level: intermediate

@*/
int STGetSLES(ST st,SLES* sles)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE);
  if (!st->type_name) { SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call STSetType first"); }
  if (sles)  *sles = st->sles;
  PetscFunctionReturn(0);
}


