
/*
    The ST (spectral transformation) interface routines related to the
    KSP object associated to it.
*/

#include "src/st/stimpl.h"            /*I "slepcst.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "STAssociatedKSPSolve"
/*@C
   STAssociatedKSPSolve - Solve the linear system of equations associated
   to the spectral transformation.

   Collective on ST

   Input Parameters:
.  st - the spectral transformation context
.  b  - right hand side vector

   Output  Parameter:
.  x - computed solution

   Level: developer

.seealso: STGetKSP(), KSPSolve()
@*/
int STAssociatedKSPSolve(ST st,Vec b,Vec x)
{
  int   ierr;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (!st->ksp) { SETERRQ(PETSC_ERR_SUP,"ST has no associated KSP"); }
  ierr = KSPSetRhs(st->ksp,b);CHKERRQ(ierr);
  ierr = KSPSetSolution(st->ksp,x);CHKERRQ(ierr);
  ierr = KSPSolve(st->ksp);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(st->ksp,&reason);CHKERRQ(ierr);
  if (reason<0) { SETERRQ1(0,"Warning: KSP did not converge (%d)",reason); }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetKSP"
/*@
   STSetKSP - Sets the KSP object associated with the spectral 
   transformation.

   Not collective

   Input Parameters:
+  st   - the spectral transformation context
-  ksp  - the linear system context

   Level: advanced

@*/
int STSetKSP(ST st,KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE);
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  PetscCheckSameComm(st,ksp);
  st->ksp = ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetKSP"
/*@
   STGetKSP - Gets the KSP object associated with the spectral
   transformation.

   Not collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  ksp  - the linear system context

   Notes:
   On output, the value of ksp can be PETSC_NULL if the combination of 
   eigenproblem type and selected transformation does not require to 
   solve a linear system of equations.
   
   Level: intermediate

@*/
int STGetKSP(ST st,KSP* ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE);
  if (!st->type_name) { SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call STSetType first"); }
  if (ksp)  *ksp = st->ksp;
  PetscFunctionReturn(0);
}


