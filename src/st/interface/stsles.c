
/*
    The ST (spectral transformation) interface routines related to the
    KSP object associated to it.
*/

#include "src/st/stimpl.h"            /*I "slepcst.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "STAssociatedKSPSolve"
/*
   STAssociatedKSPSolve - Solves the linear system of equations associated
   to the spectral transformation.

   Input Parameters:
.  st - the spectral transformation context
.  b  - right hand side vector

   Output  Parameter:
.  x - computed solution
*/
PetscErrorCode STAssociatedKSPSolve(ST st,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       its;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(b,VEC_COOKIE,2);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  if (!st->ksp) { SETERRQ(PETSC_ERR_SUP,"ST has no associated KSP"); }
  ierr = KSPSolve(st->ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(st->ksp,&reason);CHKERRQ(ierr);
  if (reason<0) { SETERRQ1(0,"Warning: KSP did not converge (%d)",reason); }
  ierr = KSPGetIterationNumber(st->ksp,&its);CHKERRQ(ierr);  
  st->lineariterations += its;
  PetscLogInfo((st,"ST: linear solve iterations=%d\n",its));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STAssociatedKSPSolveTranspose"
/*
   STAssociatedKSPSolveTranspose - Solves the transpose of the linear 
   system of equations associated to the spectral transformation.

   Input Parameters:
.  st - the spectral transformation context
.  b  - right hand side vector

   Output  Parameter:
.  x - computed solution
*/
PetscErrorCode STAssociatedKSPSolveTranspose(ST st,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       its;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(b,VEC_COOKIE,2);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  if (!st->ksp) { SETERRQ(PETSC_ERR_SUP,"ST has no associated KSP"); }
  ierr = KSPSolveTranspose(st->ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(st->ksp,&reason);CHKERRQ(ierr);
  if (reason<0) { SETERRQ1(0,"Warning: KSP did not converge (%d)",reason); }
  ierr = KSPGetIterationNumber(st->ksp,&its);CHKERRQ(ierr);  
  st->lineariterations += its;
  PetscLogInfo((st,"ST: linear solve iterations=%d\n",its));
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
PetscErrorCode STSetKSP(ST st,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,2);
  PetscCheckSameComm(st,1,ksp,2);
  if (st->ksp) {
    ierr = KSPDestroy(st->ksp);CHKERRQ(ierr);
  }
  st->ksp = ksp;
  ierr = PetscObjectReference((PetscObject)st->ksp);CHKERRQ(ierr);
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
PetscErrorCode STGetKSP(ST st,KSP* ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (!st->type_name) { SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call STSetType first"); }
  if (ksp)  *ksp = st->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetNumberLinearIterations"
/*@
   STGetNumberLinearIterations - Gets the total number of linear iterations
   used by the ST object.

   Not Collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  lits - number of linear iterations

   Level: intermediate

.seealso: STResetNumberLinearIterations()
@*/
PetscErrorCode STGetNumberLinearIterations(ST st,int* lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidIntPointer(lits,2);
  *lits = st->lineariterations;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STResetNumberLinearIterations"
/*@
   STResetNumberLinearIterations - Resets the counter for total number of 
   linear iterations used by the ST object.

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Level: intermediate

.seealso: STGetNumberLinearIterations()
@*/
PetscErrorCode STResetNumberLinearIterations(ST st)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  st->lineariterations = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCheckNullSpace_Default"
PetscErrorCode STCheckNullSpace_Default(ST st,int n,Vec* V)
{
  PetscErrorCode ierr;
  int            i,c;
  PetscReal      norm;
  Vec            *T,w;
  Mat            A;
  PC             pc;
  MatNullSpace   nullsp;
  
  PetscFunctionBegin;
  ierr = PetscMalloc(n*sizeof(Vec),&T);CHKERRQ(ierr);
  ierr = KSPGetPC(st->ksp,&pc);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&A,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetVecs(A,PETSC_NULL,&w);CHKERRQ(ierr);
  c = 0;
  for (i=0;i<n;i++) {
    ierr = MatMult(A,V[i],w);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
    if (norm < 1e-8) {
      PetscLogInfo((st,"STCheckNullSpace: vector %i norm=%g\n",i,norm));
      T[c] = V[i];
      c++;
    }
  }
  ierr = VecDestroy(w);CHKERRQ(ierr);
  if (c>0) {
    ierr = MatNullSpaceCreate(st->comm,PETSC_FALSE,c,T,&nullsp);CHKERRQ(ierr);
    ierr = KSPSetNullSpace(st->ksp,nullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(nullsp);CHKERRQ(ierr);
  }
  ierr = PetscFree(T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCheckNullSpace"
/*@C
   STCheckNullSpace - Given a set of vectors, this function test each of
   them to be a nullspace vector of the coefficient matrix of the associated
   KSP object. All these nullspace vectors are passed to the KSP object.

   Collective on ST

   Input Parameters:
+  st - the spectral transformation context
.  n  - number of vectors
-  V  - vectors to be checked

   Note:
   This function allows to handle singular pencils and to solve some problems
   in which the nullspace is important (see the users guide for details).
   
   Level: developer

.seealso: EPSAttachDeflationSpace()
@*/
PetscErrorCode STCheckNullSpace(ST st,int n,Vec* V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n>0 && st->checknullspace) {
    ierr = (*st->checknullspace)(st,n,V);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


