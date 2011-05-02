/*
    The ST (spectral transformation) interface routines related to the
    KSP object associated to it.

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

#include <private/stimpl.h>            /*I "slepcst.h" I*/

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
  PetscErrorCode     ierr;
  PetscInt           its;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (!st->ksp) { SETERRQ(((PetscObject)st)->comm,PETSC_ERR_SUP,"ST has no associated KSP"); }
  ierr = KSPSolve(st->ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(st->ksp,&reason);CHKERRQ(ierr);
  if (reason<0) { SETERRQ1(((PetscObject)st)->comm,0,"Warning: KSP did not converge (%d)",reason); }
  ierr = KSPGetIterationNumber(st->ksp,&its);CHKERRQ(ierr);  
  st->lineariterations += its;
  PetscInfo1(st,"Linear solve iterations=%d\n",its);
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
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (!st->ksp) { SETERRQ(((PetscObject)st)->comm,PETSC_ERR_SUP,"ST has no associated KSP"); }
  ierr = KSPSolveTranspose(st->ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(st->ksp,&reason);CHKERRQ(ierr);
  if (reason<0) { SETERRQ1(((PetscObject)st)->comm,0,"Warning: KSP did not converge (%d)",reason); }
  ierr = KSPGetIterationNumber(st->ksp,&its);CHKERRQ(ierr);  
  st->lineariterations += its;
  PetscInfo1(st,"Linear solve iterations=%d\n",its);
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
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(st,1,ksp,2);
  ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPDestroy(&st->ksp);CHKERRQ(ierr);
  st->ksp = ksp;
  ierr = PetscLogObjectParent(st,st->ksp);CHKERRQ(ierr);
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
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  if (!((PetscObject)st)->type_name) { SETERRQ(((PetscObject)st)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call STSetType first"); }
  PetscValidPointer(ksp,2);
  *ksp = st->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetOperationCounters"
/*@
   STGetOperationCounters - Gets the total number of operator applications
   and linear solver iterations used by the ST object.

   Not Collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
+  ops  - number of operator applications
-  lits - number of linear solver iterations

   Notes:
   Any output parameter may be PETSC_NULL on input if not needed. 
   
   Level: intermediate

.seealso: STResetOperationCounters()
@*/
PetscErrorCode STGetOperationCounters(ST st,PetscInt* ops,PetscInt* lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  if (ops) *ops = st->applys;
  if (lits) *lits = st->lineariterations;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STResetOperationCounters"
/*@
   STResetOperationCounters - Resets the counters for operator applications,
   inner product operations and total number of linear iterations used by 
   the ST object.

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Level: intermediate

.seealso: STGetOperationCounters()
@*/
PetscErrorCode STResetOperationCounters(ST st)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  st->lineariterations = 0;
  st->applys = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCheckNullSpace_Default"
PetscErrorCode STCheckNullSpace_Default(ST st,PetscInt n,const Vec V[])
{
  PetscErrorCode ierr;
  PetscInt       i,c;
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
      PetscInfo2(st,"Vector %i norm=%g\n",i,norm);
      T[c] = V[i];
      c++;
    }
  }
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  if (c>0) {
    ierr = MatNullSpaceCreate(((PetscObject)st)->comm,PETSC_FALSE,c,T,&nullsp);CHKERRQ(ierr);
    ierr = KSPSetNullSpace(st->ksp,nullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
  }
  ierr = PetscFree(T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCheckNullSpace"
/*@
   STCheckNullSpace - Given a set of vectors, this function tests each of
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

.seealso: EPSSetDeflationSpace()
@*/
PetscErrorCode STCheckNullSpace(ST st,PetscInt n,const Vec V[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n>0 && st->checknullspace) {
    ierr = (*st->checknullspace)(st,n,V);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


