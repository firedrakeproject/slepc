/*
    Routines to set ST methods and options.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/stimpl.h>      /*I "slepcst.h" I*/

PetscBool         STRegisterAllCalled = PETSC_FALSE;
PetscFunctionList STList = 0;

#undef __FUNCT__
#define __FUNCT__ "STSetType"
/*@C
   STSetType - Builds ST for a particular spectral transformation.

   Logically Collective on ST

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

   Level: beginner

.seealso: EPSSetType()

@*/
PetscErrorCode STSetType(ST st,STType type)
{
  PetscErrorCode ierr,(*r)(ST);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)st,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(STList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested ST type %s",type);

  if (st->ops->destroy) { ierr = (*st->ops->destroy)(st);CHKERRQ(ierr); }
  ierr = PetscMemzero(st->ops,sizeof(struct _STOps));CHKERRQ(ierr);

  st->state = ST_STATE_INITIAL;
  ierr = PetscObjectChangeTypeName((PetscObject)st,type);CHKERRQ(ierr);
  ierr = (*r)(st);CHKERRQ(ierr);
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
PetscErrorCode STGetType(ST st,STType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)st)->type_name;
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
@*/
PetscErrorCode STSetFromOptions(ST st)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    s;
  char           type[256];
  PetscBool      flg;
  const char     *mode_list[3] = {"copy","inplace","shell"};
  const char     *structure_list[3] = {"same","different","subset"};

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  ierr = STRegisterAll();CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)st);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-st_type","Spectral Transformation type","STSetType",STList,(char*)(((PetscObject)st)->type_name?((PetscObject)st)->type_name:STSHIFT),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = STSetType(st,type);CHKERRQ(ierr);
    }
    /*
      Set the type if it was never set.
    */
    if (!((PetscObject)st)->type_name) {
      ierr = STSetType(st,STSHIFT);CHKERRQ(ierr);
    }

    ierr = PetscOptionsScalar("-st_shift","Value of the shift","STSetShift",st->sigma,&s,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = STSetShift(st,s);CHKERRQ(ierr);
    }

    ierr = PetscOptionsEList("-st_matmode","Matrix mode for transformed matrices","STSetMatMode",mode_list,3,mode_list[st->shift_matrix],&i,&flg);CHKERRQ(ierr);
    if (flg) st->shift_matrix = (STMatMode)i;

    ierr = PetscOptionsEList("-st_matstructure","Shift nonzero pattern","STSetMatStructure",structure_list,3,structure_list[st->str],&i,&flg);CHKERRQ(ierr);
    if (flg) {
      switch (i) {
        case 0: ierr = STSetMatStructure(st,SAME_NONZERO_PATTERN);CHKERRQ(ierr); break;
        case 1: ierr = STSetMatStructure(st,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); break;
        case 2: ierr = STSetMatStructure(st,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr); break;
      }
    }

    ierr = PetscOptionsBool("-st_transform","Whether transformed matrices are computed or not","STSetTransform",st->transform,&st->transform,&flg);CHKERRQ(ierr);

    if (st->ops->setfromoptions) {
      ierr = (*st->ops->setfromoptions)(PetscOptionsObject,st);CHKERRQ(ierr);
    }
    ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)st);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
  ierr = KSPSetFromOptions(st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetMatStructure"
/*@
   STSetMatStructure - Sets an internal MatStructure attribute to
   indicate which is the relation of the sparsity pattern of all ST matrices.

   Logically Collective on ST

   Input Parameters:
+  st  - the spectral transformation context
-  str - either SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN or
         SUBSET_NONZERO_PATTERN

   Options Database Key:
.  -st_matstructure <str> - Indicates the structure flag, where <str> is one
         of 'same' (matrices have the same nonzero pattern), 'different'
         (different nonzero pattern) or 'subset' (pattern is a subset of the
         first one).

   Notes:
   By default, the sparsity patterns are assumed to be different. If the
   patterns are equal or a subset then it is recommended to set this attribute
   for efficiency reasons (in particular, for internal MatAXPY() operations).

   This function has no effect in the case of standard eigenproblems.

   Level: advanced

.seealso: STSetOperators(), MatAXPY()
@*/
PetscErrorCode STSetMatStructure(ST st,MatStructure str)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveEnum(st,str,2);
  switch (str) {
    case SAME_NONZERO_PATTERN:
    case DIFFERENT_NONZERO_PATTERN:
    case SUBSET_NONZERO_PATTERN:
      st->str = str;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_OUTOFRANGE,"Invalid matrix structure flag");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STGetMatStructure"
/*@
   STGetMatStructure - Gets the internal MatStructure attribute to
   indicate which is the relation of the sparsity pattern of the matrices.

   Not Collective

   Input Parameters:
.  st  - the spectral transformation context

   Output Parameters:
.  str - either SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN or
         SUBSET_NONZERO_PATTERN

   Level: advanced

.seealso: STSetMatStructure(), STSetOperators(), MatAXPY()
@*/
PetscErrorCode STGetMatStructure(ST st,MatStructure *str)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(str,2);
  *str = st->str;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetMatMode"
/*@
   STSetMatMode - Sets a flag to indicate how the transformed matrices are
   being stored in the spectral transformations.

   Logically Collective on ST

   Input Parameters:
+  st - the spectral transformation context
-  mode - the mode flag, one of ST_MATMODE_COPY,
          ST_MATMODE_INPLACE, or ST_MATMODE_SHELL

   Options Database Key:
.  -st_matmode <mode> - Indicates the mode flag, where <mode> is one of
          'copy', 'inplace', 'shell' (see explanation below).

   Notes:
   By default (ST_MATMODE_COPY), a copy of matrix A is made and then
   this copy is modified explicitly, e.g. A <- (A - s B).

   With ST_MATMODE_INPLACE, the original matrix A is modified at STSetUp()
   and changes are reverted at the end of the computations. With respect to
   the previous one, this mode avoids a copy of matrix A. However, a
   drawback is that the recovered matrix might be slightly different
   from the original one (due to roundoff).

   With ST_MATMODE_SHELL, the solver works with an implicit shell
   matrix that represents the shifted matrix. This mode is the most efficient
   in creating the shifted matrix but it places serious limitations to the
   linear solves performed in each iteration of the eigensolver (typically,
   only interative solvers with Jacobi preconditioning can be used).

   In the two first modes the efficiency of the computation 
   can be controlled with STSetMatStructure().

   Level: intermediate

.seealso: STSetOperators(), STSetMatStructure(), STGetMatMode(), STMatMode
@*/
PetscErrorCode STSetMatMode(ST st,STMatMode mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveEnum(st,mode,2);
  st->shift_matrix = mode;
  st->state = ST_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STGetMatMode"
/*@
   STGetMatMode - Gets a flag that indicates how the transformed matrices
   are stored in spectral transformations.

   Not Collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  mode - the mode flag

   Level: intermediate

.seealso: STSetMatMode(), STMatMode
@*/
PetscErrorCode STGetMatMode(ST st,STMatMode *mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(mode,2);
  *mode = st->shift_matrix;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetTransform"
/*@
   STSetTransform - Sets a flag to indicate whether the transformed matrices are
   computed or not.

   Logically Collective on ST

   Input Parameters:
+  st  - the spectral transformation context
-  flg - the boolean flag

   Options Database Key:
.  -st_transform <bool> - Activate/deactivate the computation of matrices.

   Notes:
   This flag is intended for the case of polynomial eigenproblems solved
   via linearization. If this flag is off (default) the spectral transformation
   is applied to the linearization (handled by the eigensolver), otherwise
   it is applied to the original problem.

   Level: developer

.seealso: STMatSolve(), STMatMult(), STSetMatStructure(), STGetTransform()
@*/
PetscErrorCode STSetTransform(ST st,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveBool(st,flg,2);
  if (st->transform != flg) {
    st->transform = flg;
    st->state = ST_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STGetTransform"
/*@
   STGetTransform - Gets a flag that that indicates whether the transformed
   matrices are computed or not.

   Not Collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  flg - the flag

   Level: developer

.seealso: STSetTransform()
@*/
PetscErrorCode STGetTransform(ST st,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(flg,2);
  *flg = st->transform;
  PetscFunctionReturn(0);
}

