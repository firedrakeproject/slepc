/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This provides a simple shell interface for programmers to create
   their own spectral transformations without writing much interface code
*/

#include <slepc/private/stimpl.h>        /*I "slepcst.h" I*/

typedef struct {
  void           *ctx;                       /* user provided context */
  PetscErrorCode (*apply)(ST,Vec,Vec);
  PetscErrorCode (*applytrans)(ST,Vec,Vec);
  PetscErrorCode (*backtransform)(ST,PetscInt n,PetscScalar*,PetscScalar*);
} ST_SHELL;

/*@C
   STShellGetContext - Returns the user-provided context associated with a shell ST

   Not Collective

   Input Parameter:
.  st - spectral transformation context

   Output Parameter:
.  ctx - the user provided context

   Level: advanced

   Notes:
   This routine is intended for use within various shell routines

.seealso: STShellSetContext()
@*/
PetscErrorCode STShellGetContext(ST st,void *ctx)
{
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(ctx,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)st,STSHELL,&flg));
  if (!flg) *(void**)ctx = NULL;
  else      *(void**)ctx = ((ST_SHELL*)(st->data))->ctx;
  PetscFunctionReturn(0);
}

/*@
   STShellSetContext - Sets the context for a shell ST

   Logically Collective on st

   Input Parameters:
+  st - the shell ST
-  ctx - the context

   Level: advanced

   Fortran Notes:
   To use this from Fortran you must write a Fortran interface definition
   for this function that tells Fortran the Fortran derived data type that
   you are passing in as the ctx argument.

.seealso: STShellGetContext()
@*/
PetscErrorCode STShellSetContext(ST st,void *ctx)
{
  ST_SHELL       *shell = (ST_SHELL*)st->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)st,STSHELL,&flg));
  if (flg) shell->ctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode STApply_Shell(ST st,Vec x,Vec y)
{
  ST_SHELL         *shell = (ST_SHELL*)st->data;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  PetscCheck(shell->apply,PetscObjectComm((PetscObject)st),PETSC_ERR_USER,"No apply() routine provided to Shell ST");
  PetscCall(PetscObjectStateGet((PetscObject)y,&instate));
  PetscCallBack("STSHELL user function apply()",(*shell->apply)(st,x,y));
  PetscCall(PetscObjectStateGet((PetscObject)y,&outstate));
  if (instate == outstate) {
    /* user forgot to increase the state of the output vector */
    PetscCall(PetscObjectStateIncrease((PetscObject)y));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STApplyTranspose_Shell(ST st,Vec x,Vec y)
{
  ST_SHELL       *shell = (ST_SHELL*)st->data;
  PetscObjectState instate,outstate;

  PetscFunctionBegin;
  PetscCheck(shell->applytrans,PetscObjectComm((PetscObject)st),PETSC_ERR_USER,"No applytranspose() routine provided to Shell ST");
  PetscCall(PetscObjectStateGet((PetscObject)y,&instate));
  PetscCallBack("STSHELL user function applytrans()",(*shell->applytrans)(st,x,y));
  PetscCall(PetscObjectStateGet((PetscObject)y,&outstate));
  if (instate == outstate) {
    /* user forgot to increase the state of the output vector */
    PetscCall(PetscObjectStateIncrease((PetscObject)y));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STBackTransform_Shell(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  ST_SHELL       *shell = (ST_SHELL*)st->data;

  PetscFunctionBegin;
  if (shell->backtransform) PetscCallBack("STSHELL user function backtransform()",(*shell->backtransform)(st,n,eigr,eigi));
  PetscFunctionReturn(0);
}

/*
   STIsInjective_Shell - Check if the user has provided the backtransform operation.
*/
PetscErrorCode STIsInjective_Shell(ST st,PetscBool* is)
{
  ST_SHELL *shell = (ST_SHELL*)st->data;

  PetscFunctionBegin;
  *is = shell->backtransform? PETSC_TRUE: PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode STDestroy_Shell(ST st)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(st->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)st,"STShellSetApply_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)st,"STShellSetApplyTranspose_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)st,"STShellSetBackTransform_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode STShellSetApply_Shell(ST st,PetscErrorCode (*apply)(ST,Vec,Vec))
{
  ST_SHELL *shell = (ST_SHELL*)st->data;

  PetscFunctionBegin;
  shell->apply = apply;
  PetscFunctionReturn(0);
}

/*@C
   STShellSetApply - Sets routine to use as the application of the
   operator to a vector in the user-defined spectral transformation.

   Logically Collective on st

   Input Parameters:
+  st    - the spectral transformation context
-  apply - the application-provided transformation routine

   Calling sequence of apply:
$  PetscErrorCode apply(ST st,Vec xin,Vec xout)

+  st   - the spectral transformation context
.  xin  - input vector
-  xout - output vector

   Level: advanced

.seealso: STShellSetBackTransform(), STShellSetApplyTranspose()
@*/
PetscErrorCode STShellSetApply(ST st,PetscErrorCode (*apply)(ST,Vec,Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscTryMethod(st,"STShellSetApply_C",(ST,PetscErrorCode (*)(ST,Vec,Vec)),(st,apply));
  PetscFunctionReturn(0);
}

static PetscErrorCode STShellSetApplyTranspose_Shell(ST st,PetscErrorCode (*applytrans)(ST,Vec,Vec))
{
  ST_SHELL *shell = (ST_SHELL*)st->data;

  PetscFunctionBegin;
  shell->applytrans = applytrans;
  PetscFunctionReturn(0);
}

/*@C
   STShellSetApplyTranspose - Sets routine to use as the application of the
   transposed operator to a vector in the user-defined spectral transformation.

   Logically Collective on st

   Input Parameters:
+  st    - the spectral transformation context
-  applytrans - the application-provided transformation routine

   Calling sequence of applytrans:
$  PetscErrorCode applytrans(ST st,Vec xin,Vec xout)

+  st   - the spectral transformation context
.  xin  - input vector
-  xout - output vector

   Level: advanced

.seealso: STShellSetApply(), STShellSetBackTransform()
@*/
PetscErrorCode STShellSetApplyTranspose(ST st,PetscErrorCode (*applytrans)(ST,Vec,Vec))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscTryMethod(st,"STShellSetApplyTranspose_C",(ST,PetscErrorCode (*)(ST,Vec,Vec)),(st,applytrans));
  PetscFunctionReturn(0);
}

static PetscErrorCode STShellSetBackTransform_Shell(ST st,PetscErrorCode (*backtr)(ST,PetscInt,PetscScalar*,PetscScalar*))
{
  ST_SHELL *shell = (ST_SHELL*)st->data;

  PetscFunctionBegin;
  shell->backtransform = backtr;
  PetscFunctionReturn(0);
}

/*@C
   STShellSetBackTransform - Sets the routine to be called after the
   eigensolution process has finished in order to transform back the
   computed eigenvalues.

   Logically Collective on st

   Input Parameters:
+  st     - the spectral transformation context
-  backtr - the application-provided backtransform routine

   Calling sequence of backtr:
$  PetscErrorCode backtr(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)

+  st   - the spectral transformation context
.  n    - number of eigenvalues to be backtransformed
.  eigr - pointer ot the real parts of the eigenvalues to transform back
-  eigi - pointer ot the imaginary parts

   Level: advanced

.seealso: STShellSetApply(), STShellSetApplyTranspose()
@*/
PetscErrorCode STShellSetBackTransform(ST st,PetscErrorCode (*backtr)(ST,PetscInt,PetscScalar*,PetscScalar*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscTryMethod(st,"STShellSetBackTransform_C",(ST,PetscErrorCode (*)(ST,PetscInt,PetscScalar*,PetscScalar*)),(st,backtr));
  PetscFunctionReturn(0);
}

/*MC
   STSHELL - User-defined spectral transformation via callback functions
   for the application of the operator to a vector and (optionally) the
   backtransform operation.

   Level: advanced

   Usage:
$             extern PetscErrorCode (*apply)(void*,Vec,Vec);
$             extern PetscErrorCode (*applytrans)(void*,Vec,Vec);
$             extern PetscErrorCode (*backtr)(void*,PetscScalar*,PetscScalar*);
$
$             STCreate(comm,&st);
$             STSetType(st,STSHELL);
$             STShellSetContext(st,ctx);
$             STShellSetApply(st,apply);
$             STShellSetApplyTranspose(st,applytrans);  (optional)
$             STShellSetBackTransform(st,backtr);       (optional)

M*/

SLEPC_EXTERN PetscErrorCode STCreate_Shell(ST st)
{
  ST_SHELL       *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  st->data = (void*)ctx;

  st->usesksp = PETSC_FALSE;

  st->ops->apply           = STApply_Shell;
  st->ops->applytrans      = STApplyTranspose_Shell;
  st->ops->backtransform   = STBackTransform_Shell;
  st->ops->destroy         = STDestroy_Shell;

  PetscCall(PetscObjectComposeFunction((PetscObject)st,"STShellSetApply_C",STShellSetApply_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)st,"STShellSetApplyTranspose_C",STShellSetApplyTranspose_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)st,"STShellSetBackTransform_C",STShellSetBackTransform_Shell));
  PetscFunctionReturn(0);
}
