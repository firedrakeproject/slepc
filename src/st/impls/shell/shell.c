/*
   This provides a simple shell interface for programmers to 
   create their own spectral transformations without writing much 
   interface code.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "private/stimpl.h"        /*I "slepcst.h" I*/
#include "slepceps.h"

EXTERN_C_BEGIN 
typedef struct {
  void           *ctx;                       /* user provided context */
  PetscErrorCode (*apply)(void *,Vec,Vec);
  PetscErrorCode (*applytrans)(void *,Vec,Vec);
  PetscErrorCode (*backtr)(void *,PetscScalar*,PetscScalar*);
  char           *name;
} ST_Shell;
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "STShellGetContext"
/*@C
    STShellGetContext - Returns the user-provided context associated with a shell ST

    Not Collective

    Input Parameter:
.   st - spectral transformation context

    Output Parameter:
.   ctx - the user provided context

    Level: advanced

    Notes:
    This routine is intended for use within various shell routines
    
.seealso: STShellSetContext()
@*/
PetscErrorCode STShellGetContext(ST st,void **ctx)
{
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidPointer(ctx,2); 
  ierr = PetscTypeCompare((PetscObject)st,STSHELL,&flg);CHKERRQ(ierr);
  if (!flg) *ctx = 0; 
  else      *ctx = ((ST_Shell*)(st->data))->ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STShellSetContext"
/*@C
    STShellSetContext - sets the context for a shell ST

   Collective on ST

    Input Parameters:
+   st - the shell ST
-   ctx - the context

   Level: advanced

   Fortran Notes: The context can only be an integer or a PetscObject;
      unfortunately it cannot be a Fortran array or derived type.

.seealso: STShellGetContext()
@*/
PetscErrorCode STShellSetContext(ST st,void *ctx)
{
  ST_Shell      *shell = (ST_Shell*)st->data;
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscTypeCompare((PetscObject)st,STSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    shell->ctx = ctx;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApply_Shell"
PetscErrorCode STApply_Shell(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;
  ST_Shell       *shell = (ST_Shell*)st->data;

  PetscFunctionBegin;
  if (!shell->apply) SETERRQ(PETSC_ERR_USER,"No apply() routine provided to Shell ST");
  PetscStackPush("PCSHELL user function");
  CHKMEMQ;
  ierr  = (*shell->apply)(shell->ctx,x,y);CHKERRQ(ierr);
  CHKMEMQ;
  PetscStackPop;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyTranspose_Shell"
PetscErrorCode STApplyTranspose_Shell(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;
  ST_Shell       *shell = (ST_Shell*)st->data;

  PetscFunctionBegin;
  if (!shell->applytrans) SETERRQ(PETSC_ERR_USER,"No applytranspose() routine provided to Shell ST");
  ierr  = (*shell->applytrans)(shell->ctx,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBackTransform_Shell"
PetscErrorCode STBackTransform_Shell(ST st,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscErrorCode ierr;
  ST_Shell       *shell = (ST_Shell*)st->data;

  PetscFunctionBegin;
  if (shell->backtr) {
    ierr  = (*shell->backtr)(shell->ctx,eigr,eigi);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STDestroy_Shell"
PetscErrorCode STDestroy_Shell(ST st)
{
  PetscErrorCode ierr;
  ST_Shell       *shell = (ST_Shell*)st->data;

  PetscFunctionBegin;
  ierr = PetscFree(shell->name);CHKERRQ(ierr);
  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STView_Shell"
PetscErrorCode STView_Shell(ST st,PetscViewer viewer)
{
  PetscErrorCode ierr;
  ST_Shell       *ctx = (ST_Shell*)st->data;
  PetscTruth     isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (ctx->name) {ierr = PetscViewerASCIIPrintf(viewer,"  ST Shell: %s\n",ctx->name);CHKERRQ(ierr);}
    else           {ierr = PetscViewerASCIIPrintf(viewer,"  ST Shell: no name\n");CHKERRQ(ierr);}
  } else {
    SETERRQ1(1,"Viewer type %s not supported for STShell",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STShellSetApply_Shell"
PetscErrorCode STShellSetApply_Shell(ST st,PetscErrorCode (*apply)(void*,Vec,Vec))
{
  ST_Shell *shell = (ST_Shell*)st->data;

  PetscFunctionBegin;
  shell->apply = apply;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STShellSetApplyTranspose_Shell"
PetscErrorCode STShellSetApplyTranspose_Shell(ST st,PetscErrorCode (*applytrans)(void*,Vec,Vec))
{
  ST_Shell *shell = (ST_Shell*)st->data;

  PetscFunctionBegin;
  shell->applytrans = applytrans;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STShellSetBackTransform_Shell"
PetscErrorCode STShellSetBackTransform_Shell(ST st,PetscErrorCode (*backtr)(void*,PetscScalar*,PetscScalar*))
{
  ST_Shell *shell = (ST_Shell *) st->data;

  PetscFunctionBegin;
  shell->backtr = backtr;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STShellSetName_Shell"
PetscErrorCode STShellSetName_Shell(ST st,const char name[])
{
  ST_Shell *shell = (ST_Shell*)st->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrfree(shell->name);CHKERRQ(ierr);    
  ierr = PetscStrallocpy(name,&shell->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STShellGetName_Shell"
PetscErrorCode STShellGetName_Shell(ST st,char *name[])
{
  ST_Shell *shell = (ST_Shell*)st->data;

  PetscFunctionBegin;
  *name  = shell->name;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "STShellSetApply"
/*@C
   STShellSetApply - Sets routine to use as the application of the 
   operator to a vector in the user-defined spectral transformation.

   Collective on ST

   Input Parameters:
+  st    - the spectral transformation context
-  apply - the application-provided transformation routine

   Calling sequence of apply:
.vb
   PetscErrorCode apply (void *ptr,Vec xin,Vec xout)
.ve

+  ptr  - the application context
.  xin  - input vector
-  xout - output vector

   Level: developer

.seealso: STShellSetBackTransform(), STShellSetApplyTranspose()
@*/
PetscErrorCode STShellSetApply(ST st,PetscErrorCode (*apply)(void*,Vec,Vec))
{
  PetscErrorCode ierr, (*f)(ST,PetscErrorCode (*)(void*,Vec,Vec));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STShellSetApply_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,apply);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STShellSetApplyTranspose"
/*@C
   STShellSetApplyTranspose - Sets routine to use as the application of the 
   transposed operator to a vector in the user-defined spectral transformation.

   Collective on ST

   Input Parameters:
+  st    - the spectral transformation context
-  applytrans - the application-provided transformation routine

   Calling sequence of apply:
.vb
   PetscErrorCode applytrans (void *ptr,Vec xin,Vec xout)
.ve

+  ptr  - the application context
.  xin  - input vector
-  xout - output vector

   Level: developer

.seealso: STShellSetApply(), STShellSetBackTransform()
@*/
PetscErrorCode STShellSetApplyTranspose(ST st,PetscErrorCode (*applytrans)(void*,Vec,Vec))
{
  PetscErrorCode ierr, (*f)(ST,PetscErrorCode (*)(void*,Vec,Vec));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STShellSetApplyTranspose_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,applytrans);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STShellSetBackTransform"
/*@C
   STShellSetBackTransform - Sets the routine to be called after the 
   eigensolution process has finished in order to transform back the
   computed eigenvalues.

   Collective on ST

   Input Parameters:
+  st     - the spectral transformation context
-  backtr - the application-provided backtransform routine

   Calling sequence of backtr:
.vb
   PetscErrorCode backtr (void *ptr,PetscScalar *eigr,PetscScalar *eigi)
.ve

+  ptr  - the application context
.  eigr - pointer ot the real part of the eigenvalue to transform back
-  eigi - pointer ot the imaginary part 

   Level: developer

.seealso: STShellSetApply(), STShellSetApplyTranspose()
@*/
PetscErrorCode STShellSetBackTransform(ST st,PetscErrorCode (*backtr)(void*,PetscScalar*,PetscScalar*))
{
  PetscErrorCode ierr, (*f)(ST,PetscErrorCode (*)(void*,PetscScalar*,PetscScalar*));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STShellSetBackTransform_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,(PetscErrorCode (*)(void*,PetscScalar*,PetscScalar*))backtr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STShellSetName"
/*@C
   STShellSetName - Sets an optional name to associate with a shell
   spectral transformation.

   Not Collective

   Input Parameters:
+  st   - the spectral transformation context
-  name - character string describing the shell spectral transformation

   Level: developer

.seealso: STShellGetName()
@*/
PetscErrorCode STShellSetName(ST st,const char name[])
{
  PetscErrorCode ierr, (*f)(ST,const char []);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STShellSetName_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STShellGetName"
/*@C
   STShellGetName - Gets an optional name that the user has set for a shell
   spectral transformation.

   Not Collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  name - character string describing the shell spectral transformation 
          (you should not free this)

   Level: developer

.seealso: STShellSetName()
@*/
PetscErrorCode STShellGetName(ST st,char *name[])
{
  PetscErrorCode ierr, (*f)(ST,char *[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STShellGetName_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,name);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Not shell spectral transformation, cannot get name");
  }
  PetscFunctionReturn(0);
}

/*MC
   STSHELL - Creates a new spectral transformation class.
          This is intended to provide a simple class to use with EPS.
	  You should not use this if you plan to make a complete class.

  Level: advanced

  Usage:
$             PetscErrorCode (*apply)(void*,Vec,Vec);
$             PetscErrorCode (*applytrans)(void*,Vec,Vec);
$             PetscErrorCode (*backtr)(void*,PetscScalar*,PetscScalar*);
$             STCreate(comm,&st);
$             STSetType(st,STSHELL);
$             STShellSetApply(st,apply);
$             STShellSetApplyTranspose(st,applytrans);
$             STShellSetBackTransform(st,backtr);    (optional)

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STCreate_Shell"
PetscErrorCode STCreate_Shell(ST st)
{
  PetscErrorCode ierr;
  ST_Shell       *shell;

  PetscFunctionBegin;
  st->ops->destroy = STDestroy_Shell;
  ierr = PetscNew(ST_Shell,&shell);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(st,sizeof(ST_Shell));CHKERRQ(ierr);

  st->data           = (void *) shell;
  ((PetscObject)st)->name           = 0;

  st->ops->apply     = STApply_Shell;
  st->ops->applytrans= STApplyTranspose_Shell;
  st->ops->backtr    = STBackTransform_Shell;
  st->ops->view      = STView_Shell;

  shell->apply       = 0;
  shell->applytrans  = 0;
  shell->backtr      = 0;
  shell->name        = 0;
  shell->ctx         = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STShellSetApply_C","STShellSetApply_Shell",
                    STShellSetApply_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STShellSetApplyTranspose_C","STShellSetApplyTranspose_Shell",
                    STShellSetApplyTranspose_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STShellSetBackTransform_C","STShellSetBackTransform_Shell",
                    STShellSetBackTransform_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STShellSetName_C","STShellSetName_Shell",
                    STShellSetName_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STShellGetName_C","STShellGetName_Shell",
                    STShellGetName_Shell);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

