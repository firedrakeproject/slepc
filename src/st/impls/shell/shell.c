
/*
   This provides a simple shell interface for programmers to 
   create their own spectral transformations without writing much 
   interface code.
*/

#include "src/st/stimpl.h"        /*I "slepcst.h" I*/
#include "slepceps.h"

typedef struct {
  void *ctx;                            /* user provided context */
  int  (*apply)(void *,Vec,Vec);
  int  (*backtr)(void *,PetscScalar*,PetscScalar*);
  char *name;
} ST_Shell;

#undef __FUNCT__  
#define __FUNCT__ "STApply_Shell"
static int STApply_Shell(ST st,Vec x,Vec y)
{
  ST_Shell *shell;
  int       ierr;

  PetscFunctionBegin;
  shell = (ST_Shell *) st->data;
  if (!shell->apply) SETERRQ(1,"No apply() routine provided to Shell ST");
  ierr  = (*shell->apply)(shell->ctx,x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBackTransform_Shell"
static int STBackTransform_Shell(ST st,PetscScalar *eigr,PetscScalar *eigi)
{
  ST_Shell *shell;
  int       ierr;

  PetscFunctionBegin;
  shell = (ST_Shell *) st->data;
  if (shell->backtr) {
    ierr  = (*shell->backtr)(shell->ctx,eigr,eigi); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STDestroy_Shell"
static int STDestroy_Shell(ST st)
{
  ST_Shell *shell = (ST_Shell *) st->data;
  int      ierr;

  PetscFunctionBegin;
  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STView_Shell"
static int STView_Shell(ST st,PetscViewer viewer)
{
  ST_Shell  *ctx = (ST_Shell*)st->data;
  int        ierr;
  PetscTruth isascii;

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
int STShellSetApply_Shell(ST st, int (*apply)(void*,Vec,Vec),void *ptr)
{
  ST_Shell *shell;

  PetscFunctionBegin;
  shell        = (ST_Shell *) st->data;
  shell->apply = apply;
  shell->ctx   = ptr;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STShellSetBackTransform_Shell"
int STShellSetBackTransform_Shell(ST st, int (*backtr)(void*,PetscScalar*,PetscScalar*))
{
  ST_Shell *shell;

  PetscFunctionBegin;
  shell         = (ST_Shell *) st->data;
  shell->backtr = backtr;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STShellSetName_Shell"
int STShellSetName_Shell(ST st,char *name)
{
  ST_Shell *shell;

  PetscFunctionBegin;
  shell       = (ST_Shell *) st->data;
  shell->name = name;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STShellGetName_Shell"
int STShellGetName_Shell(ST st,char **name)
{
  ST_Shell *shell;

  PetscFunctionBegin;
  shell  = (ST_Shell *) st->data;
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
.  apply - the application-provided transformation routine
-  ptr   - pointer to data needed by this routine

   Calling sequence of apply:
.vb
   int apply (void *ptr,Vec xin,Vec xout)
.ve

+  ptr  - the application context
.  xin  - input vector
-  xout - output vector

   Level: developer

.seealso: STShellSetBackTransform()
@*/
int STShellSetApply(ST st, int (*apply)(void*,Vec,Vec),void *ptr)
{
  int ierr, (*f)(ST,int (*)(void*,Vec,Vec),void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STShellSetApply_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,apply,ptr);CHKERRQ(ierr);
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
.  backtr - the application-provided routine

   Calling sequence of backtr:
.vb
   int backtr (void *ptr, PetscScalar *eigr, PetscScalar *eigi)
.ve

.  ptr  - the application context
.  eigr - pointer ot the real part of the eigenvalue to transform back
.  eigi - pointer ot the imaginary part 

   Level: developer

.seealso: STShellSetApply()
@*/
int STShellSetBackTransform(ST st, int (*backtr)(void*,PetscScalar*,PetscScalar*))
{
  int ierr, (*f)(ST,int (*)(void*,PetscScalar*,PetscScalar*));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STShellSetBackTransform_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,(int (*)(void*,PetscScalar*,PetscScalar*))backtr);CHKERRQ(ierr);
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
int STShellSetName(ST st,char *name)
{
  int ierr, (*f)(ST,char *);

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

   Level: developer

.seealso: STShellSetName()
@*/
int STShellGetName(ST st,char **name)
{
  int ierr, (*f)(ST,char **);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)st,"STShellGetName_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(st,name);CHKERRQ(ierr);
  } else {
    SETERRQ(1,"Not shell spectral transformation, cannot get name");
  }
  PetscFunctionReturn(0);
}

/*
   STCreate_Shell - creates a new spectral transformation class.
          This is intended to provide a simple class to use with EPS.
	  You should not use this if you plan to make a complete class.

  Usage:
$             int (*apply)(void *,Vec,Vec);
$             int (*backtr)(void *,PetscScalar*,PetscScalar*);
$             STCreate(comm,&st);
$             STSetType(st,STSHELL);
$             STShellSetApply(st,apply,ctx);
$             STShellSetBackTransform(st,backtr);    (optional)

*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STCreate_Shell"
int STCreate_Shell(ST st)
{
  int      ierr;
  ST_Shell *shell;

  PetscFunctionBegin;
  st->ops->destroy = STDestroy_Shell;
  ierr             = PetscNew(ST_Shell,&shell); CHKERRQ(ierr);
  PetscLogObjectMemory(st,sizeof(ST_Shell));

  st->data           = (void *) shell;
  st->name           = 0;
  st->numberofshifts = 0;

  st->ops->apply     = STApply_Shell;
  st->ops->backtr    = STBackTransform_Shell;
  st->ops->view      = STView_Shell;

  shell->apply       = 0;
  shell->name        = 0;
  shell->ctx         = 0;
  shell->backtr      = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STShellSetApply_C","STShellSetApply_Shell",
                    STShellSetApply_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STShellSetBackTransform_C","STShellSetBackTransform_Shell",
                    STShellSetBackTransform_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STShellSetName_C","STShellSetName_Shell",
                    STShellSetName_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)st,"STShellGetName_C","STShellGetName_Shell",
                    STShellGetName_Shell);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

