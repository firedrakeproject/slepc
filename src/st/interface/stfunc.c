/*
    The ST (spectral transformation) interface routines, callable by users.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/st/stimpl.h"            /*I "slepcst.h" I*/

PetscCookie ST_COOKIE = 0;
PetscLogEvent ST_SetUp = 0, ST_Apply = 0, ST_ApplyTranspose = 0;

#undef __FUNCT__  
#define __FUNCT__ "STInitializePackage"
/*@C
  STInitializePackage - This function initializes everything in the ST package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to STCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode STInitializePackage(char *path) {
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char             *className;
  PetscTruth        opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Spectral Transform",&ST_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = STRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("STSetUp",ST_COOKIE,&ST_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("STApply",ST_COOKIE,&ST_Apply);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("STApplyTranspose",ST_COOKIE,&ST_ApplyTranspose); CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "st", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(ST_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "st", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(ST_COOKIE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STDestroy"
/*@
   STDestroy - Destroys ST context that was created with STCreate().

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Level: beginner

.seealso: STCreate(), STSetUp()
@*/
PetscErrorCode STDestroy(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (--((PetscObject)st)->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(st);CHKERRQ(ierr);

  if (st->ops->destroy) { ierr = (*st->ops->destroy)(st);CHKERRQ(ierr); }
  if (st->ksp) { ierr = KSPDestroy(st->ksp);CHKERRQ(ierr); } 
  if (st->w) { ierr = VecDestroy(st->w);CHKERRQ(ierr); } 
  if (st->shift_matrix != STMATMODE_INPLACE && st->mat) { 
    ierr = MatDestroy(st->mat);CHKERRQ(ierr); 
  }

  PetscHeaderDestroy(st);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STCreate"
/*@C
   STCreate - Creates a spectral transformation context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator 

   Output Parameter:
.  st - location to put the spectral transformation context

   Level: beginner

.seealso: STSetUp(), STApply(), STDestroy(), ST
@*/
PetscErrorCode STCreate(MPI_Comm comm,ST *newst)
{
  PetscErrorCode ierr;
  ST             st;
  const char     *prefix;

  PetscFunctionBegin;
  PetscValidPointer(newst,2);
  *newst = 0;

  ierr = PetscHeaderCreate(st,_p_ST,struct _STOps,ST_COOKIE,-1,"ST",comm,STDestroy,STView);CHKERRQ(ierr);
  ierr = PetscMemzero(st->ops,sizeof(struct _STOps));CHKERRQ(ierr);

  st->A                   = 0;
  st->B                   = 0;
  st->sigma               = 0.0;
  st->data                = 0;
  st->setupcalled         = 0;
  st->w                   = 0;
  st->shift_matrix        = STMATMODE_COPY;
  st->str                 = DIFFERENT_NONZERO_PATTERN;
  
  ierr = KSPCreate(((PetscObject)st)->comm,&st->ksp);CHKERRQ(ierr);
  ierr = STGetOptionsPrefix(st,&prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(st->ksp,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(st->ksp,"st_");CHKERRQ(ierr);
  
  *newst                  = st;
  ierr = PetscPublishAll(st);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

#undef __FUNCT__  
#define __FUNCT__ "STSetOperators"
/*@
   STSetOperators - Sets the matrices associated with the eigenvalue problem. 

   Collective on ST and Mat

   Input Parameters:
+  st - the spectral transformation context
.  A  - the matrix associated with the eigensystem
-  B  - the second matrix in the case of generalized eigenproblems

   Notes:
   To specify a standard eigenproblem, use PETSC_NULL for B.

   Level: intermediate

.seealso: STGetOperators()
 @*/
PetscErrorCode STSetOperators(ST st,Mat A,Mat B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(A,MAT_COOKIE,2);
  if (B) PetscValidHeaderSpecific(B,MAT_COOKIE,3);
  PetscCheckSameComm(st,1,A,2);
  if (B) PetscCheckSameComm(st,1,B,3);
  st->A = A;
  st->B = B;
  st->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetOperators"
/*@C
   STGetOperators - Gets the matrices associated with the eigensystem.

   Not collective, though parallel Mats are returned if the ST is parallel

   Input Parameter:
.  st - the spectral transformation context

   Output Parameters:
.  A - the matrix associated with the eigensystem
-  B - the second matrix in the case of generalized eigenproblems

   Level: intermediate

.seealso: STSetOperators()
@*/
PetscErrorCode STGetOperators(ST st,Mat *A,Mat *B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (A) *A = st->A;
  if (B) *B = st->B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetShift"
/*@
   STSetShift - Sets the shift associated with the spectral transformation

   Not collective

   Input Parameters:
+  st - the spectral transformation context
-  shift - the value of the shift

   Note:
   In some spectral transformations, changing the shift may have associated
   a lot of work, for example recomputing a factorization.
   
   Level: beginner

@*/
PetscErrorCode STSetShift(ST st,PetscScalar shift)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (st->sigma != shift) {
    if (st->ops->setshift) {
      ierr = (*st->ops->setshift)(st,shift); CHKERRQ(ierr);
    }
  }
  st->sigma = shift;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetShift"
/*@
   STGetShift - Gets the shift associated with the spectral transformation.

   Not collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  shift - the value of the shift

   Level: beginner

@*/
PetscErrorCode STGetShift(ST st,PetscScalar* shift)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (shift)  *shift = st->sigma;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetOptionsPrefix"
/*@C
   STSetOptionsPrefix - Sets the prefix used for searching for all 
   ST options in the database.

   Collective on ST

   Input Parameters:
+  st     - the spectral transformation context
-  prefix - the prefix string to prepend to all ST option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: STAppendOptionsPrefix(), STGetOptionsPrefix()
@*/
PetscErrorCode STSetOptionsPrefix(ST st,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)st,prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(st->ksp,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(st->ksp,"st_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STAppendOptionsPrefix"
/*@C
   STAppendOptionsPrefix - Appends to the prefix used for searching for all 
   ST options in the database.

   Collective on ST

   Input Parameters:
+  st     - the spectral transformation context
-  prefix - the prefix string to prepend to all ST option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: STSetOptionsPrefix(), STGetOptionsPrefix()
@*/
PetscErrorCode STAppendOptionsPrefix(ST st,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)st,prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(st->ksp,((PetscObject)st)->prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(st->ksp,"st_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetOptionsPrefix"
/*@C
   STGetOptionsPrefix - Gets the prefix used for searching for all 
   ST options in the database.

   Not Collective

   Input Parameters:
.  st - the spectral transformation context

   Output Parameters:
.  prefix - pointer to the prefix string used, is returned

   Notes: On the Fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: STSetOptionsPrefix(), STAppendOptionsPrefix()
@*/
PetscErrorCode STGetOptionsPrefix(ST st,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)st, prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STView"
/*@C
   STView - Prints the ST data structure.

   Collective on ST

   Input Parameters:
+  ST - the ST context
-  viewer - optional visualization context

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization contexts with
   PetscViewerASCIIOpen() (output to a specified file).

   Level: beginner

.seealso: EPSView(), PetscViewerASCIIOpen()
@*/
PetscErrorCode STView(ST st,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  STType            cstr;
  const char*       str;
  PetscTruth        isascii,isstring;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)st)->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2); 
  PetscCheckSameComm(st,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"ST Object:\n");CHKERRQ(ierr);
    ierr = STGetType(st,&cstr);CHKERRQ(ierr);
    if (cstr) {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",cstr);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: not yet set\n");CHKERRQ(ierr);
    }
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer,"  shift: %g\n",st->sigma);CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer,"  shift: %g+%g i\n",PetscRealPart(st->sigma),PetscImaginaryPart(st->sigma));CHKERRQ(ierr);
#endif
    switch (st->shift_matrix) {
    case STMATMODE_COPY:
      break;
    case STMATMODE_INPLACE:
      ierr = PetscViewerASCIIPrintf(viewer,"Shifting the matrix and unshifting at exit\n");CHKERRQ(ierr);
      break;
    case STMATMODE_SHELL:
      ierr = PetscViewerASCIIPrintf(viewer,"Using a shell matrix\n");CHKERRQ(ierr);
      break;
    }
    if (st->B && st->shift_matrix != STMATMODE_SHELL) { 
      switch (st->str) {
        case SAME_NONZERO_PATTERN:      str = "same nonzero pattern";break;
        case DIFFERENT_NONZERO_PATTERN: str = "different nonzero pattern";break;
        case SUBSET_NONZERO_PATTERN:    str = "subset nonzero pattern";break;
        default:                        SETERRQ(1,"Wrong structure flag");
      }
      ierr = PetscViewerASCIIPrintf(viewer,"Matrices A and B have %s\n",str);CHKERRQ(ierr);
    }
    if (st->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*st->ops->view)(st,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = STGetType(st,&cstr);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," %-7.7s",cstr);CHKERRQ(ierr);
    if (st->ops->view) {ierr = (*st->ops->view)(st,viewer);CHKERRQ(ierr);}
  } else {
    SETERRQ1(1,"Viewer type %s not supported by ST",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STView_Default"
PetscErrorCode STView_Default(ST st,PetscViewer viewer) 
{
  PetscErrorCode ierr;
  PetscTruth     isascii,isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Associated KSP object\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"------------------------------\n");CHKERRQ(ierr);
    ierr = KSPView(st->ksp,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = KSPView(st->ksp,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   STRegisterDynamic - Adds a method to the spectral transformation package.

   Synopsis:
   STRegisterDynamic(char *name_solver,char *path,char *name_create,int (*routine_create)(ST))

   Not collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   STRegisterDynamic() may be called multiple times to add several user-defined spectral transformations.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   STRegisterDynamic("my_solver","/home/username/my_lib/lib/libO/solaris/mylib.a",
              "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     STSetType(st,"my_solver")
   or at runtime via the option
$     -st_type my_solver

   Level: advanced

   $PETSC_DIR, $PETSC_ARCH and $PETSC_LIB_DIR occuring in pathname will be replaced with appropriate values.

.seealso: STRegisterDestroy(), STRegisterAll()
M*/

#undef __FUNCT__  
#define __FUNCT__ "STRegister"
/*@C
  STRegister - See STRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode STRegister(const char *sname,const char *path,const char *name,int (*function)(ST))
{
  PetscErrorCode ierr;
  char           fullname[256];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&STList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STRegisterDestroy"
/*@
   STRegisterDestroy - Frees the list of ST methods that were
   registered by STRegisterDynamic().

   Not Collective

   Level: advanced

.seealso: STRegisterDynamic(), STRegisterAll()
@*/
PetscErrorCode STRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&STList);CHKERRQ(ierr);
  ierr = STRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
