
/*
    The ST (spectral transformation) interface routines, callable by users.
*/

#include "src/st/stimpl.h"            /*I "slepcst.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "STDestroy"
/*@C
   STDestroy - Destroys ST context that was created with STCreate().

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Level: beginner

.seealso: STCreate(), STSetUp()
@*/
int STDestroy(ST st)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (--st->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(st);CHKERRQ(ierr);

  if (st->ops->destroy) { ierr = (*st->ops->destroy)(st);CHKERRQ(ierr); }
  if (st->ksp) { ierr = KSPDestroy(st->ksp);CHKERRQ(ierr); } 

  PetscLogObjectDestroy(st);
  PetscHeaderDestroy(st);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STPublish_Petsc"
static int STPublish_Petsc(PetscObject object)
{
#if defined(PETSC_HAVE_AMS)
  ST          v = (ST) object;
  int         ierr;
#endif
  
  PetscFunctionBegin;

#if defined(PETSC_HAVE_AMS)
  /* if it is already published then return */
  if (v->amem >=0 ) PetscFunctionReturn(0);

  ierr = PetscObjectPublishBaseBegin(object);CHKERRQ(ierr);
  ierr = PetscObjectPublishBaseEnd(object);CHKERRQ(ierr);
#endif

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

.seealso: STSetUp(), STApply(), STDestroy()
@*/
int STCreate(MPI_Comm comm,ST *newst)
{
  ST      st;
  int     ierr;

  PetscFunctionBegin;
  *newst = 0;

  PetscHeaderCreate(st,_p_ST,struct _STOps,ST_COOKIE,-1,"ST",comm,STDestroy,STView);
  PetscLogObjectCreate(st);
  st->bops->publish       = STPublish_Petsc;
  st->ops->destroy        = 0;
  st->ops->apply          = 0;
  st->ops->applyB         = STDefaultApplyB;
  st->ops->applynoB       = 0;
  st->ops->setshift       = 0;
  st->ops->view           = 0;

  st->A                   = 0;
  st->B                   = 0;
  st->sigma               = 0.0;
  st->vec                 = 0;
  st->ksp                 = 0;
  st->data                = 0;
  st->setupcalled         = 0;
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
int STSetOperators(ST st,Mat A,Mat B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(A,MAT_COOKIE,2);
  if (B) PetscValidHeaderSpecific(B,MAT_COOKIE,3);
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
int STGetOperators(ST st,Mat *A,Mat *B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (A) *A = st->A;
  if (B) *B = st->B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetVector"
/*@
   STSetVector - Sets a vector associated with the ST object.

   Collective on ST and Vec

   Input Parameters:
+  st  - the spectral transformation context
-  vec - the vector

   Notes:
   The vector must be set so that the ST object knows what type
   of vector to allocate if necessary.

   Level: intermediate

.seealso: STGetVector()

@*/
int STSetVector(ST st,Vec vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2);
  PetscCheckSameComm(st,1,vec,2);
  st->vec = vec;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetVector"
/*@
   STGetVector - Gets a vector associated with the ST object; if the 
   vector was not yet set it will return a nul pointer.

   Not collective, but vector is shared by all processors that share the ST

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  vec - the vector

   Level: intermediate

.seealso: STSetVector()

@*/
int STGetVector(ST st,Vec *vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  *vec = st->vec;
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
int STSetShift(ST st,PetscScalar shift)
{
  int         ierr;

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
int STGetShift(ST st,PetscScalar* shift)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (shift)  *shift = st->sigma;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetNumberOfShifts"
/*@
   STGetNumberOfShifts - Returns the number of shifts used in this
   spectral transformation type.

   Not collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  nshifts - the number of shifts

   Note:
   Currently, the returned value will always be either 1 (for STSHIFT 
   and STSINV). Future versions of SLEPc may provide other ST which 
   requires more than one shift.

   Level: advanced

@*/
int STGetNumberOfShifts(ST st,int* nshifts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (nshifts)  *nshifts = st->numberofshifts;
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
int STSetOptionsPrefix(ST st,char *prefix)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)st, prefix);CHKERRQ(ierr);
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
int STAppendOptionsPrefix(ST st,char *prefix)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)st, prefix);CHKERRQ(ierr);
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
int STGetOptionsPrefix(ST st,char **prefix)
{
  int ierr;

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
int STView(ST st,PetscViewer viewer)
{
  STType            cstr;
  int               ierr;
  PetscTruth        isascii,isstring;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(st->comm);
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
    if (st->numberofshifts>0) {
#if !defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer,"  shift: %g\n",st->sigma);CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer,"  shift: %g+%g i\n",PetscRealPart(st->sigma),PetscImaginaryPart(st->sigma));CHKERRQ(ierr);
#endif
    }
    if (st->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*st->ops->view)(st,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (st->ksp) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Associated KSP object\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"------------------------------\n");CHKERRQ(ierr);
      ierr = KSPView(st->ksp,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = STGetType(st,&cstr);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," %-7.7s",cstr);CHKERRQ(ierr);
    if (st->ksp) {ierr = KSPView(st->ksp,viewer);CHKERRQ(ierr);}
    if (st->ops->view) {ierr = (*st->ops->view)(st,viewer);CHKERRQ(ierr);}
  } else {
    SETERRQ1(1,"Viewer type %s not supported by ST",((PetscObject)viewer)->type_name);
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

   $SLEPC_DIR, $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.

.seealso: STRegisterAll(), STRegisterDestroy(), STRegister()
M*/

#undef __FUNCT__  
#define __FUNCT__ "STRegister"
int STRegister(char *sname,char *path,char *name,int (*function)(ST))
{
  int  ierr;
  char fullname[256];

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&STList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

