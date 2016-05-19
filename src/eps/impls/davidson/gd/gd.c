/*

   SLEPc eigensolver: "gd"

   Method: Generalized Davidson

   Algorithm:

       Generalized Davidson with various subspace extraction and
       restart techniques.

   References:

       [1] E.R. Davidson, "Super-matrix methods", Comput. Phys. Commun.
           53(2):49-60, 1989.

       [2] E. Romero and J.E. Roman, "A parallel implementation of
           Davidson methods for large-scale eigenvalue problems in
           SLEPc", ACM Trans. Math. Software 40(2), Article 13, 2014.

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

#include <slepc/private/epsimpl.h>                /*I "slepceps.h" I*/
#include <../src/eps/impls/davidson/davidson.h>

#undef __FUNCT__
#define __FUNCT__ "EPSSetFromOptions_GD"
PetscErrorCode EPSSetFromOptions_GD(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      flg,op;
  PetscInt       opi,opi0;
  KSP            ksp;
  PetscBool      orth;
  const char     *orth_list[2] = {"I","B"};

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS Generalized Davidson (GD) Options");CHKERRQ(ierr);

  ierr = EPSGDGetKrylovStart(eps,&op);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-eps_gd_krylov_start","Start the searching subspace with a krylov basis","EPSGDSetKrylovStart",op,&op,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetKrylovStart(eps,op);CHKERRQ(ierr); }

  ierr = EPSGDGetBOrth(eps,&orth);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-eps_gd_borth","orthogonalization used in the search subspace","EPSGDSetBOrth",orth_list,2,orth_list[orth?1:0],&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetBOrth(eps,opi==1?PETSC_TRUE:PETSC_FALSE);CHKERRQ(ierr); }

  ierr = EPSGDGetBlockSize(eps,&opi);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_gd_blocksize","Number vectors add to the searching subspace","EPSGDSetBlockSize",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetBlockSize(eps,opi);CHKERRQ(ierr); }

  ierr = EPSGDGetRestart(eps,&opi,&opi0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_gd_minv","Set the size of the searching subspace after restarting","EPSGDSetRestart",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetRestart(eps,opi,opi0);CHKERRQ(ierr); }

  ierr = PetscOptionsInt("-eps_gd_plusk","Set the number of saved eigenvectors from the previous iteration when restarting","EPSGDSetRestart",opi0,&opi0,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetRestart(eps,opi,opi0);CHKERRQ(ierr); }

  ierr = EPSGDGetInitialSize(eps,&opi);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_gd_initial_size","Set the initial size of the searching subspace","EPSGDSetInitialSize",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetInitialSize(eps,opi);CHKERRQ(ierr); }

  ierr = EPSGDGetWindowSizes(eps,&opi,&opi0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_gd_pwindow","(Experimental!) Set the number of converged vectors in the projector","EPSGDSetWindowSizes",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetWindowSizes(eps,opi,opi0);CHKERRQ(ierr); }

  ierr = PetscOptionsInt("-eps_gd_qwindow","(Experimental!) Set the number of converged vectors in the projected problem","EPSGDSetWindowSizes",opi0,&opi0,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetWindowSizes(eps,opi,opi0);CHKERRQ(ierr); }

  ierr = PetscOptionsBool("-eps_gd_double_expansion","use the doble-expansion variant of GD","EPSGDSetDoubleExpansion",PETSC_FALSE,&op,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSGDSetDoubleExpansion(eps,op);CHKERRQ(ierr); }

  /* Set STPrecond as the default ST */
  if (!((PetscObject)eps->st)->type_name) {
    ierr = STSetType(eps->st,STPRECOND);CHKERRQ(ierr);
  }
  ierr = STPrecondSetKSPHasMat(eps->st,PETSC_FALSE);CHKERRQ(ierr);

  /* Set the default options of the KSP */
  ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_GD"
PetscErrorCode EPSSetUp_GD(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      t;
  KSP            ksp;

  PetscFunctionBegin;
  /* Set KSPPREONLY as default */
  ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  }

  /* Setup common for all davidson solvers */
  ierr = EPSSetUp_XD(eps);CHKERRQ(ierr);

  /* Check some constraints */
  ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPPREONLY,&t);CHKERRQ(ierr);
  if (!t) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"EPSGD only works with KSPPREONLY");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSView_GD"
PetscErrorCode EPSView_GD(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii,opb;
  PetscInt       opi,opi0;
  PetscBool      borth;
  EPS_DAVIDSON   *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (data->doubleexp) {
      ierr = PetscViewerASCIIPrintf(viewer,"  GD: using double expansion variant (GD2)\n");CHKERRQ(ierr);
    }
    ierr = EPSXDGetBOrth_XD(eps,&borth);CHKERRQ(ierr);
    if (borth) {
      ierr = PetscViewerASCIIPrintf(viewer,"  GD: search subspace is B-orthogonalized\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  GD: search subspace is orthogonalized\n");CHKERRQ(ierr);
    }
    ierr = EPSXDGetBlockSize_XD(eps,&opi);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  GD: block size=%D\n",opi);CHKERRQ(ierr);
    ierr = EPSXDGetKrylovStart_XD(eps,&opb);CHKERRQ(ierr);
    if (!opb) {
      ierr = PetscViewerASCIIPrintf(viewer,"  GD: type of the initial subspace: non-Krylov\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  GD: type of the initial subspace: Krylov\n");CHKERRQ(ierr);
    }
    ierr = EPSXDGetRestart_XD(eps,&opi,&opi0);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  GD: size of the subspace after restarting: %D\n",opi);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  GD: number of vectors after restarting from the previous iteration: %D\n",opi0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_GD"
PetscErrorCode EPSDestroy_GD(EPS eps)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetKrylovStart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetKrylovStart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetBOrth_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetBOrth_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetInitialSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetInitialSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetWindowSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetWindowSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetDoubleExpansion_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetDoubleExpansion_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDSetKrylovStart"
/*@
   EPSGDSetKrylovStart - Activates or deactivates starting the searching
   subspace with a Krylov basis.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  krylovstart - boolean flag

   Options Database Key:
.  -eps_gd_krylov_start - Activates starting the searching subspace with a
    Krylov basis

   Level: advanced

.seealso: EPSGDGetKrylovStart()
@*/
PetscErrorCode EPSGDSetKrylovStart(EPS eps,PetscBool krylovstart)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,krylovstart,2);
  ierr = PetscTryMethod(eps,"EPSGDSetKrylovStart_C",(EPS,PetscBool),(eps,krylovstart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDGetKrylovStart"
/*@
   EPSGDGetKrylovStart - Returns a flag indicating if the search subspace is started with a
   Krylov basis.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  krylovstart - boolean flag indicating if the search subspace is started
   with a Krylov basis

   Level: advanced

.seealso: EPSGDGetKrylovStart()
@*/
PetscErrorCode EPSGDGetKrylovStart(EPS eps,PetscBool *krylovstart)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(krylovstart,2);
  ierr = PetscUseMethod(eps,"EPSGDGetKrylovStart_C",(EPS,PetscBool*),(eps,krylovstart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDSetBlockSize"
/*@
   EPSGDSetBlockSize - Sets the number of vectors to be added to the searching space
   in every iteration.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  blocksize - number of vectors added to the search space in every iteration

   Options Database Key:
.  -eps_gd_blocksize - number of vectors added to the search space in every iteration

   Level: advanced

.seealso: EPSGDSetKrylovStart()
@*/
PetscErrorCode EPSGDSetBlockSize(EPS eps,PetscInt blocksize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,blocksize,2);
  ierr = PetscTryMethod(eps,"EPSGDSetBlockSize_C",(EPS,PetscInt),(eps,blocksize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDGetBlockSize"
/*@
   EPSGDGetBlockSize - Returns the number of vectors to be added to the searching space
   in every iteration.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  blocksize - number of vectors added to the search space in every iteration

   Level: advanced

.seealso: EPSGDSetBlockSize()
@*/
PetscErrorCode EPSGDGetBlockSize(EPS eps,PetscInt *blocksize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(blocksize,2);
  ierr = PetscUseMethod(eps,"EPSGDGetBlockSize_C",(EPS,PetscInt*),(eps,blocksize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDGetRestart"
/*@
   EPSGDGetRestart - Gets the number of vectors of the searching space after
   restarting and the number of vectors saved from the previous iteration.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
+  minv - number of vectors of the searching subspace after restarting
-  plusk - number of vectors saved from the previous iteration

   Level: advanced

.seealso: EPSGDSetRestart()
@*/
PetscErrorCode EPSGDGetRestart(EPS eps,PetscInt *minv,PetscInt *plusk)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSGDGetRestart_C",(EPS,PetscInt*,PetscInt*),(eps,minv,plusk));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDSetRestart"
/*@
   EPSGDSetRestart - Sets the number of vectors of the searching space after
   restarting and the number of vectors saved from the previous iteration.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  minv - number of vectors of the searching subspace after restarting
-  plusk - number of vectors saved from the previous iteration

   Options Database Keys:
+  -eps_gd_minv - number of vectors of the searching subspace after restarting
-  -eps_gd_plusk - number of vectors saved from the previous iteration

   Level: advanced

.seealso: EPSGDSetRestart()
@*/
PetscErrorCode EPSGDSetRestart(EPS eps,PetscInt minv,PetscInt plusk)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,minv,2);
  PetscValidLogicalCollectiveInt(eps,plusk,2);
  ierr = PetscTryMethod(eps,"EPSGDSetRestart_C",(EPS,PetscInt,PetscInt),(eps,minv,plusk));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDGetInitialSize"
/*@
   EPSGDGetInitialSize - Returns the initial size of the searching space.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  initialsize - number of vectors of the initial searching subspace

   Notes:
   If EPSGDGetKrylovStart() is PETSC_FALSE and the user provides vectors with
   EPSSetInitialSpace(), up to initialsize vectors will be used; and if the
   provided vectors are not enough, the solver completes the subspace with
   random vectors. In the case of EPSGDGetKrylovStart() being PETSC_TRUE, the solver
   gets the first vector provided by the user or, if not available, a random vector,
   and expands the Krylov basis up to initialsize vectors.

   Level: advanced

.seealso: EPSGDSetInitialSize(), EPSGDGetKrylovStart()
@*/
PetscErrorCode EPSGDGetInitialSize(EPS eps,PetscInt *initialsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(initialsize,2);
  ierr = PetscUseMethod(eps,"EPSGDGetInitialSize_C",(EPS,PetscInt*),(eps,initialsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDSetInitialSize"
/*@
   EPSGDSetInitialSize - Sets the initial size of the searching space.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  initialsize - number of vectors of the initial searching subspace

   Options Database Key:
.  -eps_gd_initial_size - number of vectors of the initial searching subspace

   Notes:
   If EPSGDGetKrylovStart() is PETSC_FALSE and the user provides vectors with
   EPSSetInitialSpace(), up to initialsize vectors will be used; and if the
   provided vectors are not enough, the solver completes the subspace with
   random vectors. In the case of EPSGDGetKrylovStart() being PETSC_TRUE, the solver
   gets the first vector provided by the user or, if not available, a random vector,
   and expands the Krylov basis up to initialsize vectors.

   Level: advanced

.seealso: EPSGDGetInitialSize(), EPSGDGetKrylovStart()
@*/
PetscErrorCode EPSGDSetInitialSize(EPS eps,PetscInt initialsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,initialsize,2);
  ierr = PetscTryMethod(eps,"EPSGDSetInitialSize_C",(EPS,PetscInt),(eps,initialsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDSetBOrth"
/*@
   EPSGDSetBOrth - Selects the orthogonalization that will be used in the search
   subspace in case of generalized Hermitian problems.

   Logically Collective on EPS

   Input Parameters:
+  eps   - the eigenproblem solver context
-  borth - whether to B-orthogonalize the search subspace

   Options Database Key:
.  -eps_gd_borth - Set the orthogonalization used in the search subspace

   Level: advanced

.seealso: EPSGDGetBOrth()
@*/
PetscErrorCode EPSGDSetBOrth(EPS eps,PetscBool borth)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,borth,2);
  ierr = PetscTryMethod(eps,"EPSGDSetBOrth_C",(EPS,PetscBool),(eps,borth));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDGetBOrth"
/*@
   EPSGDGetBOrth - Returns the orthogonalization used in the search
   subspace in case of generalized Hermitian problems.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  borth - whether to B-orthogonalize the search subspace

   Level: advanced

.seealso: EPSGDSetBOrth()
@*/
PetscErrorCode EPSGDGetBOrth(EPS eps,PetscBool *borth)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(borth,2);
  ierr = PetscUseMethod(eps,"EPSGDGetBOrth_C",(EPS,PetscBool*),(eps,borth));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDGetWindowSizes"
/*@
   EPSGDGetWindowSizes - Gets the number of converged vectors in the projected
   problem (or Rayleigh quotient) and in the projector employed in the correction
   equation.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
+  pwindow - number of converged vectors in the projector
-  qwindow - number of converged vectors in the projected problem

   Level: advanced

.seealso: EPSGDSetWindowSizes()
@*/
PetscErrorCode EPSGDGetWindowSizes(EPS eps,PetscInt *pwindow,PetscInt *qwindow)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSGDGetWindowSizes_C",(EPS,PetscInt*,PetscInt*),(eps,pwindow,qwindow));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDSetWindowSizes"
/*@
   EPSGDSetWindowSizes - Sets the number of converged vectors in the projected
   problem (or Rayleigh quotient) and in the projector employed in the correction
   equation.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  pwindow - number of converged vectors in the projector
-  qwindow - number of converged vectors in the projected problem

   Options Database Keys:
+  -eps_gd_pwindow - set the number of converged vectors in the projector
-  -eps_gd_qwindow - set the number of converged vectors in the projected problem

   Level: advanced

.seealso: EPSGDGetWindowSizes()
@*/
PetscErrorCode EPSGDSetWindowSizes(EPS eps,PetscInt pwindow,PetscInt qwindow)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,pwindow,2);
  PetscValidLogicalCollectiveInt(eps,qwindow,3);
  ierr = PetscTryMethod(eps,"EPSGDSetWindowSizes_C",(EPS,PetscInt,PetscInt),(eps,pwindow,qwindow));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDGetDoubleExpansion_GD"
static PetscErrorCode EPSGDGetDoubleExpansion_GD(EPS eps,PetscBool *doubleexp)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  *doubleexp = data->doubleexp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDGetDoubleExpansion"
/*@
   EPSGDGetDoubleExpansion - Gets a flag indicating whether the double
   expansion variant has been activated or not.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  doubleexp - the flag

   Level: advanced

.seealso: EPSGDSetDoubleExpansion()
@*/
PetscErrorCode EPSGDGetDoubleExpansion(EPS eps,PetscBool *doubleexp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(doubleexp,2);
  ierr = PetscUseMethod(eps,"EPSGDGetDoubleExpansion_C",(EPS,PetscBool*),(eps,doubleexp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDSetDoubleExpansion_GD"
static PetscErrorCode EPSGDSetDoubleExpansion_GD(EPS eps,PetscBool doubleexp)
{
  EPS_DAVIDSON *data = (EPS_DAVIDSON*)eps->data;

  PetscFunctionBegin;
  data->doubleexp = doubleexp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGDSetDoubleExpansion"
/*@
   EPSGDSetDoubleExpansion - Activate a variant where the search subspace is
   expanded with K*[A*x B*x] (double expansion) instead of the classic K*r,
   where K is the preconditioner, x the selected approximate eigenvector and
   r its associated residual vector.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  doubleexp - the boolean flag

   Options Database Keys:
.  -eps_gd_double_expansion - activate the double-expansion variant of GD

   Level: advanced
@*/
PetscErrorCode EPSGDSetDoubleExpansion(EPS eps,PetscBool doubleexp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,doubleexp,2);
  ierr = PetscTryMethod(eps,"EPSGDSetDoubleExpansion_C",(EPS,PetscBool),(eps,doubleexp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_GD"
PETSC_EXTERN PetscErrorCode EPSCreate_GD(EPS eps)
{
  PetscErrorCode  ierr;
  EPS_DAVIDSON    *data;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&data);CHKERRQ(ierr);
  eps->data = (void*)data;

  data->blocksize   = 1;
  data->initialsize = 6;
  data->minv        = 6;
  data->plusk       = 0;
  data->ipB         = PETSC_TRUE;
  data->fix         = 0.0;
  data->krylovstart = PETSC_FALSE;
  data->dynamic     = PETSC_FALSE;
  data->cX_in_proj  = 0;
  data->cX_in_impr  = 0;

  eps->ops->solve          = EPSSolve_XD;
  eps->ops->setup          = EPSSetUp_XD;
  eps->ops->reset          = EPSReset_XD;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->computevectors = EPSComputeVectors_XD;
  eps->ops->view           = EPSView_GD;
  eps->ops->setfromoptions = EPSSetFromOptions_GD;
  eps->ops->setup          = EPSSetUp_GD;
  eps->ops->destroy        = EPSDestroy_GD;

  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetKrylovStart_C",EPSXDSetKrylovStart_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetKrylovStart_C",EPSXDGetKrylovStart_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetBOrth_C",EPSXDSetBOrth_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetBOrth_C",EPSXDGetBOrth_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetBlockSize_C",EPSXDSetBlockSize_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetBlockSize_C",EPSXDGetBlockSize_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetRestart_C",EPSXDSetRestart_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetRestart_C",EPSXDGetRestart_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetInitialSize_C",EPSXDSetInitialSize_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetInitialSize_C",EPSXDGetInitialSize_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetWindowSizes_C",EPSXDSetWindowSizes_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetWindowSizes_C",EPSXDGetWindowSizes_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDSetDoubleExpansion_C",EPSGDSetDoubleExpansion_GD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGDGetDoubleExpansion_C",EPSGDGetDoubleExpansion_GD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

