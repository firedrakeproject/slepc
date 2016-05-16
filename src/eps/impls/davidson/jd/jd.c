/*

   SLEPc eigensolver: "jd"

   Method: Jacobi-Davidson

   Algorithm:

       Jacobi-Davidson with various subspace extraction and
       restart techniques.

   References:

       [1] G.L.G. Sleijpen and H.A. van der Vorst, "A Jacobi-Davidson
           iteration method for linear eigenvalue problems", SIAM J.
           Matrix Anal. Appl. 17(2):401-425, 1996.

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
#define __FUNCT__ "EPSSetFromOptions_JD"
PetscErrorCode EPSSetFromOptions_JD(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      flg,op;
  PetscInt       opi,opi0;
  PetscReal      opf;
  KSP            ksp;
  PetscBool      orth;
  const char     *orth_list[2] = {"I","B"};

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS Jacobi-Davidson (JD) Options");CHKERRQ(ierr);

  ierr = EPSJDGetKrylovStart(eps,&op);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-eps_jd_krylov_start","Start the searching subspace with a krylov basis","EPSJDSetKrylovStart",op,&op,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSJDSetKrylovStart(eps,op);CHKERRQ(ierr); }

  ierr = EPSJDGetBlockSize(eps,&opi);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_jd_blocksize","Number vectors add to the searching subspace","EPSJDSetBlockSize",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSJDSetBlockSize(eps,opi);CHKERRQ(ierr); }

  ierr = EPSJDGetRestart(eps,&opi,&opi0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_jd_minv","Set the size of the searching subspace after restarting","EPSJDSetRestart",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSJDSetRestart(eps,opi,opi0);CHKERRQ(ierr); }

  ierr = PetscOptionsInt("-eps_jd_plusk","Set the number of saved eigenvectors from the previous iteration when restarting","EPSJDSetRestart",opi0,&opi0,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSJDSetRestart(eps,opi,opi0);CHKERRQ(ierr); }

  ierr = EPSJDGetInitialSize(eps,&opi);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_jd_initial_size","Set the initial size of the searching subspace","EPSJDSetInitialSize",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSJDSetInitialSize(eps,opi);CHKERRQ(ierr); }

  ierr = EPSJDGetFix(eps,&opf);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eps_jd_fix","Set the tolerance for changing the target in the correction equation","EPSJDSetFix",opf,&opf,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSJDSetFix(eps,opf);CHKERRQ(ierr); }

  ierr = EPSJDGetBOrth(eps,&orth);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-eps_jd_borth","orthogonalization used in the search subspace","EPSJDSetBOrth",orth_list,2,orth_list[orth?1:0],&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSJDSetBOrth(eps,opi==1?PETSC_TRUE:PETSC_FALSE);CHKERRQ(ierr); }

  ierr = EPSJDGetConstCorrectionTol(eps,&op);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-eps_jd_const_correction_tol","Disable the dynamic stopping criterion when solving the correction equation","EPSJDSetConstCorrectionTol",op,&op,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSJDSetConstCorrectionTol(eps,op);CHKERRQ(ierr); }

  ierr = EPSJDGetWindowSizes(eps,&opi,&opi0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-eps_jd_pwindow","(Experimental!) Set the number of converged vectors in the projector","EPSJDSetWindowSizes",opi,&opi,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSJDSetWindowSizes(eps,opi,opi0);CHKERRQ(ierr); }

  ierr = PetscOptionsInt("-eps_jd_qwindow","(Experimental!) Set the number of converged vectors in the projected problem","EPSJDSetWindowSizes",opi0,&opi0,&flg);CHKERRQ(ierr);
  if (flg) { ierr = EPSJDSetWindowSizes(eps,opi,opi0);CHKERRQ(ierr); }

  /* Set STPrecond as the default ST */
  if (!((PetscObject)eps->st)->type_name) {
    ierr = STSetType(eps->st,STPRECOND);CHKERRQ(ierr);
  }
  ierr = STPrecondSetKSPHasMat(eps->st,PETSC_FALSE);CHKERRQ(ierr);

  /* Set the default options of the KSP */
  ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPBCGSL);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,1e-4,PETSC_DEFAULT,PETSC_DEFAULT,90);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetUp_JD"
PetscErrorCode EPSSetUp_JD(EPS eps)
{
  PetscErrorCode ierr;
  PetscBool      t;
  KSP            ksp;

  PetscFunctionBegin;
  /* Setup common for all davidson solvers */
  ierr = EPSSetUp_XD(eps);CHKERRQ(ierr);

  /* Set the default options of the KSP */
  ierr = STGetKSP(eps->st,&ksp);CHKERRQ(ierr);
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPBCGSL);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,1e-4,PETSC_DEFAULT,PETSC_DEFAULT,90);CHKERRQ(ierr);
  }

  /* Check some constraints */
  ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPPREONLY,&t);CHKERRQ(ierr);
  if (t) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"EPSJD does not work with KSPPREONLY");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSView_JD"
PetscErrorCode EPSView_JD(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii,opb;
  PetscInt       opi,opi0;
  PetscBool      borth;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = EPSXDGetBOrth_XD(eps,&borth);CHKERRQ(ierr);
    if (borth) {
      ierr = PetscViewerASCIIPrintf(viewer,"  JD: search subspace is B-orthogonalized\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  JD: search subspace is orthogonalized\n");CHKERRQ(ierr);
    }
    ierr = EPSXDGetBlockSize_XD(eps,&opi);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  JD: block size=%D\n",opi);CHKERRQ(ierr);
    ierr = EPSXDGetKrylovStart_XD(eps,&opb);CHKERRQ(ierr);
    if (!opb) {
      ierr = PetscViewerASCIIPrintf(viewer,"  JD: type of the initial subspace: non-Krylov\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  JD: type of the initial subspace: Krylov\n");CHKERRQ(ierr);
    }
    ierr = EPSXDGetRestart_XD(eps,&opi,&opi0);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  JD: size of the subspace after restarting: %D\n",opi);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  JD: number of vectors after restarting from the previous iteration: %D\n",opi0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy_JD"
PetscErrorCode EPSDestroy_JD(EPS eps)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetKrylovStart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetKrylovStart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetBlockSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetRestart_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetInitialSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetInitialSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetFix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetFix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetConstCorrectionTol_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetConstCorrectionTol_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetWindowSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetWindowSizes_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetBOrth_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetBOrth_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDSetKrylovStart"
/*@
   EPSJDSetKrylovStart - Activates or deactivates starting the searching
   subspace with a Krylov basis.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  krylovstart - boolean flag

   Options Database Key:
.  -eps_jd_krylov_start - Activates starting the searching subspace with a
    Krylov basis

   Level: advanced

.seealso: EPSJDGetKrylovStart()
@*/
PetscErrorCode EPSJDSetKrylovStart(EPS eps,PetscBool krylovstart)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,krylovstart,2);
  ierr = PetscTryMethod(eps,"EPSJDSetKrylovStart_C",(EPS,PetscBool),(eps,krylovstart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDGetKrylovStart"
/*@
   EPSJDGetKrylovStart - Returns a flag indicating if the searching subspace is started with a
   Krylov basis.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  krylovstart - boolean flag indicating if the searching subspace is started
   with a Krylov basis

   Level: advanced

.seealso: EPSJDGetKrylovStart()
@*/
PetscErrorCode EPSJDGetKrylovStart(EPS eps,PetscBool *krylovstart)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(krylovstart,2);
  ierr = PetscUseMethod(eps,"EPSJDGetKrylovStart_C",(EPS,PetscBool*),(eps,krylovstart));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDSetBlockSize"
/*@
   EPSJDSetBlockSize - Sets the number of vectors to be added to the searching space
   in every iteration.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  blocksize - number of vectors added to the search space in every iteration

   Options Database Key:
.  -eps_jd_blocksize - number of vectors added to the searching space every iteration

   Level: advanced

.seealso: EPSJDSetKrylovStart()
@*/
PetscErrorCode EPSJDSetBlockSize(EPS eps,PetscInt blocksize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,blocksize,2);
  ierr = PetscTryMethod(eps,"EPSJDSetBlockSize_C",(EPS,PetscInt),(eps,blocksize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDGetBlockSize"
/*@
   EPSJDGetBlockSize - Returns the number of vectors to be added to the searching space
   in every iteration.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  blocksize - number of vectors added to the search space in every iteration

   Level: advanced

.seealso: EPSJDSetBlockSize()
@*/
PetscErrorCode EPSJDGetBlockSize(EPS eps,PetscInt *blocksize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(blocksize,2);
  ierr = PetscUseMethod(eps,"EPSJDGetBlockSize_C",(EPS,PetscInt*),(eps,blocksize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDGetRestart"
/*@
   EPSJDGetRestart - Gets the number of vectors of the searching space after
   restarting and the number of vectors saved from the previous iteration.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
+  minv - number of vectors of the searching subspace after restarting
-  plusk - number of vectors saved from the previous iteration

   Level: advanced

.seealso: EPSJDSetRestart()
@*/
PetscErrorCode EPSJDGetRestart(EPS eps,PetscInt *minv,PetscInt *plusk)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSJDGetRestart_C",(EPS,PetscInt*,PetscInt*),(eps,minv,plusk));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDSetRestart"
/*@
   EPSJDSetRestart - Sets the number of vectors of the searching space after
   restarting and the number of vectors saved from the previous iteration.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  minv - number of vectors of the searching subspace after restarting
-  plusk - number of vectors saved from the previous iteration

   Options Database Keys:
+  -eps_jd_minv - number of vectors of the searching subspace after restarting
-  -eps_jd_plusk - number of vectors saved from the previous iteration

   Level: advanced

.seealso: EPSJDGetRestart()
@*/
PetscErrorCode EPSJDSetRestart(EPS eps,PetscInt minv,PetscInt plusk)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,minv,2);
  PetscValidLogicalCollectiveInt(eps,plusk,3);
  ierr = PetscTryMethod(eps,"EPSJDSetRestart_C",(EPS,PetscInt,PetscInt),(eps,minv,plusk));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDGetInitialSize"
/*@
   EPSJDGetInitialSize - Returns the initial size of the searching space.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  initialsize - number of vectors of the initial searching subspace

   Notes:
   If EPSJDGetKrylovStart() is PETSC_FALSE and the user provides vectors with
   EPSSetInitialSpace(), up to initialsize vectors will be used; and if the
   provided vectors are not enough, the solver completes the subspace with
   random vectors. In the case of EPSJDGetKrylovStart() being PETSC_TRUE, the solver
   gets the first vector provided by the user or, if not available, a random vector,
   and expands the Krylov basis up to initialsize vectors.

   Level: advanced

.seealso: EPSJDSetInitialSize(), EPSJDGetKrylovStart()
@*/
PetscErrorCode EPSJDGetInitialSize(EPS eps,PetscInt *initialsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(initialsize,2);
  ierr = PetscUseMethod(eps,"EPSJDGetInitialSize_C",(EPS,PetscInt*),(eps,initialsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDSetInitialSize"
/*@
   EPSJDSetInitialSize - Sets the initial size of the searching space.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  initialsize - number of vectors of the initial searching subspace

   Options Database Key:
.  -eps_jd_initial_size - number of vectors of the initial searching subspace

   Notes:
   If EPSJDGetKrylovStart() is PETSC_FALSE and the user provides vectors with
   EPSSetInitialSpace(), up to initialsize vectors will be used; and if the
   provided vectors are not enough, the solver completes the subspace with
   random vectors. In the case of EPSJDGetKrylovStart() being PETSC_TRUE, the solver
   gets the first vector provided by the user or, if not available, a random vector,
   and expands the Krylov basis up to initialsize vectors.

   Level: advanced

.seealso: EPSJDGetInitialSize(), EPSJDGetKrylovStart()
@*/
PetscErrorCode EPSJDSetInitialSize(EPS eps,PetscInt initialsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,initialsize,2);
  ierr = PetscTryMethod(eps,"EPSJDSetInitialSize_C",(EPS,PetscInt),(eps,initialsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDGetFix"
/*@
   EPSJDGetFix - Returns the threshold for changing the target in the correction
   equation.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  fix - threshold for changing the target

   Note:
   The target in the correction equation is fixed at the first iterations.
   When the norm of the residual vector is lower than the fix value,
   the target is set to the corresponding eigenvalue.

   Level: advanced

.seealso: EPSJDSetFix()
@*/
PetscErrorCode EPSJDGetFix(EPS eps,PetscReal *fix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(fix,2);
  ierr = PetscUseMethod(eps,"EPSJDGetFix_C",(EPS,PetscReal*),(eps,fix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDSetFix"
/*@
   EPSJDSetFix - Sets the threshold for changing the target in the correction
   equation.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  fix - threshold for changing the target

   Options Database Key:
.  -eps_jd_fix - the fix value

   Note:
   The target in the correction equation is fixed at the first iterations.
   When the norm of the residual vector is lower than the fix value,
   the target is set to the corresponding eigenvalue.

   Level: advanced

.seealso: EPSJDGetFix()
@*/
PetscErrorCode EPSJDSetFix(EPS eps,PetscReal fix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(eps,fix,2);
  ierr = PetscTryMethod(eps,"EPSJDSetFix_C",(EPS,PetscReal),(eps,fix));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDSetConstCorrectionTol"
/*@
   EPSJDSetConstCorrectionTol - If true, deactivates the dynamic stopping criterion
   (also called Newton) that sets the KSP relative tolerance
   to 0.5**i, where i is the number of EPS iterations from the last converged value.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  constant - if false, the KSP relative tolerance is set to 0.5**i.

   Options Database Key:
.  -eps_jd_const_correction_tol - Deactivates the dynamic stopping criterion.

   Level: advanced

.seealso: EPSJDGetConstCorrectionTol()
@*/
PetscErrorCode EPSJDSetConstCorrectionTol(EPS eps,PetscBool constant)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,constant,2);
  ierr = PetscTryMethod(eps,"EPSJDSetConstCorrectionTol_C",(EPS,PetscBool),(eps,constant));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDGetConstCorrectionTol"
/*@
   EPSJDGetConstCorrectionTol - Returns a flag indicating if the dynamic stopping is being used for
   solving the correction equation. If the flag is false the KSP relative tolerance is set
   to 0.5**i, where i is the number of EPS iterations from the last converged value.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  constant - boolean flag indicating if the dynamic stopping criterion is not being used.

   Level: advanced

.seealso: EPSJDGetConstCorrectionTol()
@*/
PetscErrorCode EPSJDGetConstCorrectionTol(EPS eps,PetscBool *constant)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(constant,2);
  ierr = PetscUseMethod(eps,"EPSJDGetConstCorrectionTol_C",(EPS,PetscBool*),(eps,constant));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDGetWindowSizes"
/*@
   EPSJDGetWindowSizes - Gets the number of converged vectors in the projected
   problem (or Rayleigh quotient) and in the projector employed in the correction
   equation.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
+  pwindow - number of converged vectors in the projector
-  qwindow - number of converged vectors in the projected problem

   Level: advanced

.seealso: EPSJDSetWindowSizes()
@*/
PetscErrorCode EPSJDGetWindowSizes(EPS eps,PetscInt *pwindow,PetscInt *qwindow)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSJDGetWindowSizes_C",(EPS,PetscInt*,PetscInt*),(eps,pwindow,qwindow));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDSetWindowSizes"
/*@
   EPSJDSetWindowSizes - Sets the number of converged vectors in the projected
   problem (or Rayleigh quotient) and in the projector employed in the correction
   equation.

   Logically Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
.  pwindow - number of converged vectors in the projector
-  qwindow - number of converged vectors in the projected problem

   Options Database Keys:
+  -eps_jd_pwindow - set the number of converged vectors in the projector
-  -eps_jd_qwindow - set the number of converged vectors in the projected problem

   Level: advanced

.seealso: EPSJDGetWindowSizes()
@*/
PetscErrorCode EPSJDSetWindowSizes(EPS eps,PetscInt pwindow,PetscInt qwindow)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,pwindow,2);
  PetscValidLogicalCollectiveInt(eps,qwindow,3);
  ierr = PetscTryMethod(eps,"EPSJDSetWindowSizes_C",(EPS,PetscInt,PetscInt),(eps,pwindow,qwindow));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDSetBOrth"
/*@
   EPSJDSetBOrth - Selects the orthogonalization that will be used in the search
   subspace in case of generalized Hermitian problems.

   Logically Collective on EPS

   Input Parameters:
+  eps   - the eigenproblem solver context
-  borth - whether to B-orthogonalize the search subspace

   Options Database Key:
.  -eps_jd_borth - Set the orthogonalization used in the search subspace

   Level: advanced

.seealso: EPSJDGetBOrth()
@*/
PetscErrorCode EPSJDSetBOrth(EPS eps,PetscBool borth)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveBool(eps,borth,2);
  ierr = PetscTryMethod(eps,"EPSJDSetBOrth_C",(EPS,PetscBool),(eps,borth));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSJDGetBOrth"
/*@
   EPSJDGetBOrth - Returns the orthogonalization used in the search
   subspace in case of generalized Hermitian problems.

   Not Collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameters:
.  borth - whether to B-orthogonalize the search subspace

   Level: advanced

.seealso: EPSJDSetBOrth()
@*/
PetscErrorCode EPSJDGetBOrth(EPS eps,PetscBool *borth)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(borth,2);
  ierr = PetscUseMethod(eps,"EPSJDGetBOrth_C",(EPS,PetscBool*),(eps,borth));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate_JD"
PETSC_EXTERN PetscErrorCode EPSCreate_JD(EPS eps)
{
  PetscErrorCode ierr;
  EPS_DAVIDSON   *data;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&data);CHKERRQ(ierr);
  eps->data = (void*)data;

  data->blocksize   = 1;
  data->initialsize = 6;
  data->minv        = 6;
  data->plusk       = 0;
  data->ipB         = PETSC_TRUE;
  data->fix         = 0.01;
  data->krylovstart = PETSC_FALSE;
  data->dynamic     = PETSC_FALSE;
  data->cX_in_proj  = 0;
  data->cX_in_impr  = 0;

  eps->ops->solve          = EPSSolve_XD;
  eps->ops->setup          = EPSSetUp_XD;
  eps->ops->reset          = EPSReset_XD;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->computevectors = EPSComputeVectors_XD;
  eps->ops->view           = EPSView_JD;
  eps->ops->setfromoptions = EPSSetFromOptions_JD;
  eps->ops->setup          = EPSSetUp_JD;
  eps->ops->destroy        = EPSDestroy_JD;

  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetKrylovStart_C",EPSXDSetKrylovStart_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetKrylovStart_C",EPSXDGetKrylovStart_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetBlockSize_C",EPSXDSetBlockSize_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetBlockSize_C",EPSXDGetBlockSize_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetRestart_C",EPSXDSetRestart_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetRestart_C",EPSXDGetRestart_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetInitialSize_C",EPSXDSetInitialSize_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetInitialSize_C",EPSXDGetInitialSize_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetFix_C",EPSJDSetFix_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetFix_C",EPSXDGetFix_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetConstCorrectionTol_C",EPSJDSetConstCorrectionTol_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetConstCorrectionTol_C",EPSJDGetConstCorrectionTol_JD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetWindowSizes_C",EPSXDSetWindowSizes_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetWindowSizes_C",EPSXDGetWindowSizes_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDSetBOrth_C",EPSXDSetBOrth_XD);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSJDGetBOrth_C",EPSXDGetBOrth_XD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

