/*
   Options of polynomial JD solver.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/pepimpl.h>    /*I "slepcpep.h" I*/
#include "pjdp.h"

#undef __FUNCT__
#define __FUNCT__ "PEPJDSetTolerances_JD"
PetscErrorCode PEPJDSetTolerances_JD(PEP pep,PetscReal mtol,PetscReal htol,PetscReal stol)
{
  PEP_JD *pjd = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  if (mtol==PETSC_DEFAULT) pjd->mtol = 1e-5;
  else {
    if (mtol<0.0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of mtol. Must be > 0");
    pjd->mtol = mtol;
  }
  if (htol==PETSC_DEFAULT) pjd->htol = 1e-2;
  else {
    if (htol<0.0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of htol. Must be > 0");
    pjd->htol = htol;
  }
  if (stol==PETSC_DEFAULT) pjd->stol = 1e-2;
  else {
    if (stol<0.0) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of stol. Must be > 0");
    pjd->stol = stol;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDSetTolerances"
/*@
   PEPJDSetTolerances - Sets various tolerance parameters for the Jacobi-Davidson
   method.

   Logically Collective on PEP

   Input Parameters:
+  pep  - the eigenproblem solver context
.  mtol - the multiplicity tolerance
.  htol - the tolerance for harmonic extraction
-  stol - the tolerance for harmonic shift

   Options Database Key:
+  -pep_jd_mtol - Sets the multiplicity tolerance
.  -pep_jd_htol - Sets the harmonic tolerance
-  -pep_jd_stol - Sets the shift tolerance

   Level: advanced

.seealso: PEPJDGetTolerances()
@*/
PetscErrorCode PEPJDSetTolerances(PEP pep,PetscReal mtol,PetscReal htol,PetscReal stol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,mtol,2);
  PetscValidLogicalCollectiveReal(pep,htol,3);
  PetscValidLogicalCollectiveReal(pep,stol,4);
  ierr = PetscTryMethod(pep,"PEPJDSetTolerances_C",(PEP,PetscReal,PetscReal,PetscReal),(pep,mtol,htol,stol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDGetTolerances_JD"
PetscErrorCode PEPJDGetTolerances_JD(PEP pep,PetscReal *mtol,PetscReal *htol,PetscReal *stol)
{
  PEP_JD *pjd = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  *mtol = pjd->mtol;
  *htol = pjd->htol;
  *stol = pjd->stol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDGetTolerances"
/*@
   PEPJDGetTolerances - Gets various tolerances in the Jacobi-Davidson method.

   Not Collective

   Input Parameter:
.  pep - the eigenproblem solver context

   Output Parameter:
+  mtol - the multiplicity tolerance
.  htol - the harmonic tolerance
-  stol - the shift tolerance

   Level: advanced

.seealso: PEPJDSetTolerances()
@*/
PetscErrorCode PEPJDGetTolerances(PEP pep,PetscReal *mtol,PetscReal *htol,PetscReal *stol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(mtol,2);
  PetscValidPointer(htol,3);
  PetscValidPointer(stol,4);
  ierr = PetscTryMethod(pep,"PEPJDGetTolerances_C",(PEP,PetscReal*,PetscReal*,PetscReal*),(pep,mtol,htol,stol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDSetRestart_JD"
PetscErrorCode PEPJDSetRestart_JD(PEP pep,PetscReal keep)
{
  PEP_JD *pjd = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  if (keep==PETSC_DEFAULT) pjd->keep = 0.5;
  else {
    if (keep<0.1 || keep>0.9) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"The keep argument must be in the range [0.1,0.9]");
    pjd->keep = keep;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDSetRestart"
/*@
   PEPJDSetRestart - Sets the restart parameter for the Jacobi-Davidson
   method, in particular the proportion of basis vectors that must be kept
   after restart.

   Logically Collective on PEP

   Input Parameters:
+  pep  - the eigenproblem solver context
-  keep - the number of vectors to be kept at restart

   Options Database Key:
.  -pep_jd_restart - Sets the restart parameter

   Notes:
   Allowed values are in the range [0.1,0.9]. The default is 0.5.

   Level: advanced

.seealso: PEPJDGetRestart()
@*/
PetscErrorCode PEPJDSetRestart(PEP pep,PetscReal keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveReal(pep,keep,2);
  ierr = PetscTryMethod(pep,"PEPJDSetRestart_C",(PEP,PetscReal),(pep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDGetRestart_JD"
PetscErrorCode PEPJDGetRestart_JD(PEP pep,PetscReal *keep)
{
  PEP_JD *pjd = (PEP_JD*)pep->data;

  PetscFunctionBegin;
  *keep = pjd->keep;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPJDGetRestart"
/*@
   PEPJDGetRestart - Gets the restart parameter used in the Jacobi-Davidson method.

   Not Collective

   Input Parameter:
.  pep - the eigenproblem solver context

   Output Parameter:
.  keep - the restart parameter

   Level: advanced

.seealso: PEPJDSetRestart()
@*/
PetscErrorCode PEPJDGetRestart(PEP pep,PetscReal *keep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(keep,2);
  ierr = PetscTryMethod(pep,"PEPJDGetRestart_C",(PEP,PetscReal*),(pep,keep));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetFromOptions_JD"
PetscErrorCode PEPSetFromOptions_JD(PetscOptions *PetscOptionsObject,PEP pep)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscBool      flg,flg1,flg2,flg3;
  PetscReal      r1,r2,r3;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PEP JD Options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pep_jd_restart","Proportion of vectors kept after restart","PEPJDSetRestart",0.5,&r1,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PEPJDSetRestart(pep,r1);CHKERRQ(ierr);
  }

  r1 = pjd->mtol;
  ierr = PetscOptionsReal("-pep_jd_mtol","Multiplicity tolerance","PEPJDSetTolerances",pjd->mtol,&r1,&flg1);CHKERRQ(ierr);
  r2 = pjd->htol;
  ierr = PetscOptionsReal("-pep_jd_htol","Harmonic tolerance","PEPJDSetTolerances",pjd->htol,&r2,&flg2);CHKERRQ(ierr);
  r3 = pjd->stol;
  ierr = PetscOptionsReal("-pep_jd_stol","Shift tolerance","PEPJDSetTolerances",pjd->stol,&r3,&flg3);CHKERRQ(ierr);
  if (flg1 || flg2 || flg3) {
    ierr = PEPJDSetTolerances(pep,r1,r2,r3);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPView_JD"
PetscErrorCode PEPView_JD(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PEP_JD         *pjd = (PEP_JD*)pep->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  JD: %d%% of basis vectors kept after restart\n",(int)(100*pjd->keep));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  JD: multiplicity tolerance = %g\n",(double)pjd->mtol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  JD: harmonic tolerance = %g\n",(double)pjd->htol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  JD: shift tolerance = %g\n",(double)pjd->stol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

