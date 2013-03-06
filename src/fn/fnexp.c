/*
   Exponential function  f(x) = beta*exp(alpha*x).

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/fnimpl.h>      /*I "slepcfn.h" I*/

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunction_Exp"
PetscErrorCode FNEvaluateFunction_Exp(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscScalar arg;

  PetscFunctionBegin;
  if (!fn->na) arg = x;
  else arg = fn->alpha[0]*x;
  if (!fn->nb) *y = PetscExpScalar(arg);
  else *y = fn->beta[0]*PetscExpScalar(arg);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateDerivative_Exp"
PetscErrorCode FNEvaluateDerivative_Exp(FN fn,PetscScalar x,PetscScalar *yp)
{
  PetscScalar arg,scal;

  PetscFunctionBegin;
  if (!fn->na) {
    arg = x;
    scal = 1.0;
  } else {
    arg = fn->alpha[0]*x;
    scal = fn->alpha[0];
  }
  if (fn->nb) scal *= fn->beta[0];
  *yp = scal*PetscExpScalar(arg);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNView_Exp"
PetscErrorCode FNView_Exp(FN fn,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!fn->nb) {
      if (!fn->na) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: exp(x)\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: exp(%3.1F*x)\n",fn->alpha[0]);CHKERRQ(ierr);
      }
    } else {
      if (!fn->na) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: %3.1F*exp(x)\n",fn->beta[0]);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  Exponential: %3.1F*exp(%3.1F*x)\n",fn->beta[0],fn->alpha[0]);CHKERRQ(ierr);
      }
    }
  }  
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "FNCreate_Exp"
PetscErrorCode FNCreate_Exp(FN fn)
{
  PetscFunctionBegin;
  fn->ops->evaluatefunction   = FNEvaluateFunction_Exp;
  fn->ops->evaluatederivative = FNEvaluateDerivative_Exp;
  fn->ops->view               = FNView_Exp;
  PetscFunctionReturn(0);
}
EXTERN_C_END

