/*
   Rational function  r(x) = p(x)/q(x), where p(x) is a polynomial of
   degree na and q(x) is a polynomial of degree nb (can be 0).

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
#define __FUNCT__ "FNEvaluateFunction_Rational"
PetscErrorCode FNEvaluateFunction_Rational(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscInt    i;
  PetscScalar p,q;

  PetscFunctionBegin;
  if (!fn->na) p = 1.0;
  else {
    p = fn->alpha[0];
    for (i=1;i<fn->na;i++)
      p = fn->alpha[i]+x*p;
  }
  if (!fn->nb) *y = p;
  else {
    q = fn->beta[0];
    for (i=1;i<fn->nb;i++)
      q = fn->beta[i]+x*q;
    *y = p/q;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateDerivative_Rational"
PetscErrorCode FNEvaluateDerivative_Rational(FN fn,PetscScalar x,PetscScalar *yp)
{
  PetscInt    i;
  PetscScalar p,q,pp,qp;

  PetscFunctionBegin;
  if (!fn->na) {
    p = 1.0;
    pp = 0.0;
  } else {
    p = fn->alpha[0];
    pp = 0.0;
    for (i=1;i<fn->na;i++) {
      pp = p+x*pp;
      p = fn->alpha[i]+x*p;
    }
  }
  if (!fn->nb) *yp = pp;
  else {
    q = fn->beta[0];
    qp = 0.0;
    for (i=1;i<fn->nb;i++) {
      qp = q+x*qp;
      q = fn->beta[i]+x*q;
    }
    *yp = (pp*q-p*qp)/(q*q);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNView_Rational"
PetscErrorCode FNView_Rational(FN fn,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (!fn->nb) {
      if (fn->na==1) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Constant: %3.1F\n",fn->alpha[0]);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  Polynomial: ");CHKERRQ(ierr);
        for (i=0;i<fn->na-1;i++) {
          ierr = PetscViewerASCIIPrintf(viewer,"%+3.1F*x^%1D",fn->alpha[i],fn->na-i-1);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"%+3.1F\n",fn->alpha[fn->na-1]);CHKERRQ(ierr);
      }
    } else if (!fn->na) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Inverse polinomial: 1 / (");CHKERRQ(ierr);
      for (i=0;i<fn->nb-1;i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%+3.1F*x^%1D",fn->beta[i],fn->nb-i-1);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"%+3.1F)\n",fn->beta[fn->nb-1]);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Rational function: (");CHKERRQ(ierr);
      for (i=0;i<fn->na-1;i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%+3.1F*x^%1D",fn->alpha[i],fn->na-i-1);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"%+3.1F) / (",fn->alpha[fn->na-1]);CHKERRQ(ierr);
      for (i=0;i<fn->nb-1;i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%+3.1F*x^%1D",fn->beta[i],fn->nb-i-1);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"%+3.1F)\n",fn->beta[fn->nb-1]);CHKERRQ(ierr);
    }
  }  
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "FNCreate_Rational"
PetscErrorCode FNCreate_Rational(FN fn)
{
  PetscFunctionBegin;
  fn->ops->evaluatefunction   = FNEvaluateFunction_Rational;
  fn->ops->evaluatederivative = FNEvaluateDerivative_Rational;
  fn->ops->view               = FNView_Rational;
  PetscFunctionReturn(0);
}
EXTERN_C_END

