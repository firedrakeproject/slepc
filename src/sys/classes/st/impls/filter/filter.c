/*
    Filter spectral transformation, to encapsulate polynomial filters

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

#include <slepc/private/stimpl.h>
#include "./filter.h"

PetscErrorCode STApply_Filter(ST st,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode STSetUp_Filter(ST st)
{
  PetscErrorCode ierr;
  ST_FILTER      *ctx = (ST_FILTER*)st->data;

  PetscFunctionBegin;
  if (st->nmat>1) SETERRQ(PetscObjectComm((PetscObject)st),1,"Only implemented for standard eigenvalue problem");
  if (ctx->intb >= PETSC_MAX_REAL && ctx->inta <= PETSC_MIN_REAL) SETERRQ(PetscObjectComm((PetscObject)st),1,"Must pass an interval with STFilterSetInterval()");
  if (!ctx->polyDegree) ctx->polyDegree = 100;
  ierr = MatCreateVecs(st->A[0],&ctx->w,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode STSetFromOptions_Filter(PetscOptionItems *PetscOptionsObject,ST st)
{
  PetscErrorCode ierr;
  PetscReal      array[2]={0,0};
  PetscInt       k;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"ST Filter Options");CHKERRQ(ierr);

    k = 2;
    ierr = PetscOptionsRealArray("-st_filter_interval","Interval containing the desired eigenvalues (two real values separated with a comma without spaces)","STFilterSetInterval",array,&k,&flg);CHKERRQ(ierr);
    if (flg) {
      if (k<2) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_SIZ,"Must pass two values in -st_filter_interval (comma-separated without spaces)");
      ierr = STFilterSetInterval(st,array[0],array[1]);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-st_filter_degree","Degree of filter polynomial","STFilterSetDegree",100,&k,&flg);CHKERRQ(ierr);
    if (flg) { ierr = STFilterSetDegree(st,k);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode STFilterSetInterval_Filter(ST st,PetscReal inta,PetscReal intb)
{
  ST_FILTER *ctx = (ST_FILTER*)st->data;

  PetscFunctionBegin;
  if (inta>intb) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_WRONG,"Badly defined interval, must be inta<intb");
  if (ctx->inta != inta || ctx->intb != intb) {
    ctx->inta = inta;
    ctx->intb = intb;
    st->state = ST_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   STFilterSetInterval - Defines the interval containing the desired eigenvalues.

   Logically Collective on ST

   Input Parameters:
+  st   - the spectral transformation context
.  inta - left end of the interval
-  intb - right end of the interval

   Options Database Key:
.  -st_filter_interval <a,b> - set [a,b] as the interval of interest

   Level: intermediate

   Notes:
   The filter will be configured to emphasize eigenvalues contained in the given
   interval, and damp out eigenvalues outside it. If the interval is open, then
   the filter is low- or high-pass, otherwise it is mid-pass.

   Common usage is to set the interval in EPS with EPSSetInterval().

.seealso: STFilterGetInterval(), EPSSetInterval()
@*/
PetscErrorCode STFilterSetInterval(ST st,PetscReal inta,PetscReal intb)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveReal(st,inta,2);
  PetscValidLogicalCollectiveReal(st,intb,3);
  ierr = PetscTryMethod(st,"STFilterSetInterval_C",(ST,PetscReal,PetscReal),(st,inta,intb));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode STFilterGetInterval_Filter(ST st,PetscReal *inta,PetscReal *intb)
{
  ST_FILTER *ctx = (ST_FILTER*)st->data;

  PetscFunctionBegin;
  if (inta) *inta = ctx->inta;
  if (intb) *intb = ctx->intb;
  PetscFunctionReturn(0);
}

/*@
   STFilterGetInterval - Gets the interval containing the desired eigenvalues.

   Not Collective

   Input Parameter:
.  st  - the spectral transformation context

   Output Parameter:
+  inta - left end of the interval
-  intb - right end of the interval

   Level: intermediate

.seealso: STFilterSetInterval()
@*/
PetscErrorCode STFilterGetInterval(ST st,PetscReal *inta,PetscReal *intb)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  ierr = PetscUseMethod(st,"STFilterGetInterval_C",(ST,PetscReal*,PetscReal*),(st,inta,intb));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode STFilterSetDegree_Filter(ST st,PetscInt deg)
{
  ST_FILTER *ctx = (ST_FILTER*)st->data;

  PetscFunctionBegin;
  if (deg == PETSC_DEFAULT || deg == PETSC_DECIDE) {
    ctx->polyDegree = 0;
    st->state = ST_STATE_INITIAL;
  } else {
    if (deg<=0) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_OUTOFRANGE,"Illegal value of degree. Must be > 0");
    if (ctx->polyDegree != deg) {
      ctx->polyDegree = deg;
      st->state = ST_STATE_INITIAL;
    }
  }
  PetscFunctionReturn(0);
}

/*@
   STFilterSetDegree - Sets the degree of the filter polynomial.

   Logically Collective on ST

   Input Parameters:
+  st  - the spectral transformation context
-  deg - polynomial degree

   Options Database Key:
.  -st_filter_degree <deg> - sets the degree of the filter polynomial

   Level: intermediate

.seealso: STFilterGetDegree()
@*/
PetscErrorCode STFilterSetDegree(ST st,PetscInt deg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveInt(st,deg,2);
  ierr = PetscTryMethod(st,"STFilterSetDegree_C",(ST,PetscInt),(st,deg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode STFilterGetDegree_Filter(ST st,PetscInt *deg)
{
  ST_FILTER *ctx = (ST_FILTER*)st->data;

  PetscFunctionBegin;
  *deg = ctx->polyDegree;
  PetscFunctionReturn(0);
}

/*@
   STFilterGetDegree - Gets the degree of the filter polynomial.

   Not Collective

   Input Parameter:
.  st  - the spectral transformation context

   Output Parameter:
.  deg - polynomial degree

   Level: intermediate

.seealso: STFilterSetDegree()
@*/
PetscErrorCode STFilterGetDegree(ST st,PetscInt *deg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(deg,2);
  ierr = PetscUseMethod(st,"STFilterGetDegree_C",(ST,PetscInt*),(st,deg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode STView_Filter(ST st,PetscViewer viewer)
{
  PetscErrorCode ierr;
  ST_FILTER      *ctx = (ST_FILTER*)st->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Filter: interval of desired eigenvalues is [%g,%g]\n",(double)ctx->inta,(double)ctx->intb);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Filter: degree of filter polynomial is %D\n",ctx->polyDegree);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STReset_Filter(ST st)
{
  PetscErrorCode ierr;
  ST_FILTER      *ctx = (ST_FILTER*)st->data;

  PetscFunctionBegin;
  ierr = VecDestroy(&ctx->w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode STDestroy_Filter(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(st->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STFilterSetInterval_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STFilterGetInterval_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STFilterSetDegree_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STFilterGetDegree_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode STCreate_Filter(ST st)
{
  PetscErrorCode ierr;
  ST_FILTER      *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(st,&ctx);CHKERRQ(ierr);
  st->data = (void*)ctx;

  ctx->inta               = PETSC_MIN_REAL;
  ctx->intb               = PETSC_MAX_REAL;
  ctx->polyDegree         = 0;

  st->ops->apply           = STApply_Filter;
  st->ops->setfromoptions  = STSetFromOptions_Filter;
  st->ops->setup           = STSetUp_Filter;
  st->ops->destroy         = STDestroy_Filter;
  st->ops->reset           = STReset_Filter;
  st->ops->view            = STView_Filter;

  ierr = PetscObjectComposeFunction((PetscObject)st,"STFilterSetInterval_C",STFilterSetInterval_Filter);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STFilterGetInterval_C",STFilterGetInterval_Filter);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STFilterSetDegree_C",STFilterSetDegree_Filter);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)st,"STFilterGetDegree_C",STFilterGetDegree_Filter);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
