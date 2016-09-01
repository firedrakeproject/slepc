/*
   Interval of the real axis or more generally a (possibly open) rectangle
   of the complex plane.

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

#include <slepc/private/rgimpl.h>      /*I "slepcrg.h" I*/

typedef struct {
  PetscReal   a,b;     /* interval in the real axis */
  PetscReal   c,d;     /* interval in the imaginary axis */
} RG_INTERVAL;

#undef __FUNCT__
#define __FUNCT__ "RGIntervalSetEndpoints_Interval"
static PetscErrorCode RGIntervalSetEndpoints_Interval(RG rg,PetscReal a,PetscReal b,PetscReal c,PetscReal d)
{
  RG_INTERVAL *ctx = (RG_INTERVAL*)rg->data;

  PetscFunctionBegin;
  if (!a && !b && !c && !d) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_WRONG,"At least one argument must be nonzero");
  if (a==b && a) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_WRONG,"Badly defined interval, endpoints must be distinct (or both zero)");
  if (a>b) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_WRONG,"Badly defined interval, must be a<b");
  if (c==d && c) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_WRONG,"Badly defined interval, endpoints must be distinct (or both zero)");
  if (c>d) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_WRONG,"Badly defined interval, must be c<d");
#if !defined(PETSC_USE_COMPLEX)
  if (c!=-d) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_WRONG,"In real scalars the region must be symmetric wrt real axis");
#endif
  ctx->a = a;
  ctx->b = b;
  ctx->c = c;
  ctx->d = d;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGIntervalSetEndpoints"
/*@
   RGIntervalSetEndpoints - Sets the parameters defining the interval region.

   Logically Collective on RG

   Input Parameters:
+  rg  - the region context
.  a,b - endpoints of the interval in the real axis
-  c,d - endpoints of the interval in the imaginary axis

   Options Database Keys:
.  -rg_interval_endpoints - the four endpoints

   Note:
   The region is defined as [a,b]x[c,d]. Particular cases are an interval on
   the real axis (c=d=0), similar for the imaginary axis (a=b=0), the whole
   complex plane (a=-inf,b=inf,c=-inf,d=inf), and so on.

   Level: advanced

.seealso: RGIntervalGetEndpoints()
@*/
PetscErrorCode RGIntervalSetEndpoints(RG rg,PetscReal a,PetscReal b,PetscReal c,PetscReal d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidLogicalCollectiveReal(rg,a,2);
  PetscValidLogicalCollectiveReal(rg,b,3);
  PetscValidLogicalCollectiveReal(rg,c,4);
  PetscValidLogicalCollectiveReal(rg,d,5);
  ierr = PetscTryMethod(rg,"RGIntervalSetEndpoints_C",(RG,PetscReal,PetscReal,PetscReal,PetscReal),(rg,a,b,c,d));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGIntervalGetEndpoints_Interval"
static PetscErrorCode RGIntervalGetEndpoints_Interval(RG rg,PetscReal *a,PetscReal *b,PetscReal *c,PetscReal *d)
{
  RG_INTERVAL *ctx = (RG_INTERVAL*)rg->data;

  PetscFunctionBegin;
  if (a) *a = ctx->a;
  if (b) *b = ctx->b;
  if (c) *c = ctx->c;
  if (d) *d = ctx->d;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGIntervalGetEndpoints"
/*@
   RGIntervalGetEndpoints - Gets the parameters that define the interval region.

   Not Collective

   Input Parameter:
.  rg - the region context

   Output Parameters:
+  a,b - endpoints of the interval in the real axis
-  c,d - endpoints of the interval in the imaginary axis

   Level: advanced

.seealso: RGIntervalSetEndpoints()
@*/
PetscErrorCode RGIntervalGetEndpoints(RG rg,PetscReal *a,PetscReal *b,PetscReal *c,PetscReal *d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  ierr = PetscUseMethod(rg,"RGIntervalGetEndpoints_C",(RG,PetscReal*,PetscReal*,PetscReal*,PetscReal*),(rg,a,b,c,d));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGView_Interval"
PetscErrorCode RGView_Interval(RG rg,PetscViewer viewer)
{
  PetscErrorCode ierr;
  RG_INTERVAL    *ctx = (RG_INTERVAL*)rg->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"region: [%g,%g]x[%g,%g]\n",RGShowReal(ctx->a),RGShowReal(ctx->b),RGShowReal(ctx->c),RGShowReal(ctx->d));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGIsTrivial_Interval"
PetscErrorCode RGIsTrivial_Interval(RG rg,PetscBool *trivial)
{
  RG_INTERVAL *ctx = (RG_INTERVAL*)rg->data;

  PetscFunctionBegin;
  if (rg->complement) *trivial = (ctx->a==ctx->b && ctx->c==ctx->d)? PETSC_TRUE: PETSC_FALSE;
  else *trivial = (ctx->a<=-PETSC_MAX_REAL && ctx->b>=PETSC_MAX_REAL && ctx->c<=-PETSC_MAX_REAL && ctx->d>=PETSC_MAX_REAL)? PETSC_TRUE: PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGComputeContour_Interval"
PetscErrorCode RGComputeContour_Interval(RG rg,PetscInt n,PetscScalar *cr,PetscScalar *ci)
{
  RG_INTERVAL *ctx = (RG_INTERVAL*)rg->data;
  PetscInt    i,pt,idx,j;
  PetscReal   hr[4],hi[4],h,off,d[4];
  PetscScalar vr[4],vi[4];

  PetscFunctionBegin;
  if (!(ctx->a>-PETSC_MAX_REAL && ctx->b<PETSC_MAX_REAL && ctx->c>-PETSC_MAX_REAL && ctx->d<PETSC_MAX_REAL)) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_SUP,"Contour not defined in unbounded regions");
  if (ctx->a==ctx->b || ctx->c==ctx->d) {
    if (ctx->a==ctx->b) {hi[0] = (ctx->d-ctx->c)/(n-1); hr[0] = 0.0;}
    else {hr[0] = (ctx->b-ctx->a)/(n-1); hi[0] = 0.0;}
    for (i=0;i<n;i++) {
#if defined(PETSC_USE_COMPLEX)
      cr[i] = ctx->a+hr[0]*i + (ctx->c+hi[0]*i)*PETSC_i;
#else
      cr[i] = ctx->a+hr[0]*i; ci[i] = ctx->c+hi[0]*i;
#endif
    }
  } else {
    d[1] = d[3] = ctx->d-ctx->c; d[0] = d[2] = ctx->b-ctx->a;
    h = (2*(d[0]+d[1]))/n;
    vr[0] = ctx->a; vr[1] = ctx->b; vr[2] = ctx->b; vr[3] = ctx->a;
    vi[0] = ctx->c; vi[1] = ctx->c; vi[2] = ctx->d; vi[3] = ctx->d;
    hr[0] = h;   hr[1] = 0.0; hr[2] = -h;  hr[3] = 0.0;
    hi[0] = 0.0; hi[1] = h;   hi[2] = 0.0; hi[3] = -h;
    off = 0.0; idx = 0;
    for (i=0;i<4;i++) {
#if defined(PETSC_USE_COMPLEX)
      cr[idx] = vr[i]+off*(hr[i]/h)+ (vi[i]+off*(hi[i]/h))*PETSC_i; 
#else
      cr[idx] = vr[i]+off*(hr[i]/h); ci[idx]=vi[i]+off*(hi[i]/h);
#endif 
      idx++;
      pt = (d[i]-off)/h+1;
      for (j=1;j<pt && idx<n;j++) {
#if defined(PETSC_USE_COMPLEX)
        cr[idx] = cr[idx-1]+(hr[i]+hi[i]*PETSC_i);
#else
        cr[idx] = cr[idx-1]+hr[i]; ci[idx] = ci[idx-1]+hi[i];
#endif
        idx++;
      }
      off = off+pt*h-d[i];
      if (off>=d[i+1]) {off -= d[i+1]; i++;}
    }  
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCheckInside_Interval"
PetscErrorCode RGCheckInside_Interval(RG rg,PetscReal dx,PetscReal dy,PetscInt *inside)
{
  RG_INTERVAL *ctx = (RG_INTERVAL*)rg->data;

  PetscFunctionBegin;
  if (dx>ctx->a && dx<ctx->b) *inside = 1;
  else if (dx==ctx->a || dx==ctx->b) *inside = 0;
  else *inside = -1;
  if (*inside>=0) {
    if (dy>ctx->c && dy<ctx->d) ;
    else if (dy==ctx->c || dy==ctx->d) *inside = 0;
    else *inside = -1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGSetFromOptions_Interval"
PetscErrorCode RGSetFromOptions_Interval(PetscOptionItems *PetscOptionsObject,RG rg)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscInt       k;
  PetscReal      array[4]={0,0,0,0};

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"RG Interval Options");CHKERRQ(ierr);

  k = 4;
  ierr = PetscOptionsRealArray("-rg_interval_endpoints","Interval endpoints (four real values separated with a comma without spaces)","RGIntervalSetEndpoints",array,&k,&flg);CHKERRQ(ierr);
  if (flg) {
    if (k<2) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_SIZ,"Must pass at leat two values in -rg_interval_endpoints (comma-separated without spaces)");
    ierr = RGIntervalSetEndpoints(rg,array[0],array[1],array[2],array[3]);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGDestroy_Interval"
PetscErrorCode RGDestroy_Interval(RG rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(rg->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGIntervalSetEndpoints_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGIntervalGetEndpoints_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCreate_Interval"
PETSC_EXTERN PetscErrorCode RGCreate_Interval(RG rg)
{
  RG_INTERVAL    *interval;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(rg,&interval);CHKERRQ(ierr);
  interval->a = -PETSC_MAX_REAL;
  interval->b = PETSC_MAX_REAL;
  interval->c = -PETSC_MAX_REAL;
  interval->d = PETSC_MAX_REAL;
  rg->data = (void*)interval;

  rg->ops->istrivial      = RGIsTrivial_Interval;
  rg->ops->computecontour = RGComputeContour_Interval;
  rg->ops->checkinside    = RGCheckInside_Interval;
  rg->ops->setfromoptions = RGSetFromOptions_Interval;
  rg->ops->view           = RGView_Interval;
  rg->ops->destroy        = RGDestroy_Interval;
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGIntervalSetEndpoints_C",RGIntervalSetEndpoints_Interval);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGIntervalGetEndpoints_C",RGIntervalGetEndpoints_Interval);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

