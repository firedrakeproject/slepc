/*
   Arc region, similar to the ellipse but with a start and end angle for
   the arc, together with the width.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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
  PetscScalar center;     /* center of the ellipse */
  PetscReal   radius;     /* radius of the ellipse */
  PetscReal   vscale;     /* vertical scale of the ellipse */
  PetscReal   start_ang;  /* start angle of the arc */
  PetscReal   end_ang;    /* end angle of the arc */
  PetscReal   width;      /* arc width */
} RG_ARC;

#undef __FUNCT__
#define __FUNCT__ "RGArcSetParameters_Arc"
static PetscErrorCode RGArcSetParameters_Arc(RG rg,PetscScalar center,PetscReal radius,PetscReal vscale,PetscReal start_ang,PetscReal end_ang,PetscReal width)
{
  RG_ARC *ctx = (RG_ARC*)rg->data;

  PetscFunctionBegin;
  ctx->center = center;
  if (radius == PETSC_DEFAULT) {
    ctx->radius = 1.0;
  } else {
    if (radius<=0.0) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_OUTOFRANGE,"The radius argument must be > 0.0");
    ctx->radius = radius;
  }
  if (vscale<=0.0) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_OUTOFRANGE,"The vscale argument must be > 0.0");
  ctx->vscale = vscale;
  if (start_ang == PETSC_DEFAULT) {
    ctx->start_ang = 0.0;
  } else {
    if (start_ang<0.0) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_OUTOFRANGE,"The right-hand side angle argument must be >= 0.0");
    if (start_ang>1.0) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_OUTOFRANGE,"The right-hand side angle argument must be <= 1.0");
    ctx->start_ang = start_ang;
  }
  if (end_ang == PETSC_DEFAULT) {
    ctx->end_ang = 1.0;
  } else {
    if (end_ang<0.0) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_OUTOFRANGE,"The left-hand side angle argument must be >= 0.0");
    if (end_ang>1.0) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_OUTOFRANGE,"The left-hand side angle argument must be <= 1.0");
    ctx->end_ang = end_ang;
  }
  if (ctx->start_ang>ctx->end_ang) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_OUTOFRANGE,"The right-hand side angle argument must be smaller than left one");
  if (width == PETSC_DEFAULT) {
    ctx->width = 0.1;
  } else {
    if (width<=0.0) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_OUTOFRANGE,"The width argument must be > 0.0");
    ctx->width = width;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGArcSetParameters"
/*@
   RGArcSetParameters - Sets the parameters defining the arc region.

   Logically Collective on RG

   Input Parameters:
+  rg        - the region context
.  center    - center of the ellipse
.  radius    - radius of the ellipse
.  vscale    - vertical scale of the ellipse
.  start_ang - the right-hand side angle of the arc
.  end_ang   - the left-hand side angle of the arc
-  width     - width of the arc

   Options Database Keys:
+  -rg_arc_center     - Sets the center
.  -rg_arc_radius     - Sets the radius
.  -rg_arc_vscale     - Sets the vertical scale
.  -rg_arc_startangle - Sets the right-hand side angle of the arc 
.  -rg_arc_endangle   - Sets the left-hand side angle of the arc 
-  -rg_arc_width      - Sets the width of the arc

   Notes:
   The values of center, radius and vscale have the same meaning as in the
   ellipse region. The startangle and endangle define the span of the arc
   (by default it is the whole ring), while the width is the separation
   between the two concentric ellipses (above and below the radius by
   width/2). The start and end angles are expressed as a fraction of the
   circumference: the allowed range is [0..1], with 0 corresponding to 0
   radians, 0.25 to pi/2 radians, and so on.

   In the case of complex scalars, a complex center can be provided in the
   command line with [+/-][realnumber][+/-]realnumberi with no spaces, e.g.
   -rg_arc_center 1.0+2.0i

   When PETSc is built with real scalars, the center is restricted to a real value.

   Level: advanced

.seealso: RGArcGetParameters()
@*/
PetscErrorCode RGArcSetParameters(RG rg,PetscScalar center,PetscReal radius,PetscReal vscale,PetscReal start_ang,PetscReal end_ang,PetscReal width)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidLogicalCollectiveScalar(rg,center,2);
  PetscValidLogicalCollectiveReal(rg,radius,3);
  PetscValidLogicalCollectiveReal(rg,vscale,4);
  PetscValidLogicalCollectiveReal(rg,start_ang,5);
  PetscValidLogicalCollectiveReal(rg,end_ang,6);
  PetscValidLogicalCollectiveReal(rg,width,7);
  ierr = PetscTryMethod(rg,"RGArcSetParameters_C",(RG,PetscScalar,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal),(rg,center,radius,vscale,start_ang,end_ang,width));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGArcGetParameters_Arc"
static PetscErrorCode RGArcGetParameters_Arc(RG rg,PetscScalar *center,PetscReal *radius,PetscReal *vscale,PetscReal *start_ang,PetscReal *end_ang,PetscReal *width)
{
  RG_ARC *ctx = (RG_ARC*)rg->data;

  PetscFunctionBegin;
  if (center)    *center    = ctx->center;
  if (radius)    *radius    = ctx->radius;
  if (vscale)    *vscale    = ctx->vscale;
  if (start_ang) *start_ang = ctx->start_ang;
  if (end_ang)   *end_ang   = ctx->end_ang;
  if (width)     *width     = ctx->width;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGArcGetParameters"
/*@
   RGArcGetParameters - Gets the parameters that define the arc region.

   Not Collective

   Input Parameter:
.  rg     - the region context

   Output Parameters:
+  center    - center of the region
.  radius    - radius of the region
.  vscale    - vertical scale of the region
.  start_ang - the right-hand side angle of the arc
.  end_ang   - the left-hand side angle of the arc
-  width     - width of the arc

   Level: advanced

.seealso: RGArcSetParameters()
@*/
PetscErrorCode RGArcGetParameters(RG rg,PetscScalar *center,PetscReal *radius,PetscReal *vscale,PetscReal *start_ang,PetscReal *end_ang,PetscReal *width)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  ierr = PetscTryMethod(rg,"RGArcGetParameters_C",(RG,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*),(rg,center,radius,vscale,start_ang,end_ang,width));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGView_Arc"
PetscErrorCode RGView_Arc(RG rg,PetscViewer viewer)
{
  PetscErrorCode ierr;
  RG_ARC         *ctx = (RG_ARC*)rg->data;
  PetscBool      isascii;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = SlepcSNPrintfScalar(str,50,ctx->center,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"center: %s, radius: %g, vscale: %g, start angle: %g, end angle: %g, arc width: %g\n",str,RGShowReal(ctx->radius),RGShowReal(ctx->vscale),ctx->start_ang,ctx->end_ang,ctx->width);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGIsTrivial_Arc"
PetscErrorCode RGIsTrivial_Arc(RG rg,PetscBool *trivial)
{
  RG_ARC *ctx = (RG_ARC*)rg->data;

  PetscFunctionBegin;
  if (rg->complement) *trivial = (ctx->radius==0.0)? PETSC_TRUE: PETSC_FALSE;
  else *trivial = (ctx->radius>=PETSC_MAX_REAL)? PETSC_TRUE: PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGComputeContour_Arc"
PetscErrorCode RGComputeContour_Arc(RG rg,PetscInt n,PetscScalar *cr,PetscScalar *ci)
{
  RG_ARC      *ctx = (RG_ARC*)rg->data;
  PetscReal   theta,theta2;
  PetscInt    i;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    theta = PETSC_PI*(i+0.5)/n;
    theta2 = (ctx->start_ang*2.0+(ctx->end_ang-ctx->start_ang)*(PetscCosReal(theta)+1.0))*PETSC_PI;
#if defined(PETSC_USE_COMPLEX)
    cr[i] = ctx->center + ctx->radius*(PetscCosReal(theta2)+ctx->vscale*PetscSinReal(theta2)*PETSC_i);
#else
    cr[i] = ctx->center + ctx->radius*PetscCosReal(theta2);
    ci[i] = ctx->radius*ctx->vscale*PetscSinReal(theta2);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCheckInside_Arc"
PetscErrorCode RGCheckInside_Arc(RG rg,PetscInt n,PetscScalar *ar,PetscScalar *ai,PetscInt *inside)
{
  RG_ARC      *ctx = (RG_ARC*)rg->data;
  PetscInt    i;
  PetscReal   dx,dy,r;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar d;
#endif

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    /* outer ellipse */
#if defined(PETSC_USE_COMPLEX)
    d = ar[i]-ctx->center;
    dx = PetscRealPart(d)/(ctx->radius+ctx->width/2.0);
    dy = PetscImaginaryPart(d)/(ctx->radius+ctx->width/2.0);
#else
    dx = (ar[i]-ctx->center)/(ctx->radius+ctx->width/2.0);
    dy = ai[i]/(ctx->radius+ctx->width/2.0);
#endif
    r = 1.0-dx*dx-(dy*dy)/(ctx->vscale*ctx->vscale);
    inside[i] = PetscSign(r);
    /* inner ellipse */
#if defined(PETSC_USE_COMPLEX)
    dx = PetscRealPart(d)/(ctx->radius-ctx->width/2.0);
    dy = PetscImaginaryPart(d)/(ctx->radius-ctx->width/2.0);
#else
    dx = (ar[i]-ctx->center)/(ctx->radius-ctx->width/2.0);
    dy = ai[i]/(ctx->radius-ctx->width/2.0);
#endif
    r = -1.0+dx*dx+(dy*dy)/(ctx->vscale*ctx->vscale);
    inside[i] *= PetscSign(r);
    /* check angles */
#if defined(PETSC_USE_COMPLEX)
    dx = PetscRealPart(d);
    dy = PetscImaginaryPart(d);
#else
    dx = (ar[i]-ctx->center);
    dy = ai[i];
#endif
    if (dx == 0) {
      if (dy == 0) r = -1;
      else if (dy > 0) r = 0.25;
      else r = 0.75;
    } else if (dx > 0) {
      r = PetscAtanReal((dy/ctx->vscale)/dx);
      if (dy >= 0) r /= 2*PETSC_PI;
      else r = r/(2*PETSC_PI)+1;
    } else r = PetscAtanReal((dy/ctx->vscale)/dx)/(2*PETSC_PI)+0.5;
    if (r>=ctx->start_ang && r<=ctx->end_ang && inside[i] == 1) inside[i] = 1;
    else inside[i] = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGSetFromOptions_Arc"
PetscErrorCode RGSetFromOptions_Arc(PetscOptions *PetscOptionsObject,RG rg)
{
  PetscErrorCode ierr;
  PetscScalar    s;
  PetscReal      r1,r2,r3,r4,r5;
  PetscBool      flg1,flg2,flg3,flg4,flg5,flg6;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"RG Arc Options");CHKERRQ(ierr);

  ierr = RGArcGetParameters(rg,&s,&r1,&r2,&r3,&r4,&r5);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-rg_arc_center","Center of ellipse","RGArcSetParameters",s,&s,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_arc_radius","Radius of ellipse","RGArcSetParameters",r1,&r1,&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_arc_vscale","Vertical scale of ellipse","RGArcSetParameters",r2,&r2,&flg3);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_arc_startangle","Right-hand side angle of the arc","RGArcSetParameters",r3,&r3,&flg4);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_arc_endangle","Left-hand side angle of the arc","RGArcSetParameters",r4,&r4,&flg5);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_arc_width","Width of arc","RGArcSetParameters",r5,&r5,&flg6);CHKERRQ(ierr);
  if (flg1 || flg2 || flg3 || flg4 || flg5 || flg6) {
    ierr = RGArcSetParameters(rg,s,r1,r2,r3,r4,r5);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGDestroy_Arc"
PetscErrorCode RGDestroy_Arc(RG rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(rg->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGArcSetParameters_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGArcGetParameters_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCreate_Arc"
PETSC_EXTERN PetscErrorCode RGCreate_Arc(RG rg)
{
  RG_ARC         *arc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(rg,&arc);CHKERRQ(ierr);
  arc->center    = 0.0;
  arc->radius    = 1.0;
  arc->vscale    = 1.0;
  arc->start_ang = 0.0;
  arc->end_ang   = 1.0;
  arc->width     = 0.1;
  rg->data = (void*)arc;

  rg->ops->istrivial      = RGIsTrivial_Arc;
  rg->ops->computecontour = RGComputeContour_Arc;
  rg->ops->checkinside    = RGCheckInside_Arc;
  rg->ops->setfromoptions = RGSetFromOptions_Arc;
  rg->ops->view           = RGView_Arc;
  rg->ops->destroy        = RGDestroy_Arc;
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGArcSetParameters_C",RGArcSetParameters_Arc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGArcGetParameters_C",RGArcGetParameters_Arc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

