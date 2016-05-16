/*
   Ring region, similar to the ellipse but with a start and end angle,
   together with the width.

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
  PetscScalar center;     /* center of the ellipse */
  PetscReal   radius;     /* radius of the ellipse */
  PetscReal   vscale;     /* vertical scale of the ellipse */
  PetscReal   start_ang;  /* start angle */
  PetscReal   end_ang;    /* end angle */
  PetscReal   width;      /* ring width */
} RG_RING;

#undef __FUNCT__
#define __FUNCT__ "RGRingSetParameters_Ring"
static PetscErrorCode RGRingSetParameters_Ring(RG rg,PetscScalar center,PetscReal radius,PetscReal vscale,PetscReal start_ang,PetscReal end_ang,PetscReal width)
{
  RG_RING *ctx = (RG_RING*)rg->data;

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
#define __FUNCT__ "RGRingSetParameters"
/*@
   RGRingSetParameters - Sets the parameters defining the ring region.

   Logically Collective on RG

   Input Parameters:
+  rg        - the region context
.  center    - center of the ellipse
.  radius    - radius of the ellipse
.  vscale    - vertical scale of the ellipse
.  start_ang - the right-hand side angle
.  end_ang   - the left-hand side angle
-  width     - width of the ring

   Options Database Keys:
+  -rg_ring_center     - Sets the center
.  -rg_ring_radius     - Sets the radius
.  -rg_ring_vscale     - Sets the vertical scale
.  -rg_ring_startangle - Sets the right-hand side angle
.  -rg_ring_endangle   - Sets the left-hand side angle
-  -rg_ring_width      - Sets the width of the ring

   Notes:
   The values of center, radius and vscale have the same meaning as in the
   ellipse region. The startangle and endangle define the span of the ring
   (by default it is the whole ring), while the width is the separation
   between the two concentric ellipses (above and below the radius by
   width/2). The start and end angles are expressed as a fraction of the
   circumference: the allowed range is [0..1], with 0 corresponding to 0
   radians, 0.25 to pi/2 radians, and so on.

   In the case of complex scalars, a complex center can be provided in the
   command line with [+/-][realnumber][+/-]realnumberi with no spaces, e.g.
   -rg_ring_center 1.0+2.0i

   When PETSc is built with real scalars, the center is restricted to a real value.

   Level: advanced

.seealso: RGRingGetParameters()
@*/
PetscErrorCode RGRingSetParameters(RG rg,PetscScalar center,PetscReal radius,PetscReal vscale,PetscReal start_ang,PetscReal end_ang,PetscReal width)
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
  ierr = PetscTryMethod(rg,"RGRingSetParameters_C",(RG,PetscScalar,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal),(rg,center,radius,vscale,start_ang,end_ang,width));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGRingGetParameters_Ring"
static PetscErrorCode RGRingGetParameters_Ring(RG rg,PetscScalar *center,PetscReal *radius,PetscReal *vscale,PetscReal *start_ang,PetscReal *end_ang,PetscReal *width)
{
  RG_RING *ctx = (RG_RING*)rg->data;

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
#define __FUNCT__ "RGRingGetParameters"
/*@
   RGRingGetParameters - Gets the parameters that define the ring region.

   Not Collective

   Input Parameter:
.  rg     - the region context

   Output Parameters:
+  center    - center of the region
.  radius    - radius of the region
.  vscale    - vertical scale of the region
.  start_ang - the right-hand side angle
.  end_ang   - the left-hand side angle
-  width     - width of the ring

   Level: advanced

.seealso: RGRingSetParameters()
@*/
PetscErrorCode RGRingGetParameters(RG rg,PetscScalar *center,PetscReal *radius,PetscReal *vscale,PetscReal *start_ang,PetscReal *end_ang,PetscReal *width)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  ierr = PetscUseMethod(rg,"RGRingGetParameters_C",(RG,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*),(rg,center,radius,vscale,start_ang,end_ang,width));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGView_Ring"
PetscErrorCode RGView_Ring(RG rg,PetscViewer viewer)
{
  PetscErrorCode ierr;
  RG_RING        *ctx = (RG_RING*)rg->data;
  PetscBool      isascii;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = SlepcSNPrintfScalar(str,50,ctx->center,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"center: %s, radius: %g, vscale: %g, start angle: %g, end angle: %g, ring width: %g\n",str,RGShowReal(ctx->radius),RGShowReal(ctx->vscale),ctx->start_ang,ctx->end_ang,ctx->width);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGIsTrivial_Ring"
PetscErrorCode RGIsTrivial_Ring(RG rg,PetscBool *trivial)
{
  RG_RING *ctx = (RG_RING*)rg->data;

  PetscFunctionBegin;
  if (rg->complement) *trivial = PetscNot(ctx->radius);
  else *trivial = PetscNot(ctx->radius<PETSC_MAX_REAL);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGComputeContour_Ring"
PetscErrorCode RGComputeContour_Ring(RG rg,PetscInt n,PetscScalar *cr,PetscScalar *ci)
{
  RG_RING   *ctx = (RG_RING*)rg->data;
  PetscReal theta;
  PetscInt  i,n2=n/2;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    if (i < n2) {
      theta = ((ctx->end_ang-ctx->start_ang)*i/n2 + ctx->start_ang)*2.0*PETSC_PI;
#if defined(PETSC_USE_COMPLEX)
      cr[i] = ctx->center + (ctx->radius+ctx->width/2.0)*(PetscCosReal(theta)+ctx->vscale*PetscSinReal(theta)*PETSC_i);
#else
      cr[i] = ctx->center + (ctx->radius+ctx->width/2.0)*PetscCosReal(theta);
      ci[i] = (ctx->radius+ctx->width/2.0)*ctx->vscale*PetscSinReal(theta);
#endif
    } else {
      theta = ((ctx->end_ang-ctx->start_ang)*(n-i)/n2 + ctx->start_ang)*2.0*PETSC_PI;
#if defined(PETSC_USE_COMPLEX)
      cr[i] = ctx->center + (ctx->radius-ctx->width/2.0)*(PetscCosReal(theta)+ctx->vscale*PetscSinReal(theta)*PETSC_i);
#else
      cr[i] = ctx->center + (ctx->radius-ctx->width/2.0)*PetscCosReal(theta);
      ci[i] = (ctx->radius-ctx->width/2.0)*ctx->vscale*PetscSinReal(theta);
#endif
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCheckInside_Ring"
PetscErrorCode RGCheckInside_Ring(RG rg,PetscReal px,PetscReal py,PetscInt *inside)
{
  RG_RING   *ctx = (RG_RING*)rg->data;
  PetscReal dx,dy,r;

  PetscFunctionBegin;
  /* outer ellipse */
#if defined(PETSC_USE_COMPLEX)
  dx = (px-PetscRealPart(ctx->center))/(ctx->radius+ctx->width/2.0);
  dy = (py-PetscImaginaryPart(ctx->center))/(ctx->radius+ctx->width/2.0);
#else
  dx = (px-ctx->center)/(ctx->radius+ctx->width/2.0);
  dy = py/(ctx->radius+ctx->width/2.0);
#endif
  r = 1.0-dx*dx-(dy*dy)/(ctx->vscale*ctx->vscale);
  *inside = PetscSign(r);
  /* inner ellipse */
#if defined(PETSC_USE_COMPLEX)
  dx = (px-PetscRealPart(ctx->center))/(ctx->radius-ctx->width/2.0);
  dy = (py-PetscImaginaryPart(ctx->center))/(ctx->radius-ctx->width/2.0);
#else
  dx = (px-ctx->center)/(ctx->radius-ctx->width/2.0);
  dy = py/(ctx->radius-ctx->width/2.0);
#endif
  r = -1.0+dx*dx+(dy*dy)/(ctx->vscale*ctx->vscale);
  *inside *= PetscSign(r);
  /* check angles */
#if defined(PETSC_USE_COMPLEX)
  dx = (px-PetscRealPart(ctx->center));
  dy = (py-PetscImaginaryPart(ctx->center));
#else
  dx = px-ctx->center;
  dy = py;
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
  if (r>=ctx->start_ang && r<=ctx->end_ang && *inside == 1) *inside = 1;
  else *inside = -1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGSetFromOptions_Ring"
PetscErrorCode RGSetFromOptions_Ring(PetscOptionItems *PetscOptionsObject,RG rg)
{
  PetscErrorCode ierr;
  PetscScalar    s;
  PetscReal      r1,r2,r3,r4,r5;
  PetscBool      flg1,flg2,flg3,flg4,flg5,flg6;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"RG Ring Options");CHKERRQ(ierr);

  ierr = RGRingGetParameters(rg,&s,&r1,&r2,&r3,&r4,&r5);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-rg_ring_center","Center of ellipse","RGRingSetParameters",s,&s,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_ring_radius","Radius of ellipse","RGRingSetParameters",r1,&r1,&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_ring_vscale","Vertical scale of ellipse","RGRingSetParameters",r2,&r2,&flg3);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_ring_startangle","Right-hand side angle","RGRingSetParameters",r3,&r3,&flg4);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_ring_endangle","Left-hand side angle","RGRingSetParameters",r4,&r4,&flg5);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_ring_width","Width of ring","RGRingSetParameters",r5,&r5,&flg6);CHKERRQ(ierr);
  if (flg1 || flg2 || flg3 || flg4 || flg5 || flg6) {
    ierr = RGRingSetParameters(rg,s,r1,r2,r3,r4,r5);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGDestroy_Ring"
PetscErrorCode RGDestroy_Ring(RG rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(rg->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGRingSetParameters_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGRingGetParameters_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCreate_Ring"
PETSC_EXTERN PetscErrorCode RGCreate_Ring(RG rg)
{
  RG_RING        *ring;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(rg,&ring);CHKERRQ(ierr);
  ring->center    = 0.0;
  ring->radius    = 1.0;
  ring->vscale    = 1.0;
  ring->start_ang = 0.0;
  ring->end_ang   = 1.0;
  ring->width     = 0.1;
  rg->data = (void*)ring;

  rg->ops->istrivial      = RGIsTrivial_Ring;
  rg->ops->computecontour = RGComputeContour_Ring;
  rg->ops->checkinside    = RGCheckInside_Ring;
  rg->ops->setfromoptions = RGSetFromOptions_Ring;
  rg->ops->view           = RGView_Ring;
  rg->ops->destroy        = RGDestroy_Ring;
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGRingSetParameters_C",RGRingSetParameters_Ring);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGRingGetParameters_C",RGRingGetParameters_Ring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

