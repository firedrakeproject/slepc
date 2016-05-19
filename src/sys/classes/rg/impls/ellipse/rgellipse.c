/*
   Region enclosed in an ellipse (aligned with the coordinate axes).

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
} RG_ELLIPSE;

#undef __FUNCT__
#define __FUNCT__ "RGEllipseSetParameters_Ellipse"
static PetscErrorCode RGEllipseSetParameters_Ellipse(RG rg,PetscScalar center,PetscReal radius,PetscReal vscale)
{
  RG_ELLIPSE *ctx = (RG_ELLIPSE*)rg->data;

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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGEllipseSetParameters"
/*@
   RGEllipseSetParameters - Sets the parameters defining the ellipse region.

   Logically Collective on RG

   Input Parameters:
+  rg     - the region context
.  center - center of the ellipse
.  radius - radius of the ellipse
-  vscale - vertical scale of the ellipse

   Options Database Keys:
+  -rg_ellipse_center - Sets the center
.  -rg_ellipse_radius - Sets the radius
-  -rg_ellipse_vscale - Sets the vertical scale

   Notes:
   In the case of complex scalars, a complex center can be provided in the
   command line with [+/-][realnumber][+/-]realnumberi with no spaces, e.g.
   -rg_ellipse_center 1.0+2.0i

   When PETSc is built with real scalars, the center is restricted to a real value.

   Level: advanced

.seealso: RGEllipseGetParameters()
@*/
PetscErrorCode RGEllipseSetParameters(RG rg,PetscScalar center,PetscReal radius,PetscReal vscale)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidLogicalCollectiveScalar(rg,center,2);
  PetscValidLogicalCollectiveReal(rg,radius,3);
  PetscValidLogicalCollectiveReal(rg,vscale,4);
  ierr = PetscTryMethod(rg,"RGEllipseSetParameters_C",(RG,PetscScalar,PetscReal,PetscReal),(rg,center,radius,vscale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGEllipseGetParameters_Ellipse"
static PetscErrorCode RGEllipseGetParameters_Ellipse(RG rg,PetscScalar *center,PetscReal *radius,PetscReal *vscale)
{
  RG_ELLIPSE *ctx = (RG_ELLIPSE*)rg->data;

  PetscFunctionBegin;
  if (center) *center = ctx->center;
  if (radius) *radius = ctx->radius;
  if (vscale) *vscale = ctx->vscale;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGEllipseGetParameters"
/*@
   RGEllipseGetParameters - Gets the parameters that define the ellipse region.

   Not Collective

   Input Parameter:
.  rg     - the region context

   Output Parameters:
+  center - center of the region
.  radius - radius of the region
-  vscale - vertical scale of the region

   Level: advanced

.seealso: RGEllipseSetParameters()
@*/
PetscErrorCode RGEllipseGetParameters(RG rg,PetscScalar *center,PetscReal *radius,PetscReal *vscale)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  ierr = PetscUseMethod(rg,"RGEllipseGetParameters_C",(RG,PetscScalar*,PetscReal*,PetscReal*),(rg,center,radius,vscale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGView_Ellipse"
PetscErrorCode RGView_Ellipse(RG rg,PetscViewer viewer)
{
  PetscErrorCode ierr;
  RG_ELLIPSE     *ctx = (RG_ELLIPSE*)rg->data;
  PetscBool      isascii;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = SlepcSNPrintfScalar(str,50,ctx->center,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"center: %s, radius: %g, vscale: %g\n",str,RGShowReal(ctx->radius),RGShowReal(ctx->vscale));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGIsTrivial_Ellipse"
PetscErrorCode RGIsTrivial_Ellipse(RG rg,PetscBool *trivial)
{
  RG_ELLIPSE *ctx = (RG_ELLIPSE*)rg->data;

  PetscFunctionBegin;
  if (rg->complement) *trivial = PetscNot(ctx->radius);
  else *trivial = PetscNot(ctx->radius<PETSC_MAX_REAL);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGComputeContour_Ellipse"
PetscErrorCode RGComputeContour_Ellipse(RG rg,PetscInt n,PetscScalar *cr,PetscScalar *ci)
{
  RG_ELLIPSE *ctx = (RG_ELLIPSE*)rg->data;
  PetscReal  theta;
  PetscInt   i;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    theta = 2.0*PETSC_PI*(i+0.5)/n;
#if defined(PETSC_USE_COMPLEX)
    cr[i] = ctx->center + ctx->radius*(PetscCosReal(theta)+ctx->vscale*PetscSinReal(theta)*PETSC_i);
#else
    cr[i] = ctx->center + ctx->radius*PetscCosReal(theta);
    ci[i] = ctx->radius*ctx->vscale*PetscSinReal(theta);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCheckInside_Ellipse"
PetscErrorCode RGCheckInside_Ellipse(RG rg,PetscReal px,PetscReal py,PetscInt *inside)
{
  RG_ELLIPSE *ctx = (RG_ELLIPSE*)rg->data;
  PetscReal  dx,dy,r;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  dx = (px-PetscRealPart(ctx->center))/ctx->radius;
  dy = (py-PetscImaginaryPart(ctx->center))/ctx->radius;
#else
  dx = (px-ctx->center)/ctx->radius;
  dy = py/ctx->radius;
#endif
  r = 1.0-dx*dx-(dy*dy)/(ctx->vscale*ctx->vscale);
  *inside = PetscSign(r);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGSetFromOptions_Ellipse"
PetscErrorCode RGSetFromOptions_Ellipse(PetscOptionItems *PetscOptionsObject,RG rg)
{
  PetscErrorCode ierr;
  PetscScalar    s;
  PetscReal      r1,r2;
  PetscBool      flg1,flg2,flg3;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"RG Ellipse Options");CHKERRQ(ierr);

  ierr = RGEllipseGetParameters(rg,&s,&r1,&r2);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-rg_ellipse_center","Center of ellipse","RGEllipseSetParameters",s,&s,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_ellipse_radius","Radius of ellipse","RGEllipseSetParameters",r1,&r1,&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rg_ellipse_vscale","Vertical scale of ellipse","RGEllipseSetParameters",r2,&r2,&flg3);CHKERRQ(ierr);
  if (flg1 || flg2 || flg3) {
    ierr = RGEllipseSetParameters(rg,s,r1,r2);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGDestroy_Ellipse"
PetscErrorCode RGDestroy_Ellipse(RG rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(rg->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGEllipseSetParameters_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGEllipseGetParameters_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCreate_Ellipse"
PETSC_EXTERN PetscErrorCode RGCreate_Ellipse(RG rg)
{
  RG_ELLIPSE     *ellipse;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(rg,&ellipse);CHKERRQ(ierr);
  ellipse->center = 0.0;
  ellipse->radius = 1.0;
  ellipse->vscale = 1.0;
  rg->data = (void*)ellipse;

  rg->ops->istrivial      = RGIsTrivial_Ellipse;
  rg->ops->computecontour = RGComputeContour_Ellipse;
  rg->ops->checkinside    = RGCheckInside_Ellipse;
  rg->ops->setfromoptions = RGSetFromOptions_Ellipse;
  rg->ops->view           = RGView_Ellipse;
  rg->ops->destroy        = RGDestroy_Ellipse;
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGEllipseSetParameters_C",RGEllipseSetParameters_Ellipse);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGEllipseGetParameters_C",RGEllipseGetParameters_Ellipse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

