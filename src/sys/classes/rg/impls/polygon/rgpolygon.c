/*
   Region defined by a set of vertices.

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

#define VERTMAX 30

typedef struct {
  PetscInt    n;         /* number of vertices */
  PetscScalar *vr,*vi;   /* array of vertices (vi not used in complex scalars) */
} RG_POLYGON;

#undef __FUNCT__
#define __FUNCT__ "RGPolygonSetVertices_Polygon"
static PetscErrorCode RGPolygonSetVertices_Polygon(RG rg,PetscInt n,PetscScalar *vr,PetscScalar *vi)
{
  PetscErrorCode ierr;
  PetscInt       i;
  RG_POLYGON     *ctx = (RG_POLYGON*)rg->data;

  PetscFunctionBegin;
  if (n<3) SETERRQ1(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_OUTOFRANGE,"At least 3 vertices required, you provided %s",n);
  if (n>VERTMAX) SETERRQ1(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_OUTOFRANGE,"Too many points, maximum allowed is %d",VERTMAX);
  if (ctx->n) {
    ierr = PetscFree(ctx->vr);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscFree(ctx->vi);CHKERRQ(ierr);
#endif
  }
  ctx->n = n;
  ierr = PetscMalloc1(n,&ctx->vr);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(n,&ctx->vi);CHKERRQ(ierr);
#endif
  for (i=0;i<n;i++) {
    ctx->vr[i] = vr[i];
#if !defined(PETSC_USE_COMPLEX)
    ctx->vi[i] = vi[i];
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGPolygonSetVertices"
/*@
   RGPolygonSetVertices - Sets the vertices that define the polygon region.

   Logically Collective on RG

   Input Parameters:
+  rg - the region context
.  n  - number of vertices
.  vr - array of vertices
-  vi - array of vertices (imaginary part)

   Options Database Keys:
+  -rg_polygon_vertices - Sets the vertices
-  -rg_polygon_verticesi - Sets the vertices (imaginary part)

   Notes:
   In the case of complex scalars, only argument vr is used, containing
   the complex vertices; the list of vertices can be provided in the
   command line with a comma-separated list of complex values
   [+/-][realnumber][+/-]realnumberi with no spaces.

   When PETSc is built with real scalars, the real and imaginary parts of
   the vertices must be provided in two separate arrays (or two lists in
   the command line).

   Level: advanced

.seealso: RGPolygonGetVertices()
@*/
PetscErrorCode RGPolygonSetVertices(RG rg,PetscInt n,PetscScalar *vr,PetscScalar *vi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  PetscValidLogicalCollectiveInt(rg,n,2);
  PetscValidPointer(vr,3);
#if !defined(PETSC_USE_COMPLEX)
  PetscValidPointer(vi,4);
#endif
  ierr = PetscTryMethod(rg,"RGPolygonSetVertices_C",(RG,PetscInt,PetscScalar*,PetscScalar*),(rg,n,vr,vi));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGPolygonGetVertices_Polygon"
static PetscErrorCode RGPolygonGetVertices_Polygon(RG rg,PetscInt *n,PetscScalar **vr,PetscScalar **vi)
{
  RG_POLYGON *ctx = (RG_POLYGON*)rg->data;

  PetscFunctionBegin;
  if (n)  *n  = ctx->n;
  if (vr) *vr = ctx->vr;
  if (vi) *vi = ctx->vi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGPolygonGetVertices"
/*@
   RGPolygonGetVertices - Gets the vertices that define the polygon region.

   Not Collective

   Input Parameter:
.  rg     - the region context

   Output Parameters:
+  n  - number of vertices
.  vr - array of vertices
-  vi - array of vertices (imaginary part)

   Notes:
   The returned arrays must NOT be freed by the calling application.

   Level: advanced

.seealso: RGPolygonSetVertices()
@*/
PetscErrorCode RGPolygonGetVertices(RG rg,PetscInt *n,PetscScalar **vr,PetscScalar **vi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rg,RG_CLASSID,1);
  ierr = PetscUseMethod(rg,"RGPolygonGetVertices_C",(RG,PetscInt*,PetscScalar**,PetscScalar**),(rg,n,vr,vi));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGView_Polygon"
PetscErrorCode RGView_Polygon(RG rg,PetscViewer viewer)
{
  PetscErrorCode ierr;
  RG_POLYGON     *ctx = (RG_POLYGON*)rg->data;
  PetscBool      isascii;
  PetscInt       i;
  char           str[50];

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"vertices: ");CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0;i<ctx->n;i++) {
#if defined(PETSC_USE_COMPLEX)
      ierr = SlepcSNPrintfScalar(str,50,ctx->vr[i],PETSC_FALSE);CHKERRQ(ierr);
#else
      if (ctx->vi[i]!=0.0) {
        ierr = PetscSNPrintf(str,50,"%g%+gi",(double)ctx->vr[i],(double)ctx->vi[i]);CHKERRQ(ierr);
      } else {
        ierr = PetscSNPrintf(str,50,"%g",(double)ctx->vr[i]);CHKERRQ(ierr);
      }
#endif
      ierr = PetscViewerASCIIPrintf(viewer,"%s%s",str,(i<ctx->n-1)?",":"");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGIsTrivial_Polygon"
PetscErrorCode RGIsTrivial_Polygon(RG rg,PetscBool *trivial)
{
  RG_POLYGON *ctx = (RG_POLYGON*)rg->data;

  PetscFunctionBegin;
  *trivial = PetscNot(ctx->n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGComputeContour_Polygon"
PetscErrorCode RGComputeContour_Polygon(RG rg,PetscInt n,PetscScalar *cr,PetscScalar *ci)
{
  RG_POLYGON  *ctx = (RG_POLYGON*)rg->data;
  PetscReal   length,h,d,rem=0.0;
  PetscInt    k=1,idx=ctx->n-1,i;
  PetscBool   ini=PETSC_FALSE;
  PetscScalar incr;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar inci;
#endif

  PetscFunctionBegin;
  if (!ctx->n) SETERRQ(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_WRONGSTATE,"No vertices have been set yet");
  length = SlepcAbsEigenvalue(ctx->vr[0]-ctx->vr[ctx->n-1],ctx->vi[0]-ctx->vi[ctx->n-1]);
  for (i=0;i<ctx->n-1;i++) length += SlepcAbsEigenvalue(ctx->vr[i]-ctx->vr[i+1],ctx->vi[i]-ctx->vi[i+1]);
  h = length/n;
  cr[0] = ctx->vr[0];
#if !defined(PETSC_USE_COMPLEX)
  ci[0] = ctx->vi[0];
#endif
  incr = ctx->vr[ctx->n-1]-ctx->vr[0];
#if !defined(PETSC_USE_COMPLEX)
  inci = ctx->vi[ctx->n-1]-ctx->vi[0];
#endif
  d = SlepcAbsEigenvalue(incr,inci);
  incr /= d;
#if !defined(PETSC_USE_COMPLEX)
  inci /= d;
#endif
  while (k<n) {
    if (ini) {
      incr = ctx->vr[idx]-ctx->vr[idx+1];
#if !defined(PETSC_USE_COMPLEX)
      inci = ctx->vi[idx]-ctx->vi[idx+1];
#endif
      d = SlepcAbsEigenvalue(incr,inci);
      incr /= d;
#if !defined(PETSC_USE_COMPLEX)
      inci /= d;
#endif
      if (rem+d>h) {
        cr[k] = ctx->vr[idx+1]+incr*(h-rem);
#if !defined(PETSC_USE_COMPLEX)
        ci[k] = ctx->vi[idx+1]+inci*(h-rem);
#endif
        k++;
        ini = PETSC_FALSE;
      } else {rem += d; idx--;}
    } else {
#if !defined(PETSC_USE_COMPLEX)
      rem = SlepcAbsEigenvalue(ctx->vr[idx]-cr[k-1],ctx->vi[idx]-ci[k-1]);
#else
      rem = PetscAbsScalar(ctx->vr[idx]-cr[k-1]);
#endif
      if (rem>h) {
        cr[k] = cr[k-1]+incr*h;
#if !defined(PETSC_USE_COMPLEX)
        ci[k] = ci[k-1]+inci*h;
#endif
        k++;
      } else {ini = PETSC_TRUE; idx--;}
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCheckInside_Polygon"
PetscErrorCode RGCheckInside_Polygon(RG rg,PetscReal px,PetscReal py,PetscInt *inout)
{
  RG_POLYGON *ctx = (RG_POLYGON*)rg->data;
  PetscReal  val,x[VERTMAX],y[VERTMAX];
  PetscBool  mx,my,nx,ny;
  PetscInt   i,j;

  PetscFunctionBegin;
  for (i=0;i<ctx->n;i++) {
#if defined(PETSC_USE_COMPLEX)
    x[i] = PetscRealPart(ctx->vr[i])-px;
    y[i] = PetscImaginaryPart(ctx->vr[i])-py;
#else
    x[i] = ctx->vr[i]-px;
    y[i] = ctx->vi[i]-py;
#endif
  }
  *inout = -1;
  for (i=0;i<ctx->n;i++) {
    j = (i+1)%ctx->n;
    mx = PetscNot(x[i]<0.0);
    nx = PetscNot(x[j]<0.0);
    my = PetscNot(y[i]<0.0);
    ny = PetscNot(y[j]<0.0);
    if (!((my||ny) && (mx||nx)) || (mx&&nx)) continue;
    if (((my && ny && (mx||nx)) && (!(mx&&nx)))) {
      *inout = -*inout;
      continue;
    }
    val = (y[i]*x[j]-x[i]*y[j])/(x[j]-x[i]);
    if (PetscAbs(val)<10*PETSC_MACHINE_EPSILON) {
      *inout = 0;
      PetscFunctionReturn(0);
    } else if (val>0.0) *inout = -*inout;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGSetFromOptions_Polygon"
PetscErrorCode RGSetFromOptions_Polygon(PetscOptionItems *PetscOptionsObject,RG rg)
{
  PetscErrorCode ierr;
  PetscScalar    array[VERTMAX];
  PetscInt       i,k;
  PetscBool      flg,flgi=PETSC_FALSE;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    arrayi[VERTMAX];
  PetscInt       ki;
#else
  PetscScalar    *arrayi=NULL;
#endif

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"RG Polygon Options");CHKERRQ(ierr);

  k = VERTMAX;
  for (i=0;i<k;i++) array[i] = 0;
  ierr = PetscOptionsScalarArray("-rg_polygon_vertices","Vertices of polygon","RGPolygonSetVertices",array,&k,&flg);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ki = VERTMAX;
  for (i=0;i<ki;i++) arrayi[i] = 0;
  ierr = PetscOptionsScalarArray("-rg_polygon_verticesi","Vertices of polygon (imaginary part)","RGPolygonSetVertices",arrayi,&ki,&flgi);CHKERRQ(ierr);
  if (ki!=k) SETERRQ2(PetscObjectComm((PetscObject)rg),PETSC_ERR_ARG_SIZ,"The number of real %D and imaginary %D parts do not match",k,ki);
#endif
  if (flg || flgi) {
    ierr = RGPolygonSetVertices(rg,k,array,arrayi);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGDestroy_Polygon"
PetscErrorCode RGDestroy_Polygon(RG rg)
{
  PetscErrorCode ierr;
  RG_POLYGON     *ctx = (RG_POLYGON*)rg->data;

  PetscFunctionBegin;
  if (ctx->n) {
    ierr = PetscFree(ctx->vr);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    ierr = PetscFree(ctx->vi);CHKERRQ(ierr);
#endif
  }
  ierr = PetscFree(rg->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGPolygonSetVertices_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGPolygonGetVertices_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RGCreate_Polygon"
PETSC_EXTERN PetscErrorCode RGCreate_Polygon(RG rg)
{
  RG_POLYGON     *polygon;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(rg,&polygon);CHKERRQ(ierr);
  rg->data = (void*)polygon;

  rg->ops->istrivial      = RGIsTrivial_Polygon;
  rg->ops->computecontour = RGComputeContour_Polygon;
  rg->ops->checkinside    = RGCheckInside_Polygon;
  rg->ops->setfromoptions = RGSetFromOptions_Polygon;
  rg->ops->view           = RGView_Polygon;
  rg->ops->destroy        = RGDestroy_Polygon;
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGPolygonSetVertices_C",RGPolygonSetVertices_Polygon);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)rg,"RGPolygonGetVertices_C",RGPolygonGetVertices_Polygon);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

