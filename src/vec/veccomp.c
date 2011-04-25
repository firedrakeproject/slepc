/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include "private/vecimpl.h"          /*I  "petscvec.h"   I*/
#include "veccomp_private.h"
#include <slepcvec.h>

typedef struct {
  PetscInt      n,        /* number of active subvectors */
                N,        /* virtual global size */
                lN,       /* virtual local size */
                friends;  /* number of vectors sharing this structure */
} Vec_Comp_N;

typedef struct {
  Vec           *x;       /* the vectors */
  PetscInt      nx;       /* number of available subvectors */
  Vec_Comp_N    *n;       /* structure shared by friend vectors */
} Vec_Comp;

#if defined(PETSC_USE_DEBUG)
#define PetscValidVecComp(y) \
  if (((Vec_Comp*)(y)->data)->nx < ((Vec_Comp*)(y)->data)->n->n) { \
    return PetscError(((PetscObject)(*((Vec_Comp*)(y)->data)->x))->comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,__SDIR__,PETSC_ERR_ARG_WRONG,PETSC_ERROR_INITIAL,"Invalid number of subvectors required!");}
#else
#define PetscValidVecComp(y)
#endif

static PetscErrorCode VecCreate_Comp_Private(Vec v, Vec *x, PetscInt nx,
                                             PetscBool x_to_me, Vec_Comp_N* n);

#include "veccomp0.h"

#define __WITH_MPI__
#include "veccomp0.h"

#undef __FUNCT__  
#define __FUNCT__ "VecRegister_Comp"
PetscErrorCode VecRegister_Comp(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = VecRegisterDynamic(VECCOMP, path, "VecCreate_Comp", VecCreate_Comp);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecDestroy_Comp"
PetscErrorCode VecDestroy_Comp(Vec v)
{
  Vec_Comp       *vs = (Vec_Comp*)v->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(v);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  for(i=0; i<vs->nx; i++) {
    ierr = VecDestroy(&vs->x[i]); CHKERRQ(ierr);
  }
  if(--vs->n->friends <= 0) {
    ierr = PetscFree(vs->n); CHKERRQ(ierr);
  }
  ierr = PetscFree(vs->x); CHKERRQ(ierr);
  ierr = PetscFree(vs); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


static struct _VecOps DvOps = {VecDuplicate_Comp, /* 1 */
            VecDuplicateVecs_Default,
            VecDestroyVecs_Default,
            VecDot_Comp_MPI,
            VecMDot_Comp_MPI,
            VecNorm_Comp_MPI,
            0,
            0,
            VecScale_Comp,
            VecCopy_Comp, /* 10 */
            VecSet_Comp,
            VecSwap_Comp,
            VecAXPY_Comp,
            VecAXPBY_Comp,
            VecMAXPY_Comp,
            VecAYPX_Comp,
            VecWAXPY_Comp,
            VecAXPBYPCZ_Comp,
            VecPointwiseMult_Comp,
            VecPointwiseDivide_Comp, 
            0, /* 20 */
            0,0,
            0 /*VecGetArray_Seq*/,
            VecGetSize_Comp,
            VecGetLocalSize_Comp,
            0/*VecRestoreArray_Seq*/,
            VecMax_Comp,
            VecMin_Comp,
            VecSetRandom_Comp,
            0, /* 30 */
            0,
            VecDestroy_Comp,
            VecView_Comp,
            0/*VecPlaceArray_Seq*/,
            0/*VecReplaceArray_Seq*/,
            VecDot_Comp_Seq,
            0,
            VecNorm_Comp_Seq,
            VecMDot_Comp_Seq,
            0, /* 40 */
            0,
            VecReciprocal_Comp,
            VecConjugate_Comp,
            0,0,
            0/*VecResetArray_Seq*/,
            0,
            VecMaxPointwiseDivide_Comp,
            VecPointwiseMax_Comp,
            VecPointwiseMaxAbs_Comp,
            VecPointwiseMin_Comp,
            0,
            VecSqrtAbs_Comp,
            VecAbs_Comp,
            VecExp_Comp,
            VecLog_Comp,
            0/*VecShift_Comp*/,
            0,
            0,
            0,
            VecDotNorm2_Comp_MPI
          };

#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Comp_Private"
static PetscErrorCode VecCreate_Comp_Private(Vec v, Vec *x, PetscInt nx,
                                             PetscBool x_to_me, Vec_Comp_N *n)
{
  Vec_Comp        *s;
  PetscErrorCode  ierr;
  PetscInt        N=0, lN=0, i, k;

  PetscFunctionBegin;

  /* Allocate a new Vec_Comp */
  if (v->data) { ierr = PetscFree(v->data); CHKERRQ(ierr); }
  ierr = PetscNewLog(v, Vec_Comp, &s); CHKERRQ(ierr);
  ierr = PetscMemcpy(v->ops, &DvOps, sizeof(DvOps)); CHKERRQ(ierr);
  v->data  = (void*)s;
  v->petscnative     = PETSC_FALSE;

  /* Allocate the array of Vec, if it is needed to be done */
  if (x_to_me != PETSC_TRUE) {
    ierr = PetscMalloc(sizeof(Vec)*nx, &s->x); CHKERRQ(ierr);
    ierr = PetscMemcpy(s->x, x, sizeof(Vec)*nx); CHKERRQ(ierr);
  } else
    s->x = x;

  s->nx = nx;
  for(i=0; i<nx; i++) {
    ierr = VecGetSize(x[i], &k); CHKERRQ(ierr); N+= k;
    ierr = VecGetLocalSize(x[i], &k); CHKERRQ(ierr); lN+= k;
  }
 
  /* Allocate the shared structure, if it is not given */
  if (!n) {
    ierr = PetscNewLog(v, Vec_Comp_N, &n); CHKERRQ(ierr);
    s->n = n;
    n->n = nx;
    n->N = N;
    n->lN = lN;
    n->friends = 1;
  }

  /* If not, check in the vector in the shared structure */
  else {
    s->n = n;
    s->n->friends++;
    s->n->n = nx;
  }

  /* Set the virtual sizes as the real sizes of the vector */
  ierr = VecSetSizes(v, s->n->lN, s->n->N); CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)v,VECCOMP); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "VecCreate_Comp"
PetscErrorCode VecCreate_Comp(Vec V)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = VecCreate_Comp_Private(V, PETSC_NULL, 0, PETSC_FALSE, PETSC_NULL);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "VecCreateComp"
PetscErrorCode VecCreateComp(MPI_Comm comm, PetscInt *Nx,
                                                PetscInt n, const VecType t,
                                                Vec Vparent, Vec *V)
{
  PetscErrorCode  ierr;
  Vec             *x;
  PetscInt        i;

  PetscFunctionBegin;

  ierr = VecCreate(comm, V); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(Vec)*n, &x); CHKERRQ(ierr);
  for(i=0; i<n; i++) {
    ierr = VecCreate(comm, &x[i]); CHKERRQ(ierr);
    ierr = VecSetSizes(x[i], PETSC_DECIDE, Nx[i]); CHKERRQ(ierr);
    ierr = VecSetType(x[i], t); CHKERRQ(ierr);
  }
  ierr = VecCreate_Comp_Private(*V, x, n, PETSC_TRUE,
                           Vparent?((Vec_Comp*)Vparent->data)->n:PETSC_NULL);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecCreateCompWithVecs"
PetscErrorCode VecCreateCompWithVecs(Vec *x, PetscInt n,
                                                        Vec Vparent, Vec *V)
{
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;

  ierr = VecCreate(((PetscObject)x[0])->comm, V); CHKERRQ(ierr);
  for(i=0; i<n; i++) {
    ierr = PetscObjectReference((PetscObject)x[i]); CHKERRQ(ierr);
  }
  ierr = VecCreate_Comp_Private(*V, x, n, PETSC_FALSE,
                           Vparent?((Vec_Comp*)Vparent->data)->n:PETSC_NULL);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_Comp"
PetscErrorCode VecDuplicate_Comp(Vec win, Vec *V)
{
  PetscErrorCode  ierr;
  Vec             *x;
  PetscInt        i;
  Vec_Comp        *s = (Vec_Comp*)win->data;

  PetscFunctionBegin;

  PetscValidVecComp(win);

  ierr = VecCreate(((PetscObject)win)->comm, V); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(Vec)*s->nx, &x); CHKERRQ(ierr);
  for(i=0; i<s->nx; i++) {
    ierr = VecDuplicate(s->x[i], &x[i]); CHKERRQ(ierr);
  }
  ierr = VecCreate_Comp_Private(*V, x, s->nx, PETSC_TRUE, s->n); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecCompGetVecs"
PetscErrorCode VecCompGetVecs(Vec win, const Vec **x, PetscInt *n)
{
  Vec_Comp        *s = (Vec_Comp*)win->data;

  PetscFunctionBegin;

  PetscValidVecComp(win);

  if(x) *x = s->x;
  if(n) *n = s->nx;

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecCompSetVecs"
PetscErrorCode VecCompSetVecs(Vec win, Vec *x, PetscInt n)
{
  Vec_Comp        *s = (Vec_Comp*)win->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  PetscValidVecComp(win);

  if(x) {
    if (n > s->nx) {
      ierr = PetscFree(s->x); CHKERRQ(ierr);
      ierr = PetscMalloc(sizeof(Vec)*n, &s->x); CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(s->x, x, sizeof(Vec)*n); CHKERRQ(ierr);
    s->nx = n;
  }
  s->n->n = n;

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecAXPY_Comp"
PetscErrorCode VecAXPY_Comp(Vec v, PetscScalar alpha, Vec w)
{
  PetscErrorCode  ierr;
  Vec_Comp        *vs = (Vec_Comp*)v->data,
                  *ws = (Vec_Comp*)w->data;
  PetscInt        i;

  PetscFunctionBegin;

  PetscValidVecComp(v);
  PetscValidVecComp(w);

  for(i=0; i<vs->n->n; i++) {
    ierr = VecAXPY(vs->x[i], alpha, ws->x[i]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecAYPX_Comp"
PetscErrorCode VecAYPX_Comp(Vec v, PetscScalar alpha, Vec w)
{
  PetscErrorCode  ierr;
  Vec_Comp        *vs = (Vec_Comp*)v->data,
                  *ws = (Vec_Comp*)w->data;
  PetscInt        i;

  PetscFunctionBegin;

  PetscValidVecComp(v);
  PetscValidVecComp(w);

  for(i=0; i<vs->n->n; i++) {
    ierr = VecAYPX(vs->x[i], alpha, ws->x[i]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecAXPBY_Comp"
PetscErrorCode VecAXPBY_Comp(Vec v, PetscScalar alpha, PetscScalar beta, Vec w)
{
  PetscErrorCode  ierr;
  Vec_Comp        *vs = (Vec_Comp*)v->data,
                  *ws = (Vec_Comp*)w->data;
  PetscInt        i;

  PetscFunctionBegin;

  PetscValidVecComp(v);
  PetscValidVecComp(w);

  for(i=0; i<vs->n->n; i++) {
    ierr = VecAXPBY(vs->x[i], alpha, beta, ws->x[i]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecMAXPY_Comp"
PetscErrorCode VecMAXPY_Comp(Vec v, PetscInt n, const PetscScalar *alpha,
                             Vec *w)
{
  PetscErrorCode  ierr;
  Vec_Comp        *vs = (Vec_Comp*)v->data;
  Vec             *wx;
  PetscInt        i, j;

  PetscFunctionBegin;

  PetscValidVecComp(v);
  for(i=0; i<n; i++) PetscValidVecComp(w[i]);

  ierr = PetscMalloc(sizeof(Vec)*n, &wx); CHKERRQ(ierr);

  for(j=0; j<vs->n->n; j++) {
    for(i=0; i<n; i++) wx[i] = ((Vec_Comp*)w[i]->data)->x[j];
    ierr = VecMAXPY(vs->x[j], n, alpha, wx); CHKERRQ(ierr);
  }

  ierr = PetscFree(wx); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecWAXPY_Comp"
PetscErrorCode VecWAXPY_Comp(Vec v, PetscScalar alpha, Vec w, Vec z)
{
  PetscErrorCode  ierr;
  Vec_Comp        *vs = (Vec_Comp*)v->data,
                  *ws = (Vec_Comp*)w->data,
                  *zs = (Vec_Comp*)z->data;
  PetscInt        i;

  PetscFunctionBegin;

  PetscValidVecComp(v);
  PetscValidVecComp(w);
  PetscValidVecComp(z);

  for(i=0; i<vs->n->n; i++) {
    ierr = VecWAXPY(vs->x[i], alpha, ws->x[i], zs->x[i]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecAXPBYPCZ_Comp"
PetscErrorCode VecAXPBYPCZ_Comp(Vec v, PetscScalar alpha, PetscScalar beta,
                                PetscScalar gamma, Vec w, Vec z)
{
  PetscErrorCode  ierr;
  Vec_Comp        *vs = (Vec_Comp*)v->data,
                  *ws = (Vec_Comp*)w->data,
                  *zs = (Vec_Comp*)z->data;
  PetscInt        i;

  PetscFunctionBegin;

  PetscValidVecComp(v);
  PetscValidVecComp(w);
  PetscValidVecComp(z);

  for(i=0; i<vs->n->n; i++) {
    ierr = VecAXPBYPCZ(vs->x[i], alpha, beta, gamma, ws->x[i], zs->x[i]);
    CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecGetSize_Comp"
PetscErrorCode VecGetSize_Comp(Vec v, PetscInt *size)
{
  Vec_Comp        *vs = (Vec_Comp*)v->data;

  PetscFunctionBegin;

  PetscValidVecComp(v);

  if (size) *size = vs->n->N;

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecGetLocalSize_Comp"
PetscErrorCode VecGetLocalSize_Comp(Vec v, PetscInt *size)
{
  Vec_Comp        *vs = (Vec_Comp*)v->data;

  PetscFunctionBegin;

  PetscValidVecComp(v);

  if (size) *size = vs->n->lN;

  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "VecMax_Comp"
PetscErrorCode VecMax_Comp(Vec v, PetscInt *idx, PetscReal *z)
{
  PetscErrorCode  ierr;
  Vec_Comp        *vs = (Vec_Comp*)v->data;
  PetscInt        idxp, s=0, s0;
  PetscReal       zp, z0;
  PetscInt        i;

  PetscFunctionBegin;

  PetscValidVecComp(v);

  if(!idx && !z)
    PetscFunctionReturn(0);

  if (vs->n->n > 0) {
    ierr = VecMax(vs->x[0], idx?&idxp:PETSC_NULL, &zp); CHKERRQ(ierr);
  }
  for (i=1; i<vs->n->n; i++) {
    ierr = VecGetSize(vs->x[i-1], &s0); CHKERRQ(ierr); s+= s0;
    ierr = VecMax(vs->x[i], idx?&idxp:PETSC_NULL, &z0); CHKERRQ(ierr);
    if(zp < z0) {
      if(idx) *idx = s+idxp;
      zp = z0;
    }
  }
  if (z) *z = zp;

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecMin_Comp"
PetscErrorCode VecMin_Comp(Vec v, PetscInt *idx, PetscReal *z)
{
  PetscErrorCode  ierr;
  Vec_Comp        *vs = (Vec_Comp*)v->data;
  PetscInt        idxp, s=0, s0;
  PetscReal       zp, z0;
  PetscInt        i;

  PetscFunctionBegin;

  PetscValidVecComp(v);

  if(!idx && !z)
    PetscFunctionReturn(0);

  if (vs->n->n > 0) {
    ierr = VecMin(vs->x[0], idx?&idxp:PETSC_NULL, &zp); CHKERRQ(ierr);
  }
  for (i=1; i<vs->n->n; i++) {
    ierr = VecGetSize(vs->x[i-1], &s0); CHKERRQ(ierr); s+= s0;
    ierr = VecMin(vs->x[i], idx?&idxp:PETSC_NULL, &z0); CHKERRQ(ierr);
    if(zp > z0) {
      if(idx) *idx = s+idxp;
      zp = z0;
    }
  }
  if (z) *z = zp;

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecMaxPointwiseDivide_Comp"
PetscErrorCode VecMaxPointwiseDivide_Comp(Vec v, Vec w, PetscReal *m)
{
  PetscErrorCode  ierr;
  Vec_Comp        *vs = (Vec_Comp*)v->data,
                  *ws = (Vec_Comp*)w->data;
  PetscReal       work;
  PetscInt        i;

  PetscFunctionBegin;

  PetscValidVecComp(v);
  PetscValidVecComp(w);

  if(!m || vs->n->n == 0)
    PetscFunctionReturn(0);
  ierr = VecMaxPointwiseDivide(vs->x[0], ws->x[0], m); CHKERRQ(ierr);
  for (i=1; i<vs->n->n; i++) {
    ierr = VecMaxPointwiseDivide(vs->x[i], ws->x[i], &work); CHKERRQ(ierr);
    *m = PetscMax(*m, work);
  }

  PetscFunctionReturn(0);
}


#define __QUOTEME__(x) #x
#define __COMPOSE2__(A,B) A##B
#define __COMPOSE3__(A,B,C) A##B##C

#define __FUNC_TEMPLATE1__(NAME) \
PetscErrorCode __COMPOSE3__(Vec,NAME,_Comp)(Vec v) \
{ \
  PetscErrorCode  ierr; \
  Vec_Comp        *vs = (Vec_Comp*)v->data; \
  PetscInt        i; \
\
  PetscFunctionBegin; \
\
  PetscValidVecComp(v); \
\
  for(i=0; i<vs->n->n; i++) { \
    ierr = __COMPOSE2__(Vec,NAME)(vs->x[i]); CHKERRQ(ierr); \
  } \
\
  PetscFunctionReturn(0);\
} \

#undef __FUNCT__  
#define __FUNCT__ "VecConjugate_Comp"
__FUNC_TEMPLATE1__(Conjugate)

#undef __FUNCT__  
#define __FUNCT__ "VecReciprocal_Comp"
__FUNC_TEMPLATE1__(Reciprocal)

#undef __FUNCT__  
#define __FUNCT__ "VecSqrtAbs_Comp"
__FUNC_TEMPLATE1__(SqrtAbs)

#undef __FUNCT__  
#define __FUNCT__ "VecAbs_Comp"
__FUNC_TEMPLATE1__(Abs)

#undef __FUNCT__  
#define __FUNCT__ "VecExp_Comp"
__FUNC_TEMPLATE1__(Exp)

#undef __FUNCT__  
#define __FUNCT__ "VecLog_Comp"
__FUNC_TEMPLATE1__(Log)


#define __FUNC_TEMPLATE2__(NAME,T0) \
PetscErrorCode __COMPOSE3__(Vec,NAME,_Comp)(Vec v, T0 __a) \
{ \
  PetscErrorCode  ierr; \
  Vec_Comp        *vs = (Vec_Comp*)v->data; \
  PetscInt        i; \
\
  PetscFunctionBegin; \
\
  PetscValidVecComp(v); \
\
  for(i=0; i<vs->n->n; i++) { \
    ierr = __COMPOSE2__(Vec,NAME)(vs->x[i], __a); CHKERRQ(ierr); \
  } \
\
  PetscFunctionReturn(0);\
} \

#undef __FUNCT__  
#define __FUNCT__ "VecSet_Comp"
__FUNC_TEMPLATE2__(Set, PetscScalar)

#undef __FUNCT__  
#define __FUNCT__ "VecView_Comp"
__FUNC_TEMPLATE2__(View, PetscViewer)

#undef __FUNCT__  
#define __FUNCT__ "VecScale_Comp"
__FUNC_TEMPLATE2__(Scale, PetscScalar)

#undef __FUNCT__  
#define __FUNCT__ "VecSetRandom_Comp"
__FUNC_TEMPLATE2__(SetRandom, PetscRandom)

#undef __FUNCT__  
#define __FUNCT__ "VecShift_Comp"
__FUNC_TEMPLATE2__(Shift, PetscScalar)


#define __FUNC_TEMPLATE3__(NAME) \
PetscErrorCode __COMPOSE3__(Vec,NAME,_Comp)(Vec v, Vec w) \
{ \
  PetscErrorCode  ierr; \
  Vec_Comp        *vs = (Vec_Comp*)v->data, \
                  *ws = (Vec_Comp*)w->data; \
  PetscInt        i; \
\
  PetscFunctionBegin; \
\
  PetscValidVecComp(v); \
  PetscValidVecComp(w); \
\
  for(i=0; i<vs->n->n; i++) { \
    ierr = __COMPOSE2__(Vec,NAME)(vs->x[i], ws->x[i]); CHKERRQ(ierr); \
  } \
\
  PetscFunctionReturn(0);\
} \

#undef __FUNCT__  
#define __FUNCT__ "VecCopy_Comp"
__FUNC_TEMPLATE3__(Copy)

#undef __FUNCT__  
#define __FUNCT__ "VecSwap_Comp"
__FUNC_TEMPLATE3__(Swap)


#define __FUNC_TEMPLATE4__(NAME) \
PetscErrorCode __COMPOSE3__(Vec,NAME,_Comp)(Vec v, Vec w, Vec z) \
{ \
  PetscErrorCode  ierr; \
  Vec_Comp        *vs = (Vec_Comp*)v->data, \
                  *ws = (Vec_Comp*)w->data, \
                  *zs = (Vec_Comp*)z->data; \
  PetscInt        i; \
\
  PetscFunctionBegin; \
\
  PetscValidVecComp(v); \
  PetscValidVecComp(w); \
  PetscValidVecComp(z); \
\
  for(i=0; i<vs->n->n; i++) { \
    ierr = __COMPOSE2__(Vec,NAME)(vs->x[i], ws->x[i], zs->x[i]); \
    CHKERRQ(ierr); \
  } \
\
  PetscFunctionReturn(0);\
} \

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMax_Comp"
__FUNC_TEMPLATE4__(PointwiseMax)

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMaxAbs_Comp"
__FUNC_TEMPLATE4__(PointwiseMaxAbs)

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMin_Comp"
__FUNC_TEMPLATE4__(PointwiseMin)

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult_Comp"
__FUNC_TEMPLATE4__(PointwiseMult)

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_Comp"
__FUNC_TEMPLATE4__(PointwiseDivide)
