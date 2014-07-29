/*
  SLEPc eigensolver: "davidson"

  Some utils

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

#include "davidson.h"

typedef struct {
  PC pc;
} dvdPCWrapper;

/*
  Configure the harmonics.
  switch (mode) {
  DVD_HARM_RR:    harmonic RR
  DVD_HARM_RRR:   relative harmonic RR
  DVD_HARM_REIGS: rightmost eigenvalues
  DVD_HARM_LEIGS: largest eigenvalues
  }
  fixedTarged, if true use the target instead of the best eigenvalue
  target, the fixed target to be used
*/
typedef struct {
  PetscScalar
    Wa, Wb,       /* span{W} = span{Wa*AV - Wb*BV} */
    Pa, Pb;       /* H=W'*(Pa*AV - Pb*BV), G=W'*(Wa*AV - Wb*BV) */
  PetscBool
    withTarget;
  HarmType_t
    mode;
} dvdHarmonic;

static PetscErrorCode dvd_static_precond_PC_0(dvdDashboard*,PetscInt,Vec,Vec);
static PetscErrorCode dvd_jacobi_precond_0(dvdDashboard*,PetscInt,Vec,Vec);
static PetscErrorCode dvd_jacobi_precond_d(dvdDashboard*);
static PetscErrorCode dvd_precond_none(dvdDashboard*,PetscInt,Vec,Vec);
static PetscErrorCode dvd_improvex_precond_d(dvdDashboard*);
static PetscErrorCode dvd_initV_prof(dvdDashboard*);
static PetscErrorCode dvd_calcPairs_prof(dvdDashboard*);
static PetscErrorCode dvd_improveX_prof(dvdDashboard*,PetscInt,PetscInt,PetscInt*);
static PetscErrorCode dvd_updateV_prof(dvdDashboard*);
static PetscErrorCode dvd_profiler_d(dvdDashboard*);
static PetscErrorCode dvd_harm_d(dvdDashboard*);
static PetscErrorCode dvd_harm_transf(dvdHarmonic*,PetscScalar);
static PetscErrorCode dvd_harm_updateW(dvdDashboard*);
static PetscErrorCode dvd_harm_proj(dvdDashboard*);
static PetscErrorCode dvd_harm_eigs_trans(dvdDashboard*);
static PetscErrorCode dvd_harm_eig_backtrans(dvdDashboard*,PetscScalar,PetscScalar,PetscScalar*,PetscScalar*);
static PetscErrorCode dvd_harm_backtrans(dvdHarmonic*,PetscScalar*,PetscScalar*);


/*
  Create a static preconditioner from a PC
*/
#undef __FUNCT__
#define __FUNCT__ "dvd_static_precond_PC"
PetscErrorCode dvd_static_precond_PC(dvdDashboard *d,dvdBlackboard *b,PC pc)
{
  PetscErrorCode ierr;
  dvdPCWrapper   *dvdpc;
  Mat            P;
  PetscBool      t0,t1,t2;

  PetscFunctionBegin;
  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    /* If the preconditioner is valid */
    if (pc) {
      ierr = PetscMalloc(sizeof(dvdPCWrapper),&dvdpc);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)d->eps,sizeof(dvdPCWrapper));CHKERRQ(ierr);
      dvdpc->pc = pc;
      ierr = PetscObjectReference((PetscObject)pc);CHKERRQ(ierr);
      d->improvex_precond_data = dvdpc;
      d->improvex_precond = dvd_static_precond_PC_0;

      /* PC saves the matrix associated with the linear system, and it has to
         be initialize to a valid matrix */
      ierr = PCGetOperatorsSet(pc,NULL,&t0);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)pc,PCNONE,&t1);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&t2);CHKERRQ(ierr);
      if (t0 && !t1) {
        ierr = PCGetOperators(pc,NULL,&P);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)P);CHKERRQ(ierr);
        ierr = PCSetOperators(pc,P,P);CHKERRQ(ierr);
        ierr = PCSetReusePreconditioner(pc,PETSC_TRUE);CHKERRQ(ierr);
        ierr = MatDestroy(&P);CHKERRQ(ierr);
      } else if (t2) {
        ierr = PCSetOperators(pc,d->A,d->A);CHKERRQ(ierr);
        ierr = PCSetReusePreconditioner(pc,PETSC_TRUE);CHKERRQ(ierr);
      } else {
        d->improvex_precond = dvd_precond_none;
      }

      ierr = EPSDavidsonFLAdd(&d->destroyList,dvd_improvex_precond_d);CHKERRQ(ierr);

    /* Else, use no preconditioner */
    } else d->improvex_precond = dvd_precond_none;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_precond_d"
static PetscErrorCode dvd_improvex_precond_d(dvdDashboard *d)
{
  PetscErrorCode ierr;
  dvdPCWrapper   *dvdpc = (dvdPCWrapper*)d->improvex_precond_data;

  PetscFunctionBegin;
  /* Free local data */
  if (dvdpc->pc) { ierr = PCDestroy(&dvdpc->pc);CHKERRQ(ierr); }
  ierr = PetscFree(d->improvex_precond_data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_static_precond_PC_0"
static PetscErrorCode dvd_static_precond_PC_0(dvdDashboard *d,PetscInt i,Vec x,Vec Px)
{
  PetscErrorCode ierr;
  dvdPCWrapper   *dvdpc = (dvdPCWrapper*)d->improvex_precond_data;

  PetscFunctionBegin;
  ierr = PCApply(dvdpc->pc, x, Px);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  Vec diagA, diagB;
} dvdJacobiPrecond;

#undef __FUNCT__
#define __FUNCT__ "dvd_jacobi_precond"
/*
  Create the Jacobi preconditioner for Generalized Eigenproblems
*/
PetscErrorCode dvd_jacobi_precond(dvdDashboard *d,dvdBlackboard *b)
{
  PetscErrorCode   ierr;
  dvdJacobiPrecond *dvdjp;
  PetscBool        t;

  PetscFunctionBegin;
  /* Check if the problem matrices support GetDiagonal */
  ierr = MatHasOperation(d->A, MATOP_GET_DIAGONAL, &t);CHKERRQ(ierr);
  if (t && d->B) {
    ierr = MatHasOperation(d->B, MATOP_GET_DIAGONAL, &t);CHKERRQ(ierr);
  }

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscMalloc(sizeof(dvdJacobiPrecond), &dvdjp);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)d->eps,sizeof(dvdJacobiPrecond));CHKERRQ(ierr);
    if (t) {
      ierr = MatGetVecs(d->A,&dvdjp->diagA,NULL);CHKERRQ(ierr);
      ierr = MatGetDiagonal(d->A,dvdjp->diagA);CHKERRQ(ierr);
      if (d->B) {
        ierr = MatGetVecs(d->B,&dvdjp->diagB,NULL);CHKERRQ(ierr);
        ierr = MatGetDiagonal(d->B,dvdjp->diagB);CHKERRQ(ierr);
      } else dvdjp->diagB = 0;
      d->improvex_precond_data = dvdjp;
      d->improvex_precond = dvd_jacobi_precond_0;

      ierr = EPSDavidsonFLAdd(&d->destroyList,dvd_jacobi_precond_d);CHKERRQ(ierr);

    /* Else, use no preconditioner */
    } else {
      dvdjp->diagA = dvdjp->diagB = 0;
      d->improvex_precond = dvd_precond_none;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_jacobi_precond_0"
static PetscErrorCode dvd_jacobi_precond_0(dvdDashboard *d,PetscInt i,Vec x,Vec Px)
{
  PetscErrorCode   ierr;
  dvdJacobiPrecond *dvdjp = (dvdJacobiPrecond*)d->improvex_precond_data;

  PetscFunctionBegin;
  /* Compute inv(D - eig)*x */
  if (dvdjp->diagB == 0) {
    /* Px <- diagA - l */
    ierr = VecCopy(dvdjp->diagA, Px);CHKERRQ(ierr);
    ierr = VecShift(Px, -d->eigr[i]);CHKERRQ(ierr);
  } else {
    /* Px <- diagA - l*diagB */
    ierr = VecWAXPY(Px, -d->eigr[i], dvdjp->diagB, dvdjp->diagA);CHKERRQ(ierr);
  }

  /* Px(i) <- x/Px(i) */
  ierr = VecPointwiseDivide(Px, x, Px);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_jacobi_precond_d"
static PetscErrorCode dvd_jacobi_precond_d(dvdDashboard *d)
{
  PetscErrorCode   ierr;
  dvdJacobiPrecond *dvdjp = (dvdJacobiPrecond*)d->improvex_precond_data;

  PetscFunctionBegin;
  if (dvdjp->diagA) {ierr = VecDestroy(&dvdjp->diagA);CHKERRQ(ierr);}
  if (dvdjp->diagB) {ierr = VecDestroy(&dvdjp->diagB);CHKERRQ(ierr);}
  ierr = PetscFree(d->improvex_precond_data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_precond_none"
/*
  Create a trivial preconditioner
*/
static PetscErrorCode dvd_precond_none(dvdDashboard *d,PetscInt i,Vec x,Vec Px)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(x, Px);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Use of PETSc profiler functions
*/

/* Define stages */
#define DVD_STAGE_INITV 0
#define DVD_STAGE_NEWITER 1
#define DVD_STAGE_CALCPAIRS 2
#define DVD_STAGE_IMPROVEX 3
#define DVD_STAGE_UPDATEV 4
#define DVD_STAGE_ORTHV 5

typedef struct {
  PetscErrorCode (*old_initV)(struct _dvdDashboard*);
  PetscErrorCode (*old_calcPairs)(struct _dvdDashboard*);
  PetscErrorCode (*old_improveX)(struct _dvdDashboard*,PetscInt r_s,PetscInt r_e,PetscInt *size_D);
  PetscErrorCode (*old_updateV)(struct _dvdDashboard*);
  PetscErrorCode (*old_orthV)(struct _dvdDashboard*);
} DvdProfiler;

static PetscLogStage stages[6] = {0,0,0,0,0,0};

/*** Other things ****/

#undef __FUNCT__
#define __FUNCT__ "dvd_prof_init"
PetscErrorCode dvd_prof_init()
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!stages[0]) {
    ierr = PetscLogStageRegister("Dvd_step_initV",&stages[DVD_STAGE_INITV]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Dvd_step_calcPairs",&stages[DVD_STAGE_CALCPAIRS]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Dvd_step_improveX",&stages[DVD_STAGE_IMPROVEX]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Dvd_step_updateV",&stages[DVD_STAGE_UPDATEV]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Dvd_step_orthV",&stages[DVD_STAGE_ORTHV]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_initV_prof"
PetscErrorCode dvd_initV_prof(dvdDashboard* d)
{
  DvdProfiler     *p = (DvdProfiler*)d->prof_data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscLogStagePush(stages[DVD_STAGE_INITV]);
  ierr = p->old_initV(d);CHKERRQ(ierr);
  PetscLogStagePop();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_calcPairs_prof"
static PetscErrorCode dvd_calcPairs_prof(dvdDashboard* d)
{
  DvdProfiler    *p = (DvdProfiler*)d->prof_data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscLogStagePush(stages[DVD_STAGE_CALCPAIRS]);
  ierr = p->old_calcPairs(d);CHKERRQ(ierr);
  PetscLogStagePop();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improveX_prof"
static PetscErrorCode dvd_improveX_prof(dvdDashboard *d,PetscInt r_s,PetscInt r_e,PetscInt *size_D)
{
  DvdProfiler    *p = (DvdProfiler*)d->prof_data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscLogStagePush(stages[DVD_STAGE_IMPROVEX]);
  ierr = p->old_improveX(d, r_s, r_e, size_D);CHKERRQ(ierr);
  PetscLogStagePop();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_updateV_prof"
static PetscErrorCode dvd_updateV_prof(dvdDashboard *d)
{
  DvdProfiler    *p = (DvdProfiler*)d->prof_data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscLogStagePush(stages[DVD_STAGE_UPDATEV]);
  ierr = p->old_updateV(d);CHKERRQ(ierr);
  PetscLogStagePop();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_profiler"
PetscErrorCode dvd_profiler(dvdDashboard *d,dvdBlackboard *b)
{
  PetscErrorCode ierr;
  DvdProfiler    *p;

  PetscFunctionBegin;
  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscFree(d->prof_data);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(DvdProfiler),&p);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)d->eps,sizeof(DvdProfiler));CHKERRQ(ierr);
    d->prof_data = p;
    p->old_initV = d->initV; d->initV = dvd_initV_prof;
    p->old_calcPairs = d->calcPairs; d->calcPairs = dvd_calcPairs_prof;
    p->old_improveX = d->improveX; d->improveX = dvd_improveX_prof;
    p->old_updateV = d->updateV; d->updateV = dvd_updateV_prof;

    ierr = EPSDavidsonFLAdd(&d->destroyList,dvd_profiler_d);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_profiler_d"
static PetscErrorCode dvd_profiler_d(dvdDashboard *d)
{
  PetscErrorCode ierr;
  DvdProfiler    *p = (DvdProfiler*)d->prof_data;

  PetscFunctionBegin;
  /* Free local data */
  ierr = PetscFree(p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_conf"
PetscErrorCode dvd_harm_conf(dvdDashboard *d,dvdBlackboard *b,HarmType_t mode,PetscBool fixedTarget,PetscScalar t)
{
  PetscErrorCode ierr;
  dvdHarmonic    *dvdh;

  PetscFunctionBegin;
  /* Set the problem to GNHEP:
     d->G maybe is upper triangular due to biorthogonality of V and W */
  d->sEP = d->sA = d->sB = 0;

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscMalloc(sizeof(dvdHarmonic),&dvdh);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)d->eps,sizeof(dvdHarmonic));CHKERRQ(ierr);
    dvdh->withTarget = fixedTarget;
    dvdh->mode = mode;
    if (fixedTarget) dvd_harm_transf(dvdh, t);
    d->calcpairs_W_data = dvdh;
    d->calcpairs_W = dvd_harm_updateW;
    d->calcpairs_proj_trans = dvd_harm_proj;
    d->calcpairs_eigs_trans = dvd_harm_eigs_trans;
    d->calcpairs_eig_backtrans = dvd_harm_eig_backtrans;

    ierr = EPSDavidsonFLAdd(&d->destroyList,dvd_harm_d);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_d"
static PetscErrorCode dvd_harm_d(dvdDashboard *d)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Free local data */
  ierr = PetscFree(d->calcpairs_W_data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_transf"
static PetscErrorCode dvd_harm_transf(dvdHarmonic *dvdh,PetscScalar t)
{
  PetscFunctionBegin;
  switch (dvdh->mode) {
  case DVD_HARM_RR:    /* harmonic RR */
    dvdh->Wa = 1.0; dvdh->Wb = t;   dvdh->Pa = 0.0; dvdh->Pb = -1.0; break;
  case DVD_HARM_RRR:   /* relative harmonic RR */
    dvdh->Wa = 1.0; dvdh->Wb = t;   dvdh->Pa = 1.0; dvdh->Pb = 0.0; break;
  case DVD_HARM_REIGS: /* rightmost eigenvalues */
    dvdh->Wa = 1.0; dvdh->Wb = t;   dvdh->Pa = 1.0; dvdh->Pb = -PetscConj(t);
    break;
  case DVD_HARM_LEIGS: /* largest eigenvalues */
    dvdh->Wa = 0.0; dvdh->Wb = 1.0; dvdh->Pa = 1.0; dvdh->Pb = 0.0; break;
  case DVD_HARM_NONE:
  default:
    SETERRQ(PETSC_COMM_SELF,1, "Harmonic type not supported");
  }

  /* Check the transformation does not change the sign of the imaginary part */
#if !defined(PETSC_USE_COMPLEX)
  if (dvdh->Pb*dvdh->Wa - dvdh->Wb*dvdh->Pa < 0.0)
    dvdh->Pa*= -1.0, dvdh->Pb*= -1.0;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_updateW"
static PetscErrorCode dvd_harm_updateW(dvdDashboard *d)
{
  dvdHarmonic    *data = (dvdHarmonic*)d->calcpairs_W_data;
  PetscErrorCode ierr;
  PetscInt       l,k,i;
  BV             BX = d->BX?d->BX:d->eps->V;

  PetscFunctionBegin;
  /* Update the target if it is necessary */
  if (!data->withTarget) {
    ierr = dvd_harm_transf(data,d->eigr[0]);CHKERRQ(ierr);
  }

  /* W(i) <- Wa*AV(i) - Wb*BV(i) */
  ierr = BVGetActiveColumns(d->eps->V,&l,&k);CHKERRQ(ierr);
  if (k != l+d->V_new_s) SETERRQ(PETSC_COMM_SELF,1, "Consistency broken");
  ierr = BVSetActiveColumns(d->W,l+d->V_new_s,l+d->V_new_e);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(d->AX,l+d->V_new_s,l+d->V_new_e);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(BX,l+d->V_new_s,l+d->V_new_e);CHKERRQ(ierr);
  ierr = BVCopy(d->AX,d->W);CHKERRQ(ierr);
  /* Work around bug in BVScale
  ierr = BVScale(d->W,data->Wa);CHKERRQ(ierr); */
  for (i=l+d->V_new_s;i<l+d->V_new_e; ++i) {
    ierr = BVScaleColumn(d->W,i,data->Wa);CHKERRQ(ierr);
  }
  ierr = BVAXPY(d->W,-data->Wb,BX);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(d->W,l,k);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(d->AX,l,k);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(BX,l,k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_proj"
static PetscErrorCode dvd_harm_proj(dvdDashboard *d)
{
  PetscErrorCode ierr;
  dvdHarmonic    *data = (dvdHarmonic*)d->calcpairs_W_data;
  PetscInt       i,j,l0,l,k,ld;
  PetscScalar    h,g,*H,*G;

  PetscFunctionBegin;
  ierr = BVGetActiveColumns(d->eps->V,&l0,&k);CHKERRQ(ierr);
  l = l0 + d->V_new_s;
  k = l0 + d->V_new_e;
  ierr = MatGetSize(d->H,&ld,NULL);CHKERRQ(ierr);
  ierr = MatDenseGetArray(d->H,&H);CHKERRQ(ierr);
  ierr = MatDenseGetArray(d->G,&G);CHKERRQ(ierr);
  /* [H G] <- [Pa*H - Pb*G, Wa*H - Wb*G] */
  /* Right part */
  for (i=l; i<k; i++) {
    for (j=l0; j<k; j++) {
      h = H[ld*i+j]; g = G[ld*i+j];
      H[ld*i+j] = data->Pa*h - data->Pb*g;
      G[ld*i+j] = data->Wa*h - data->Wb*g;
    }
  }
  /* Left part */
  for (i=l0; i<l; i++) {
    for (j=l; j<k; j++) {
      h = H[ld*i+j]; g = G[ld*i+j];
      H[ld*i+j] = data->Pa*h - data->Pb*g;
      G[ld*i+j] = data->Wa*h - data->Wb*g;
    }
  }
  ierr = MatDenseRestoreArray(d->H,&H);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(d->G,&G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_updateproj"
PetscErrorCode dvd_harm_updateproj(dvdDashboard *d)
{
  PetscErrorCode ierr;
  dvdHarmonic    *data = (dvdHarmonic*)d->calcpairs_W_data;
  PetscInt       i,j,l,k,ld;
  PetscScalar    h,g,*H,*G;

  PetscFunctionBegin;
  ierr = BVGetActiveColumns(d->eps->V,&l,&k);CHKERRQ(ierr);
  k = l + d->V_tra_s;
  ierr = MatGetSize(d->H,&ld,NULL);CHKERRQ(ierr);
  ierr = MatDenseGetArray(d->H,&H);CHKERRQ(ierr);
  ierr = MatDenseGetArray(d->G,&G);CHKERRQ(ierr);
  /* [H G] <- [Pa*H - Pb*G, Wa*H - Wb*G] */
  /* Right part */
  for (i=l; i<k; i++) {
    for (j=0; j<l; j++) {
      h = H[ld*i+j]; g = G[ld*i+j];
      H[ld*i+j] = data->Pa*h - data->Pb*g;
      G[ld*i+j] = data->Wa*h - data->Wb*g;
    }
  }
  /* Lower triangular part */
  for (i=0; i<l; i++) {
    for (j=l; j<k; j++) {
      h = H[ld*i+j]; g = G[ld*i+j];
      H[ld*i+j] = data->Pa*h - data->Pb*g;
      G[ld*i+j] = data->Wa*h - data->Wb*g;
    }
  }
  ierr = MatDenseRestoreArray(d->H,&H);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(d->G,&G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_backtrans"
static PetscErrorCode dvd_harm_backtrans(dvdHarmonic *data,PetscScalar *ar,PetscScalar *ai)
{
  PetscScalar xr;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar xi, k;
#endif

  PetscFunctionBegin;
  PetscValidPointer(ar,2);
  xr = *ar;
#if !defined(PETSC_USE_COMPLEX)
  PetscValidPointer(ai,3);
  xi = *ai;

  if (xi != 0.0) {
    k = (data->Pa - data->Wa*xr)*(data->Pa - data->Wa*xr) +
        data->Wa*data->Wa*xi*xi;
    *ar = (data->Pb*data->Pa - (data->Pb*data->Wa + data->Wb*data->Pa)*xr +
           data->Wb*data->Wa*(xr*xr + xi*xi))/k;
    *ai = (data->Pb*data->Wa - data->Wb*data->Pa)*xi/k;
  } else
#endif
    *ar = (data->Pb - data->Wb*xr) / (data->Pa - data->Wa*xr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_eig_backtrans"
static PetscErrorCode dvd_harm_eig_backtrans(dvdDashboard *d,PetscScalar ar,PetscScalar ai,PetscScalar *br,PetscScalar *bi)
{
  dvdHarmonic    *data = (dvdHarmonic*)d->calcpairs_W_data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = dvd_harm_backtrans(data,&ar,&ai);CHKERRQ(ierr);
  *br = ar;
  *bi = ai;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_eigs_trans"
static PetscErrorCode dvd_harm_eigs_trans(dvdDashboard *d)
{
  dvdHarmonic    *data = (dvdHarmonic*)d->calcpairs_W_data;
  PetscInt       i,l,k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = BVGetActiveColumns(d->eps->V,&l,&k);CHKERRQ(ierr);
  for (i=0;i<k-l;i++) {
    ierr = dvd_harm_backtrans(data,&d->eigr[i],&d->eigi[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultS"
/* H = [H              Y(old)'*X(new);
        Y(new)'*X(old) Y(new)'*X(new) ],
     being old=0:l-1, new=l:k-1 */
PetscErrorCode BVMultS(BV X,BV Y,PetscScalar *H,PetscInt ldh)
{
  PetscErrorCode ierr;
  PetscInt       j,lx,ly,kx,ky;
  PetscScalar    *array;
  Mat            M;

  PetscFunctionBegin;
  ierr = BVGetActiveColumns(X,&lx,&kx);CHKERRQ(ierr);
  ierr = BVGetActiveColumns(Y,&ly,&ky);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,ky,kx,NULL,&M);CHKERRQ(ierr);
  ierr = BVMatProject(X,NULL,Y,M);CHKERRQ(ierr);
  ierr = MatDenseGetArray(M,&array);CHKERRQ(ierr);
  /* upper part */
  for (j=lx;j<kx;j++) {
    ierr = PetscMemcpy(&H[ldh*j],&array[j*ky],ly*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  /* lower part */
  for (j=0;j<kx;j++) {
    ierr = PetscMemcpy(&H[ldh*j+ly],&array[j*ky+ly],(ky-ly)*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(M,&array);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcMatDenseCopy"
/*
   SlepcMatDenseCopy - Copy a submatrix from A to B.

   Not Collective

   Input Parameters:
+  A    - source seq dense matrix
.  Ar0  - first row to copy from A
.  Ac0  - first column to copy from A
.  Br0  - first row to copy on B
.  Bc0  - first column to copy on B
.  rows - number of rows to copy
-  cols - number of columns to copy

   Level: advanced
*/
PetscErrorCode SlepcMatDenseCopy(Mat A,PetscInt Ar0,PetscInt Ac0,Mat B,PetscInt Br0,PetscInt Bc0,PetscInt rows,PetscInt cols)
{
  PetscErrorCode ierr;
  PetscInt       i,n,m,ldA,ldB;
  PetscScalar    *pA,*pB;

  PetscFunctionBegin;
  if (!rows || !cols) PetscFunctionReturn(0);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr); ldA=m;
  if (Ar0<0 || Ar0>=m) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid initial row in A");
  if (Ac0<0 || Ac0>=n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid initial column in A");
  if (Ar0+rows>m) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid number of rows");
  if (Ac0+cols>n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid number of columns");
  ierr = MatGetSize(B,&m,&n);CHKERRQ(ierr); ldB=m;
  if (Br0<0 || Br0>=m) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid initial row in B");
  if (Bc0<0 || Bc0>=n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid initial column in B");
  if (Br0+rows>m) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid number of rows");
  if (Bc0+cols>n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid number of columns");
  ierr = MatDenseGetArray(A,&pA);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&pB);CHKERRQ(ierr);
  for (i=0;i<cols;i++) {
    ierr = PetscMemcpy(&pB[ldB*(Bc0+i)+Br0],&pA[ldA*(Ac0+i)+Ar0],sizeof(PetscScalar)*rows);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(A,&pA);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&pB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

