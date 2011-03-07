/*
  SLEPc eigensolver: "davidson"

  Some utils

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

#include "davidson.h"

PetscErrorCode dvd_static_precond_PC_0(dvdDashboard *d, PetscInt i, Vec x,
                                       Vec Px);
PetscErrorCode dvd_jacobi_precond_0(dvdDashboard *d, PetscInt i, Vec x, Vec Px);
PetscErrorCode dvd_precond_none(dvdDashboard *d, PetscInt i, Vec x, Vec Px);
PetscErrorCode dvd_improvex_precond_d(dvdDashboard *d);

typedef struct {
  PC pc;
} dvdPCWrapper;
/*
  Create a static preconditioner from a PC
*/
#undef __FUNCT__
#define __FUNCT__ "dvd_static_precond_PC"
PetscErrorCode dvd_static_precond_PC(dvdDashboard *d, dvdBlackboard *b, PC pc)
{
  PetscErrorCode  ierr;
  dvdPCWrapper    *dvdpc;
  Mat             P;
  MatStructure    str;

  PetscFunctionBegin;

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    /* If the preconditioner is valid */
    if (pc) {
      ierr = PetscMalloc(sizeof(dvdPCWrapper), &dvdpc); CHKERRQ(ierr);
      dvdpc->pc = pc;
      ierr = PetscObjectReference((PetscObject)pc); CHKERRQ(ierr);
      d->improvex_precond_data = dvdpc;
      d->improvex_precond = dvd_static_precond_PC_0;

      /* PC saves the matrix associated with the linear system, and it has to
         be initialize to a valid matrix */
      ierr = PCGetOperators(pc, PETSC_NULL, &P, &str); CHKERRQ(ierr);
      if (P) {
        ierr = PetscObjectReference((PetscObject)P); CHKERRQ(ierr);
        ierr = PCSetOperators(pc, P, P, str); CHKERRQ(ierr);
        ierr = MatDestroy(P); CHKERRQ(ierr);
      } else
        d->improvex_precond = dvd_precond_none;

      DVD_FL_ADD(d->destroyList, dvd_improvex_precond_d);

    /* Else, use no preconditioner */
    } else
      d->improvex_precond = dvd_precond_none;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_precond_d"
PetscErrorCode dvd_improvex_precond_d(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  dvdPCWrapper    *dvdpc = (dvdPCWrapper*)d->improvex_precond_data;

  PetscFunctionBegin;

  /* Free local data */
  if (dvdpc->pc) { ierr = PCDestroy(dvdpc->pc); CHKERRQ(ierr); }
  ierr = PetscFree(d->improvex_precond_data); CHKERRQ(ierr);
  d->improvex_precond_data = PETSC_NULL;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "dvd_static_precond_PC_0"
PetscErrorCode dvd_static_precond_PC_0(dvdDashboard *d, PetscInt i, Vec x,
                                       Vec Px)
{
  PetscErrorCode  ierr;
  dvdPCWrapper    *dvdpc = (dvdPCWrapper*)d->improvex_precond_data;

  PetscFunctionBegin;

  ierr = PCApply(dvdpc->pc, x, Px); CHKERRQ(ierr);
  
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
PetscErrorCode dvd_jacobi_precond(dvdDashboard *d, dvdBlackboard *b)
{
  PetscErrorCode  ierr;
  dvdJacobiPrecond
                  *dvdjp;
  PetscBool       t;

  PetscFunctionBegin;

  /* Check if the problem matrices support GetDiagonal */
  ierr = MatHasOperation(d->A, MATOP_GET_DIAGONAL, &t); CHKERRQ(ierr);
  if (t && d->B) {
    ierr = MatHasOperation(d->B, MATOP_GET_DIAGONAL, &t); CHKERRQ(ierr);
  }

  /* Setting configuration constrains */
  b->own_vecs+= t?( (d->B == 0)?1:2 ) : 0;

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    if (t) {
      ierr = PetscMalloc(sizeof(dvdJacobiPrecond), &dvdjp); CHKERRQ(ierr);
      dvdjp->diagA = *b->free_vecs; b->free_vecs++;
      ierr = MatGetDiagonal(d->A,dvdjp->diagA); CHKERRQ(ierr);
      if (d->B) {
        dvdjp->diagB = *b->free_vecs; b->free_vecs++;
        ierr = MatGetDiagonal(d->B, dvdjp->diagB); CHKERRQ(ierr);
      } else
        dvdjp->diagB = 0;
      d->improvex_precond_data = dvdjp;
      d->improvex_precond = dvd_jacobi_precond_0;

      DVD_FL_ADD(d->destroyList, dvd_improvex_precond_d);

    /* Else, use no preconditioner */
    } else
      d->improvex_precond = dvd_precond_none;
  } 

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_jacobi_precond_0"
PetscErrorCode dvd_jacobi_precond_0(dvdDashboard *d, PetscInt i, Vec x, Vec Px)
{
  PetscErrorCode  ierr;
  dvdJacobiPrecond
                  *dvdjp = (dvdJacobiPrecond*)d->improvex_precond_data;

  PetscFunctionBegin;

  /* Compute inv(D - eig)*x */
  if (dvdjp->diagB == 0) {
    /* Px <- diagA - l */
    ierr = VecCopy(dvdjp->diagA, Px); CHKERRQ(ierr);
    ierr = VecShift(Px, -d->eigr[i]); CHKERRQ(ierr);
  } else {
    /* Px <- diagA - l*diagB */
    ierr = VecWAXPY(Px, -d->eigr[i], dvdjp->diagB, dvdjp->diagA);
    CHKERRQ(ierr);
  }

  /* Px(i) <- x/Px(i) */
  ierr = VecPointwiseDivide(Px, x, Px); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_precond_none"
/*
  Create a trivial preconditioner
*/
PetscErrorCode dvd_precond_none(dvdDashboard *d, PetscInt i, Vec x, Vec Px)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = VecCopy(x, Px); CHKERRQ(ierr);

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

PetscErrorCode dvd_profiler_d(dvdDashboard *d);

typedef struct {
  PetscErrorCode (*old_initV)(struct _dvdDashboard*);
  PetscErrorCode (*old_calcPairs)(struct _dvdDashboard*);
  PetscErrorCode (*old_improveX)(struct _dvdDashboard*, Vec *D,
                                 PetscInt max_size_D, PetscInt r_s,
                                 PetscInt r_e, PetscInt *size_D);
  PetscErrorCode (*old_updateV)(struct _dvdDashboard*);
  PetscErrorCode (*old_orthV)(struct _dvdDashboard*);
} DvdProfiler;

PetscLogStage stages[6] = {0,0,0,0,0,0};

/*** Other things ****/

#undef __FUNCT__
#define __FUNCT__ "dvd_prof_init"
PetscErrorCode dvd_prof_init()
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  if (!stages[0]) {
    ierr = PetscLogStageRegister("Dvd_step_initV", &stages[DVD_STAGE_INITV]);
    CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Dvd_step_calcPairs",
  		               &stages[DVD_STAGE_CALCPAIRS]); CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Dvd_step_improveX",
  		               &stages[DVD_STAGE_IMPROVEX]); CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Dvd_step_updateV",
  		               &stages[DVD_STAGE_UPDATEV]); CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Dvd_step_orthV",
  		               &stages[DVD_STAGE_ORTHV]); CHKERRQ(ierr);
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
  ierr = p->old_initV(d); CHKERRQ(ierr);
  PetscLogStagePop();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_calcPairs_prof"
PetscErrorCode dvd_calcPairs_prof(dvdDashboard* d)
{
  DvdProfiler     *p = (DvdProfiler*)d->prof_data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  PetscLogStagePush(stages[DVD_STAGE_CALCPAIRS]);
  ierr = p->old_calcPairs(d); CHKERRQ(ierr);
  PetscLogStagePop();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_improveX_prof"
PetscErrorCode dvd_improveX_prof(dvdDashboard* d, Vec *D, PetscInt max_size_D,
                       PetscInt r_s, PetscInt r_e, PetscInt *size_D)
{
  DvdProfiler     *p = (DvdProfiler*)d->prof_data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  PetscLogStagePush(stages[DVD_STAGE_IMPROVEX]);
  ierr = p->old_improveX(d, D, max_size_D, r_s, r_e, size_D); CHKERRQ(ierr);
  PetscLogStagePop();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_updateV_prof"
PetscErrorCode dvd_updateV_prof(dvdDashboard *d)
{
  DvdProfiler     *p = (DvdProfiler*)d->prof_data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  PetscLogStagePush(stages[DVD_STAGE_UPDATEV]);
  ierr = p->old_updateV(d); CHKERRQ(ierr);
  PetscLogStagePop();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_orthV_prof"
PetscErrorCode dvd_orthV_prof(dvdDashboard *d)
{
  DvdProfiler     *p = (DvdProfiler*)d->prof_data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  PetscLogStagePush(stages[DVD_STAGE_ORTHV]);
  ierr = p->old_orthV(d); CHKERRQ(ierr);
  PetscLogStagePop();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_profiler"
PetscErrorCode dvd_profiler(dvdDashboard *d, dvdBlackboard *b)
{
  PetscErrorCode  ierr;
  DvdProfiler     *p;

  PetscFunctionBegin;

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    if (d->prof_data) {
      ierr = PetscFree(d->prof_data); CHKERRQ(ierr);
    }
    ierr = PetscMalloc(sizeof(DvdProfiler), &p); CHKERRQ(ierr);
    d->prof_data = p;
    p->old_initV = d->initV; d->initV = dvd_initV_prof;
    p->old_calcPairs = d->calcPairs; d->calcPairs = dvd_calcPairs_prof;
    p->old_improveX = d->improveX; d->improveX = dvd_improveX_prof;
    p->old_updateV = d->updateV; d->updateV = dvd_updateV_prof;

    DVD_FL_ADD(d->destroyList, dvd_profiler_d);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_profiler_d"
PetscErrorCode dvd_profiler_d(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  DvdProfiler     *p = (DvdProfiler*)d->prof_data;

  PetscFunctionBegin;

  /* Free local data */
  ierr = PetscFree(p); CHKERRQ(ierr);
  d->prof_data = PETSC_NULL;

  PetscFunctionReturn(0);
}



/*
  Configure the harmonics.
  switch(mode) {
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

  /* old values of eps */
  EPSWhich
    old_which;
  PetscErrorCode
    (*old_which_func)(EPS,PetscScalar,PetscScalar,PetscScalar,PetscScalar,
                      PetscInt*,void*);
  void
    *old_which_ctx;
} dvdHarmonic;

PetscErrorCode dvd_harm_start(dvdDashboard *d);
PetscErrorCode dvd_harm_end(dvdDashboard *d);
PetscErrorCode dvd_harm_d(dvdDashboard *d);
PetscErrorCode dvd_harm_transf(dvdHarmonic *dvdh, PetscScalar t);
PetscErrorCode dvd_harm_updateW(dvdDashboard *d);
PetscErrorCode dvd_harm_proj(dvdDashboard *d);
PetscErrorCode dvd_harm_sort(EPS eps, PetscScalar ar, PetscScalar ai,
                             PetscScalar br, PetscScalar bi, PetscInt *r,
                             void *ctx);
PetscErrorCode dvd_harm_eigs_trans(dvdDashboard *d);

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_conf"
PetscErrorCode dvd_harm_conf(dvdDashboard *d, dvdBlackboard *b,
                             HarmType_t mode, PetscBool fixedTarget,
                             PetscScalar t)
{
  PetscErrorCode  ierr;
  dvdHarmonic     *dvdh;

  PetscFunctionBegin;

  /* Set the problem to GNHEP */
  // TODO: d->G maybe is upper triangular due to biorthogonality of V and W
  d->sEP = d->sA = d->sB = 0;

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    ierr = PetscMalloc(sizeof(dvdHarmonic), &dvdh); CHKERRQ(ierr);
    dvdh->withTarget = fixedTarget;
    dvdh->mode = mode;
    if (fixedTarget) dvd_harm_transf(dvdh, t);
    d->calcpairs_W_data = dvdh;
    d->calcpairs_W = dvd_harm_updateW;
    d->calcpairs_proj_trans = dvd_harm_proj;
    d->calcpairs_eigs_trans = dvd_harm_eigs_trans;

    DVD_FL_ADD(d->startList, dvd_harm_start);
    DVD_FL_ADD(d->endList, dvd_harm_end);
    DVD_FL_ADD(d->destroyList, dvd_harm_d);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "dvd_harm_d"
PetscErrorCode dvd_harm_d(dvdDashboard *d)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* Free local data */
  ierr = PetscFree(d->calcpairs_W_data); CHKERRQ(ierr);
  d->calcpairs_W_data = PETSC_NULL;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "dvd_harm_start"
PetscErrorCode dvd_harm_start(dvdDashboard *d)
{
  dvdHarmonic     *data = (dvdHarmonic*)d->calcpairs_W_data;

  PetscFunctionBegin;

  /* Overload the eigenpairs selection routine */
  data->old_which = d->eps->which;
  data->old_which_func = d->eps->which_func;
  data->old_which_ctx = d->eps->which_ctx;
  d->eps->which = EPS_WHICH_USER;
  d->eps->which_func = dvd_harm_sort;
  d->eps->which_ctx = data;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "dvd_harm_end"
PetscErrorCode dvd_harm_end(dvdDashboard *d)
{
  dvdHarmonic     *data = (dvdHarmonic*)d->calcpairs_W_data;

  PetscFunctionBegin;

  /* Restore the eigenpairs selection routine */
  d->eps->which = data->old_which;
  d->eps->which_func = data->old_which_func;
  d->eps->which_ctx = data->old_which_ctx;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "dvd_harm_transf"
PetscErrorCode dvd_harm_transf(dvdHarmonic *dvdh, PetscScalar t)
{
  PetscFunctionBegin;

  switch(dvdh->mode) {
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
    SETERRQ(PETSC_COMM_WORLD,1, "Harmonic type not supported");
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
PetscErrorCode dvd_harm_updateW(dvdDashboard *d)
{
  dvdHarmonic     *data = (dvdHarmonic*)d->calcpairs_W_data;
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;

  /* Update the target if it is necessary */
  if (!data->withTarget) dvd_harm_transf(data, d->eigr[0]);
    
  for(i=d->V_new_s; i<d->V_new_e; i++) {
    /* W(i) <- Wa*AV(i) - Wb*BV(i) */
    ierr = VecCopy(d->AV[i], d->W[i]); CHKERRQ(ierr);
    ierr = VecAXPBY(d->W[i], -data->Wb, data->Wa, (d->BV?d->BV:d->V)[i]);
    CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_proj"
PetscErrorCode dvd_harm_proj(dvdDashboard *d)
{
  dvdHarmonic     *data = (dvdHarmonic*)d->calcpairs_W_data;
  PetscInt        i,j;

  PetscFunctionBegin;

  if (d->sH != d->sG) {
    SETERRQ(PETSC_COMM_WORLD,1,"Projected matrices H and G must have the same structure");
    PetscFunctionReturn(1);
  }

  /* [H G] <- [Pa*H - Pb*G, Wa*H - Wb*G] */
  if (DVD_ISNOT(d->sH,DVD_MAT_LTRIANG))     /* Upper triangular part */
    for(i=d->V_new_s; i<d->V_new_e; i++)
      for(j=0; j<=i; j++) {
        PetscScalar h = d->H[d->ldH*i+j], g = d->G[d->ldH*i+j];
        d->H[d->ldH*i+j] = data->Pa*h - data->Pb*g;
        d->G[d->ldH*i+j] = data->Wa*h - data->Wb*g;
      }
  if (DVD_ISNOT(d->sH,DVD_MAT_UTRIANG))     /* Lower triangular part */
    for(i=0; i<d->V_new_e; i++)
      for(j=PetscMax(d->V_new_s,i+(DVD_ISNOT(d->sH,DVD_MAT_LTRIANG)?1:0));
          j<d->V_new_e; j++) {
        PetscScalar h = d->H[d->ldH*i+j], g = d->G[d->ldH*i+j];
        d->H[d->ldH*i+j] = data->Pa*h - data->Pb*g;
        d->G[d->ldH*i+j] = data->Wa*h - data->Wb*g;
      }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_backtrans"
PetscErrorCode dvd_harm_backtrans(dvdHarmonic *data, PetscScalar *ar,
                                  PetscScalar *ai)
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
#define __FUNCT__ "dvd_harm_sort"
PetscErrorCode dvd_harm_sort(EPS eps, PetscScalar ar, PetscScalar ai,
                             PetscScalar br, PetscScalar bi, PetscInt *r,
                             void *ctx)
{
  dvdHarmonic     *data = (dvdHarmonic*)ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* Back-transform the harmonic values */
  dvd_harm_backtrans(data, &ar, &ai);
  dvd_harm_backtrans(data, &br, &bi);

  /* Compare values using the user options for the eigenpairs selection */
  eps->which = data->old_which;
  eps->which_func = data->old_which_func;
  eps->which_ctx = data->old_which_ctx;
  ierr = EPSCompareEigenvalues(eps, ar, ai, br, bi, r); CHKERRQ(ierr);

  /* Restore the eps values */
  eps->which = EPS_WHICH_USER;
  eps->which_func = dvd_harm_sort;
  eps->which_ctx = data;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dvd_harm_eigs_trans"
PetscErrorCode dvd_harm_eigs_trans(dvdDashboard *d)
{
  dvdHarmonic     *data = (dvdHarmonic*)d->calcpairs_W_data;
  PetscInt        i;

  PetscFunctionBegin;

  for(i=0; i<d->size_H; i++)
    dvd_harm_backtrans(data, &d->eigr[i], &d->eigi[i]);

  PetscFunctionReturn(0);
}


