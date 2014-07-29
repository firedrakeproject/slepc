/*
  Method: General Davidson Method (includes GD and JD)

  References:
    - Ernest R. Davidson. Super-matrix methods. Computer Physics Communications,
      53:49–60, May 1989.

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

#include <slepc-private/epsimpl.h>
#include <slepc-private/stimpl.h>
#include <slepc-private/vecimplslepc.h>

struct _dvdDashboard;
typedef PetscErrorCode (*dvdCallback)(struct _dvdDashboard*);
typedef struct _dvdFunctionList {
  dvdCallback f;
  struct _dvdFunctionList *next;
} dvdFunctionList;

typedef enum {
  DVD_HARM_NONE,
  DVD_HARM_RR,
  DVD_HARM_RRR,
  DVD_HARM_REIGS,
  DVD_HARM_LEIGS
} HarmType_t;

typedef enum {
  DVD_INITV_CLASSIC,
  DVD_INITV_KRYLOV
} InitType_t;

typedef enum {
  DVD_PROJ_KXX,
  DVD_PROJ_KZX
} ProjType_t;

typedef enum {
  DVD_METH_GD,
  DVD_METH_JD,
  DVD_METH_GD2
} Method_t;

/*
   Dashboard struct: contains the methods that will be employed and the tunning
   options.
*/

typedef struct _dvdDashboard {
  /**** Function steps ****/
  /* Initialize V */
  PetscErrorCode (*initV)(struct _dvdDashboard*);
  void *initV_data;

  /* Find the approximate eigenpairs from V */
  PetscErrorCode (*calcPairs)(struct _dvdDashboard*);
  void *calcPairs_data;

  /* Eigenpair test for convergence */
  PetscBool (*testConv)(struct _dvdDashboard*,PetscScalar eigvr,PetscScalar eigvi,PetscReal res,PetscReal *error);
  void *testConv_data;

  /* Number of converged eigenpairs */
  PetscInt nconv;

  /* Number of pairs ready to converge */
  PetscInt npreconv;

  /* Improve the selected eigenpairs */
  PetscErrorCode (*improveX)(struct _dvdDashboard*,PetscInt r_s,PetscInt r_e,PetscInt *size_D);
  void *improveX_data;

  /* Check for restarting */
  PetscErrorCode (*isRestarting)(struct _dvdDashboard*,PetscBool*);
  void *isRestarting_data;

  /* Perform restarting */
  PetscErrorCode (*restartV)(struct _dvdDashboard*);
  void *restartV_data;

  /* Update V */
  PetscErrorCode (*updateV)(struct _dvdDashboard*);
  void *updateV_data;

  /**** Problem specification ****/
  Mat A, B;         /* Problem matrices */
  MatType_t sA, sB; /* Matrix specifications */
  EPType_t sEP;     /* Problem specifications */
  PetscInt nev;     /* number of eigenpairs */
  EPSWhich which;   /* spectrum selection */
  PetscBool
    withTarget;     /* if there is a target */
  PetscScalar
    target[2];         /* target value */
  PetscReal tol;    /* tolerance */
  PetscBool
    correctXnorm;   /* if true, norm of X are computed */

  /**** Subspaces specification ****/
  BV W;             /* left basis for harmonic case */
  BV AX;            /* A*V */
  BV BX;            /* B*V */
  PetscInt size_D,  /* active vectors */
    max_size_proj,  /* max size projected problem */
    max_cX_in_proj, /* max vectors from cX in the projected problem */
    max_cX_in_impr, /* max vectros from cX in the projector */
    max_size_P,     /* max unconverged vectors in the projector */
    bs;             /* max vectors that expands the subspace every iteration */
  EPS eps;          /* Connection to SLEPc */

  /**** Auxiliary space ****/
  VecPool auxV;     /* auxiliary vectors */
  BV  auxBV;        /* auxiliary vectors */

  /**** Eigenvalues and errors ****/
  PetscScalar
    *ceigr, *ceigi, /* converged eigenvalues */
    *eigr, *eigi;   /* current eigenvalues */
  PetscReal
    *nR,            /* residual norm */
    *real_nR,       /* original nR */
    *nX,            /* X norm */
    *real_nX,       /* original nX */
    *errest,        /* relative error eigenpairs */
    *nBds;          /* B-norms of projected problem  */

  /**** Shared function and variables ****/
  PetscErrorCode (*e_Vchanged)(struct _dvdDashboard*,PetscInt s_imm,PetscInt e_imm,PetscInt s_new,PetscInt e_new);
  void *e_Vchanged_data;

  PetscErrorCode (*calcpairs_residual)(struct _dvdDashboard*,PetscInt s,PetscInt e);
  PetscErrorCode (*calcpairs_selectPairs)(struct _dvdDashboard*,PetscInt n);
  void *calcpairs_residual_data;
  PetscErrorCode (*improvex_precond)(struct _dvdDashboard*,PetscInt i,Vec x,Vec Px);
  void *improvex_precond_data;
  PetscErrorCode (*improvex_jd_proj_uv)(struct _dvdDashboard*,PetscInt i_s,PetscInt i_e,Vec *u,Vec *v,Vec *kr,PetscScalar *theta,PetscScalar *thetai,PetscScalar *pX,PetscScalar *pY,PetscInt ld);
  PetscErrorCode (*improvex_jd_lit)(struct _dvdDashboard*,PetscInt i,PetscScalar* theta,PetscScalar* thetai,PetscInt *maxits,PetscReal *tol);
  PetscErrorCode (*calcpairs_W)(struct _dvdDashboard*);
  void *calcpairs_W_data;
  PetscErrorCode (*calcpairs_proj_trans)(struct _dvdDashboard*);
  PetscErrorCode (*calcpairs_eigs_trans)(struct _dvdDashboard*);
  PetscErrorCode (*calcpairs_eig_backtrans)(struct _dvdDashboard*,PetscScalar,PetscScalar,PetscScalar*,PetscScalar*);
  PetscErrorCode (*calcpairs_proj_res)(struct _dvdDashboard*,PetscInt r_s,PetscInt r_e,Vec *R);
  PetscErrorCode (*preTestConv)(struct _dvdDashboard*,PetscInt s,PetscInt pre,PetscInt e,PetscInt *nConv);

  PetscErrorCode (*e_newIteration)(struct _dvdDashboard*);
  void *e_newIteration_data;

  dvdFunctionList
    *startList,     /* starting list */
    *endList,       /* ending list */
    *destroyList;   /* destructor list */

  Mat H, G;         /* Projected problem matrices */
  Mat auxM;         /* auxiliary dense matrix */
  PetscInt size_MT; /* rows in MT */

  PetscInt
    V_tra_s,
    V_tra_e,        /* cX <- [cX V*MT(0:V_tra_s-1)], V <- V*MT(V_tra_s:V_tra_e) */
    V_new_s,
    V_new_e;        /* added to V the columns V_new_s:V_new_e */
  PetscBool
    BV_shift,       /* if true BV is shifted when vectors converge */
    W_shift;        /* if true W is shifted when vectors converge */

  void* prof_data;  /* profiler data */
} dvdDashboard;

#undef __FUNCT__
#define __FUNCT__ "EPSDavidsonFLAdd"
PETSC_STATIC_INLINE PetscErrorCode EPSDavidsonFLAdd(dvdFunctionList **fl,dvdCallback f)
{
  PetscErrorCode ierr;
  dvdFunctionList *l;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(dvdFunctionList),&l);CHKERRQ(ierr);
  l->f = f;
  l->next = *fl;
  *fl = l;
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "EPSDavidsonFLCall"
PETSC_STATIC_INLINE PetscErrorCode EPSDavidsonFLCall(dvdFunctionList *fl,dvdDashboard *d)
{
  PetscErrorCode ierr;
  dvdFunctionList *l;

  PetscFunctionBegin;
  for (l=fl; l; l=l->next) {ierr = (l->f)(d);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDavidsonFLDestroy"
PETSC_STATIC_INLINE PetscErrorCode EPSDavidsonFLDestroy(dvdFunctionList **fl)
{
  PetscErrorCode ierr;
  dvdFunctionList *l,*l0;

  PetscFunctionBegin;
  for (l=*fl; l; l=l0) {
    l0 = l->next;
    ierr = PetscFree(l);CHKERRQ(ierr);
  }
  *fl = NULL;
  PetscFunctionReturn(0);
}

/*
  The blackboard configuration structure: saves information about the memory
  and other requirements.

  The starting memory structure:

  V           W?          AV          BV?          tKZ
  |-----------|-----------|-----------|------------|-------|
  nev+mpd     nev+mpd     scP+mpd     nev?+mpd     sP+scP
              scP+mpd                 scP+mpd

  The final memory structure considering W_shift and BV_shift:

  cX  V       cY?  W?     cAV AV      BcX? BV?     KZ  tKZ
  |---|-------|----|------|---|-------|----|-------|---|---|
  nev mpd     nev  mpd    scP mpd     nev  mpd     scP sP    <- shift
              scP                     scP                    <- !shift
*/
typedef struct {
  PetscInt max_size_V,  /* max size of the searching subspace (mpd) */
    max_size_X,         /* max size of X (bs) */
    size_V,             /* real size of V (nev+size_P+mpd) */
    max_size_oldX,      /* max size of oldX */
    max_nev,            /* max number of converged pairs */
    max_size_P,         /* number of computed vectors for the projector */
    max_size_cP,        /* number of converged vectors in the projectors */
    max_size_proj,      /* max size projected problem */
    max_size_cX_proj;   /* max converged vectors in the projected problem */
  PetscInt state;       /* method states:
                            0: preconfiguring
                            1: configuring
                            2: running
                        */
} dvdBlackboard;

#define DVD_STATE_PRECONF 0
#define DVD_STATE_CONF 1
#define DVD_STATE_RUN 2

/* Shared types */
typedef PetscErrorCode (*dvdPrecond)(dvdDashboard*,PetscInt i,Vec x,Vec Px);
typedef PetscErrorCode (*e_Vchanged_type)(dvdDashboard*,PetscInt s_imm,PetscInt e_imm,PetscInt s_new,PetscInt e_new);
typedef PetscErrorCode (*isRestarting_type)(dvdDashboard*,PetscBool*);
typedef PetscErrorCode (*e_newIteration_type)(dvdDashboard*);
typedef PetscErrorCode (*improveX_type)(dvdDashboard*,Vec *D,PetscInt max_size_D,PetscInt r_s,PetscInt r_e,PetscInt *size_D);

/* Routines for initV step */
PETSC_INTERN PetscErrorCode dvd_initV(dvdDashboard *d,dvdBlackboard *b,PetscInt k,PetscInt user,PetscBool krylov);

/* Routines for calcPairs step */
PETSC_INTERN PetscErrorCode dvd_calcpairs_qz(dvdDashboard *d,dvdBlackboard *b,PetscBool orth,PetscInt cX_proj,PetscBool harm);
PETSC_INTERN PetscErrorCode dvd_harm_updateproj(dvdDashboard *d);

/* Routines for improveX step */
PETSC_INTERN PetscErrorCode dvd_improvex_jd(dvdDashboard *d,dvdBlackboard *b,KSP ksp,PetscInt max_bs,PetscInt cX_impr,PetscBool dynamic);
PETSC_INTERN PetscErrorCode dvd_improvex_jd_proj_uv(dvdDashboard *d,dvdBlackboard *b,ProjType_t p);
PETSC_INTERN PetscErrorCode dvd_improvex_jd_lit_const(dvdDashboard *d,dvdBlackboard *b,PetscInt maxits,PetscReal tol,PetscReal fix);
PETSC_INTERN PetscErrorCode dvd_improvex_gd2(dvdDashboard *d,dvdBlackboard *b,KSP ksp,PetscInt max_bs);

/* Routines for testConv step */
PETSC_INTERN PetscErrorCode dvd_testconv_basic(dvdDashboard *d,dvdBlackboard *b);
PETSC_INTERN PetscErrorCode dvd_testconv_slepc(dvdDashboard *d,dvdBlackboard *b);

/* Routines for management of V */
PETSC_INTERN PetscErrorCode dvd_managementV_basic(dvdDashboard *d,dvdBlackboard *b,PetscInt bs,PetscInt mpd,PetscInt min_size_V,PetscInt plusk,PetscBool harm,PetscBool allResiduals);

/* Some utilities */
PETSC_INTERN PetscErrorCode dvd_static_precond_PC(dvdDashboard *d,dvdBlackboard *b,PC pc);
PETSC_INTERN PetscErrorCode dvd_jacobi_precond(dvdDashboard *d,dvdBlackboard *b);
PETSC_INTERN PetscErrorCode dvd_profiler(dvdDashboard *d,dvdBlackboard *b);
PETSC_INTERN PetscErrorCode dvd_prof_init();
PETSC_INTERN PetscErrorCode dvd_harm_conf(dvdDashboard *d,dvdBlackboard *b,HarmType_t mode,PetscBool fixedTarget,PetscScalar t);
PETSC_INTERN PetscErrorCode BVMultS(BV X,BV Y,PetscScalar *H,PetscInt ldh);
PETSC_INTERN PetscErrorCode SlepcMatDenseCopy(Mat,PetscInt,PetscInt,Mat,PetscInt,PetscInt,PetscInt,PetscInt);

/* Methods */
PETSC_INTERN PetscErrorCode dvd_schm_basic_preconf(dvdDashboard *d,dvdBlackboard *b,PetscInt mpd,PetscInt min_size_V,PetscInt bs,PetscInt ini_size_V,PetscInt size_initV,PetscInt plusk,HarmType_t harmMode,KSP ksp,InitType_t init,PetscBool allResiduals,PetscBool orth,PetscInt cX_proj,PetscInt cX_impr,Method_t method);
PETSC_INTERN PetscErrorCode dvd_schm_basic_conf(dvdDashboard *d,dvdBlackboard *b,PetscInt mpd,PetscInt min_size_V,PetscInt bs,PetscInt ini_size_V,PetscInt size_initV,PetscInt plusk,HarmType_t harmMode,PetscBool fixedTarget,PetscScalar t,KSP ksp,PetscReal fix,InitType_t init,PetscBool allResiduals,PetscBool orth,PetscInt cX_proj,PetscInt cX_impr,PetscBool dynamic,Method_t method);

/* Orthogonalization routines */
PETSC_INTERN PetscErrorCode dvd_orthV(BV bv,PetscInt V_new_s,PetscInt V_new_e,PetscRandom rand);

/* SLEPc interface routines */
PETSC_INTERN PetscErrorCode SLEPcNotImplemented();
PETSC_INTERN PetscErrorCode EPSCreate_XD(EPS eps);
PETSC_INTERN PetscErrorCode EPSReset_XD(EPS eps);
PETSC_INTERN PetscErrorCode EPSSetUp_XD(EPS eps);
PETSC_INTERN PetscErrorCode EPSSolve_XD(EPS eps);
PETSC_INTERN PetscErrorCode EPSComputeVectors_XD(EPS eps);
PETSC_INTERN PetscErrorCode EPSXDSetKrylovStart_XD(EPS eps,PetscBool krylovstart);
PETSC_INTERN PetscErrorCode EPSXDGetKrylovStart_XD(EPS eps,PetscBool *krylovstart);
PETSC_INTERN PetscErrorCode EPSXDSetBlockSize_XD(EPS eps,PetscInt blocksize);
PETSC_INTERN PetscErrorCode EPSXDGetBlockSize_XD(EPS eps,PetscInt *blocksize);
PETSC_INTERN PetscErrorCode EPSXDSetRestart_XD(EPS eps,PetscInt minv,PetscInt plusk);
PETSC_INTERN PetscErrorCode EPSXDGetRestart_XD(EPS eps,PetscInt *minv,PetscInt *plusk);
PETSC_INTERN PetscErrorCode EPSXDGetInitialSize_XD(EPS eps,PetscInt *initialsize);
PETSC_INTERN PetscErrorCode EPSXDSetInitialSize_XD(EPS eps,PetscInt initialsize);
PETSC_INTERN PetscErrorCode EPSXDGetFix_XD(EPS eps,PetscReal *fix);
PETSC_INTERN PetscErrorCode EPSJDSetFix_JD(EPS eps,PetscReal fix);
PETSC_INTERN PetscErrorCode EPSXDSetBOrth_XD(EPS eps,PetscBool borth);
PETSC_INTERN PetscErrorCode EPSXDGetBOrth_XD(EPS eps,PetscBool *borth);
PETSC_INTERN PetscErrorCode EPSJDSetConstCorrectionTol_JD(EPS eps,PetscBool constant);
PETSC_INTERN PetscErrorCode EPSJDGetConstCorrectionTol_JD(EPS eps,PetscBool *constant);
PETSC_INTERN PetscErrorCode EPSXDSetWindowSizes_XD(EPS eps,PetscInt pwindow,PetscInt qwindow);
PETSC_INTERN PetscErrorCode EPSXDGetWindowSizes_XD(EPS eps,PetscInt *pwindow,PetscInt *qwindow);
PETSC_INTERN PetscErrorCode EPSXDSetMethod(EPS eps,Method_t method);
PETSC_INTERN PetscErrorCode EPSXDGetMethod_XD(EPS eps,Method_t *method);

/* Common inline function */
#undef __FUNCT__
#define __FUNCT__ "dvd_improvex_compute_X"
PETSC_STATIC_INLINE PetscErrorCode dvd_improvex_compute_X(dvdDashboard *d,PetscInt i_s,PetscInt i_e,Vec *u_,PetscScalar *pX,PetscInt ld)
{
  PetscErrorCode  ierr;
  PetscInt        n = i_e - i_s,i;
  Vec             *u;

  PetscFunctionBegin;
  if (u_) {
    u = u_;
  } else if (d->correctXnorm) {
    ierr = SlepcVecPoolGetVecs(d->auxV,i_e-i_s,&u);CHKERRQ(ierr);
  }
  if (u_ || d->correctXnorm) {
    for (i=0; i<n; i++) {
      ierr = BVMultVec(d->eps->V,1.0,0.0,u[i],&pX[ld*(i+i_s)]);CHKERRQ(ierr);
    }
  }
  /* nX(i) <- ||X(i)|| */
  if (d->correctXnorm) {
    for (i=0; i<n; i++) {
      ierr = VecNormBegin(u[i],NORM_2,&d->nX[i_s+i]);CHKERRQ(ierr);
    }
    for (i=0; i<n; i++) {
      ierr = VecNormEnd(u[i],NORM_2,&d->nX[i_s+i]);CHKERRQ(ierr);
    }
#if !defined(PETSC_USE_COMPLEX)
    for (i=0;i<n;i++) {
      if (d->eigi[i_s+i] != 0.0) {
        d->nX[i_s+i] = d->nX[i_s+i+1] = PetscSqrtScalar(d->nX[i_s+i]*d->nX[i_s+i]+d->nX[i_s+i+1]*d->nX[i_s+i+1]);
        i++;
      }
    }
#endif
  } else {
    for (i=0; i<n; i++) d->nX[i_s+i] = 1.0;
  }
  if (d->correctXnorm && !u_) {
    ierr = SlepcVecPoolRestoreVecs(d->auxV,i_e-i_s,&u);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#define _Ceil(A,B) ((A)/(B)+((A)%(B)==0?0:1))
#define FromIntToScalar(S) ((PetscInt)_Ceil((S)*sizeof(PetscBLASInt),sizeof(PetscScalar)))
#define FromRealToScalar(S) ((PetscInt)_Ceil((S)*sizeof(PetscReal),sizeof(PetscScalar)))
