/*
  Method: General Davidson Method (includes GD and JD)

  References:
    - Ernest R. Davidson. Super-matrix methods. Computer Physics Communications,
      53:49–60, May 1989.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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


/* 
   Dashboard struct: contains the methods that will be employed and the tunning
   options.
*/

#include <private/epsimpl.h>         /*I "slepceps.h" I*/
#include <private/stimpl.h>          /*I "slepcst.h" I*/
#include <slepcblaslapack.h>

typedef struct _dvdFunctionList {
  PetscErrorCode (*f)(void*);
  void *d;
  struct _dvdFunctionList *next;
} dvdFunctionList;

typedef PetscInt MatType_t;
#define DVD_MAT_HERMITIAN (1<<1)
#define DVD_MAT_NEG_DEF (1<<2)
#define DVD_MAT_POS_DEF (1<<3)
#define DVD_MAT_SINGULAR (1<<4)
#define DVD_MAT_COMPLEX (1<<5)
#define DVD_MAT_IMPLICIT (1<<6)
#define DVD_MAT_IDENTITY (1<<7)
#define DVD_MAT_DIAG (1<<8)
#define DVD_MAT_TRIANG (1<<9)
#define DVD_MAT_UTRIANG (1<<9)
#define DVD_MAT_LTRIANG (1<<10)
#define DVD_MAT_UNITARY (1<<11)

typedef PetscInt EPType_t;
#define DVD_EP_STD (1<<1)
#define DVD_EP_HERMITIAN (1<<2)

#define DVD_IS(T,P) ((T) & (P))
#define DVD_ISNOT(T,P) (((T) & (P)) ^ (P))

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
    DVD_ORTHOV_I,        /* V is orthonormalized */     
    DVD_ORTHOV_B,        /* V is B-orthonormalized */
    DVD_ORTHOV_BOneMV    /* V is B-orthonormalized using IPBOrthogonalize */
} orthoV_type_t;


typedef struct _dvdDashboard {
  /**** Function steps ****/
  /* Initialize V */
  PetscErrorCode (*initV)(struct _dvdDashboard*);
  void *initV_data;
  
  /* Find the approximate eigenpairs from V */
  PetscErrorCode (*calcPairs)(struct _dvdDashboard*);
  void *calcPairs_data;

  /* Eigenpair test for convergence */
  PetscBool (*testConv)(struct _dvdDashboard*, PetscScalar eigvr,
       PetscScalar eigvi, PetscReal res, PetscReal *error);
  void *testConv_data;

  /* Number of converged eigenpairs */
  PetscInt nconv;

  /* Number of pairs ready to converge */
  PetscInt npreconv;

  /* Improve the selected eigenpairs */
  PetscErrorCode (*improveX)(struct _dvdDashboard*, Vec *D, PetscInt max_size_D,
                       PetscInt r_s, PetscInt r_e, PetscInt *size_D);
  void *improveX_data;

  /* Check for restarting */
  PetscBool (*isRestarting)(struct _dvdDashboard*);
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
  Vec *V,           /* searching subspace */
    *real_V,        /* original V */
    *W,             /* testing subspace */
    *real_W,        /* original W */
    *cX,            /* converged right eigenvectors */
    *cY,            /* converged left eigenvectors */
    *BcX,           /* basis of B*cX */
    *AV,            /* A*V */
    *real_AV,       /* original A*V space */
    *BV,            /* B*V */
    *real_BV,       /* original B*V space */
    *BDS;           /* B * eps->DV */
  PetscInt size_V,  /* size of V */
    size_real_V,    /* original size of V */
    size_W,         /* size of W */
    size_real_W,    /* original size of W */
    size_AV,        /* size of AV */
    size_real_AV,   /* original size of AV */
    size_BV,        /* size of BV */
    size_BDS,       /* size of BDS */
    size_real_BV,   /* original size of BV */
    size_cX,        /* size of cX */
    size_cY,        /* size of cY */
    size_D,         /* active vectors */
    size_BcX,       /* size of BcX */
    size_real_eigr, /* size of original eigr, eigi, nR, errest */
    max_size_V,     /* max size of V */
    max_size_W,     /* max size of W */
    max_size_X,     /* max new vectors in V */
    max_size_AV,    /* max size of AV */
    max_size_BV,    /* max size of BV */
    max_size_proj,  /* max size projected problem */
    max_size_cX_in_proj,
                    /* max vectors from cX in the projected problem */
    max_size_cX_in_impr,
                    /* max vectros from cX in the projector */
    max_size_P,     /* max unconverged vectors in the projector */
    bs;             /* max vectors that expands the subspace every iteration */
  EPS eps;          /* Connection to SLEPc */
 
  /**** Auxiliary space ****/
  Vec *auxV;        /* auxiliary vectors */
  PetscScalar
    *auxS;          /* auxiliary scalars */
  PetscInt
    size_auxV,      /* max size of auxV */
    size_auxS;      /* max size of auxS */

  /**** Eigenvalues and errors ****/
  PetscScalar
    *ceigr, *ceigi, /* converged eigenvalues */
    *eigr, *eigi,   /* current eigenvalues */
    *real_eigr,
    *real_eigi;     /* original eigr and eigi */
  PetscReal
    *nR,            /* residual norm */
    *real_nR,       /* original nR */
    *nX,            /* X norm */
    *real_nX,       /* original nX */
    *errest,        /* relative error eigenpairs */
    *real_errest;   /* original errest */

  /**** Shared function and variables ****/
  PetscErrorCode (*e_Vchanged)(struct _dvdDashboard*, PetscInt s_imm,
                         PetscInt e_imm, PetscInt s_new, PetscInt e_new);
  void *e_Vchanged_data;

  PetscErrorCode (*calcpairs_residual)(struct _dvdDashboard*, PetscInt s, PetscInt e,
                                 Vec *R);
  PetscErrorCode (*calcpairs_selectPairs)(struct _dvdDashboard*, PetscInt n);
  void *calcpairs_residual_data;
  PetscErrorCode (*improvex_precond)(struct _dvdDashboard*, PetscInt i, Vec x,
                  Vec Px);
  void *improvex_precond_data;
  PetscErrorCode (*improvex_jd_proj_uv)(struct _dvdDashboard*, PetscInt i_s,
                                        PetscInt i_e, Vec *u, Vec *v,
                                        Vec *kr, Vec *auxV,
                                        PetscScalar *theta,
                                        PetscScalar *thetai,
                                        PetscScalar *pX, PetscScalar *pY,
                                        PetscInt ld);
  PetscErrorCode (*improvex_jd_lit)(struct _dvdDashboard*, PetscInt i,
                                    PetscScalar* theta, PetscScalar* thetai,
                                    PetscInt *maxits, PetscReal *tol);
  PetscErrorCode (*calcpairs_W)(struct _dvdDashboard*);
  void *calcpairs_W_data;
  PetscErrorCode (*calcpairs_proj_trans)(struct _dvdDashboard*);
  PetscErrorCode (*calcpairs_eigs_trans)(struct _dvdDashboard*);
  PetscErrorCode (*calcpairs_proj_res)(struct _dvdDashboard*, PetscInt r_s,
                  PetscInt r_e, Vec *R);
  PetscErrorCode (*preTestConv)(struct _dvdDashboard*, PetscInt s, PetscInt pre,
                                PetscInt e, Vec *auxV, PetscScalar *auxS,
	                              PetscInt *nConv);

  PetscErrorCode (*e_newIteration)(struct _dvdDashboard*);
  void *e_newIteration_data;

  IP ipI;
  IP ipV,           /* orthogonal routine options for V subspace */
    ipW;            /* orthogonal routine options for W subspace */

  dvdFunctionList
    *startList,     /* starting list */
    *endList,       /* ending list */
    *destroyList;   /* destructor list */

  PetscScalar *H,   /* Projected problem matrix A*/
    *real_H,        /* original H */
    *G,             /* Projected problem matrix B*/
    *real_G,        /* original G */
    *pX,            /* projected problem right eigenvectors */
    *pY,            /* projected problem left eigenvectors */
    *MTX,           /* right transformation matrix */
    *MTY,           /* left transformation matrix */
    *S,             /* first Schur matrix, S = pY'*H*pX */
    *T,             /* second Schur matrix, T = pY'*G*pX */
    *cS,            /* first Schur matrix of converged pairs */
    *cT;            /* second Schur matrix of converged pairs */
  MatType_t
    sH,             /* H properties */
    sG;             /* G properties */
  PetscInt ldH,     /* leading dimension of H */
    ldpX,           /* leading dimension of pX */
    ldpY,           /* leading dimension of pY */
    ldMTX,          /* leading dimension of MTX */
    ldMTY,          /* leading dimension of MTY */
    ldS,            /* leading dimension of S */
    ldT,            /* leading dimension of T */
    ldcS,           /* leading dimension of cS */
    ldcT,           /* leading dimension of cT */
    size_H,         /* rows and columns in H */
    size_G,         /* rows and columns in G */
    size_MT,        /* rows in MT */
    size_cS,        /* dimension of cS */
    size_cT,        /* dimension of cT */
    max_size_cS,    /* max size of cS and cT */
    cX_in_V,        /* number of converged vectors in V */
    cX_in_W,        /* number of converged vectors in W */
    cX_in_AV,       /* number of converged vectors in AV */
    cX_in_BV,       /* number of converged vectors in BV */
    cX_in_H,        /* number of converged vectors in H */
    cX_in_G,        /* number of converged vectors in G */
    max_cX_in_proj; /* max number of converged vectors in the proj. probl. */

  PetscInt 
    V_tra_s,
    V_tra_e,        /* cX <- [cX V*MT(0:V_tra_s-1)], V <- V*MT(V_tra_s:V_tra_e) */
    V_new_s,
    V_new_e;        /* added to V the columns V_new_s:V_new_e */
  PetscBool
    BV_shift,       /* if true BV is shifted when vectors converge */
    W_shift;        /* if true W is shifted when vectors converge */

  orthoV_type_t orthoV_type;

  void* prof_data;  /* profiler data */
} dvdDashboard;

/* Add the function fun at the beginning of list */
#define DVD_FL_ADD_BEGIN(list, fun) { \
  dvdFunctionList *fl=(list); \
  PetscErrorCode ierr; \
  ierr = PetscMalloc(sizeof(dvdFunctionList), &(list)); CHKERRQ(ierr); \
  (list)->f = (PetscErrorCode(*)(void*))(fun); \
  (list)->next = fl; }

/* Add the function fun at the end of list */
#define DVD_FL_ADD_END(list, fun) { \
  if ((list)) {DVD_FL_ADD_END0(list, fun);} \
  else {DVD_FL_ADD_BEGIN(list, fun);} }

#define DVD_FL_ADD_END0(list, fun) { \
  dvdFunctionList *fl=(list); \
  PetscErrorCode ierr; \
  for(;fl->next; fl = fl->next); \
  ierr = PetscMalloc(sizeof(dvdFunctionList), &fl->next); CHKERRQ(ierr); \
  fl->next->f = (PetscErrorCode(*)(void*))(fun); \
  fl->next->next = PETSC_NULL; }

#define DVD_FL_ADD(list, fun) DVD_FL_ADD_END(list, fun)

#define DVD_FL_CALL(list, arg0) { \
  dvdFunctionList *fl; \
  for(fl=(list); fl; fl=fl->next) \
    if(*(dvdCallback)fl->f) (*(dvdCallback)fl->f)((arg0)); }

#define DVD_FL_DEL(list) { \
  dvdFunctionList *fl=(list), *oldfl; \
  PetscErrorCode ierr; \
  while(fl) { \
    oldfl = fl; fl = fl->next; ierr = PetscFree(oldfl); CHKERRQ(ierr); } \
  (list) = PETSC_NULL;}

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
    max_size_auxV,      /* max size of auxiliary vecs */
    max_size_auxS,      /* max size of auxiliary scalars */ 
    max_nev,            /* max number of converged pairs */
    max_size_P,         /* number of computed vectors for the projector */
    max_size_cP,        /* number of converged vectors in the projectors */
    max_size_proj,      /* max size projected problem */
    own_vecs,           /* number of global vecs */
    own_scalars;        /* number of local scalars */
  Vec *free_vecs;       /* free global vectors */
  PetscScalar
    *free_scalars;      /* free scalars */
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
typedef PetscErrorCode (*dvdPrecond)(dvdDashboard*, PetscInt i, Vec x, Vec Px);
typedef PetscErrorCode (*dvdCallback)(dvdDashboard*);
typedef PetscErrorCode (*e_Vchanged_type)(dvdDashboard*, PetscInt s_imm,
                         PetscInt e_imm, PetscInt s_new, PetscInt e_new);
typedef PetscBool (*isRestarting_type)(dvdDashboard*);
typedef PetscErrorCode (*e_newIteration_type)(dvdDashboard*);
typedef PetscErrorCode (*improveX_type)(dvdDashboard*, Vec *D, PetscInt max_size_D,
                                  PetscInt r_s, PetscInt r_e, PetscInt *size_D);

/* Structures for blas */
typedef PetscErrorCode (*DvdReductionPostF)(PetscScalar*,PetscInt,void*);
typedef struct {
  PetscScalar
    *out;               /* final vector */
  PetscInt
    size_out;           /* size of out */
  DvdReductionPostF
    f;                  /* function called after the reduction */
  void *ptr;
} DvdReductionChunk;  

typedef struct {
  PetscScalar
    *in,                /* vector to sum-up with more nodes */
    *out;               /* final vector */
  PetscInt size_in,     /* size of in */
    max_size_in;        /* max size of in */
  DvdReductionChunk
    *ops;               /* vector of reduction operations */
  PetscInt
    size_ops,           /* size of ops */
    max_size_ops;       /* max size of ops */
  MPI_Comm comm;        /* MPI communicator */
} DvdReduction;

typedef struct {
  PetscInt i0, i1, i2, ld, s0, e0, s1, e1;
  PetscScalar *M;
} DvdMult_copy_func;

/* Routines for initV step */
PetscErrorCode dvd_initV(dvdDashboard *d, dvdBlackboard *b, PetscInt k,
                         PetscInt user, PetscBool krylov);

/* Routines for calcPairs step */
PetscErrorCode dvd_calcpairs_qz(dvdDashboard *d, dvdBlackboard *b,
                                orthoV_type_t orth, IP ipI,
                                PetscInt cX_proj, PetscBool harm);

/* Routines for improveX step */
PetscErrorCode dvd_improvex_jd(dvdDashboard *d, dvdBlackboard *b, KSP ksp,
                         PetscInt max_bs, PetscInt cX_impr, PetscBool dynamic);
PetscErrorCode dvd_improvex_jd_proj_uv(dvdDashboard *d, dvdBlackboard *b,
                                 ProjType_t p);
PetscErrorCode dvd_improvex_jd_lit_const(dvdDashboard *d, dvdBlackboard *b,
                                   PetscInt maxits, PetscReal tol,
                                   PetscReal fix);

/* Routines for testConv step */
PetscErrorCode dvd_testconv_basic(dvdDashboard *d, dvdBlackboard *b);
PetscErrorCode dvd_testconv_slepc(dvdDashboard *d, dvdBlackboard *b);

/* Routines for management of V */
PetscErrorCode dvd_managementV_basic(dvdDashboard *d, dvdBlackboard *b,
                                     PetscInt bs, PetscInt mpd,
                                     PetscInt min_size_V,
                                     PetscInt plusk, PetscBool harm,
                                     PetscBool allResiduals);

/* Some utilities */
PetscErrorCode dvd_static_precond_PC(dvdDashboard *d, dvdBlackboard *b, PC pc);
PetscErrorCode dvd_jacobi_precond(dvdDashboard *d, dvdBlackboard *b);
PetscErrorCode dvd_profiler(dvdDashboard *d, dvdBlackboard *b);
PetscErrorCode dvd_prof_init();
PetscErrorCode dvd_harm_conf(dvdDashboard *d, dvdBlackboard *b,
                             HarmType_t mode, PetscBool fixedTarget,
                             PetscScalar t);

/* Methods */
PetscErrorCode dvd_schm_basic_preconf(dvdDashboard *d, dvdBlackboard *b,
  PetscInt mpd, PetscInt min_size_V, PetscInt bs,
  PetscInt ini_size_V, PetscInt size_initV, PetscInt plusk,
  HarmType_t harmMode, KSP ksp, InitType_t init, PetscBool allResiduals,
  orthoV_type_t orth, PetscInt cX_proj, PetscInt cX_impr);
PetscErrorCode dvd_schm_basic_conf(dvdDashboard *d, dvdBlackboard *b,
  PetscInt mpd, PetscInt min_size_V, PetscInt bs,
  PetscInt ini_size_V, PetscInt size_initV, PetscInt plusk,
  IP ip, HarmType_t harmMode, PetscBool fixedTarget, PetscScalar t, KSP ksp,
  PetscReal fix, InitType_t init, PetscBool allResiduals, orthoV_type_t orth,
  PetscInt cX_proj, PetscInt cX_impr, PetscBool dynamic);

/* BLAS routines */
PetscErrorCode SlepcDenseMatProd(PetscScalar *C, PetscInt _ldC, PetscScalar b,
  PetscScalar a,
  const PetscScalar *A, PetscInt _ldA, PetscInt rA, PetscInt cA, PetscBool At,
  const PetscScalar *B, PetscInt _ldB, PetscInt rB, PetscInt cB, PetscBool Bt);
PetscErrorCode SlepcDenseMatProdTriang(
  PetscScalar *C, MatType_t sC, PetscInt ldC,
  const PetscScalar *A, MatType_t sA, PetscInt ldA, PetscInt rA, PetscInt cA,
  PetscBool At,
  const PetscScalar *B, MatType_t sB, PetscInt ldB, PetscInt rB, PetscInt cB,
  PetscBool Bt);
PetscErrorCode SlepcDenseNorm(PetscScalar *A, PetscInt ldA, PetscInt _rA,
                              PetscInt cA, PetscScalar *eigi);
PetscErrorCode SlepcDenseOrth(PetscScalar *A, PetscInt _ldA, PetscInt _rA,
                              PetscInt _cA, PetscScalar *auxS, PetscInt _lauxS,
                              PetscInt *ncA);
PetscErrorCode SlepcDenseCopy(PetscScalar *Y, PetscInt ldY, PetscScalar *X,
                              PetscInt ldX, PetscInt rX, PetscInt cX);
PetscErrorCode SlepcDenseCopyTriang(PetscScalar *Y, MatType_t sY, PetscInt ldY,
                                    PetscScalar *X, MatType_t sX, PetscInt ldX,
                                    PetscInt rX, PetscInt cX);
PetscErrorCode SlepcUpdateVectorsZ(Vec *Y, PetscScalar beta, PetscScalar alpha,
  Vec *X, PetscInt cX, const PetscScalar *M, PetscInt ldM, PetscInt rM,
  PetscInt cM);
PetscErrorCode SlepcUpdateVectorsS(Vec *Y, PetscInt dY, PetscScalar beta,
  PetscScalar alpha, Vec *X, PetscInt cX, PetscInt dX, const PetscScalar *M,
  PetscInt ldM, PetscInt rM, PetscInt cM);
PetscErrorCode SlepcUpdateVectorsD(Vec *X, PetscInt cX, PetscScalar alpha,
  const PetscScalar *M, PetscInt ldM, PetscInt rM, PetscInt cM,
  PetscScalar *work, PetscInt lwork);
PetscErrorCode VecsMult(PetscScalar *M, MatType_t sM, PetscInt ldM,
                        Vec *U, PetscInt sU, PetscInt eU,
                        Vec *V, PetscInt sV, PetscInt eV,
                        PetscScalar *workS0, PetscScalar *workS1);
PetscErrorCode VecsMultS(PetscScalar *M, MatType_t sM, PetscInt ldM,
                         Vec *U, PetscInt sU, PetscInt eU,
                         Vec *V, PetscInt sV, PetscInt eV, DvdReduction *r,
                         DvdMult_copy_func *sr);
PetscErrorCode VecsMultIb(PetscScalar *M, MatType_t sM, PetscInt ldM,
                          PetscInt rM, PetscInt cM, PetscScalar *auxS,
                          Vec V);
PetscErrorCode VecsMultIa(PetscScalar *M, MatType_t sM, PetscInt ldM,
                          Vec *U, PetscInt sU, PetscInt eU,
                          Vec *V, PetscInt sV, PetscInt eV);
PetscErrorCode SlepcAllReduceSumBegin(DvdReductionChunk *ops,
                                      PetscInt max_size_ops,
                                      PetscScalar *in, PetscScalar *out,
                                      PetscInt max_size_in, DvdReduction *r,
                                      MPI_Comm comm);
PetscErrorCode SlepcAllReduceSum(DvdReduction *r, PetscInt size_in,
                                 DvdReductionPostF f, void *ptr,
                                 PetscScalar **in);
PetscErrorCode SlepcAllReduceSumEnd(DvdReduction *r);
PetscErrorCode dvd_orthV(IP ip, Vec *DS, PetscInt size_DS, Vec *cX,
                         PetscInt size_cX, Vec *V, PetscInt V_new_s,
                         PetscInt V_new_e, PetscScalar *auxS,
                         PetscRandom rand);
PetscErrorCode dvd_BorthV(IP ip, Vec *DS, Vec *BDS, PetscInt size_DS, Vec *cX,
                         Vec *BcX, PetscInt size_cX, Vec *V, Vec *BV,
                         PetscInt V_new_s, PetscInt V_new_e,
                         PetscScalar *auxS, PetscRandom rand);
PetscErrorCode dvd_compute_eigenvectors(PetscInt n_, PetscScalar *S,
  PetscInt ldS_, PetscScalar *T, PetscInt ldT_, PetscScalar *pX,
  PetscInt ldpX_, PetscScalar *pY, PetscInt ldpY_, PetscScalar *auxS,
  PetscInt size_auxS, PetscBool doProd);
PETSC_EXTERN_CXX_BEGIN
PetscErrorCode EPSSortDenseHEP(EPS eps, PetscInt n, PetscInt k, PetscScalar *w, PetscScalar *V, PetscInt ldV);
PETSC_EXTERN_CXX_END

/* SLEPc interface routines */
PetscErrorCode SLEPcNotImplemented();
PetscErrorCode EPSCreate_Davidson(EPS eps);
PetscErrorCode EPSReset_Davidson(EPS eps);
PetscErrorCode EPSSetUp_Davidson(EPS eps);
PetscErrorCode EPSSolve_Davidson(EPS eps);
PetscErrorCode EPSComputeVectors_Davidson(EPS eps);
PetscErrorCode EPSDavidsonSetKrylovStart_Davidson(EPS eps,PetscBool krylovstart);
PetscErrorCode EPSDavidsonGetKrylovStart_Davidson(EPS eps,PetscBool *krylovstart);
PetscErrorCode EPSDavidsonSetBlockSize_Davidson(EPS eps,PetscInt blocksize);
PetscErrorCode EPSDavidsonGetBlockSize_Davidson(EPS eps,PetscInt *blocksize);
PetscErrorCode EPSDavidsonSetRestart_Davidson(EPS eps,PetscInt minv,PetscInt plusk);
PetscErrorCode EPSDavidsonGetRestart_Davidson(EPS eps,PetscInt *minv,PetscInt *plusk);
PetscErrorCode EPSDavidsonGetInitialSize_Davidson(EPS eps,PetscInt *initialsize);
PetscErrorCode EPSDavidsonSetInitialSize_Davidson(EPS eps,PetscInt initialsize);
PetscErrorCode EPSDavidsonGetFix_Davidson(EPS eps,PetscReal *fix);
PetscErrorCode EPSDavidsonSetFix_Davidson(EPS eps,PetscReal fix);
PetscErrorCode EPSDavidsonSetBOrth_Davidson(EPS eps,PetscBool borth);
PetscErrorCode EPSDavidsonGetBOrth_Davidson(EPS eps,PetscBool *borth);
PetscErrorCode EPSDavidsonSetConstantCorrectionTolerance_Davidson(EPS eps,PetscBool dynamic);
PetscErrorCode EPSDavidsonGetConstantCorrectionTolerance_Davidson(EPS eps,PetscBool *dynamic);
