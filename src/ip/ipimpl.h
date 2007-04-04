#ifndef _IPIMPL
#define _IPIMPL

#include "slepcip.h"

extern PetscCookie IP_COOKIE;
extern PetscEvent IP_InnerProduct,IP_Orthogonalize,IP_ApplyMatrix;

typedef struct _IPOps *IPOps;

struct _IPOps {
  int dummy;
};

struct _p_IP {
  PETSCHEADER(struct _IPOps);
  IPOrthogonalizationType orthog_type; /* which orthogonalization to use */
  IPOrthogonalizationRefinementType orthog_ref;   /* refinement method */
  PetscReal orthog_eta;
  IPBilinearForm bilinear_form;
  Mat matrix;
  Vec work; /* workspace */
  int innerproducts;
};

#endif
