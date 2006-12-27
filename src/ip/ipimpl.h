#ifndef _IPIMPL
#define _IPIMPL

#include "slepcip.h"

extern PetscCookie IP_COOKIE;
extern PetscEvent IP_InnerProduct,IP_Orthogonalize;

typedef struct _IPOps *IPOps;

struct _IPOps {
};

struct _p_IP {
  PETSCHEADER(struct _IPOps);
  IPOrthogonalizationType orthog_type; /* which orthogonalization to use */
  IPOrthogonalizationRefinementType orthog_ref;   /* refinement method */
  PetscReal orthog_eta;
  IPBilinearForm bilinear_form;
  int innerproducts;
  Vec w;
};

#endif
