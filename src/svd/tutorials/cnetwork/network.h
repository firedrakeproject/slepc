/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Utilities for loading a complex network file and represent it as a graph
*/

#pragma once

#include <slepcsys.h>

typedef enum { GRAPH_UNDIRECTED,
               GRAPH_DIRECTED,
               GRAPH_BIPARTITE } GraphType;
SLEPC_EXTERN const char *GraphTypes[];

typedef enum { GRAPH_WEIGHT_UNWEIGHTED,
               GRAPH_WEIGHT_POSITIVE,
               GRAPH_WEIGHT_POSWEIGHTED,
               GRAPH_WEIGHT_SIGNED,
               GRAPH_WEIGHT_MULTISIGNED,
               GRAPH_WEIGHT_WEIGHTED,
               GRAPH_WEIGHT_MULTIWEIGHTED,
               GRAPH_WEIGHT_DYNAMIC,
               GRAPH_WEIGHT_MULTIPOSWEIGHTED } GraphWeight;
SLEPC_EXTERN const char *GraphWeights[];

struct _n_Graph {
  MPI_Comm      comm;
  GraphType     type;
  GraphWeight   weight;
  PetscInt      nvertices;
  PetscInt      nedges;
  Mat           adjacency;
};
typedef struct _n_Graph* Graph;

SLEPC_EXTERN PetscErrorCode GraphCreate(MPI_Comm,Graph*);
SLEPC_EXTERN PetscErrorCode GraphDestroy(Graph*);
SLEPC_EXTERN PetscErrorCode GraphPreload(Graph,char*);
SLEPC_EXTERN PetscErrorCode GraphPreallocate(Graph,char*);
SLEPC_EXTERN PetscErrorCode GraphLoadUnweighted(Graph,char*);
