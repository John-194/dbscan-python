#pragma once

#include <iostream>
#include "dbscan/capi.h"
#include "dbscan/point.h"
#include "dbscan/shared.h"
#include "dbscan/grid.h"
#include "dbscan/coreBccp.h"
// #include "dbscan/pbbs/gettime.h"
#include "dbscan/pbbs/parallel.h"
#include "dbscan/pbbs/sampleSort.h"
#include "dbscan/pbbs/unionFind.h"

// #define VERBOSE

template<int dim>
int DBSCAN(intT n, floatT* PF, double epsilon, intT minPts, bool* coreFlagOut, intT* coreFlag, intT* cluster) {
  if (n <= 0) return 0;
  typedef point<dim> pointT;
  typedef grid<dim, pointT> gridT;
  typedef cell<dim, pointT> cellT;

  point<dim>* PRead = (point<dim>*)PF;

#ifdef VERBOSE
  cout << "Input: " << n << " points, dimension " << dim << endl;
  printScheduler();
  timing tt; tt.start();
  timing t0; t0.start();
#endif

  floatT epsSqr = epsilon*epsilon;
  bool allFinite;
  pointT pMin = pMinParallel(PRead, n, &allFinite);
  if (!allFinite) return DBSCAN_ERR_NONFINITE;

  auto P = newA(pointT, n);
  // parallel_for(0, n, [&](intT i){P[i] = PRead[i];});
  auto G = new gridT(n+1, pMin, epsilon/sqrt(dim));
  auto I = newA(intT, n);
  G->insertParallel(PRead, P, n, I);
#ifdef VERBOSE
  cout << "num-cell = " << G->numCell() << endl;
  cout << "compute-grid = " << t0.next() << endl;
#endif
  //mark core
  parallel_for(0, n, [&](intT i) {coreFlag[i] = -1;});

  auto isCore = [&](pointT *p) {
                  coreFlag[p-P] = 1;
                  return false;
                };

  parallel_for(0, G->numCell(), [&](intT i) {
      cellT* c = G->getCell(i);
      if (c->size() >= minPts) c->pointMap(isCore);
    });

  parallel_for(0, n, [&](intT i) {
                       if (coreFlag[i] < 0) {
                         intT count = 0;
                         auto isCore = [&] (pointT *p) {
                                         if(count >= minPts) return true;
                                         if(p->distSqr(P[i]) <= epsSqr) {
                                           count ++;}
                                         return false;};
                         G->nghPointMap(P[i].coordinate(), isCore);
                         if (count >= minPts) coreFlag[i] = 1;
                         else coreFlag[i] = 0;
                       }
                     });
#ifdef VERBOSE
  cout << "mark-core-time = " << t0.next() << endl;
#endif
  //cluster core
  auto ccFlag = newA(intT, G->numCell());
  parallel_for(0, G->numCell(), [&](intT i) {
      auto ci = G->getCell(i);
      ccFlag[i] = 0;
      auto hasCore = [&](pointT *p) {
                       if (coreFlag[p-P]) {
                         ccFlag[i] = 1;
                         return true;
                       }
                       return false;
                     };
      ci->pointMap(hasCore);
    });

  typedef kdTree<dim, pointT> treeT;
  auto trees = newA(treeT*, G->numCell());

  parallel_for(0, G->numCell(), [&](intT i) {
    if (ccFlag[i]) {
        trees[i] = new treeT(G->getCell(i)->getItem(), G->getCell(i)->size(), false);
    } else {
        trees[i] = NULL;
    }
});

  // auto degCmp = [&](intT i, intT j) {
  //                 return G->getCell(i)->size() < G->getCell(j)->size();
  //               };
  // auto ordering = newA(intT, G->numCell());
  // par_for(intT i=0; i<G->numCell(); ++i) ordering[i] = i;
  //sampleSort(ordering, G->numCell(), degCmp);

  auto uf = unionFind(G->numCell());

  parallel_for(0, G->numCell(), [&](intT i) {
      if (ccFlag[i]) {
        auto procTj = [&](cellT* cj) {
                        intT j = cj - G->getCell(0);
                        if (j < i && ccFlag[j] &&
                            uf.find(i) != uf.find(j)) {
                          if(hasEdge<cellT, treeT, pointT>(i, j, coreFlag, P, epsilon, G->getCell(0), trees)) {
                            uf.link(i, j);
                          }
                        }
                        return false;
                      };
        //G->nghCellMap(G->getCell(ordering[i]), procTj);
        G->nghCellMap(G->getCell(i), procTj);
      }
    });

  parallel_for(0, G->numCell(), [&](intT i) {
      if (trees[i]) delete trees[i];
    });

  parallel_for(0, n, [&](intT i) {cluster[i] = -1;});

  parallel_for(0, G->numCell(), [&](intT i) {
      auto cid = G->getCell(uf.find(i))->getItem() - P;//id of first point
      auto clusterCore = [&](pointT* p){
                           if (coreFlag[p - P])
                             cluster[p - P] = cid;
                           return false;
                         };
      G->getCell(i)->pointMap(clusterCore);
    });
#ifdef VERBOSE
  cout << "cluster-core-time = " << t0.next() << endl;
#endif
  //cluster border to closest core point
  parallel_for(0, n, [&](intT i) {
                       if (!coreFlag[i]) {
                         intT cid = -1;
                         floatT cDistSqr = floatMax();
                         auto closestCore = [&] (pointT* p) {
                                              if (coreFlag[p-P]) {
                                                auto dist = p->distSqr(P[i]);
                                                if (dist <= epsSqr && dist < cDistSqr) {
                                                  cDistSqr = dist;
                                                  cid = cluster[p-P];}
                                              }
                                              return false;};
                         G->nghPointMap(P[i].coordinate(), closestCore);
                         cluster[i] = cid;
                       }
                     });
#ifdef VERBOSE
  cout << "cluster-border-time = " << t0.next() << endl;
  cout << ">> total-clustering-time = " << tt.next() << endl;
#endif
  uf.del();
  free(ccFlag);
  free(trees);
  delete G;

  //improving cluster representation
  // O(n) renumbering: cluster IDs are point indices ∈ [0,n), use prefix sum
  // instead of O(n log n) sort + hash table.
  auto idMap = newA(intT, n);
  parallel_for(0, n, [&](intT i) { idMap[i] = 0; });
  parallel_for(0, n, [&](intT i) {
    if (cluster[i] >= 0) idMap[cluster[i]] = 1;
  });
  sequence::prefixSum(idMap, 0, n); // exclusive: idMap[cid] = sequential ID for cid

  // Remap to sequential IDs (separate buffer to avoid read/write conflict)
  auto cluster2 = newA(intT, n);
  parallel_for(0, n, [&](intT i) {
    cluster2[i] = (cluster[i] >= 0) ? idMap[cluster[i]] : cluster[i];
  });

  //restoring order
  parallel_for(0, n, [&](intT i) {
    cluster[I[i]] = cluster2[i];
    coreFlagOut[I[i]] = coreFlag[i];
  });

  free(I);
  free(cluster2);
  free(idMap);
  free(P);
#ifdef VERBOSE
  cout << "output-time = " << tt.stop() << endl;
#endif
  return 0;
}
