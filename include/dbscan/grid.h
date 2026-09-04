// This code is part of the project "Theoretically Efficient and Practical
// Parallel DBSCAN"
// Copyright (c) 2020 Yiqiu Wang, Yan Gu, Julian Shun
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <atomic>
#include "cell.h"
#include "point.h"
#include "shared.h"
#include "kdTree.h"
#include "kdNode.h"
#include "pbbs/sequence.h"
#include "pbbs/ndHash.h"
#include "pbbs/sampleSort.h"
#include "pbbs/quickSort.h"
#include "pbbs/parallel.h"

//a less comparator based on grid
template<int dim, class pointT, class geoPointT>
inline bool pointGridCmp(pointT p1, pointT p2, geoPointT pMin, floatT r) {
  for(int i=0; i<dim; ++i) {
    intT xx1 = (intT) floor((p1[i]-pMin[i])/r);
    intT xx2 = (intT) floor((p2[i]-pMin[i])/r);
    if (xx1 != xx2) {
      if (xx1 > xx2) return false;
      else return true;}
  }
  return false;
}

/**
  *   A grid class, that puts dim-dimensional axis-aligned box cells on a point set.
  */
template<int dim, class objT>
struct grid {
  typedef grid<dim, objT> gridT;
  typedef double floatT;
  typedef point<dim> geoPointT;
  typedef cell<dim, objT> cellT;
  typedef hashFloatToCell<dim> cellHashT;
  typedef Table<cellHash<dim, objT>,intT> tableT;
  typedef Table<aFloatHash<dim, objT>,intT> objTableT;
  typedef kdTree<dim, cellT> treeT;
// #ifdef USEJEMALLOC
//   typedef vector<cellT*, je_allocator<intT>> cellBuf;
// #else
  typedef vector<cellT*> cellBuf;
  //#endif

  static const bool noRandom = true;

  floatT r;
  geoPointT pMin;
  cellT* cells;
  intT numCells, cellCapacity;
  cellHashT* myHash=NULL;// generic hash function
  tableT* table=NULL;
  treeT* tree=NULL;
  intT totalPoints;
  std::atomic<cellBuf*>* nbrCache;

  /**
  *   Grid constructor.
  *   @param cellMax projected maximum number of points inserted.
  *   @param pMinn global coordinate minimum.
  *   @param r box cell size.
  */
  grid(intT cellMax, geoPointT pMinn, floatT rr):
    r(rr), pMin(pMinn), cellCapacity(cellMax), totalPoints(0) {

    cells = newA(cellT, cellCapacity);
    nbrCache = new std::atomic<cellBuf*>[cellCapacity];
    // nbrCache/cells are initialized lazily in insertParallel over
    // numCells only, so nothing to initialize here.
    numCells = 0;

    myHash = new cellHashT(pMinn, r);
    // Hash table sized for expected number of cells, with safety rebuild
    // in insertParallel if numCells exceeds the estimate.
    intT tableHint = std::max((intT)2048, cellMax / 4);
    table = new tableT(tableHint, cellHash<dim, objT>(myHash));
  }

  ~grid() {
    free(cells);
    // Only [0, numCells) were initialized in insertParallel; reading beyond
    // that would touch uninitialized atomics.
    parallel_for(0, numCells, [&](intT i) {
      auto cached = nbrCache[i].load(std::memory_order_relaxed);
      if(cached) delete cached;
    });
    delete[] nbrCache;
    if(myHash) delete myHash;
    if(table) {
      table->del();
      delete table;}
    if(tree) delete tree;
  }

  inline cellT* getCell(floatT* coord) {
    cellT bait = cellT(geoPointT(coord));
    cellT* found = table->find(&bait);
    return found;}

  inline cellT* getCell(intT i) {
    return &cells[i];}

  inline intT numCell() {return numCells;}

  inline intT size() {
    return totalPoints;
  }

  static constexpr floatT nbrHop() {return 1.0000001 * sqrt((floatT)(4 * dim - 3));}

  inline cellBuf* nbrCacheFor(int idx, cellT* bait) {
    // Acquire ensures vector contents are visible if pointer is non-null
    auto cached = nbrCache[idx].load(std::memory_order_acquire);
    if (cached) return cached;
    auto fStop = [&](){return false;};
    auto fNone = [&](cellT* cell){return false;};
    auto mine = tree->rangeNeighbor(bait, r * nbrHop(), fStop, fNone, true, (cellBuf*)nullptr);
    cellBuf* expected = nullptr;
    if (nbrCache[idx].compare_exchange_strong(expected, mine,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      return mine;
    }
    delete mine;
    return expected;
  }

  template<class func>
  inline void nghPointMap(floatT* center, func& f) {
    auto bait = getCell(center);//center must be there
    if (bait == table->empty || bait < cells || bait >= cells + numCells) {
      cout << "error, nghPointMap mapped to a non-existent point, abort" << endl;
      abort();}
    auto fWrap = [&](cellT* nbr) {
                   if (!nbr->isEmpty()
                       && nbr->actualSize()>0) {
                     for(intT jj=0;jj<nbr->size();++jj) {
                         if(f(nbr->getItem(jj))) return true;
                     }
                   }
                   return false;};//todo, optimize
    int idx = bait - cells;
    for (auto accum_i : *nbrCacheFor(idx, bait)) {
      if(fWrap(accum_i)) break;
    }
  }

  template<class func>
  inline void nghCellMap(cellT* bait, func& f) {
    auto fWrap = [&](cellT* cell){
                   if(!cell->isEmpty())
                     return f(cell);
                   return false;
                 };
    int idx = bait - cells;
    for (auto accum_i : *nbrCacheFor(idx, bait)) {
      if (fWrap(accum_i)) break;
    }
  }

  void insertParallel(objT* P, objT* PP, intT nn, intT* I, intT* flag=NULL) {
    if (nn==0) return;

    bool freeFlag=false;
    if (!flag) {
      flag=newA(intT, nn+1);//todo size
      freeFlag=true;}

    parallel_for(0, nn, [&](intT i){I[i] = i;});

    // Pre-compute integer cell coordinates to avoid floor() in every sort comparison
    auto cellKeys = newA(intT, nn * dim);
    parallel_for(0, nn, [&](intT i) {
      for (int d = 0; d < dim; d++) {
        cellKeys[i * dim + d] = (intT)floor((P[i][d] - pMin[d]) / r);
      }
    });
    auto ipLess = [&] (intT a, intT b) {
                    for (int d = 0; d < dim; d++) {
                      intT ca = cellKeys[a * dim + d];
                      intT cb = cellKeys[b * dim + d];
                      if (ca != cb) return ca < cb;
                    }
                    return false;};
    sampleSort(I, nn, ipLess);
    free(cellKeys);
    parallel_for(0, nn, [&](intT i){PP[i] = P[I[i]];});

    flag[0] = 1;
    parallel_for(1, nn, [&](intT i) {
	if (table->hashStruct.diffCell(PP[i].coordinate(), PP[i-1].coordinate())) {
	  flag[i] = 1;
	} else {
	  flag[i] = 0;}
      });

    numCells = sequence::prefixSum(flag, 0, nn);
    flag[nn] = numCells;

    if (numCells > cellCapacity) {
      cout << "error, grid insert exceeded cell capacity, abort()" << endl;abort();}

    // Rebuild hash table if initial estimate was too small
    if (numCells > cellCapacity / 8) {
      table->del(); delete table;
      table = new tableT(numCells * 2, cellHash<dim, objT>(myHash));
    }

    // Initialize only the cells that will actually be used
    parallel_for(0, numCells, [&](intT i) {
      nbrCache[i].store(nullptr, std::memory_order_relaxed);
      cells[i].init();
    });

    parallel_for(0, nn, [&](intT i) {
	if (flag[i] != flag[i+1]) {
	  auto c = &cells[flag[i]];
	  c->P = &PP[i];
	  c->computeCoord(pMin, r);
	  table->insert(c);
	}
      });
    parallel_for(0, numCells-1, [&](intT i) {
	cells[i].numPoints = cells[i+1].P - cells[i].P;
      });
    cells[numCells-1].numPoints = &PP[nn] - cells[numCells-1].P;

    tree = new treeT(&cells[0], numCells, true);
    if(freeFlag) free(flag);
  }

};
