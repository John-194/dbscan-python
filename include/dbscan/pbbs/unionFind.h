#ifndef UNION_FIND_H
#define UNION_FIND_H

//adapted from PBBS
#include <atomic>
#include "utils.h"
#include "parallel.h"

struct unionFind {

  std::atomic<intT> *parents;
  std::atomic<intT> *hooks;

  unionFind(intT n) {
    parents = newA(std::atomic<intT>, n);
    parallel_for (0, n, [&](intT i) {parents[i].store(intMax(), std::memory_order_relaxed);});
    hooks = newA(std::atomic<intT>, n);
    parallel_for (0, n, [&](intT i) {hooks[i].store(intMax(), std::memory_order_relaxed);});
  }

  void del() {
    free(parents);
    free(hooks);
  }

  inline intT find(intT i) {
    intT j = i;
    if (parents[j].load(std::memory_order_acquire) == intMax()) return j;
    do j = parents[j].load(std::memory_order_relaxed);
    while (parents[j].load(std::memory_order_relaxed) < intMax());
    intT tmp;
    while((tmp=parents[i].load(std::memory_order_relaxed))<j){
      parents[i].store(j, std::memory_order_relaxed); i=tmp;}
    return j;
  }

  void link(intT u, intT v) {
    while(1){
      u = find(u);
      v = find(v);
      if(u == v) break;
      if(u > v) swap(u,v);
      intT expected = intMax();
      if(hooks[u].load(std::memory_order_relaxed) == intMax() &&
         hooks[u].compare_exchange_strong(expected, u, std::memory_order_relaxed, std::memory_order_relaxed)){
        parents[u].store(v, std::memory_order_release);
        break;
      }}
  }
};

struct edgeUnionFind {
  typedef pair<intT, intT> edgeT;
  intT *parents;
  edgeT *hooks;
  intT n;

edgeUnionFind(intT nn): n(nn) {
  parents = newA(intT, n);
  parallel_for (0, n, [&](intT i) {parents[i] = intMax();});
  hooks = newA(edgeT, n);
  parallel_for (0, n, [&](intT i) {
      hooks[i] = make_pair(intMax(), intMax());});
}

  void del() {free(parents);}

  inline intT find(intT i) {
    intT j = i;
    if (parents[j] == intMax()) return j;
    do j = parents[j];
    while (parents[j] < intMax());
    intT tmp;
    while((tmp=parents[i])<j) {parents[i]=j; i=tmp;} 
    return j;
  }

  intT link(intT u, intT v) {
    intT c_from = u;
    intT c_to = v;
    while(1) {
      u = find(u);
      v = find(v);
      if(u == v) break;
      if(u > v) swap(u,v);
      // if(hooks[u].first == intMax() && __sync_bool_compare_and_swap(&hooks[u].first, intMax(), c_from)){
      if(hooks[u].first == intMax() && utils::myCAS(&hooks[u].first, intMax(), c_from)){
        parents[u]=v;
        hooks[u].second=c_to;
        break;
      }
    }
    return parents[u];
  }

  edgeT getEdge(intT idx) {
    return hooks[idx];
  }

};
  

#endif

