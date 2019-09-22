#include <stdio.h>
#include <iostream>
#include <cstring>
#include <time.h>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <set>
#include <stack>
#include <thread>
#include <math.h>
#include "boxMuller.h"
#include "main.h"

typedef struct {
  std::vector<int> positive;
  std::vector<int> negative;
} NeedIrs;

typedef struct {
  int positive;
  int negative;
  int timeStep;
}Irs;

void perturbBalancePart(double* balances, int start, int end) {
  for(int i = start; i < end; i++){
    balances[i] += generateGaussianNoise(0,1);
  }
}

class Run {
  double* balances;
  int* defaults;
  int noSteps, noNodes, threshold, tenure, irsValue;
  int time;
  std::default_random_engine rng;
  std::vector<Irs>* irss;

  public:
    Run(int noSteps, int noNodes, int threshold, int tenure, int irsValue) {
      this->noNodes = noNodes;
      this->threshold = threshold;
      this->tenure = tenure;
      this->irsValue = irsValue;
      this->balances = new double[noNodes];
      this->defaults = new int[noNodes];
      this->time = 0;
      this->rng = std::default_random_engine {};
      this->irss = new std::vector<Irs>();
    }

    ~Run(){
      delete[] this->balances;
      delete[] this->defaults;
      delete this->irss;
    }

    double* getBalances() {
      return this->balances;
    }

    int* getDefaults() {
      return this->defaults;
    }

    bool step() {
      this->perturbBalances();
      this->checkIrsTenures();
      this->createIrss();
      this->checkDefaults();
      this->time++;
    }

  private:
    void perturbBalances() {
      /*
      int i;
      int noThreads = 5;
      std::thread threads[5];

      int noBalancesPerThread = int(std::round(this->noNodes/noThreads));

      for (i = 0; i < noThreads; i++) {
        int start = i*noBalancesPerThread;
        int end = std::min(start+noBalancesPerThread, this->noNodes);
        threads[i] = std::thread(perturbBalancePart, this->balances,start,end);;
      }

      for (i = 0; i < noThreads; i++) {
        threads[i].join();
      }
      */
      for(int i = 0; i < this->noNodes; i++) {
        balances[i] += generateGaussianNoise(0,1);
      }
    }

    void createIrss(){
      NeedIrs currentMarket = this->checkIrsNeeded();

      int noIrsCreation = 0;
      if(currentMarket.positive.size() > 0 && currentMarket.negative.size() > 0){
        std::shuffle(currentMarket.positive.begin(),currentMarket.positive.end(),this->rng);
        std::shuffle(currentMarket.negative.begin(),currentMarket.negative.end(),this->rng);

        noIrsCreation = std::min(currentMarket.negative.size(),currentMarket.positive.size());

        for(int i = 0; i < noIrsCreation; i++){
          this->irss->push_back({currentMarket.positive[i], currentMarket.negative[i], this->time});
          this->balances[currentMarket.positive[i]] -= this->irsValue;
          this->balances[currentMarket.negative[i]] += this->irsValue;
        }
      }
    }

    NeedIrs checkIrsNeeded() {
      NeedIrs res;
      res.positive = std::vector<int>();
      res.negative = std::vector<int>();

      for(int i = 0; i < this->noNodes; i++) {
        if(this->balances[i] > this->irsValue) res.positive.push_back(i);
        if(this->balances[i] < -1 * this->irsValue) res.negative.push_back(i);
      }

      return res;
    }

    void checkIrsTenures() {
      std::vector<int> toRemove;

      for(int i = 0; i < this->irss->size(); i++){
        Irs irs = this->irss->at(i);
        if(irs.timeStep + this->tenure < this->time) {
          // Destory irs
          this->balances[irs.positive] += this->irsValue;
          this->balances[irs.negative] -= this->irsValue;
          toRemove.insert(toRemove.begin(), i);
        }
      }

      for(int i = 0; i < toRemove.size(); i++){
        this->irss->erase(this->irss->begin()+toRemove[i]);
      }
    }

    void checkDefaults() {
      for(int i = 0; i < this->noNodes; i++) {
        if(this->balances[i] > this->irsValue) {
          this->defaultNode(i);
        }
      }
    }

    void defaultNode(int node) {
      #ifdef DEBUG
      std::cout << "Defaulting "<< node << std::endl;
      #endif

      int noDefaulted = 0;
      std::stack<int> defaulting;
      std::set<int> defaultingSet;
      defaulting.push(node);
      defaultingSet.insert(node);

      while(!defaulting.empty()){
        noDefaulted++;
        std::set<int> affected;
        std::vector<int> irssToRemove;
        node = defaulting.top();
        defaulting.pop();

        for(int i = 0; i < this->irss->size(); i++){
          Irs irs = this->irss->at(i);
          if(irs.positive == node && affected.count(irs.negative) == 0) {
            irssToRemove.insert(irssToRemove.begin(), i);
            affected.insert(irs.negative);
          } else if(irs.negative == node && affected.count(irs.positive) == 0) {
            irssToRemove.insert(irssToRemove.begin(), i);
            affected.insert(irs.positive);
          }
        }

        #ifdef DEBUG
        std::cout << "Affected "<< affected.size() << " with " << irssToRemove.size() << std::endl;
        #endif

        for(int i = 0; i<irssToRemove.size(); i++){
          int irsIndex = irssToRemove[i];
          Irs irs = this->irss->at(irsIndex);

          this->balances[irs.positive] += this->irsValue;
          this->balances[irs.negative] -= this->irsValue;
        }

        for(int i = 0; i<irssToRemove.size(); i++){
          int irsIndex = irssToRemove[i];
          this->irss->erase(this->irss->begin() + irsIndex);
        }

        std::set<int>::iterator it;
        for (it = affected.begin(); it != affected.end(); it++) {
          int affectedNode = *it;
          if((this->balances[affectedNode] > this->threshold
            || this->balances[affectedNode] < -1*this->threshold)
            && defaultingSet.count(affectedNode) == 0) {
              #ifdef DEBUG
              std::cout << "Affected node"<< affectedNode << " defaulted with balance " << this->balances[affectedNode]<< std::endl;
              #endif

              defaulting.push(affectedNode);
              defaultingSet.insert(affectedNode);
          }
        }

        this->balances[node] = 0;
      }

      this->defaults[noDefaulted-1] += 1;
    }
};

int main()
{
  std::clock_t start;
  start = std::clock();
  srand(time(NULL));
  int timeSteps = 100000;
  int noNodes= 5000;
  int threshold = 10;
  int irsValue = 4;
  int tenure = 400;

  Run* run = new Run(timeSteps , noNodes, threshold, tenure, irsValue);
  for(int i = 0; i < timeSteps; i++) {
    run->step();
  }

  int* defaults = run->getDefaults();
  for(int i = 0; i < noNodes; i++) {
    std::cout << "'"<<i+1<<"':"<< defaults[i]<<std::endl;
  }

  delete run;
  std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
  std::cout << "Done\n";

  return 0;
}
