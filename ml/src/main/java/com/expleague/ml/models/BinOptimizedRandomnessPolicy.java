package com.expleague.ml.models;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.randomnessAware.ProcessRandomnessPolicy;

/**
 * Created by noxoomo on 06/11/2017.
 */

public enum BinOptimizedRandomnessPolicy implements ProcessRandomnessPolicy {
  SampleBin,
  BinsExpectation,
  PointEstimateBin
}

