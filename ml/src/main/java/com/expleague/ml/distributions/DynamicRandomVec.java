package com.expleague.ml.distributions;

import com.expleague.commons.func.Action;


public interface DynamicRandomVec<U extends RandomVariable<U>> extends RandomVec<U> {
  Action<U> updater();
}
