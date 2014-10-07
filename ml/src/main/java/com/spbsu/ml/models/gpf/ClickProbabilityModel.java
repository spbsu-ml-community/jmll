package com.spbsu.ml.models.gpf;

import java.util.List;

/**
 * Created by irlab on 03.10.2014.
 */
public interface ClickProbabilityModel<Blk extends Session.Block> {
  void trainClickProbability(List<Session<Blk>> dataset);
  double getClickGivenViewProbability(Blk b);
}
