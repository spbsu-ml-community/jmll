package com.spbsu.ml.methods.greedyRegion.hierarchicalCherryRegion;

import com.spbsu.commons.func.AdditiveStatistics;

/**
 * Created by noxoomo on 24/11/14.
 */
public class  RegionStats<T extends AdditiveStatistics> {
  final T inside;
  final int feature;
  final int bin;
  final double information;
  final double score;
  public  RegionLayer basedOn;



  public RegionStats(T inside, int feature, int bin, double information, double score) {
    this.inside = inside;
    this.feature = feature;
    this.bin = bin;
    this.information = information;
    this.score = score;
  }

}