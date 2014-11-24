package com.spbsu.ml.methods.greedyRegion.hierarchicalCherryRegion;

import com.spbsu.commons.func.AdditiveStatistics;

import java.util.BitSet;

/**
 * Created by noxoomo on 24/11/14.
 */
public class  RegionLayer<T extends AdditiveStatistics> {
  final T inside;
  final BitSet conditions;
  final double information;
  final double score;
  final int[] insidePoints;
  final int[] outsidePoints;
  boolean used = false;


  public RegionLayer(T inside, BitSet conditions, double information, double score, int[] insidePoints, int[] outsidePoints) {
    this.inside = inside;
    this.conditions = conditions;
    this.information = information;
    this.score = score;
    this.insidePoints = insidePoints;
    this.outsidePoints = outsidePoints;
  }


}


