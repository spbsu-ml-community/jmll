package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public abstract class MultiClassModel extends Model {
  public abstract double value(Vec point, int classNo);

  public Vec value(DataSet ds, int classNo) {
    Vec result = new ArrayVec(ds.power());
    DSIterator dsIterator = ds.iterator();
    int i = 0;
    while (dsIterator.advance()){
      result.set(i++, value(dsIterator.x(), classNo));
    }
    return result;
  }
}
