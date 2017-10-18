package com.expleague.ml.func;

import com.expleague.commons.func.Computable;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public class FuncJoin extends TransJoin {
  public FuncJoin(final Func[] dirs) {
    super(dirs);
  }

  public Func[] dirs() {
    return ArrayTools.map(this.dirs, Func.class, new Computable<Trans, Func>() {
      @Override
      public Func compute(final Trans argument) {
        return (Func)argument;
      }
    });
  }
}
