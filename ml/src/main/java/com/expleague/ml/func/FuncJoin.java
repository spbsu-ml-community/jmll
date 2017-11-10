package com.expleague.ml.func;

import com.expleague.commons.math.Func;
import com.expleague.commons.util.ArrayTools;

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
    return ArrayTools.map(this.dirs, Func.class, argument -> (Func)argument);
  }
}
