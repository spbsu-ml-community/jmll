package com.expleague.ml;

import com.expleague.commons.func.Action;
import com.expleague.commons.math.Trans;

/**
 * User: solar
 * Date: 22.12.2010
 * Time: 17:17:41
 */
public interface ProgressHandler extends Action<Trans> {
  @Override
  void invoke(Trans partial);
}
