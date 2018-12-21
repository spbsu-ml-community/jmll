package com.expleague.ml.data.tools;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.meta.items.QURLItem;

/**
* User: solar
* Date: 07.07.14
* Time: 20:55
*/
// TODO: Why FeaturesTxtPool duplicates FakePool?
public class FeaturesTxtPool extends FakePool<QURLItem> {
  public FeaturesTxtPool(final Seq<QURLItem> items, final Mx data, final Vec target) {
    super(data, target, (count) -> items);
  }
}
