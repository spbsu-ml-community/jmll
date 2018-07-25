package com.expleague.ml.data.tools;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.PoolFeatureMeta;
import com.expleague.ml.meta.TargetMeta;
import com.expleague.ml.meta.items.QURLItem;
import com.expleague.commons.util.Pair;
import com.expleague.ml.Vectorization;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.meta.DataSetMeta;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

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
