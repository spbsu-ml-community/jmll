package com.spbsu.ml.meta;

import java.util.Date;


import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.meta.items.ViewportAnswersWeighting;

/**
 * User: solar
 * Date: 04.07.14
 * Time: 15:11
 */
public interface DataSetMeta {
  String id();
  String source();
  String author();
  Pool owner();
  Date created();
  ItemType type();

  enum ItemType {
    QURL(QURLItem.class),
    VIEWPORT_WEIGHTING(ViewportAnswersWeighting.class)
    ;

    private final Class<? extends DSItem> clazz;

    ItemType(final Class<? extends DSItem> clazz) {
      this.clazz = clazz;
    }

    public Class<? extends DSItem> clazz() {
      return clazz;
    }
  }
}
