package com.expleague.ml.meta;

import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.meta.items.FakeItem;
import com.expleague.ml.meta.items.FocusItem;
import com.expleague.ml.meta.items.QURLItem;
import com.expleague.ml.meta.items.ViewportAnswersWeighting;

import java.util.Date;

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
  Class<?> type();

  enum ItemType {
    FAKE(FakeItem.class),
    QURL(QURLItem.class),
    VIEWPORT_WEIGHTING(ViewportAnswersWeighting.class),
    FOCUS(FocusItem.class)
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
