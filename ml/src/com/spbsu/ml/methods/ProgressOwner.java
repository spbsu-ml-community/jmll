package com.spbsu.ml.methods;

import com.spbsu.ml.Model;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.models.AdditiveModel;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * User: solar
 * Date: 08.08.13
 * Time: 19:01
 */
public class ProgressOwner {
  protected List<WeakReference<ProgressHandler>> progress = new ArrayList<WeakReference<ProgressHandler>>();

  public void addProgressHandler(ProgressHandler handler) {
    this.progress.add(new WeakReference<ProgressHandler>(handler));
  }

  public void processProgress(Model result) {
    final Iterator<WeakReference<ProgressHandler>> progIter = progress.iterator();
    while (progIter.hasNext()) {
      WeakReference<ProgressHandler> next = progIter.next();
      final ProgressHandler progressHandler;
      if ((progressHandler = next.get()) != null) {
        progressHandler.progress(result);
      }
      else progIter.remove();
    }
  }
}
