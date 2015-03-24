package com.spbsu.bernulli;

public interface Learner<Model> {
  FittedModel<Model> fit();
}
