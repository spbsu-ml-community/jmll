package com.expleague.bernulli;

public interface Learner<Model> {
  FittedModel<Model> fit();
}
