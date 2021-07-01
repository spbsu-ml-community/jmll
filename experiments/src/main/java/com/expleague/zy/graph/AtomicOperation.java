package com.expleague.zy.graph;

import com.expleague.zy.Operation;

import java.util.Properties;

public interface AtomicOperation extends Operation {
  Properties provisioning();
  Container container();
}
