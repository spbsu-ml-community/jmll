package com.expleague.lzy.graph;

import com.expleague.lzy.Operation;

import java.util.Properties;

public interface AtomicOperation extends Operation {
  Properties provisioning();
  Container container();
}
