package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.models.ApplicableModel;

import java.util.List;

public interface Metric {
    double calculate(List<Event> events, ApplicableModel applicable);
}
