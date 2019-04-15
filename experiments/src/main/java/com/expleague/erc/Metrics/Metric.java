package com.expleague.erc.Metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Model;

import java.util.List;

public interface Metric {
    double calculate(List<Event> events, Model.Applicable applicable);
}
