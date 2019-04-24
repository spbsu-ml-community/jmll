package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.models.Model;

import java.util.List;

public interface Metric {
    double calculate(List<Event> events, Model.Applicable applicable);
}
