package com.expleague.erc;

public class Event {
    private final String userId;
    private final String itemId;
    private final double ts;
    private double prDelta;

    public Event(String userId, String itemId, double ts) {
        this.userId = userId;
        this.itemId = itemId;
        this.ts = ts;
        prDelta = -1;
    }

    public Event(String userId, String itemId, double ts, double prDelta) {
        this.userId = userId;
        this.itemId = itemId;
        this.ts = ts;
        this.prDelta = prDelta;
    }

    public String userId() {
        return userId;
    }

    public String itemId() {
        return itemId;
    }

    public double getTs() {
        return ts;
    }

    public double getPrDelta() {
        return prDelta;
    }

    public void setPrDelta(final double prDelta) {
        this.prDelta = prDelta;
    }

    public boolean isFinish() {
        return false;
    }
}
