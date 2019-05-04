package com.expleague.erc;

public class EventSeq {
    protected final int userId;
    protected final int itemId;
    protected final double ts;
    protected double delta;

    public EventSeq(int userId, int itemId, double ts) {
        this.userId = userId;
        this.itemId = itemId;
        this.ts = ts;
        delta = -1;
    }

    public EventSeq(int userId, int itemId, double ts, double delta) {
        this.userId = userId;
        this.itemId = itemId;
        this.ts = ts;
        this.delta = delta;
    }

    public EventSeq(Event event) {
        this(event.userId, event.itemId, event.ts, event.prDelta);
    }

    public int userId() {
        return userId;
    }

    public int itemId() {
        return itemId;
    }

    public double getStartTs() {
        return ts;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(final double delta) {
        this.delta = delta;
    }

    public long getPair() {
        return Util.combineIds(userId, itemId);
    }
}
