package com.expleague.crawl.data;

import java.util.concurrent.BlockingQueue;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */
public interface Command extends Message {
  void execute(BlockingQueue<Message> responseQueue);
}
